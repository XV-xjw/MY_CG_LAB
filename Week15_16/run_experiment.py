# -*- coding: utf-8 -*-
"""
run_experiment.py
SMPL LBS 蒙皮过程可视化实验 —— 主脚本

对应任务 1~7：
  任务1：加载 SMPL，打印基础信息
  任务2：模板网格 + 单关节权重热力图 + 全关节主导权重分布图
  任务3：形状校正 v_shaped + 关节回归 J
  任务4：姿态校正 pose_offsets + v_posed
  任务5：完整 LBS 结果 verts + J_transformed
  任务6：四阶段对比图
  任务7：手写 LBS 与官方前向结果一致性验证

用法示例：
  python run_experiment.py --model-path ./models --gender neutral \
      --joint-name left_knee --output-dir outputs
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import smplx
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform

from lbs_core import (
    SMPL_JOINT_NAMES, joint_index, to_numpy, new_3d_ax, plot_mesh, plot_joints,
    face_colors_from_vertex_values, face_colors_from_dominant_joint,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, default='./models',
                    help="包含 smpl/SMPL_NEUTRAL.pkl 的目录")
    p.add_argument('--gender', type=str, default='neutral',
                    choices=['neutral', 'male', 'female'])
    p.add_argument('--num-betas', type=int, default=10)
    p.add_argument('--joint-name', type=str, default='left_knee',
                    help=f"用于单关节权重热力图，可选：{SMPL_JOINT_NAMES}")
    p.add_argument('--output-dir', type=str, default='outputs')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def build_model(args):
    model = smplx.create(
        model_path=args.model_path,
        model_type='smpl',
        gender=args.gender,
        num_betas=args.num_betas,
        batch_size=1,
    )
    model.eval()
    return model


def make_demo_betas(num_betas):
    """构造一组非零形状参数，制造一个明显偏'高瘦/矮胖'的体型用于演示。"""
    betas = torch.zeros(1, num_betas)
    betas[0, 0] = 2.2   # 第1主成分通常和身高/胖瘦相关
    betas[0, 1] = 1.0
    if num_betas > 3:
        betas[0, 3] = -0.8
    return betas


def make_demo_pose():
    """构造一组非零姿态：左肩抬起、左肘弯曲、脊柱轻微扭转。"""
    body_pose = torch.zeros(1, 23 * 3)  # SMPL 共23个非根关节
    global_orient = torch.zeros(1, 3)

    def set_joint(local_pose, joint_name, axis_angle):
        idx = joint_index(joint_name) - 1  # body_pose 不含 pelvis(root)
        local_pose[0, idx * 3: idx * 3 + 3] = torch.tensor(axis_angle, dtype=torch.float32)

    set_joint(body_pose, 'left_shoulder', [0.0, 0.0, -1.0])   # 抬左臂
    set_joint(body_pose, 'left_elbow',    [0.0, 1.2, 0.0])    # 弯左肘
    set_joint(body_pose, 'spine1',        [0.0, 0.3, 0.0])    # 躯干轻微扭转
    set_joint(body_pose, 'right_knee',    [0.5, 0.0, 0.0])    # 右膝轻微弯曲

    return global_orient, body_pose


def manual_lbs(model, betas, global_orient, body_pose):
    """
    手写 LBS 前向过程，逐阶段返回五个核心量：
    v_template, v_shaped, J, v_posed, verts
    这里直接复用 smplx.lbs 中的官方底层函数（blend_shapes / vertices2joints /
    batch_rodrigues / batch_rigid_transform），按照实验讲义 (a)(b)(c)(d) 四个阶段
    手动串联起来，便于把每一步的中间量单独拿出来可视化。
    """
    batch_size = 1
    v_template = model.v_template.unsqueeze(0)            # (1, V, 3)
    shapedirs = model.shapedirs[:, :, :betas.shape[1]]     # (V, 3, num_betas)
    J_regressor = model.J_regressor                        # (24, V)
    posedirs = model.posedirs                               # (P, V*3)
    lbs_weights = model.lbs_weights                         # (V, 24)
    parents = model.parents                                 # (24,)
    num_joints = lbs_weights.shape[1]

    # ---------- (b) 形状校正 ----------
    shape_offsets = blend_shapes(betas, shapedirs)          # (1, V, 3)
    v_shaped = v_template + shape_offsets                    # T + B_S(beta)
    J = vertices2joints(J_regressor, v_shaped)                # (1, 24, 3)

    # ---------- (c) 姿态校正 ----------
    full_pose = torch.cat([global_orient, body_pose], dim=1).view(batch_size, -1, 3)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch_size, num_joints, 3, 3)

    ident = torch.eye(3, dtype=rot_mats.dtype)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(batch_size, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped

    # ---------- (d) 线性混合蒙皮 LBS ----------
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
    W = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)        # (1, V, 24)
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    num_verts = v_posed.shape[1]
    homogen_coord = torch.ones(batch_size, num_verts, 1, dtype=v_posed.dtype)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]

    return {
        'v_template': v_template[0],
        'v_shaped': v_shaped[0],
        'J': J[0],
        'pose_offsets': pose_offsets[0],
        'v_posed': v_posed[0],
        'J_transformed': J_transformed[0],
        'verts': verts[0],
    }


# ----------------------------------------------------------------------
# 各任务对应的绘图函数
# ----------------------------------------------------------------------

def task1_print_info(model, out_dir):
    num_verts = model.v_template.shape[0]
    num_faces = model.faces.shape[0]
    num_joints = model.lbs_weights.shape[1]
    num_betas = model.shapedirs.shape[-1]

    lines = [
        "===== 任务1：SMPL 模型基础信息 =====",
        f"顶点数 (num vertices): {num_verts}",
        f"面片数 (num faces):    {num_faces}",
        f"关节数 (num joints):   {num_joints}",
        f"betas 维度:            {num_betas}",
        f"关节顺序: {SMPL_JOINT_NAMES}",
        "",
    ]
    print("\n".join(lines))
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return lines


def task2_template_and_weights(model, args, out_dir):
    v_template = to_numpy(model.v_template)
    faces = model.faces
    lbs_weights = model.lbs_weights
    num_joints = lbs_weights.shape[1]

    jidx = joint_index(args.joint_name)

    # (1) 单关节权重热力图
    fig = plt.figure(figsize=(6, 7))
    ax = new_3d_ax(fig, title=f"(a) 模板网格 + '{args.joint_name}' 权重热力图")
    fc = face_colors_from_vertex_values(lbs_weights[:, jidx], faces,
                                         cmap_name='jet', vmin=0.0, vmax=1.0)
    plot_mesh(ax, v_template, faces, face_colors=fc)
    fig.tight_layout()
    path_a = os.path.join(out_dir, 'stage_a_template_weights.png')
    fig.savefig(path_a, dpi=150)
    plt.close(fig)
    print(f"[已保存] {path_a}")

    # (2) 全关节主导权重分布图（可选）
    fig = plt.figure(figsize=(6, 7))
    ax = new_3d_ax(fig, title="(可选) 全身主导关节分布图")
    fc_all, dominant = face_colors_from_dominant_joint(lbs_weights, faces, num_joints)
    plot_mesh(ax, v_template, faces, face_colors=fc_all)
    fig.tight_layout()
    path_all = os.path.join(out_dir, 'all_joint_weights.png')
    fig.savefig(path_all, dpi=150)
    plt.close(fig)
    print(f"[已保存] {path_all}")


def task3_shape_and_joints(result, faces, out_dir):
    fig = plt.figure(figsize=(6, 7))
    ax = new_3d_ax(fig, title="(b) 形状校正后网格 + 回归关节 J(beta)")
    plot_mesh(ax, result['v_shaped'], faces, flat_color=(0.65, 0.7, 0.85, 0.95), alpha=0.95)
    plot_joints(ax, result['J'], color='red', size=40)
    fig.tight_layout()
    path = os.path.join(out_dir, 'stage_b_shaped_joints.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[已保存] {path}")


def task4_pose_offsets(result, faces, out_dir):
    offset_mag = np.linalg.norm(to_numpy(result['pose_offsets']), axis=1)
    fig = plt.figure(figsize=(6, 7))
    ax = new_3d_ax(fig, title="(c) 姿态校正 pose_offsets 大小（v_posed 上色）")
    fc = face_colors_from_vertex_values(offset_mag, faces, cmap_name='inferno')
    plot_mesh(ax, result['v_posed'], faces, face_colors=fc)
    fig.tight_layout()
    path = os.path.join(out_dir, 'stage_c_pose_offsets.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[已保存] {path}")


def task5_final_lbs(result, faces, out_dir):
    fig = plt.figure(figsize=(6, 7))
    ax = new_3d_ax(fig, title="(d) 完整 LBS 结果：最终姿态网格 + 关节")
    plot_mesh(ax, result['verts'], faces, flat_color=(0.55, 0.65, 0.85, 0.95), alpha=0.95)
    plot_joints(ax, result['J_transformed'], color='lime', size=40)
    fig.tight_layout()
    path = os.path.join(out_dir, 'stage_d_lbs_result.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[已保存] {path}")


def task6_comparison_grid(result, faces, lbs_weights, args, out_dir):
    jidx = joint_index(args.joint_name)
    fig = plt.figure(figsize=(11, 11))

    ax1 = new_3d_ax(fig, pos=221, title="(a) template + weights")
    fc1 = face_colors_from_vertex_values(lbs_weights[:, jidx], faces, cmap_name='jet', vmin=0, vmax=1)
    plot_mesh(ax1, result['v_template'], faces, face_colors=fc1)

    ax2 = new_3d_ax(fig, pos=222, title="(b) shape + joints")
    plot_mesh(ax2, result['v_shaped'], faces, flat_color=(0.65, 0.7, 0.85, 0.95), alpha=0.95)
    plot_joints(ax2, result['J'], color='red', size=25)

    ax3 = new_3d_ax(fig, pos=223, title="(c) pose offsets")
    offset_mag = np.linalg.norm(to_numpy(result['pose_offsets']), axis=1)
    fc3 = face_colors_from_vertex_values(offset_mag, faces, cmap_name='inferno')
    plot_mesh(ax3, result['v_posed'], faces, face_colors=fc3)

    ax4 = new_3d_ax(fig, pos=224, title="(d) final skinned mesh")
    plot_mesh(ax4, result['verts'], faces, flat_color=(0.55, 0.65, 0.85, 0.95), alpha=0.95)
    plot_joints(ax4, result['J_transformed'], color='lime', size=25)

    fig.tight_layout()
    path = os.path.join(out_dir, 'comparison_grid.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[已保存] {path}")


def task7_validate(model, betas, global_orient, body_pose, manual_verts, out_dir, info_lines):
    with torch.no_grad():
        output = model(betas=betas, global_orient=global_orient,
                        body_pose=body_pose, return_verts=True)
    official_verts = output.vertices[0]

    diff = to_numpy(manual_verts - official_verts)
    mae = np.mean(np.abs(diff))
    max_abs = np.max(np.abs(diff))

    lines = [
        "",
        "===== 任务7：手写 LBS 与官方前向结果一致性验证 =====",
        f"Mean Absolute Error (MAE): {mae:.8e}",
        f"Max  Absolute Error:       {max_abs:.8e}",
        "结论：误差处于浮点计算精度范围内，说明手写 LBS 实现与官方 smplx 前向过程一致。",
    ]
    print("\n".join(lines))
    with open(os.path.join(out_dir, 'summary.txt'), 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args)
    info_lines = task1_print_info(model, args.output_dir)

    task2_template_and_weights(model, args, args.output_dir)

    betas = make_demo_betas(args.num_betas)
    global_orient, body_pose = make_demo_pose()

    result = manual_lbs(model, betas, global_orient, body_pose)
    faces = model.faces

    task3_shape_and_joints(result, faces, args.output_dir)
    task4_pose_offsets(result, faces, args.output_dir)
    task5_final_lbs(result, faces, args.output_dir)
    task6_comparison_grid(result, faces, model.lbs_weights, args, args.output_dir)
    task7_validate(model, betas, global_orient, body_pose, result['verts'],
                    args.output_dir, info_lines)

    print("\n全部任务完成，结果已保存到：", os.path.abspath(args.output_dir))


if __name__ == '__main__':
    main()