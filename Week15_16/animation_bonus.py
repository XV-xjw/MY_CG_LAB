# -*- coding: utf-8 -*-
"""
animation_bonus.py
选做内容：固定 shape 参数，让某一个关节角度从 0 逐渐旋转到目标角度，
生成多帧图片并导出为 GIF，观察权重区域如何随骨骼运动被平滑带动。

用法示例：
  python animation_bonus.py --model-path ./models --joint-name left_elbow \
      --axis 0 1 0 --max-angle 1.4 --num-frames 24 --output-dir outputs/animation
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import smplx
from lbs_core import joint_index, new_3d_ax, plot_mesh, face_colors_from_vertex_values, to_numpy
from run_experiment import manual_lbs, make_demo_betas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, default='./models')
    p.add_argument('--gender', type=str, default='neutral')
    p.add_argument('--num-betas', type=int, default=10)
    p.add_argument('--joint-name', type=str, default='left_elbow')
    p.add_argument('--axis', type=float, nargs=3, default=[0.0, 1.0, 0.0],
                    help="旋转轴方向（轴角表示中的方向向量，会被归一化）")
    p.add_argument('--max-angle', type=float, default=1.4, help="最大旋转角（弧度）")
    p.add_argument('--num-frames', type=int, default=24)
    p.add_argument('--output-dir', type=str, default='outputs/animation')
    p.add_argument('--gif-path', type=str, default='outputs/animation/joint_rotation.gif')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = smplx.create(model_path=args.model_path, model_type='smpl',
                          gender=args.gender, num_betas=args.num_betas, batch_size=1)
    model.eval()
    faces = model.faces

    betas = make_demo_betas(args.num_betas)  # 体型固定不变
    axis = np.array(args.axis, dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    jidx_local = joint_index(args.joint_name) - 1  # body_pose 中不含 root
    weight_col = model.lbs_weights[:, joint_index(args.joint_name)]

    frame_paths = []
    angles = np.linspace(0.0, args.max_angle, args.num_frames)

    for i, angle in enumerate(angles):
        global_orient = torch.zeros(1, 3)
        body_pose = torch.zeros(1, 23 * 3)
        body_pose[0, jidx_local * 3: jidx_local * 3 + 3] = torch.tensor(axis * angle)

        result = manual_lbs(model, betas, global_orient, body_pose)

        fig = plt.figure(figsize=(5, 6))
        ax = new_3d_ax(fig, title=f"{args.joint_name} 旋转角度 = {angle:.2f} rad")
        fc = face_colors_from_vertex_values(weight_col, faces, cmap_name='jet', vmin=0, vmax=1)
        plot_mesh(ax, result['verts'], faces, face_colors=fc)
        fig.tight_layout()

        frame_path = os.path.join(args.output_dir, f"frame_{i:03d}.png")
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)
        print(f"[已保存帧] {frame_path}")

    # 尝试用 Pillow 把所有帧拼成 gif
    try:
        from PIL import Image
        imgs = [Image.open(p) for p in frame_paths]
        imgs[0].save(args.gif_path, save_all=True, append_images=imgs[1:],
                      duration=80, loop=0)
        print(f"[已生成动图] {args.gif_path}")
    except ImportError:
        print("未安装 Pillow，跳过 GIF 合成，可单独 pip install Pillow 后重跑本脚本，"
              "或直接使用 frame_*.png 序列帧。")


if __name__ == '__main__':
    main()
