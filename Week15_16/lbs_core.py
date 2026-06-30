# -*- coding: utf-8 -*-
"""
lbs_core.py
通用工具函数：
- 张量/数组转换
- 网格三维可视化（matplotlib，纯 CPU，无需 OpenGL）
- SMPL 24 个关节的标准名称（便于按名字而不是数字索引指定关节）
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 让 matplotlib 支持中文标题显示（Windows 系统自带这些字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# SMPL 标准 24 个关节顺序（与 smplx 官方实现一致）
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]


def joint_index(name: str) -> int:
    """按关节名找索引，找不到则报错并列出所有可选名称。"""
    if name not in SMPL_JOINT_NAMES:
        raise ValueError(
            f"未知关节名 '{name}'，可选值为：{SMPL_JOINT_NAMES}"
        )
    return SMPL_JOINT_NAMES.index(name)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_plot_coords(x):
    """
    SMPL 坐标系是 Y 轴朝上、Z 轴朝向观察者前方；
    而 matplotlib 3D 默认把 Z 轴当成"朝上"的轴。
    这里把 (x, y, z) 重排成 (x, z, y)，让人体在图里正常站立，
    而不是被画成"躺倒/趴下"的样子。
    """
    arr = to_numpy(x)
    return arr[:, [0, 2, 1]]


def set_axes_equal(ax, vertices):
    """让三维坐标轴等比例显示，避免人体被拉伸变形。"""
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    max_range = np.array(
        [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]
    ).max() / 2.0
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def face_colors_from_vertex_values(values, faces, cmap_name='jet', vmin=None, vmax=None):
    """把顶点上的标量值（例如某关节权重、pose offset 大小）转成每个面片的颜色。"""
    values = to_numpy(values).reshape(-1)
    faces = to_numpy(faces).astype(int)
    if vmin is None:
        vmin = float(values.min())
    if vmax is None:
        vmax = float(values.max())
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    face_vals = values[faces].mean(axis=1)  # 每个面片三个顶点取平均
    return cmap(norm(face_vals))


def face_colors_from_dominant_joint(weights, faces, num_joints, cmap_name='nipy_spectral'):
    """每个顶点取权重最大的关节作为'主导关节'，再给每个面片上色。"""
    weights = to_numpy(weights)
    faces = to_numpy(faces).astype(int)
    dominant = weights.argmax(axis=1)  # (V,)
    strength = weights.max(axis=1)     # 主导权重大小，用来控制明暗
    norm = plt.Normalize(vmin=0, vmax=num_joints - 1)
    cmap = cm.get_cmap(cmap_name)
    base_colors = cmap(norm(dominant))  # (V,4)
    # 用主导权重的大小调节亮度（权重越低颜色越暗，体现"过渡区"）
    base_colors[:, :3] *= np.clip(strength, 0.35, 1.0)[:, None]
    face_colors = base_colors[faces].mean(axis=1)
    return face_colors, dominant


def plot_mesh(ax, vertices, faces, face_colors=None, alpha=1.0,
              edgecolor='none', flat_color=(0.75, 0.75, 0.8, 1.0)):
    vertices = to_plot_coords(vertices)
    faces = to_numpy(faces).astype(int)
    tris = vertices[faces]
    pc = Poly3DCollection(tris, alpha=alpha)
    pc.set_facecolor(face_colors if face_colors is not None else flat_color)
    pc.set_edgecolor(edgecolor)
    ax.add_collection3d(pc)
    set_axes_equal(ax, vertices)


def plot_joints(ax, joints, color='red', size=35, label=None):
    joints = to_plot_coords(joints)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c=color, s=size, label=label)


def new_3d_ax(fig, pos=111, title=''):
    ax = fig.add_subplot(pos, projection='3d')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # 关闭默认的灰色背景墙板，画面更干净
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    # 坐标已在 to_plot_coords 中重排为 (x, z, y)，这里用 matplotlib 默认的
    # z 轴朝上视角即可让人体正常站立、面朝观察者
    ax.view_init(elev=12, azim=-90)
    return ax