
# 基于 Taichi 的交互式局部光照渲染器

本项目是计算机图形学课程实验作业。基于 **Taichi** 框架实现了两个版本的渲染器，分别展示了经典 **Phong** 模型与进阶 **Blinn-Phong + 阴影** 模型的视觉差异，并探讨了三维空间中的物体布局与遮挡关系。

---

## 📂 项目目录结构

```text
.
├── phong_no_occlusion.py    # 版本 A：Phong 模型 + 展开式无遮挡布局
├── blinn_phong_shadow.py    # 版本 B：Blinn-Phong 模型 + 硬阴影 + 深度遮挡布局
├── README.md                # 项目说明文档
└── .gitignore               # Git 忽略文件
```

---

## 🎯 实验目标

1.  **光照模型对比**：实现并对比经典 Phong 模型（基于反射向量）与 Blinn-Phong 模型（基于半程向量）的渲染效果。
2.  **空间布局与深度测试**：通过调整物体坐标，验证 Z-buffer 深度竞争逻辑在处理物体遮挡（Occlusion）时的正确性。
3.  **高级特性实现**：在 GPU 上实现基于阴影射线（Shadow Ray）的实时硬阴影检测。
4.  **鲁棒性优化**：通过数学修正解决圆锥体底面“视觉倾斜”问题，并消除阴影算法中的自遮挡噪点（Shadow Acne）。

---

## 📐 数学原理总结

### 1. 光照模型演进
* **Phong 模型 (Version A)**：
    使用反射向量 $\mathbf{R}$ 与视线向量 $\mathbf{V}$ 的夹角计算高光。
    $$I_{spec} = K_s (\mathbf{R} \cdot \mathbf{V})^n$$
* **Blinn-Phong 模型 (Version B)**：
    使用法线 $\mathbf{N}$ 与半程向量 $\mathbf{H}$ 的夹角计算。在大角度下比 Phong 模型更自然。
    $$\mathbf{H} = \frac{\mathbf{L} + \mathbf{V}}{\|\mathbf{L} + \mathbf{V}\|}, \quad I_{spec} = K_s (\mathbf{N} \cdot \mathbf{H})^n$$

### 2. 阴影射线 (Shadow Ray)
在物体表面点 $P$ 沿光源方向 $L$ 发射射线。若在距离 $dist(L)$ 内检测到交点，则该点处于阴影。
为防止浮点误差导致物体“自遮挡”，对起点进行微量偏移：
$$P_{start} = P + \mathbf{N} \times 10^{-3}$$

---

## 💻 项目版本演示与关键代码

### 版本 A：Phong 基础版
**特点**：采用等腰三角形分布，三个物体互不遮挡，便于观察单一物体的光照表现。

```python
# 版本 A 关键渲染逻辑：Phong 反射向量计算
R = normalize(reflect(-L, N))
spec = ti.max(0.0, R.dot(V)) ** shininess[None]
color = ambient + diffuse + specular
```

### 版本 B：进阶阴影版
**特点**：采用深度重叠布局，红色球体部分遮挡蓝色圆锥，开启硬阴影以增强空间层次感。

```python
# 版本 B 关键渲染逻辑：Blinn-Phong 半程向量 + 阴影检测
H_vec = normalize(L + V)
spec = ti.max(0.0, N.dot(H_vec)) ** shininess[None]

# 阴影射线求交 (以球体为例)
t_s, _ = intersect_sphere(shadow_ro, shadow_rd, sphere_center, radius)
if 0 < t_s < light_dist: 
    color = ambient # 处于阴影，仅保留环境光
```

---

## 📊 运行结果分析

| 特性 | 版本 A (Phong) | 版本 B (Blinn-Phong + Shadow) |
| :--- | :--- | :--- |
| **高光表现** | 高光斑点较小，在边缘处衰减较快 | 高光区域更自然，符合真实物理分布 |
| **物体布局** | **无遮挡**：红球(-2.0)、蓝锥(0.0)、紫锥(2.0) | **深度遮挡**：红球(-1.2)部分挡住蓝锥(0.0) |
| **深度验证** | 仅验证了单体渲染的正确性 | 成功验证了 Z-buffer 自动处理近距离遮挡的逻辑 |
| **空间感** | 较弱，物体像悬浮在背景上 | **强**，硬阴影清晰交待了物体间的相对位置 |

**圆锥底面修复说明**：
两个版本均集成了底面封盖逻辑。通过在求交函数中增加对 $y = base\_y$ 平面的判定，确保圆锥在任何视角下底部都呈现为水平圆面，解决了单纯使用侧面解析解产生的视觉畸变。

---

## 🛠 环境依赖与运行说明

### 环境要求
* Python 3.8+
* Taichi 1.6.0+

### 安装与运行
1. **安装 Taichi**: `pip install taichi`
2. **运行无遮挡版本**: `python phong_no_occlusion.py`
3. **运行进阶阴影版本**: `python blinn_phong_shadow.py`

### 交互说明
* 拖动右下角 **Material Parameters** 面板：
    * **Ka/Kd/Ks**: 调整材质的环境/漫反射/镜面反射系数。
    * **Shininess**: 调整高光集中度。
```
