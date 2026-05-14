# 基于 Taichi 的交互式 Whitted-Style 光线追踪渲染器

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Taichi-red.svg)](https://taichi-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目为计算机图形学课程实验作业。利用 **Taichi** 高性能并行计算框架，在 GPU 上实现了一个支持递归光影效果、物理材质建模及工程级鲁棒性优化的 Whitted-Style 光线追踪系统。

---

## 📂 项目目录结构与演进逻辑

项目采用迭代开发模式，从基础几何求交逐步进化到复杂的物理现象模拟：

*   **`text1_basic_rt.py`** (阶段一：基础构建)
    *   核心：解析几何求交（Sphere/Plane）、基础 Whitted 递归模型。
    *   特性：镜面反射 + 硬阴影（Hard Shadow）。
*   **`text2_glass_aliasing.py`** (阶段二：视觉突破)
    *   核心：电介质（Dielectric）材质支持。
    *   特性：基于 Snell 定律的折射、MSAA 抗锯齿。
*   **`text3_final_refined.py`** (阶段三：工程优化)
    *   核心：Taichi 内核稳定性修复。
    *   特性：半透明阴影、显式变量解构、动态 GUI 交互。

---

## 🎯 实验核心目标

1.  **并行加速实现**：利用 Taichi 的 `@ti.kernel` 将光线与场景求交逻辑高度并行化。
2.  **物理材质建模**：
    *   **Lambertian (漫反射)**：模拟粗糙表面的等向散射。
    *   **Mirror (理想反射)**：实现完美的镜面映射。
    *   **Glass (电介质)**：结合折射与全反射（TIR）规律。
3.  **渲染质量优化**：
    *   **MSAA/SPP**：通过子像素采样消除几何边缘的“阶梯状”锯齿。
    *   **阴影平滑**：解决 Shadow Acne 现象，并实现玻璃材质的半透明遮挡效果。

---

## 📐 数学与物理原理

### 1. 镜面与折射模型
*   **反射 (Reflection)**：遵循公式 $R = I - 2(I \cdot N)N$。
*   **折射 (Refraction)**：基于斯涅尔定律（Snell's Law）：
    $$n_1 \sin \theta_1 = n_2 \sin \theta_2$$
    在代码中，通过 `refract` 函数判断全反射（TIR）并计算折射方向。

### 2. 抗锯齿 (Anti-Aliasing)
通过在每个像素内进行 `spp` (Samples Per Pixel) 次随机偏移采样，平滑物体边缘：
$$PixelColor = \frac{1}{N} \sum_{i=1}^{N} RayCast(u + \Delta u_i, v + \Delta v_i)$$

### 3. 阴影偏移 (Shadow Bias)
为防止由于浮点数精度导致的光线与自身表面相交（产生黑斑），在发射阴影射线时对起点进行法线方向的微量偏移：
$$P_{offset} = P + N \times 10^{-4}$$

---

## 💻 渲染器“进化史”细节

### 第一阶段：架构初创
实现了经典的 **Phong 模型**，包括环境光、漫反射和镜面高光。引入了动态生成的**棋盘格纹理**，用于直观地观察空间透视关系。

### 第二阶段：视觉飞跃
新增了 `MAT_GLASS` 材质。在处理折射光线时，由于 Taichi 内核暂不支持原生深层递归，项目采用了**迭代式循环**来模拟光线弹射。引入 SPP 采样，解决了球体边缘的“楼梯状”锯齿问题。

### 第三阶段：工程级鲁棒性优化
针对 Taichi 编译器特性进行了底层逻辑修复：

*   **变量解构 Bug 修复**：
    在 Taichi kernel 中使用 `_` 忽略返回值可能导致元组解析异常。解决方案为显式接收所有返回值：
    ```python
    # 显式接收，确保编译器寄存器分配正确
    t, N, obj_color, mat_id = scene_intersect(ro, rd)
