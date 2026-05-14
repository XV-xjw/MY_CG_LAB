# 基于 Taichi 的交互式 Whitted-Style 光线追踪渲染器

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Taichi-red.svg)](https://taichi-lang.org/)

本项目为计算机图形学课程实验作业。利用 **Taichi** 高性能并行计算框架，在 GPU 上实现了一个支持递归光影效果、物理材质建模及工程级鲁棒性优化的 Whitted-Style 光线追踪系统。

---

## 🖼️ 渲染效果展示 (Rendering Gallery)

通过以下三个阶段的输出结果，可以看到渲染器从基础几何求交到复杂物理模拟的阶梯式演进：

| 阶段一：基础架构 (text0.gif) | 阶段二：材质突破 (text1.gif) | 阶段三：最终精修 (text2.gif) |
| :---: | :---: | :---: |
| ![Basic RT](./text0.gif) | ![Glass & AA](./text1.gif) | ![Final Refined](./text2.gif) |
| **核心实现**：硬阴影 + 基础反射 | **核心实现**：折射 + 抗锯齿 (SPP) | **核心实现**：半透明阴影 + 鲁棒性修复 |

---

## 📂 项目目录结构与演进逻辑

*   **`text1_basic_rt.py`** (阶段一)
    *   实现解析几何求交与 Whitted 递归模型，初步搭建并行架构。
*   **`text2_glass_aliasing.py`** (阶段二)
    *   引入电介质材质与 Snell 定律，通过子像素采样消除边缘锯齿。
*   **`text3_final_refined.py`** (阶段三)
    *   针对 Taichi 编译器特性优化，修复变量解构 Bug，并实现半透明物体的光影透射。

---

## 📐 数学与物理原理

### 1. 镜面与折射模型
*   **反射 (Reflection)**：遵循公式 $R = I - 2(I \cdot N)N$。
*   **折射 (Refraction)**：基于斯涅尔定律（Snell's Law）：
    $$n_1 \sin \theta_1 = n_2 \sin \theta_2$$
    通过判断全反射（TIR）逻辑确保玻璃材质的真实感。

### 2. 抗锯齿 (Anti-Aliasing)
利用多重采样技术，在每个像素内进行 `spp` 次随机偏移采样：
$$PixelColor = \frac{1}{N} \sum_{i=1}^{N} RayCast(u + \Delta u_i, v + \Delta v_i)$$

---

## 💻 技术亮点与 Bug 修复

### 🛠 工程级鲁棒性优化
在最终版本中，我们针对并行框架的底层逻辑进行了深度加固：

*   **变量解构修复**：
    显式接收所有返回值，避免 Taichi 内核在元组解析时出现寄存器分配异常：
    ```python
    t, N, obj_color, mat_id = scene_intersect(ro, rd)
