# 基于 Taichi 的交互式 Whitted-Style 光线追踪渲染器

本项目为计算机图形学课程实验作业。基于 **Taichi** 并行计算框架，实现了一个从基础 Phong 光照到复杂折射、抗锯齿及鲁棒阴影检测的光线追踪渲染器。

---

## 📂 项目目录结构与迭代逻辑
```text
.
├── text1_basic_rt.py       # 阶段一：基础 Whitted-Style RT (镜面反射 + 硬阴影)
├── text2_glass_aliasing.py # 阶段二：材质突破 (玻璃折射 + MSAA 抗锯齿)
├── text3_final_refined.py  # 阶段三：工程优化 (修复 Taichi 变量解构 Bug + 半透明阴影)
└── README.md               # 项目说明文档
🎯 实验目标光线追踪核心实现：在 GPU 上构建 Whitted-Style 递归光线追踪算法（迭代式实现）。物理材质建模：实现理想漫反射（Lambertian）、镜面反射（Mirror）以及基于斯涅尔定律的电介质折射（Glass）。渲染质量优化：引入多重采样抗锯齿（MSAA/SPP），消除几何边缘锯齿。工程鲁棒性修复：解决光线自相交（Shadow Acne）问题，并修正 Taichi 内核中因变量占位符导致的渲染异常。📐 数学与物理原理1. 镜面与折射模型反射 (Reflection)：遵循 $R = I - 2(I \cdot N)N$。折射 (Refraction)：基于斯涅尔定律（Snell's Law）：$$n_1 \sin \theta_1 = n_2 \sin \theta_2$$在代码中，通过 refract 函数判断全反射（TIR）并计算折射方向。2. 抗锯齿 (Anti-Aliasing)通过在每个像素内进行 spp (Samples Per Pixel) 次随机偏移采样，平滑物体边缘：$$PixelColor = \frac{1}{N} \sum_{i=1}^{N} RayCast(u + \Delta u_i, v + \Delta v_i)$$3. 阴影偏移 (Shadow Bias)为防止由于浮点数精度导致的光线与自身表面相交（产生黑斑），在发射阴影射线时对起点进行法线方向的微量偏移：$$P_{offset} = P + N \times 10^{-4}$$💻 渲染器进化史第一阶段：基础架构构建 (text1.py)核心成果：实现了球体与平面的解析求交，初步搭建了支持 max_bounces 的光线递归循环。亮点：引入棋盘格动态纹理，实现了基础的硬阴影效果。第二阶段：视觉效果飞跃 (text2.py)核心成果：新增 MAT_GLASS 材质与 refract 逻辑。视觉增强：引入 SPP 采样，解决了球体边缘的“楼梯状”锯齿问题。工程发现：在此版本中发现了玻璃球阴影过重以及渲染残留的初步现象，为第三阶段的重构埋下伏笔。第三阶段：工程级鲁棒性优化 (text3.py)核心成果：针对 Taichi 编译器特性进行了底层逻辑修复。关键修复：变量解构Bug 描述：在 Taichi kernel 中使用 _ 忽略返回值可能导致元组解析异常。解决方案：显式接收所有返回值 t, N, obj_color, mat_id = scene_intersect(ro, rd)。半透明阴影 (Translucent Shadow)：修改阴影检测逻辑。当阴影射线击中玻璃材质时，光线不再被完全遮挡，而是乘以 $0.95$ 的衰减系数，模拟光线穿透玻璃的物理特性。Python# text3.py 中的半透明阴影迭代逻辑
for _shadow_b in range(4): 
    s_t, s_n, s_c, s_mat = scene_intersect(shadow_ray_orig, L)
    if s_t < rem_dist:
        if s_mat == MAT_GLASS:
            visibility *= 0.95 # 玻璃透光
            shadow_ray_orig += L * (s_t + 1e-4) # 继续穿越
📊 版本特性对比特性v1 基础版v2 进阶版v3 最终精修版材质支持漫反射、镜面+ 玻璃折射+ 玻璃透光阴影边缘处理原始锯齿采样抗锯齿 (SPP)高质量平滑阴影逻辑二进制硬阴影二进制硬阴影多级半透明阴影代码鲁棒性基础存在变量解构隐患显式解构，消除内核 Bug🛠 环境依赖与运行运行环境Python 3.9+Taichi 1.6.0+ (高性能并行计算后端)快速启动安装依赖：pip install taichi运行最终版：python text3_final_refined.py交互操作材质控制：通过 GUI 滑块实时调整 Max Bounces（光线弹射次数）观察玻璃折射的深度变化。抗锯齿调节：动态调整 SPP，在渲染性能与图像质量间平衡。光源移动：实时拖动光源位置，观察动态生成的阴影与镜面高光。BNU CG Lab 2026Author: [Your Name/ID]Key Insight: 每一个像素的真实感，都源于底层数学的精确表达与对并行框架坑点的敏锐洞察。
