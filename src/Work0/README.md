# 🌌⭐ GravitySim: 万有引力粒子模拟器

> **项目简介**：这是一个基于 C++/OpenGL 开发的粒子动力学模拟系统，通过数值积分算法还原数千颗粒子在万有引力作用下的演变过程。

---

## 📸 效果展示

| 初始星云 | 核心坍缩 | 多体系统 |
| :--- | :--- | :--- |
| ![截图1](https://via.placeholder.com/300x200) | ![截图2](https://via.placeholder.com/300x200) | ![截图3](https://via.placeholder.com/300x200) |

> *提示：建议在此处上传你的实验运行截图或 GIF 动图。*

---

## ✨ 核心特性

* **实时模拟**：支持多达 5000+ 粒子的实时引力计算。
* **交互控制**：支持鼠标旋转视角、滚轮缩放、键盘平移。
* **物理准确**：采用经典的牛顿万有引力公式，并引入 Softening Factor 防止数值爆炸。
* **视觉增强**：实现粒子发光效果（Bloom）和运动拖尾。

---

## 🔬 物理原理

本项目模拟的核心是**牛顿万有引力定律**。两个粒子之间的引力 $\vec{F}$ 计算公式如下：

$$\vec{F} = G \frac{m_1 m_2}{|\vec{r}|^2} \cdot \frac{\vec{r}}{|\vec{r}|}$$

其中：
1.  $G$ 是万有引力常数。
2.  $\vec{r}$ 是两个粒子之间的位移向量。
3.  在代码实现中，为了避免距离过近导致力趋向无穷大，我们通常使用分母修正：$\frac{1}{r^2 + \epsilon^2}$。

---

## 🛠️ 环境配置与运行

### 依赖项
* **编译器**: 支持 C++17 的 GCC 或 MSVC
* **图形库**: GLFW, GLEW
* **数学库**: GLM

### 编译步骤
1. 克隆仓库：`git clone https://github.com/yourname/project.git`
2. 进入目录：`cd project`
3. 使用 CMake 构建：
   ```bash
   mkdir build && cd build
   cmake ..
   make
