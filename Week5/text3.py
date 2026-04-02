import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# --- 核心参数调节 ---
WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 100
MAX_CURVE_POINTS = 20000 
NUM_SEGMENTS = 1000       # 贝塞尔曲线的总采样数
SEG_PER_SECTION = 100     # B样条每一段的采样数
AA_RADIUS = 1.2           # 反走样半径
POINT_RADIUS = 0.008      # 控制点绘制半径
DRAG_THRESHOLD = 0.02     # 鼠标抓取点的距离阈值

# GPU 缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)
curve_colors_field = ti.Vector.field(3, dtype=ti.f32, shape=MAX_CURVE_POINTS)

# --- 算法部分 ---

def de_casteljau(points, t):
    """递归实现贝塞尔算法"""
    if len(points) == 1: return points[0]
    next_pts = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        next_pts.append([(1.0 - t) * p0[0] + t * p1[0], (1.0 - t) * p0[1] + t * p1[1]])
    return de_casteljau(next_pts, t)

def compute_b_spline_colored(points):
    """计算分段着色的 B 样条"""
    n = len(points)
    if n < 4: return [], []
    # 均匀三次 B 样条基矩阵 M
    M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]) / 6.0
    res_pts, res_clrs = [], []
    color_pool = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.6, 1.0), (1.0, 1.0, 0.2), (1.0, 0.2, 1.0)]
    
    for i in range(n - 3):
        G = np.array([points[i], points[i+1], points[i+2], points[i+3]])
        current_color = color_pool[i % len(color_pool)]
        for t_int in range(SEG_PER_SECTION):
            t = t_int / SEG_PER_SECTION
            T = np.array([t**3, t**2, t, 1])
            res_pts.append(T @ M @ G)
            res_clrs.append(current_color)
    res_pts.append(np.array([1, 1, 1, 1]) @ M @ G)
    res_clrs.append(color_pool[(n-4) % len(color_pool)])
    return np.array(res_pts, dtype=np.float32), np.array(res_clrs, dtype=np.float32)

# GPU 渲染内核 

@ti.kernel
def clear_pixels():
    for i, j in pixels: pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pt, clr = curve_points_field[i], curve_colors_field[i]
        x_exact, y_exact = pt[0] * WIDTH, pt[1] * HEIGHT
        x_base, y_base = ti.cast(ti.floor(x_exact), ti.i32), ti.cast(ti.floor(y_exact), ti.i32)
        for dx, dy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            px, py = x_base + dx, y_base + dy
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                dist = ti.math.sqrt((px + 0.5 - x_exact)**2 + (py + 0.5 - y_exact)**2)
                if dist < AA_RADIUS:
                    pixels[px, py] += clr * (1.0 - dist / AA_RADIUS)

# --- 主程序与交互逻辑 ---

def main():
    window = ti.ui.Window("Curve-Lab: Drag to test Local Control", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    is_b_spline = True
    
    # 交互状态变量
    selected_idx = -1
    is_dragging = False

    print("操作指南:\n- 左键点击: 添加点\n- 左键长按并移动: 拖拽控制点\n- 'b': 切换 Bézier / B-Spline\n- 'c': 清空画布")

    while window.running:
        mouse_pos = window.get_cursor_pos()
        
        # 事件处理
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                # 寻找最近的点
                best_dist = DRAG_THRESHOLD
                for i, p in enumerate(control_points):
                    d = ((p[0]-mouse_pos[0])**2 + (p[1]-mouse_pos[1])**2)**0.5
                    if d < best_dist:
                        best_dist, selected_idx = d, i
                
                if selected_idx != -1:
                    is_dragging = True
                elif len(control_points) < MAX_CONTROL_POINTS:
                    control_points.append(mouse_pos)
            elif e.key == 'c': control_points = []
            elif e.key == 'b': is_b_spline = not is_b_spline

        for e in window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                is_dragging, selected_idx = False, -1

        # 更新拖拽点坐标
        if is_dragging and selected_idx != -1:
            control_points[selected_idx] = mouse_pos

        # 计算并渲染
        clear_pixels()
        n_ctrl = len(control_points)
        if n_ctrl >= 2:
            if not is_b_spline:
                pts_np = np.array([de_casteljau(control_points, i/NUM_SEGMENTS) for i in range(NUM_SEGMENTS+1)], dtype=np.float32)
                clrs_np = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (NUM_SEGMENTS+1, 1))
                curve_points_field.from_numpy(pts_np)
                curve_colors_field.from_numpy(clrs_np)
                draw_curve_kernel(NUM_SEGMENTS + 1)
            elif n_ctrl >= 4:
                pts_np, clrs_np = compute_b_spline_colored(control_points)
                n_pts = len(pts_np)
              
                p_pts, p_clrs = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32), np.zeros((MAX_CURVE_POINTS, 3), dtype=np.float32)
                p_pts[:n_pts], p_clrs[:n_pts] = pts_np, clrs_np
                curve_points_field.from_numpy(p_pts)
                curve_colors_field.from_numpy(p_clrs)
                draw_curve_kernel(n_pts)

        canvas.set_image(pixels)
        
        # 绘制控制点和连线
        if n_ctrl > 0:
            np_pts = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_pts[:n_ctrl] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_pts)
            canvas.circles(gui_points, radius=POINT_RADIUS, color=(1.0, 0.2, 0.2))
            if n_ctrl >= 2:
                indices = []
                for i in range(n_ctrl - 1): indices.extend([i, i+1])
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.4, 0.4, 0.4))
        window.show()

if __name__ == '__main__':
    main()