import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# --- 核心参数调节 ---
WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
MAX_CURVE_POINTS = 10000 
NUM_SEGMENTS = 1000       # 贝塞尔曲线的总采样数
SEG_PER_SECTION = 100     # B样条每一段的采样数
AA_RADIUS = 1.2           # 反走样半径
POINT_RADIUS = 0.006

# GPU 缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 【核心修改点 1】：新增颜色缓冲区，用于存储每一个采样点的特定颜色
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)
curve_colors_field = ti.Vector.field(3, dtype=ti.f32, shape=MAX_CURVE_POINTS)

def de_casteljau(points, t):
    """递归实现贝塞尔算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        next_points.append([(1.0 - t) * p0[0] + t * p1[0], (1.0 - t) * p0[1] + t * p1[1]])
    return de_casteljau(next_points, t)

def compute_b_spline_colored(points):
    """计算分段着色的 B 样条"""
    n = len(points)
    if n < 4: return [], []
    
    M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]) / 6.0
    
    res_pts = []
    res_clrs = []
    # 颜色池：红、绿、蓝、黄、紫
    color_pool = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.6, 1.0), (1.0, 1.0, 0.2), (1.0, 0.2, 1.0)]
    
    # 每一段由 4 个点决定 (i, i+1, i+2, i+3)
    for i in range(n - 3):
        G = np.array([points[i], points[i+1], points[i+2], points[i+3]])
        current_color = color_pool[i % len(color_pool)]
        
        for t_int in range(SEG_PER_SECTION):
            t = t_int / SEG_PER_SECTION
            T = np.array([t**3, t**2, t, 1])
            pt = T @ M @ G
            res_pts.append(pt)
            res_clrs.append(current_color)
            
    # 补上最后一个点
    res_pts.append(np.array([1, 1, 1, 1]) @ M @ G)
    res_clrs.append(color_pool[(n-4) % len(color_pool)])
    
    return np.array(res_pts, dtype=np.float32), np.array(res_clrs, dtype=np.float32)

@ti.kernel
def clear_pixels():
    for i, j in pixels: pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """【核心修改点 2】：GPU 渲染内核，支持从缓冲区读取每个点特有的颜色值"""
    for i in range(n):
        pt = curve_points_field[i]
        clr = curve_colors_field[i] # 获取当前采样点的颜色
        x_exact, y_exact = pt[0] * WIDTH, pt[1] * HEIGHT
        x_base, y_base = ti.cast(ti.floor(x_exact), ti.i32), ti.cast(ti.floor(y_exact), ti.i32)

        for dx, dy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            px, py = x_base + dx, y_base + dy
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                dist = ti.math.sqrt((px + 0.5 - x_exact)**2 + (py + 0.5 - y_exact)**2)
                if dist < AA_RADIUS:
                    weight = 1.0 - (dist / AA_RADIUS)
                    # 原子加法实现平滑颜色混合
                    pixels[px, py] += clr * weight

def main():
    window = ti.ui.Window("B-Spline Segmentation & Local Control", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    is_b_spline = False
    
    print("Controls: LMB: Add, 'c': Clear, 'b': Toggle Mode")

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                control_points.append(window.get_cursor_pos())
            elif e.key == 'c': control_points = []
            elif e.key == 'b': is_b_spline = not is_b_spline
        
        clear_pixels()
        cur_cnt = len(control_points)
        
        if cur_cnt >= 2 and not is_b_spline:
            # 贝塞尔模式：统一为绿色
            pts_np = np.array([de_casteljau(control_points, i/NUM_SEGMENTS) for i in range(NUM_SEGMENTS+1)], dtype=np.float32)
            clrs_np = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (NUM_SEGMENTS+1, 1))
            curve_points_field.from_numpy(pts_np)
            curve_colors_field.from_numpy(clrs_np)
            draw_curve_kernel(NUM_SEGMENTS + 1)
            
        elif cur_cnt >= 4 and is_b_spline:
            # B样条模式：分段变色
            pts_np, clrs_np = compute_b_spline_colored(control_points)
            n_pts = len(pts_np)
            # 填充到 GPU 缓冲区
            padded_pts = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)
            padded_clrs = np.zeros((MAX_CURVE_POINTS, 3), dtype=np.float32)
            padded_pts[:n_pts], padded_clrs[:n_pts] = pts_np, clrs_np
            
            curve_points_field.from_numpy(padded_pts)
            curve_colors_field.from_numpy(padded_clrs)
            draw_curve_kernel(n_pts)

        canvas.set_image(pixels)
        
        # 绘制辅助线和控制点
        if cur_cnt > 0:
            np_pts = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_pts[:cur_cnt] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_pts)
            canvas.circles(gui_points, radius=POINT_RADIUS, color=(1.0, 0.2, 0.2))
            if cur_cnt >= 2:
                indices = []
                for i in range(cur_cnt - 1): indices.extend([i, i+1])
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.4, 0.4, 0.4))
        window.show()

if __name__ == '__main__':
    main()