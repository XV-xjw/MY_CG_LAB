import taichi as ti
import math
import time

ti.init(arch=ti.cpu)

# --- 1. 数据定义 ---
NUM_VERTICES = 8
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c, s = ti.cos(rad), ti.sin(rad)
    # 同时绕 Y 轴和 X 轴轻微旋转，增加立体感
    return ti.Matrix([
        [c,   0.0,  s,   0.0],
        [0.1*s, 0.9, 0.1, 0.0],
        [-s,  0.0,  c,   0.0],
        [0.0, 0.0,  0.0, 1.0] 
    ])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(fov: ti.f32, aspect: ti.f32, znear: ti.f32, zfar: ti.f32):
    n, f = -znear, -zfar
    fov_rad = fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    r = aspect * t
    M_p2o = ti.Matrix([[n,0,0,0], [0,n,0,0], [0,0,n+f,-n*f], [0,0,1,0]])
    M_ortho = ti.Matrix([[1/r,0,0,0], [0,1/t,0,0], [0,0,2/(n-f),-(n+f)/(n-f)], [0,0,0,1]])
    return M_ortho @ M_p2o

# --- 2. 渲染核函数 ---
@ti.kernel
def render(angle: ti.f32, fov: ti.f32, cam_z: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, cam_z])
    mvp = get_projection_matrix(fov, 1.0, 0.1, 50.0) @ get_view_matrix(eye_pos) @ get_model_matrix(angle)
    
    for i in range(NUM_VERTICES):
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        # 视口变换：NDC [-1, 1] -> 屏幕 [0, 1]
        screen_coords[i] = (v_ndc.xy + 1.0) / 2.0

# --- 3. 主程序与光栅化逻辑 ---
def main():
    # 初始化立方体顶点 (局部坐标系)
    vertices[0], vertices[1], vertices[2], vertices[3] = [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
    vertices[4], vertices[5], vertices[6], vertices[7] = [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1]
    
    # 定义 6 个面及对应的颜色 (RGB)
    # 环绕顺序：面对相机时为逆时针 (CCW)
    face_data = [
        {'idx': (0, 1, 2, 3), 'color': 0xFF5555}, # 前 (红)
        {'idx': (1, 5, 6, 2), 'color': 0x55FF55}, # 右 (绿)
        {'idx': (5, 4, 7, 6), 'color': 0x5555FF}, # 后 (蓝)
        {'idx': (4, 0, 3, 7), 'color': 0xFFFF55}, # 左 (黄)
        {'idx': (3, 2, 6, 7), 'color': 0x55FFFF}, # 顶 (青)
        {'idx': (4, 5, 1, 0), 'color': 0xFF55FF}  # 底 (紫)
    ]

    gui = ti.GUI("Solid Rasterizer", res=(800, 800))
    start_time = time.time()

    while gui.running:
        angle = (time.time() - start_time) * 30.0
        render(angle, 45.0, 6.0)

        if gui.get_event(ti.GUI.PRESS) and gui.event.key == ti.GUI.ESCAPE:
            break

        # 遍历每个面进行光栅化填充
        for face in face_data:
            idxs = face['idx']
            v = [screen_coords[i] for i in idxs]

            # 背面剔除检测 (2D 叉乘)
            # S = (x1-x0)*(y2-y1) - (y1-y0)*(x2-x1)
            cp = (v[1][0] - v[0][0]) * (v[2][1] - v[1][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[1][0])

            if cp > 0:
                # 1. 填充面：将四边形拆分为两个三角形绘制
                # 三角形 A: (v0, v1, v2), 三角形 B: (v0, v2, v3)
                # Taichi GUI 绘制三角形的快捷写法：
                gui.triangle(v[0], v[1], v[2], color=face['color'])
                gui.triangle(v[0], v[2], v[3], color=face['color'])

                # 2. 绘制可见面的边框 (为了清晰度)
                for i in range(4):
                    gui.line(v[i], v[(i+1)%4], color=0x000000, radius=1)

                # 3. 修正：只在此处绘制可见面的顶点
                # 这样图像背后的点就不会被画出来
                for p in v:
                    gui.circle(p, radius=4, color=0xFFFFFF)

        gui.text("Solid Rasterization + Back-face Culling", (0.05, 0.95), color=0xFFFFFF)
        gui.show()

if __name__ == '__main__':
    main()