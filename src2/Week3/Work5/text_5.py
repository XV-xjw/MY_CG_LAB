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
    return ti.Matrix([
        [c,   0.0,  s,   0.0],
        [0.0, 1.0,  0.0, 0.0],
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

# --- 2. 渲染核心变换 ---
@ti.kernel
def render(angle: ti.f32, fov: ti.f32, cam_z: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, cam_z])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(fov, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    for i in range(NUM_VERTICES):
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i] = (v_ndc.xy + 1.0) / 2.0

# --- 3. 主程序 ---
def main():
    # 初始化顶点
    vertices[0], vertices[1], vertices[2], vertices[3] = [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
    vertices[4], vertices[5], vertices[6], vertices[7] = [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1]
    
    # 【核心修改 1】：废弃无序的 edges，定义 6 个面 (Faces)
    # 严格按照“逆时针 (CCW)”顺序记录每个面的 4 个顶点索引
    faces = [
        (0, 1, 2, 3), # 正面 (Front)
        (1, 5, 6, 2), # 右面 (Right)
        (5, 4, 7, 6), # 背面 (Back)  - 注意顺序，从背面看过去必须是逆时针
        (4, 0, 3, 7), # 左面 (Left)
        (3, 2, 6, 7), # 顶面 (Top)
        (4, 5, 1, 0)  # 底面 (Bottom)
    ]

    gui = ti.GUI("CG Lab - Back-face Culling", res=(800, 800))
    start_time = time.time()
    cam_z = 5.0

    while gui.running:
        current_angle = (time.time() - start_time) * 40.0
        
        if gui.is_pressed('q'): cam_z += 0.1
        if gui.is_pressed('e'): cam_z -= 0.1
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE: gui.running = False

        render(current_angle, 45.0, cam_z)

        # 【核心修改 2】：面剔除与绘制逻辑
        for face in faces:
            # 获取该面上的前三个顶点在屏幕上的 2D 坐标
            v0 = screen_coords[face[0]]
            v1 = screen_coords[face[1]]
            v2 = screen_coords[face[2]]
            v3 = screen_coords[face[3]]

            # 计算二维叉乘的 Z 分量，检测环绕顺序
            # 向量 A = v1 - v0, 向量 B = v2 - v1
            cross_product = (v1[0] - v0[0]) * (v2[1] - v1[1]) - (v1[1] - v0[1]) * (v2[0] - v1[0])

            # 如果 cross_product > 0，说明在屏幕上是逆时针，面朝向我们
            if cross_product > 0:
                # 只绘制朝向我们的面的 4 条边，背对的直接跳过（剔除）
                gui.line(v0, v1, radius=3, color=0x00FFFF)
                gui.line(v1, v2, radius=3, color=0x00FFFF)
                gui.line(v2, v3, radius=3, color=0x00FFFF)
                gui.line(v3, v0, radius=3, color=0x00FFFF)
        
        # 绘制顶点
        for i in range(NUM_VERTICES):
            gui.circle(screen_coords[i], radius=5, color=0xFFFFFF)

        gui.text("Back-face Culling Active", (0.05, 0.95), color=0x00FF00)
        gui.text("Only CCW faces are rendered", (0.05, 0.91), color=0xAAAAAA)
        gui.show()

if __name__ == '__main__':
    main()