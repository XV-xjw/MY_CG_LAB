import taichi as ti
import math

ti.init(arch=ti.cpu)

# 数据定义
NUM_VERTICES = 8
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c, s = ti.cos(rad), ti.sin(rad)
    # 标准的绕 Y 轴旋转矩阵
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
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    M_ortho = ti.Matrix([
        [1.0/r, 0.0, 0.0, 0.0],
        [0.0, 1.0/t, 0.0, 0.0],
        [0.0, 0.0, 2.0/(n-f), -(n+f)/(n-f)],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return M_ortho @ M_p2o

@ti.kernel
def render(angle: ti.f32, fov: ti.f32, cam_z: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, cam_z])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(fov, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    # 仅变换立方体顶点
    for i in range(NUM_VERTICES):
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i] = (v_ndc.xy + 1.0) / 2.0

def main():
    # 初始化立方体顶点
    vertices[0], vertices[1], vertices[2], vertices[3] = [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
    vertices[4], vertices[5], vertices[6], vertices[7] = [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    gui = ti.GUI("Pure Cube - No Axes", res=(800, 800))
    angle, fov, cam_z = 0.0, 45.0, 5.0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE: gui.running = False
        
        if gui.is_pressed('a'): angle += 2.0
        if gui.is_pressed('d'): angle -= 2.0
        if gui.is_pressed('w'): fov = max(10, fov - 1.0)
        if gui.is_pressed('s'): fov = min(120, fov + 1.0)
        if gui.is_pressed('q'): cam_z += 0.1
        if gui.is_pressed('e'): cam_z -= 0.1

        render(angle, fov, cam_z)

        # 仅绘制立方体边线
        for idx1, idx2 in edges:
            gui.line(screen_coords[idx1], screen_coords[idx2], radius=3, color=0xFFFFFF)
        
        # 仅绘制顶点
        for i in range(NUM_VERTICES):
            gui.circle(screen_coords[i], radius=5, color=0x00FFFF)

        # UI 提示
        gui.text(f"Rotation Angle: {angle:.1f}", (0.05, 0.95))
        gui.text("A/D: Rotate | W/S: FOV | Q/E: Distance", (0.05, 0.91), color=0xAAAAAA)
        
        gui.show()

if __name__ == '__main__':
    main()