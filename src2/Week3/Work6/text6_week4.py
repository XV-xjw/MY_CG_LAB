import taichi as ti
import math

# 初始化 Taichi，指定使用 CPU 后端
ti.init(arch=ti.cpu)

# 升级为正方体：8个顶点
NUM_VERTICES = 8
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

@ti.func
def get_model_matrix(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32, tx: ti.f32, ty: ti.f32, tz: ti.f32):
    """
    模型变换矩阵：支持 X, Y, Z 三轴旋转（欧拉角）以及平移
    """
    rad_x = angle_x * math.pi / 180.0
    rad_y = angle_y * math.pi / 180.0
    rad_z = angle_z * math.pi / 180.0

    cx, sx = ti.cos(rad_x), ti.sin(rad_x)
    cy, sy = ti.cos(rad_y), ti.sin(rad_y)
    cz, sz = ti.cos(rad_z), ti.sin(rad_z)

    # X轴旋转矩阵
    Rx = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx,  cx, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # Y轴旋转矩阵
    Ry = ti.Matrix([
        [ cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # Z轴旋转矩阵
    Rz = ti.Matrix([
        [cz, -sz, 0.0, 0.0],
        [sz,  cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # 平移矩阵
    T = ti.Matrix([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 组合变换：先旋转 (X -> Y -> Z) 再平移
    return T @ Rz @ Ry @ Rx

@ti.func
def get_view_matrix(eye_pos):
    """视图变换矩阵"""
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """透视投影矩阵"""
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(ax: ti.f32, ay: ti.f32, az: ti.f32, tx: ti.f32, ty: ti.f32, tz: ti.f32):
    # 将相机往后拉一点 (Z=8.0)，以便能看全多个立方体
    eye_pos = ti.Vector([0.0, 0.0, 8.0]) 
    model = get_model_matrix(ax, ay, az, tx, ty, tz)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    for i in range(NUM_VERTICES):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

# 提取绘制逻辑为独立函数，方便同屏绘制多个立方体
def draw_cube(gui, edges, color, radius=2):
    for idx1, idx2 in edges:
        pt1 = screen_coords[idx1]
        pt2 = screen_coords[idx2]
        gui.line(pt1, pt2, radius=radius, color=color)

def main():
    vertices[0] = [-1.0, -1.0,  1.0]
    vertices[1] = [ 1.0, -1.0,  1.0]
    vertices[2] = [ 1.0,  1.0,  1.0]
    vertices[3] = [-1.0,  1.0,  1.0]
    vertices[4] = [-1.0, -1.0, -1.0]
    vertices[5] = [ 1.0, -1.0, -1.0]
    vertices[6] = [ 1.0,  1.0, -1.0]
    vertices[7] = [-1.0,  1.0, -1.0]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7) 
    ]
    
    gui = ti.GUI("3D Cube Rotation Interpolation", res=(800, 600))
    
    # === 定义姿态 0 (R0)：左侧起始点 ===
    p0_rot = [0.0, 0.0, 0.0]         # 初始不旋转
    p0_pos = [-3.0, -1.5, 0.0]       # 靠左下方
    
    # === 定义姿态 1 (R1)：右侧终点 ===
    p1_rot = [60.0, 135.0, 45.0]     # 复杂的混合旋转
    p1_pos = [3.0, -1.5, 0.0]        # 靠右下方
    
    # 动画参数
    t = 0.0           # 插值进度 (0.0 到 1.0)
    dt = 0.005        # 动画速度
    direction = 1     # 1代表正向，-1代表逆向

    while gui.running:
        if gui.get_event(ti.GUI.ESCAPE):
            gui.running = False
            
        # 1. 更新插值参数 t，实现来回往复动画
        t += dt * direction
        if t >= 1.0:
            t = 1.0
            direction = -1
        elif t <= 0.0:
            t = 0.0
            direction = 1
            
        # 2. 绘制 R0 (左侧静态立方体)
        compute_transform(*p0_rot, *p0_pos)
        draw_cube(gui, edges, color=0x3333aa, radius=1)
        
        # 3. 绘制 R1 (右侧静态立方体)
        compute_transform(*p1_rot, *p1_pos)
        draw_cube(gui, edges, color=0x33aa33, radius=1)
        
        # 4. 计算并绘制 Rt (过渡中的插值立方体)
        # 欧拉角线性插值 (Lerp)
        cur_rot = [p0_rot[i] + t * (p1_rot[i] - p0_rot[i]) for i in range(3)]
        # 平移线性插值 (Lerp)
        cur_pos = [p0_pos[i] + t * (p1_pos[i] - p0_pos[i]) for i in range(3)]
        
        # 附加效果：为了契合图片里的箭头弧度，给Y轴加一个正弦轨迹，让它“跃”过去
        cur_pos[1] += math.sin(t * math.pi) * 3.0
        
        compute_transform(*cur_rot, *cur_pos)
        draw_cube(gui, edges, color=0xFFaa00, radius=3) # 亮橘色，加粗，醒目
        
        gui.show()

if __name__ == '__main__':
    main()