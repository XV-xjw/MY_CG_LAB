import taichi as ti
import math

# 初始化 Taichi，指定使用 CPU 后端
ti.init(arch=ti.cpu)

# 升级为正方体：8个顶点
NUM_VERTICES = 8
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

@ti.func
def get_model_matrix(angle: ti.f32):
    """
    为了更好地观察 3D 立方体的立体感，改为绕 Y 轴
    """
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [ c,  0.0,  s,  0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0,  c,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    """
    视图变换矩阵：将相机移动到原点
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """
    透视投影矩阵：平截头体 -> 正交长方体 -> 标准设备坐标
    """
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    # 1. 挤压矩阵: 透视平截头体 -> 正交长方体
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # 2. 正交投影矩阵: 缩放与平移至 [-1, 1]^3
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
def compute_transform(angle: ti.f32):
    """
    在并行架构上计算顶点的坐标变换
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # MVP 矩阵：严格遵循从右向左乘的原则
    mvp = proj @ view @ model
    
    for i in range(NUM_VERTICES):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换：映射到 [0, 1] 空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化正方体的8个顶点 (中心在原点，边长为2)
    vertices[0] = [-1.0, -1.0,  1.0] # 前下左
    vertices[1] = [ 1.0, -1.0,  1.0] # 前下右
    vertices[2] = [ 1.0,  1.0,  1.0] # 前上右
    vertices[3] = [-1.0,  1.0,  1.0] # 前上左
    vertices[4] = [-1.0, -1.0, -1.0] # 后下左
    vertices[5] = [ 1.0, -1.0, -1.0] # 后下右
    vertices[6] = [ 1.0,  1.0, -1.0] # 后上右
    vertices[7] = [-1.0,  1.0, -1.0] # 后上左

    # 定义立方体的12条边 (顶点索引的连接对)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # 前面 4 条边
        (4, 5), (5, 6), (6, 7), (7, 4), # 后面 4 条边
        (0, 4), (1, 5), (2, 6), (3, 7)  # 侧面 4 条连接边
    ]
    
    gui = ti.GUI("3D Cube Rotation", res=(700, 700))
    angle = 0.0
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 5.0
            elif gui.event.key == 'd':
                angle -= 5.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        compute_transform(angle)
        
        # 遍历绘制 12 条边
        for idx1, idx2 in edges:
            pt1 = screen_coords[idx1]
            pt2 = screen_coords[idx2]
            gui.line(pt1, pt2, radius=2, color=0x00FF00)
        
        gui.show()

if __name__ == '__main__':
    main()