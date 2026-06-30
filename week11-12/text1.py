import taichi as ti

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# ============ 物理与网格参数 ============
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长
k_s = 10000.0      # 结构弹簧劲度系数
k_s_shear = 7000.0 # 选做：剪切弹簧劲度系数（略弱于结构弹簧）
k_s_bend = 2500.0  # 选做：弯曲弹簧劲度系数（更弱，否则布料会变得像硬纸板）
k_d = 1.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0   # 速度上限，防止数值爆炸
particle_radius = 0.015  # 布料质点的渲染/碰撞半径

# 选做：球体碰撞参数
sphere_center = ti.Vector([0.0, 0.35, 0.0])
sphere_radius = 0.25

# ============ Taichi 数据场 ============
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧数据场（结构 + 剪切 + 弯曲，需要更大的容量）
# 结构: N*(N-1)*2  剪切: 2*(N-1)*(N-1)  弯曲: 2*N*(N-2)
max_springs = N * N * 8
spring_indices = ti.field(dtype=int, shape=max_springs * 2)  # 用于渲染画线
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_ks = ti.field(dtype=float, shape=max_springs)  # 每根弹簧自己的劲度系数
num_springs = ti.field(dtype=int, shape=())

# 选做：球心渲染用场（单元素，方便用 scene.particles 画出来）
sphere_center_field = ti.Vector.field(3, dtype=float, shape=1)
collision_enabled = ti.field(dtype=int, shape=())

# ============ 初始化 (拆分为多个 kernel 保证 GPU 同步) ============

@ti.kernel
def init_positions():
    """初始化质点位置与固定状态"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    """初始化弹簧：结构弹簧（必做） + 剪切/弯曲弹簧（选做）"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        # ---- 结构弹簧 (Structural)：右、下 ----
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_ks[c] = k_s
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_ks[c] = k_s

        # ---- 选做：剪切弹簧 (Shear)：格子内两条对角线 ----
        if i < N - 1 and j < N - 1:
            idx_diag1_a = i * N + j
            idx_diag1_b = (i + 1) * N + (j + 1)
            c1 = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c1] = ti.Vector([idx_diag1_a, idx_diag1_b])
            spring_lengths[c1] = (x[idx_diag1_a] - x[idx_diag1_b]).norm()
            spring_ks[c1] = k_s_shear

            idx_diag2_a = (i + 1) * N + j
            idx_diag2_b = i * N + (j + 1)
            c2 = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c2] = ti.Vector([idx_diag2_a, idx_diag2_b])
            spring_lengths[c2] = (x[idx_diag2_a] - x[idx_diag2_b]).norm()
            spring_ks[c2] = k_s_shear

        # ---- 选做：弯曲弹簧 (Bending)：跳过一个质点连接 ----
        if i < N - 2:
            idx_right2 = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right2])
            spring_lengths[c] = (x[idx] - x[idx_right2]).norm()
            spring_ks[c] = k_s_bend
        if j < N - 2:
            idx_down2 = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down2])
            spring_lengths[c] = (x[idx] - x[idx_down2]).norm()
            spring_ks[c] = k_s_bend

@ti.kernel
def init_spring_indices():
    """同步渲染索引"""
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    """从 Python 层按顺序调用各初始化 kernel，确保 GPU 同步"""
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()
    sphere_center_field[0] = sphere_center
    collision_enabled[None] = 1

# ============ 力计算 / 防爆 / 碰撞 (ti.func，内联进调用它的 kernel) ============

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """计算所有力 (重力 + 阻尼 + 全部弹簧力)"""
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -spring_ks[i] * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    """速度钳制，防止数值爆炸"""
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def resolve_sphere_collision(pos: ti.template(), vel: ti.template(), idx: int):
    """选做：质点与球体的简单碰撞处理（位置投影 + 法向速度清除）"""
    if collision_enabled[None] == 1:
        diff = pos[idx] - sphere_center
        dist = diff.norm()
        min_dist = sphere_radius + particle_radius
        if dist < min_dist and dist > 1e-6:
            n = diff / dist
            pos[idx] = sphere_center + n * min_dist
            vn = vel[idx].dot(n)
            if vn < 0.0:
                vel[idx] -= vn * n  # 仅去掉指向球心的速度分量，保留切向滑动

# ============ 合并的积分 kernel (每步仅 1 次 kernel 启动) ============

@ti.kernel
def step_explicit():
    """显式欧拉 (Explicit Euler) - 极易发散"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            resolve_sphere_collision(x, v, i)

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉 (Semi-Implicit Euler) - 相对稳定"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            resolve_sphere_collision(x, v, i)

@ti.kernel
def step_implicit_iter():
    """隐式欧拉 (Implicit Euler) - 定点迭代法近似求解"""
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        if is_fixed[i] == 0:
            resolve_sphere_collision(x_next, v_next, i)
        v[i] = v_next[i]
        x[i] = x_next[i]

# ============ 主函数 ============

def main():
    init_cloth()

    window = ti.ui.Window("Games101 - Mass Spring System", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1  # 0: 显式, 1: 半隐式, 2: 隐式
    paused = False

    while window.running:
        # =========== GUI 控制面板 ===========
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.4, 0.46)

        window.GUI.text("Integration Method:")
        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")

        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused

        if window.GUI.button("Reset Cloth"):
            init_cloth()

        window.GUI.text("")
        collision_label = "Disable Sphere Collision" if collision_enabled[None] == 1 else "Enable Sphere Collision"
        if window.GUI.button(collision_label):
            collision_enabled[None] = 1 - collision_enabled[None]

        window.GUI.end()
        # ====================================

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(x, radius=particle_radius, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.0, color=(0.8, 0.8, 0.8))

        # 选做：绘制碰撞球体
        if collision_enabled[None] == 1:
            scene.particles(sphere_center_field, radius=sphere_radius, color=(0.9, 0.3, 0.3))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()