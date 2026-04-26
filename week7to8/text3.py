import taichi as ti

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 窗口分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 定义全局交互参数
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

# --- 几何体相交测试函数 ---

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """测试光线与球体相交"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    """
    测试光线与竖直圆锥相交
    apex: 圆锥顶点坐标
    base_y: 圆锥底面的世界坐标 Y 值
    """
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    
    # 转换到以顶点为原点的局部坐标系
    ro_local = ro - apex
    
    # 构建一元二次方程 At^2 + Bt + C = 0 (侧面方程)
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            # 确保 t_first 是较近的交点
            t_first, t_second = t1, t2
            if t1 > t2:
                t_first, t_second = t2, t1
                
            # 验证侧面交点是否在圆锥的高度范围内 (局部坐标 y 从 -H 到 0)
            y1 = ro_local.y + t_first * rd.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + rd * t
                # 侧面的法线计算
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
                

    if ti.abs(rd.y) > 1e-5:
        # t = (平面高度 - 射线起点高度) / 射线垂直方向速度
        t_base = (-H - ro_local.y) / rd.y
        if t_base > 0:
            # 只有当底面交点比侧面交点更近（或者侧面根本没击中）时，才选用底面
            if t < 0 or t_base < t:
                p_base = ro_local + rd * t_base
                # 检查交点是否在圆盘半径范围内
                if p_base.x**2 + p_base.z**2 <= radius**2:
                    t = t_base
                    # 底面的法线永远是垂直向下的
                    normal = ti.Vector([0.0, -1.0, 0.0])
                    
    return t, normal

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        # 用于记录光线击中的最近物体 (Z-buffer 深度竞争逻辑)
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
# 1. 红色球体 (左侧) - 严格遵循老师要求的坐标建议
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.5]), 1.2) 
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            
        # 2. 紫色圆锥 (右侧) - 严格遵循老师要求的坐标建议
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.5]), -1.4, 1.2) 
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        # 3. 蓝色圆锥 (中上位置优化) 
        # 修正点：将 X 设为 0，Y 设为 2.0 左右（避开红球和紫锥的厚度），Z 设为 0.5 使其稍微靠前
        t_cone_blue, n_cone_blue = intersect_cone(ro, rd, ti.Vector([0.0, 2.8, 0.0]), 0.6, 0.8) 
        if 0 < t_cone_blue < min_t:
            min_t = t_cone_blue
            hit_normal = n_cone_blue
            hit_color = ti.Vector([0.2, 0.4, 0.8])

        # 深青色背景
        color = ti.Vector([0.05, 0.2, 0.1]) 

        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 光源设置 [任务要求]
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0]) 
            
            L = normalize(light_pos - p)
            V = normalize(ro - p)

            ambient = Ka[None] * light_color * hit_color
            
            # 截断负数，避免背面漏光
            diff = ti.max(0.0, N.dot(L))
            diffuse = Kd[None] * diff * light_color * hit_color
            
            # =========================================================================
            # [选做 1：Blinn-Phong 模型升级]
            # 核心原理：使用法线 N 与 半程向量 H 的点乘来代替经典 Phong 模型的 R 乘 V。
            # 这不仅避免了计算复杂的反射向量 R，在 grazing angle（掠射角）时高光表现也更柔和自然。
            # =========================================================================
            H_vec = normalize(L + V)
            spec = ti.max(0.0, N.dot(H_vec)) ** shininess[None]
            specular = Ks[None] * spec * light_color 
            
            # =========================================================================
            # [选做 2：硬阴影 (Hard Shadow)]
            # 将阴影射线起点沿着法线方向稍微偏移 (1e-3)，防止浮点数精度导致的“自遮挡”噪点 (Shadow Acne)
            # =========================================================================
            shadow_ro = p + N * 1e-3
            shadow_rd = L
            
            in_shadow = False
            light_dist = (light_pos - p).norm()
            
            # 阴影射线求交测试
            t_s, _ = intersect_sphere(shadow_ro, shadow_rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
            if 0 < t_s < light_dist: in_shadow = True
            
            t_c1, _ = intersect_cone(shadow_ro, shadow_rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
            if 0 < t_c1 < light_dist: in_shadow = True
            
            t_c2, _ = intersect_cone(shadow_ro, shadow_rd, ti.Vector([0.0, 1.0, -1.5]), -1.0, 0.8)
            if 0 < t_c2 < light_dist: in_shadow = True
            
            # 如果处于阴影中，只计算环境光
            if in_shadow:
                color = ambient
            else:
                color = ambient + diffuse + specular
                
        # 强制限制在合法区间内，防止颜色过曝发白导致噪点
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Blinn-Phong & Hard Shadow Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化材质参数
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 64.0 

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.22):
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 256.0)

        window.show()

if __name__ == '__main__':
    main()