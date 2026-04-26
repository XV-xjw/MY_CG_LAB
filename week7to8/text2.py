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

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

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
    
    # 构建一元二次方程 At^2 + Bt + C = 0
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    # 避免 A 为 0 时的除零错误
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first
                
            # 验证侧面交点是否在圆锥的高范围内
            y1 = ro_local.y + t_first * rd.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + rd * t
                # 圆锥表面的法线计算
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
                
    # 【修改：修补圆锥底面问题】
    # 测试光线是否与底面圆盘 (y = -H) 相交。封闭底面，消除斜面错觉。
    if ti.abs(rd.y) > 1e-5:
        t_base = (-H - ro_local.y) / rd.y
        # 如果射线击中了底面所在的平面，并且在正前方
        if t_base > 0:
            # 检查是否比侧面交点更近 (或者侧面根本没交点)
            if t < 0 or t_base < t:
                p_base = ro_local + rd * t_base
                # 检查交点是否在底面的半径范围内
                if p_base.x**2 + p_base.z**2 <= radius**2:
                    t = t_base
                    # 底面的法线垂直向下
                    normal = ti.Vector([0.0, -1.0, 0.0])
                
    return t, normal

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        # 用于记录光线击中的最近物体
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 1. 渲染红球
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-2.0, -0.5, 0.0]), 1.0) 
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            
        # 2. 渲染紫色圆锥
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([2.0, 1.2, 0.0]), -1.4, 1.0) 
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        # 3. 渲染蓝色圆锥
        t_cone_blue, n_cone_blue = intersect_cone(ro, rd, ti.Vector([0.0, 2.0, 0.0]), -1.0, 0.8) 
        if 0 < t_cone_blue < min_t:
            min_t = t_cone_blue
            hit_normal = n_cone_blue
            hit_color = ti.Vector([0.2, 0.4, 0.8]) 

        # 背景色
        color = ti.Vector([0.05, 0.2, 0.1]) 

        # 如果击中了任何物体
        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 光源设置
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0]) 
            
            L = normalize(light_pos - p)
            V = normalize(ro - p)

            # 【选做 1：Blinn-Phong 模型升级】
            ambient = Ka[None] * light_color * hit_color
            
            diff = ti.max(0.0, N.dot(L))
            diffuse = Kd[None] * diff * light_color * hit_color
            
            # 引入半程向量 H (Halfway Vector)
            H_vec = normalize(L + V)
            # 使用法线 N 与 半程向量 H 的点乘来计算高光
            spec = ti.max(0.0, N.dot(H_vec)) ** shininess[None]
            specular = Ks[None] * spec * light_color 
            
            # 【选做 2：硬阴影 (Hard Shadow)】
            # 将阴影射线起点沿着法线方向稍微偏移 (1e-3)，防止浮点数精度导致的自遮挡 (Shadow Acne)
            shadow_ro = p + N * 1e-3
            shadow_rd = L
            
            in_shadow = False
            # 计算当前点到光源的距离
            light_dist = (light_pos - p).norm()
            
            # 测试阴影射线是否被任何物体遮挡
            t_s, _ = intersect_sphere(shadow_ro, shadow_rd, ti.Vector([-2.0, -0.5, 0.0]), 1.0)
            if 0 < t_s < light_dist: in_shadow = True
            
            t_c1, _ = intersect_cone(shadow_ro, shadow_rd, ti.Vector([2.0, 1.2, 0.0]), -1.4, 1.0)
            if 0 < t_c1 < light_dist: in_shadow = True
            
            t_c2, _ = intersect_cone(shadow_ro, shadow_rd, ti.Vector([0.0, 2.0, 0.0]), -1.0, 0.8)
            if 0 < t_c2 < light_dist: in_shadow = True
            
            # 如果在阴影中，剔除漫反射和高光，仅保留环境光
            if in_shadow:
                color = ambient
            else:
                color = ambient + diffuse + specular
                
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Blinn-Phong & Hard Shadow Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化材质参数 (Blinn-Phong 通常需要更大的 shininess 来获得和 Phong 类似的高光大小)
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 64.0 # 将默认光泽度调高一点，效果更自然

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