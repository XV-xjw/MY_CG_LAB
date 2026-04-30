import taichi as ti

# 初始化 Taichi GPU 后端
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
spp = ti.field(ti.i32, shape=()) 

# 定义材质 ID 为整型常量
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2  

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta_ratio):
    cos_theta_i = -I.dot(N)
    sin2_theta_t = eta_ratio * eta_ratio * (1.0 - cos_theta_i * cos_theta_i)
    out_dir = ti.Vector([0.0, 0.0, 0.0])
    is_tir = 0 
    if sin2_theta_t > 1.0:
        is_tir = 1 
    else:
        cos_theta_t = ti.sqrt(1.0 - sin2_theta_t)
        out_dir = eta_ratio * I + (eta_ratio * cos_theta_i - cos_theta_t) * N
    return is_tir, out_dir

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        t2 = (-b + ti.sqrt(delta)) / 2.0 
        if t1 > 1e-4:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
        elif t2 > 1e-4:
            t = t2
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 1e-4:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = 1000000000.0
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 玻璃球
    t_s1, n_s1 = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t_s1 < min_t:
        min_t = t_s1
        hit_n = n_s1
        hit_c = ti.Vector([1.0, 1.0, 1.0]) 
        hit_mat = MAT_GLASS

    # 镜面球
    t_s2, n_s2 = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t_s2 < min_t:
        min_t = t_s2
        hit_n = n_s2
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 地面
    t_p, n_p = intersect_plane(ro, rd, -1.0)
    if 0 < t_p < min_t:
        min_t = t_p
        hit_n = n_p
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t_p
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        pixel_color = ti.Vector([0.0, 0.0, 0.0])
        
        for sample in range(spp[None]):
            u = (i + ti.random() - res_x / 2.0) / res_y * 2.0
            v = (j + ti.random() - res_y / 2.0) / res_y * 2.0
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            final_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            
            for bounce in range(max_bounces[None]):
                # 核心修正：不使用 '_' 占位符，显式接收所有返回值
                t, N, obj_color, mat_id = scene_intersect(ro, rd)
                
                if t > 1e9:
                    final_color += throughput * bg_color
                    break
                    
                p = ro + rd * t
                
                if mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color 
                    
                elif mat_id == MAT_GLASS:
                    dot_I_N = rd.dot(N)
                    N_eff = N
                    eta_ratio = 1.0 / 1.5 
                    
                    if dot_I_N > 0: 
                        N_eff = -N 
                        eta_ratio = 1.5 / 1.0 
                        
                    is_tir, ref_dir = refract(rd, N_eff, eta_ratio)
                    
                    if is_tir == 1:
                        ro = p + N_eff * 1e-4
                        rd = normalize(reflect(rd, N_eff))
                    else:
                        ro = p - N_eff * 1e-4 
                        rd = normalize(ref_dir)
                    
                    throughput *= 0.95 * obj_color 

                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)
                    shadow_ray_orig = p + N * 1e-4
                    rem_dist = (light_pos - p).norm()
                    
                    visibility = 1.0
                    for _shadow_b in range(4): 
                        # 核心修正：阴影射线这里也必须显式解构所有变量
                        s_t, s_n, s_c, s_mat = scene_intersect(shadow_ray_orig, L)
                        if s_t < rem_dist:
                            if s_mat == MAT_GLASS:
                                visibility *= 0.95 
                                shadow_ray_orig += L * (s_t + 1e-4) 
                                rem_dist -= s_t
                            else:
                                visibility = 0.0 
                                break
                        else:
                            break
                        
                    ambient = 0.05 * obj_color
                    direct_light = ambient 
                    
                    if visibility > 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color * visibility
                        direct_light += diffuse
                    
                    final_color += throughput * direct_light
                    break
                    
            pixel_color += final_color 

        pixels[i, j] = ti.math.clamp(pixel_color / spp[None], 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Corrected", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 4
    spp[None] = 4

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.3):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 8)
            spp[None] = gui.slider_int('Anti-Aliasing (SPP)', spp[None], 1, 16)

        window.show()

if __name__ == '__main__':
    main()