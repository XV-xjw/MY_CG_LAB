import taichi as ti
import math

ti.init(arch=ti.gpu)

# 基础配置
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 整合交互参数
light_info = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
samples_per_pixel = ti.field(ti.i32, shape=()) # 新增：抗锯齿采样数

# 材质常量
MAT_DIFFUSE = 0
MAT_MIRROR = 1
EPS = 1e-4

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    oc = ro - center
    b = oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    h = b * b - c
    if h > 0:
        t = -b - ti.sqrt(h)
    return t

@ti.func
def scene_intersect(ro, rd):
    min_t, hit_n, hit_c, hit_mat = 1e10, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0]), MAT_DIFFUSE

    # 红色漫反射球
    t_sph1 = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t_sph1 < min_t:
        min_t, hit_n, hit_c, hit_mat = t_sph1, (ro + rd * t_sph1 - ti.Vector([-1.2, 0.0, 0.0])).normalized(), ti.Vector([0.8, 0.1, 0.1]), MAT_DIFFUSE

    # 银色镜面球
    t_sph2 = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t_sph2 < min_t:
        min_t, hit_n, hit_c, hit_mat = t_sph2, (ro + rd * t_sph2 - ti.Vector([1.2, 0.0, 0.0])).normalized(), ti.Vector([0.9, 0.9, 0.9]), MAT_MIRROR

    # 棋盘格地板
    if rd.y < -EPS:
        t_plane = (-1.0 - ro.y) / rd.y
        if 0 < t_plane < min_t:
            min_t, hit_n, hit_mat = t_plane, ti.Vector([0.0, 1.0, 0.0]), MAT_DIFFUSE
            p = ro + rd * t_plane
            if (ti.floor(p.x * 2.0) + ti.floor(p.z * 2.0)) % 2 == 0:
                hit_c = ti.Vector([0.3, 0.3, 0.3])
            else:
                hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def get_visibility(ro, rd, dist):
    """轻量级阴影射线检测"""
    visible = 1.0
    # 检测逻辑：只要有任何物体在光照路径上，直接判定不可见
    t_sph1 = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t_sph1 < dist: visible = 0.0
    if visible > 0:
        t_sph2 = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
        if 0 < t_sph2 < dist: visible = 0.0
    return visible

@ti.kernel
def render():
    for i, j in pixels:
        color_acc = ti.Vector([0.0, 0.0, 0.0])
        
        # 多重采样抗锯齿循环
        spp = samples_per_pixel[None]
        for s in range(spp):
            # 添加随机抖动偏移
            offset = ti.Vector([ti.random(), ti.random()]) if spp > 1 else ti.Vector([0.5, 0.5])
            u = ((i + offset.x) - res_x / 2.0) / res_y * 2.0
            v = ((j + offset.y) - res_y / 2.0) / res_y * 2.0
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = ti.Vector([u, v - 0.2, -1.0]).normalized()
            
            tr_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            
            for bounce in range(max_bounces[None]):
                t, N, obj_c, mat_id = scene_intersect(ro, rd)
                if t > 1e9:
                    tr_color += throughput * ti.Vector([0.05, 0.15, 0.2])
                    break
                
                p = ro + rd * t
                if mat_id == MAT_MIRROR:
                    ro = p + N * EPS
                    rd = reflect(rd, N).normalized()
                    throughput *= 0.8 * obj_c
                else:
                    L = (light_info[None] - p).normalized()
                    dist_l = (light_info[None] - p).norm()
                    vis = get_visibility(p + N * EPS, L, dist_l)
                    
                    diff = ti.max(0.0, N.dot(L))
                    direct = (0.2 + 0.8 * diff * vis) * obj_c
                    tr_color += throughput * direct
                    break
            color_acc += tr_color
            
        # 平均采样并进行 Gamma 2.2 校正
        final_rgb = color_acc / spp
        pixels[i, j] = ti.pow(ti.math.clamp(final_rgb, 0.0, 1.0), 1.0/2.2)

def main():
    window = ti.ui.Window("Optimized Ray Tracing", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    light_info[None] = ti.Vector([2.0, 4.0, 3.0])
    max_bounces[None] = 3
    samples_per_pixel[None] = 4 # 默认 4 倍采样

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Setting", 0.7, 0.05, 0.28, 0.25):
            # 这里演示了如何直接通过索引解包处理 Vector field 的滑块
            l_pos = light_info[None]
            l_pos.x = gui.slider_float('Light X', l_pos.x, -5.0, 5.0)
            l_pos.y = gui.slider_float('Light Y', l_pos.y, 1.0, 8.0)
            l_pos.z = gui.slider_float('Light Z', l_pos.z, -5.0, 5.0)
            light_info[None] = l_pos
            
            max_bounces[None] = gui.slider_int('Bounces', max_bounces[None], 1, 5)
            samples_per_pixel[None] = gui.slider_int('MSAA SPP', samples_per_pixel[None], 1, 16)

        window.show()

if __name__ == '__main__':
    main()