import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# --- Configuration & Tuning Parameters ---
WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
MAX_CURVE_POINTS = 10000  # Expanded buffer to support B-Spline dynamic lengths
NUM_SEGMENTS = 1000       # Sampling resolution for Bezier
AA_RADIUS = 1.2           # Anti-aliasing radius
POINT_RADIUS = 0.006

# GPU Buffers
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# Unified buffer for both Bezier and B-Spline coordinates
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)

def de_casteljau(points, t):
    """Pure Python recursive De Casteljau algorithm for Bezier"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

def compute_b_spline(points, segments_per_section=100):
    """Matrix-based uniform cubic B-Spline calculation"""
    n = len(points)
    if n < 4:
        return np.empty((0, 2), dtype=np.float32)
    
    # Basis Matrix for Uniform Cubic B-Spline
    M = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  0,  3,  0],
        [ 1,  4,  1,  0]
    ]) / 6.0
    
    res = []
    # Loop over n-3 sections
    for i in range(n - 3):
        # Geometry matrix for the local segment
        G = np.array([points[i], points[i+1], points[i+2], points[i+3]])
        
        for t_int in range(segments_per_section):
            t = t_int / segments_per_section
            T = np.array([t**3, t**2, t, 1])
            pt = T @ M @ G
            res.append(pt)
            
    # Cap the final point of the last segment
    T_end = np.array([1, 1, 1, 1])
    pt_end = T_end @ M @ G
    res.append(pt_end)
    
    return np.array(res, dtype=np.float32)

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32, color_r: ti.f32, color_g: ti.f32, color_b: ti.f32):
    """GPU Kernel with Distance-Based Anti-Aliasing"""
    for i in range(n):
        pt = curve_points_field[i]
        x_exact = pt[0] * WIDTH
        y_exact = pt[1] * HEIGHT

        # Find the integer base pixel
        x_base = ti.cast(ti.floor(x_exact), ti.i32)
        y_base = ti.cast(ti.floor(y_exact), ti.i32)

        # Iterate over a 3x3 local neighborhood
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                px = x_base + dx
                py = y_base + dy
                
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    # Calculate center coordinate of the current pixel
                    cx = px + 0.5
                    cy = py + 0.5
                    
                    # Euclidean distance from exact geometry point to pixel center
                    dist = ti.math.sqrt((cx - x_exact)**2 + (cy - y_exact)**2)
                    
                    if dist < AA_RADIUS:
                        # Linear falloff weight
                        weight = 1.0 - (dist / AA_RADIUS)
                        color_vec = ti.Vector([color_r, color_g, color_b]) * weight
                        
                        # In Taichi, += on global fields inside a kernel is automatically atomic, preventing data races
                        pixels[px, py] += color_vec

def main():
    window = ti.ui.Window("Curve Renderer: Bezier vs B-Spline", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    # State flags
    is_b_spline = False
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added control point: {pos}")
            elif e.key == 'c': 
                control_points = []
                print("Canvas cleared.")
            elif e.key == 'b':
                is_b_spline = not is_b_spline
                mode_str = "B-Spline" if is_b_spline else "Bezier"
                print(f"Switched to mode: {mode_str}")
        
        clear_pixels()
        current_count = len(control_points)
        
        if current_count >= 2 and not is_b_spline:
            # --- Bezier Mode ---
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
            
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1, 0.0, 1.0, 0.0) # Green for Bezier
            
        elif current_count >= 4 and is_b_spline:
            # --- B-Spline Mode ---
            curve_points_np = compute_b_spline(control_points, segments_per_section=100)
            n_points = len(curve_points_np)
            if n_points > 0:
                # Pad array with zeros to match MAX_CURVE_POINTS size requirements if needed by strict Taichi versions
                # Alternatively, just load the slice
                padded_np = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)
                padded_np[:n_points] = curve_points_np
                
                curve_points_field.from_numpy(padded_np)
                draw_curve_kernel(n_points, 0.0, 0.8, 1.0) # Cyan for B-Spline

        canvas.set_image(pixels)
        
        # Draw Control Polygon
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=POINT_RADIUS, color=(1.0, 0.2, 0.2))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.4, 0.4, 0.4))
        
        window.show()

if __name__ == '__main__':
    print("Controls:\n- Left Click: Add Point\n- 'c': Clear Canvas\n- 'b': Toggle Bezier / B-Spline Mode")
    main()