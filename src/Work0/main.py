# src/Work0/main.py(它是连接配置、逻辑和界面的桥梁)
import taichi as ti
import sys

print(f"Python version: {sys.version}")
print(f"Taichi version: {ti.__version__}")

# 注意：初始化必须在最前面执行，接管底层 GPU
print("Initializing Taichi...")
ti.init(arch=ti.gpu)
print("Taichi initialized successfully!")

# 导入我们自己写的模块
from .config import WINDOW_RES, PARTICLE_COLOR, PARTICLE_RADIUS
from .physics import init_particles, update_particles, pos

print(f"Window resolution: {WINDOW_RES}")
print(f"Particle color: {PARTICLE_COLOR}")
print(f"Particle radius: {PARTICLE_RADIUS}")

def run():
    print("正在编译 GPU 内核，请稍候...")
    init_particles()
    print("Particles initialized!")
    
    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
    print("GUI created successfully!")
    print("编译完成！请在弹出的窗口中移动鼠标。")
    
    # 渲染主循环
    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()
        
        # 驱动 GPU 进行物理计算
        update_particles(mouse_x, mouse_y)
        
        # 读取显存数据并绘制
        gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
        gui.show()

if __name__ == "__main__":
    print("Starting run function...")
    run()
    print("Run function completed!")