# src/Work0/config.py

# --- 物理系统参数 ---
NUM_PARTICLES = 20000      # 粒子总数 (卡顿请调小此数值，如 5000)
GRAVITY_STRENGTH = 0.001   # 鼠标引力强度
DRAG_COEF = 0.98           # 空气阻力系数
BOUNCE_COEF = -0.8         # 边界反弹能量损耗# --- 渲染系统参数 ---
WINDOW_RES = (1280, 720)    # 窗口分辨率（建议1280*720，否则可能会导致渲染效果不佳，这样调整应该能产生更加丰富的视觉效果）    
PARTICLE_RADIUS = 1.2      # 粒子绘制半径（我做了适当的半径减小操作，谋求更加细腻的渲染效果）
PARTICLE_COLOR = 0xFFD700  # 粒子颜色 (金色)