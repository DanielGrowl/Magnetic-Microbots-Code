import numpy as np
import matplotlib.pyplot as plt

# 设定时间参数，步长为 0.035
dt = 0.035
t = np.arange(0, 30, dt)  # 生成时间序列

# 正弦 S 型轨迹
A_sin = 5      # 振幅
omega = 0.2      # 角频率
x_sin = t
y_sin = A_sin * np.sin(omega * t)

# 画图
plt.figure(figsize=(8, 6))

# 绘制 S 形轨迹
plt.plot(x_sin, y_sin, label="Sine Wave S-curve", color='b')

# 设置图形
plt.xlabel("x (time)")
plt.ylabel("y (position)")
plt.title("S-shaped Trajectory (dt=0.035)")
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
plt.legend()
plt.grid()
plt.show()
