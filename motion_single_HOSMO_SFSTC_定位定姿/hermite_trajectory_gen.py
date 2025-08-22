import numpy as np
from scipy.interpolate import CubicHermiteSpline

def theta_to_derivatives(x, y, theta):
    dx = np.cos(theta)
    dy = np.sin(theta)
    return dx, dy

def generate_follow_path_from_theta_autotime(x_points, y_points, theta_points, dt=0.035, speed=1.0):
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    theta_points = np.array(theta_points)
    n = len(x_points)

    if n < 2:
        raise ValueError("至少需要两个控制点")

    # 1. 计算相邻点间的欧几里得距离
    segment_lengths = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
    total_length = np.sum(segment_lengths)

    # 2. 计算运动总时间
    T = total_length / speed  # 匀速运动下，总时间 = 总长度 / 速度
    ref_time = np.arange(0, T + dt, dt)

    # 3. 构造 key_time（控制点时间） = 距离累加 / 总长度 * 总时间
    cumulative_length = np.insert(np.cumsum(segment_lengths), 0, 0)
    key_time = cumulative_length / total_length * T

    # 4. 导数方向
    dx, dy = theta_to_derivatives(x_points, y_points, theta_points)
    dx *= speed
    dy *= speed

    # 5. 构造 Hermite 曲线
    spline_x = CubicHermiteSpline(key_time, x_points, dx)
    spline_y = CubicHermiteSpline(key_time, y_points, dy)

    # 6. 采样轨迹
    x_ref = spline_x(ref_time)
    y_ref = spline_y(ref_time)
    vx_ref = spline_x.derivative()(ref_time)
    vy_ref = spline_y.derivative()(ref_time)
    ax_ref = spline_x.derivative(nu=2)(ref_time)
    ay_ref = spline_y.derivative(nu=2)(ref_time)

    path = np.column_stack((x_ref, y_ref))
    vel = np.column_stack((vx_ref, vy_ref))
    accel = np.column_stack((ax_ref, ay_ref))

    return path, vel, accel, ref_time

def generate_circle_trajectory(ref_time=None, r=25, v_max=4.5):
    T = 2 * np.pi * r / v_max  # 完整圆周时间
    if ref_time is None:
        dt = 0.035
        ref_time = np.arange(0, T + dt, dt)

    omega = v_max / r
    x = r * np.cos(omega * ref_time)
    y = r * np.sin(omega * ref_time)
    vx = -r * omega * np.sin(omega * ref_time)
    vy = r * omega * np.cos(omega * ref_time)
    ax = -r * omega**2 * np.cos(omega * ref_time)
    ay = -r * omega**2 * np.sin(omega * ref_time)

    path = np.column_stack((x, y))
    vel = np.column_stack((vx, vy))
    accel = np.column_stack((ax, ay))
    return path, vel, accel, ref_time


def generate_sine_trajectory(ref_time=None, r=25, v_max=0.7, theta=0.1):
    wave_length = 2 * np.pi / theta
    total_length = wave_length  # 展开两周期
    T = total_length / (v_max * np.pi)

    if ref_time is None:
        dt = 0.035
        ref_time = np.arange(0, T + dt, dt)

    x = v_max * np.pi * ref_time - r
    y = 0.5 * r * np.sin(theta * x)
    vx = v_max * np.ones_like(ref_time)
    vy = 0.5 * r * theta * v_max * np.pi * np.cos(theta * x)
    ax = np.zeros_like(ref_time)
    ay = -0.5 * r * (theta * v_max * np.pi) ** 2 * np.sin(theta * x)

    path = np.column_stack((x, y))
    vel = np.column_stack((vx, vy))
    accel = np.column_stack((ax, ay))
    return path, vel, accel, ref_time

def generate_square_trajectory(ref_time=None, r=25, speed=5.0):
    side_length = 2 * r
    total_length = 4 * side_length
    T = total_length / speed

    if ref_time is None:
        dt = 0.035
        ref_time = np.arange(0, T + dt, dt)

    n = len(ref_time)
    x_vertices = [-r, r, r, -r, -r]
    y_vertices = [-r, -r, r, r, -r]
    x = np.interp(np.linspace(0, 4, n), np.arange(5), x_vertices)
    y = np.interp(np.linspace(0, 4, n), np.arange(5), y_vertices)
    vx = np.gradient(x, ref_time)
    vy = np.gradient(y, ref_time)
    ax = np.gradient(vx, ref_time)
    ay = np.gradient(vy, ref_time)

    path = np.column_stack((x, y))
    vel = np.column_stack((vx, vy))
    accel = np.column_stack((ax, ay))
    return path, vel, accel, ref_time