import math
import time

import numpy as np
from scipy.linalg import solve_discrete_are

miu_0 = 4 * math.pi * 10 ** -7
default_current = 0

# 时间值（self.times）的变化不足或相同
# 线性回归需要时间值有变化，如果 self.times 中的值相同或变化很小，那么计算出来的速度可能会为 0 或接近 0。时间值需要是有增量的，且相差足够大才能准确地计算速度。
#
# 位置数据的变化不足
# 如果 self.x_positions 或 self.y_positions 中的位置数据变化很小，导致线性拟合计算的斜率接近 0，速度的估计值也会趋近于 0。
#
# 数据点太少
# 如果数据点数量不足以进行合理的线性回归（如数据点少于两个），结果将是速度估计为 0。
class PolynomialFilter2D:
    def __init__(self, degree=2):
        self.degree = degree
        self.times = []
        self.x_positions = []
        self.y_positions = []

    def add_measurement(self, time, x, y):
        self.times.append(time)
        self.x_positions.append(x)
        self.y_positions.append(y)
        if len(self.times) > 4:
            self.times.pop(0)
            self.x_positions.pop(0)
            self.y_positions.pop(0)

    def get_filtered_velocity(self):
        if len(self.times) < 2:
            return 0, 0  # 不足2个点无法计算
        # 根据当前点的数量，调整多项式的阶数
        fit_degree = min(self.degree, len(self.times) - 1)
        # 对x和y方向分别使用多项式拟合
        coeffs_x = np.polyfit(self.times, self.x_positions, fit_degree)
        coeffs_y = np.polyfit(self.times, self.y_positions, fit_degree)
        # 对拟合的多项式求导
        derivative_x = np.polyder(coeffs_x)
        derivative_y = np.polyder(coeffs_y)
        # 计算速度估计
        velocity_x = np.polyval(derivative_x, self.times[-1])
        velocity_y = np.polyval(derivative_y, self.times[-1])
        return velocity_x, velocity_y

# 使用示例
# filter_2d = PolynomialFilter2D(degree=2)
#
# measurements = [(0.0, 5.0, 3.0), (0.1, 5.2, 3.1), (0.2, 5.4, 3.3), (0.3, 5.1, 3.2)]
# for t, x, y in measurements:
#     filter_2d.add_measurement(t, x, y)
#     velocity_x, velocity_y = filter_2d.get_filtered_velocity()
#     print(f"当前滤波后的速度: ({velocity_x}, {velocity_y})")

# 时间值（self.times）的变化不足或相同
# 线性回归需要时间值有变化，如果 self.times 中的值相同或变化很小，那么计算出来的速度可能会为 0 或接近 0。时间值需要是有增量的，且相差足够大才能准确地计算速度。
#
# 位置数据的变化不足
# 如果 self.x_positions 或 self.y_positions 中的位置数据变化很小，导致线性拟合计算的斜率接近 0，速度的估计值也会趋近于 0。
#
# 数据点太少
# 如果数据点数量不足以进行合理的线性回归（如数据点少于两个），结果将是速度估计为 0。
class LinearRegressionFilter2D:
    def __init__(self):
        self.times = []
        self.x_positions = []
        self.y_positions = []

    def add_measurement(self, time, x, y):
        # 添加新测量值
        self.times.append(time)
        self.x_positions.append(x)
        self.y_positions.append(y)
        # 如果超过4个测量值，移除最旧的值
        if len(self.times) > 4:
            self.times.pop(0)
            self.x_positions.pop(0)
            self.y_positions.pop(0)

    def get_filtered_velocity(self):
        if len(self.times) < 2:
            return 0, 0  # 不足2个点时，返回0速度
        # 分别对x和y进行线性回归拟合
        A = np.vstack([self.times, np.ones(len(self.times))]).T
        vx, _ = np.linalg.lstsq(A, self.x_positions, rcond=None)[0]
        vy, _ = np.linalg.lstsq(A, self.y_positions, rcond=None)[0]
        return vx, vy  # 返回x和y方向的速度估计


class MovingAverageFilter2D:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.x_values = []
        self.y_values = []

    def add_value(self, x, y):
        # 添加新值
        self.x_values.append(x)
        self.y_values.append(y)
        # 如果超过窗口大小，移除最旧的值
        if len(self.x_values) > self.window_size:
            self.x_values.pop(0)
            self.y_values.pop(0)

    def get_filtered_value(self):
        # 返回当前窗口内的平均值
        if len(self.x_values) == 0:
            return 0, 0
        avg_x = sum(self.x_values) / len(self.x_values)
        avg_y = sum(self.y_values) / len(self.y_values)
        return avg_x, avg_y

class magnetic_motion_model:
    def __init__(self, dt, position=[[0],[0]], vel=[[0],[0]], accel=[[0],[0]], theta = 0, current_limit=10) -> None:
        self.current = np.array([[0], [0]])
        self.accel = np.array(accel)
        self.vel = np.array(vel)
        self.position = np.array(position)
        self.theta = theta
        self.min_limit = 1.5
        self.max_limit = 3
        self.delay_time = 50

        self.ideal_position = np.array([[0], [0]])

        self.error = np.array([[0], [0]])
        # 这个是PID用的
        self.error_total = np.array([[0], [0], [0], [0]])
        self.PID_Kp = np.array([-0.5,-0.5,0,0])
        self.PID_Ki = np.array([-0.01,-0.01,0,0])

        self.current_log = []
        self.path_log = []
        self.vel_log = []
        self.error_log = []
        self.resized_error_log = []
        self.resized_vel_error_log = []
        self.fitting_log = []
        self.fn_log = []
        self.input_log = []
        self.input_smith_log = []
        self.dead_input_log = []
        self.delay_input_log = []
        self.valid_input = np.array([[0], [0]])
        self.L_state_log = np.ones((4, 4))
        self.cost_state_log = np.ones((1, 1))
        self.exceed_flag_state_log = []
        self.current_error_robust_log = []
        self.det_L_augumented_log = []
        self.Ks_log = []
        self.theta_deg_log = []
        # self.theta_log = []

        self.dt = dt
        self.resistance = 50 * 1e-6 * 970
        self.ref_time = []
        self.ref_x = []
        self.ref_y = []
        self.ref_vx = []
        self.ref_vy = []
        self.ref_ax = []
        self.ref_ay = []
        self.ref_path = []
        self.ref_velpath = []
        self.ref_accelpath = []
        self.current_limit = current_limit
        self.alpha = 0.3  # 低通滤波器的平滑系数
        self.filter = MovingAverageFilter2D(window_size=6)

        # 线圈参数
        self.coil_number_x = 297
        self.coil_radius_x = 0.236
        self.coil_number_y = 202
        self.coil_radius_y = 0.162
        self.coil_number_z = 129
        self.coil_radius_z = 0.1

        # 磁体参数
        self.Br = 1.28
        self.density_g_per_cm3 = 7.5
        self.density_kg_per_m3 = self.density_g_per_cm3 * 1000
        self.diameter_mm = 2 / 1000  # 底面直径为 2 毫米
        self.height_mm = 2 / 1000  # 高度为 2 毫米
        self.Volume = math.pi * (self.diameter_mm / 2) ** 3 * 4/3
        # self.Volume = math.pi * (self.diameter_mm/2)**2 * self.height_mm
        self.Magnetic = self.Br/miu_0
        self.mass = self.Volume * self.density_kg_per_m3
        # self.gradient_B_x = (3 / 2) * (4 / 5) ** (5 / 2) * (miu_0 * self.coil_number_x) / (self.coil_radius_x ** 2)
        # self.gradient_B_y = (3 / 2) * (4 / 5) ** (5 / 2) * (miu_0 * self.coil_number_y) / (self.coil_radius_y ** 2)
        # self.gradient_B_z = (3 / 2) * (4 / 5) ** (5 / 2) * (miu_0 * self.coil_number_z) / (self.coil_radius_z ** 2)

        self.gradient_B_x = 0.0028
        self.gradient_B_y = 0.0039
        self.gradient_B_z = 0.0067


        # 模型为 ddot{x} = f(x)+g(x)u
        self.gx = np.array([[(self.Volume * self.Magnetic * self.gradient_B_x)/self.mass, 0], [0, (self.Volume * self.Magnetic * self.gradient_B_y)/self.mass]])
        self.fx = np.zeros((2, 1))
        self.gx_inv = np.linalg.inv(self.gx)

        # 微分平坦后的模型
        self.A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        self.state_flat_log = []
        self.dot_state_flat_log = []

        # 用于消除靠近目标点后的抽疯
        self.close_flag = 0
        # # self.optimal_Q = np.diag([50, 50, 5, 5])  # 路近点跟踪用这个，不咋管速度
        # self.optimal_Q = np.diag([1, 1, 1, 1])    #   轨迹跟踪用这个，让速度和位置一样的权重
        # self.optimal_R = np.identity(2)
        # self.P = solve_discrete_are(self.A, self.B, self.optimal_Q, self.optimal_R)
        # # 转置：np.transpose(self.B)或者self.B.T
        # # print('optimal_R is\n', self.optimal_R)
        # # print('Stable P is\n', self.P)
        # self.K_x = -np.linalg.inv(self.optimal_R + self.B.T @ self.P @ self.B) @ self.B.T @ self.P @ self.A
        # self.K_beita = -np.linalg.inv(self.optimal_R + self.B.T @ self.P @ self.B) @ self.B.T
        # # print('K_x is\n', self.K_x)
        # # print('K_beita is\n', self.K_beita)

        # # H_/infty control
        # # 这个只适用于镇定问题，也就是路近点跟踪。里头没有前馈，不适用于轨迹跟踪
        # self.E = np.array([[0, 0], [0, 0], [-1 / self.mass, 0], [0, -1 / self.mass]])
        # self.gama = 0.01
        # # 垂直拼接（上下堆叠）这两个矩阵，应该使用 np.vstack(); 水平拼接（左右合并）：使用 np.hstack([B, E])
        # self.B_aug = np.hstack([self.B, self.E])
        # print('E is\n', self.E)
        # print('B_aug is\n', self.B_aug)
        # self.robust_Q = self.optimal_Q
        # self.robust_R_beforediag = np.add([1, 1, -self.gama ** 2, -self.gama ** 2], [1, 1, 0, 0])
        # self.robust_R = np.diag(self.robust_R_beforediag)
        # print('robust_Q is\n', self.robust_Q)
        # print('robust_R is\n', self.robust_R)
        # self.robust_P = solve_discrete_are(self.A, self.B_aug, self.robust_Q, self.robust_R)
        # print('Stable robust_P is\n', self.robust_P)
        # self.robust_L_1 = np.identity(2) + self.B.T @ self.robust_P @ self.B
        # self.robust_L_2 = - self.B.T @ self.robust_P @ self.A
        # self.robust_K = np.linalg.inv(self.robust_L_1) @ self.robust_L_2
        # print('Stable robust_K is\n', self.robust_K)
        # print('K_x is\n', self.K_x)

    def update_fx(self):
        # self.fx = fx
        eta = 50 * 1e-6 * 970
        self.fx = -(6 * math.pi * eta * (self.diameter_mm/2) * self.vel) / self.mass
        # print("fx", self.fx)

    def update_accel(self, current):
        self.current = current
        self.accel = self.fx + self.gx @ current

    # def dynamic_position(self, current, dt):
    #     self.update_accel(current)
    #     self.vel = self.accel * dt + self.vel
    #     self.position = self.vel * dt + self.position

    def update_status(self, current_position):
        # 更新位置
        self.position = current_position

        # current_vel = (current_position - self.position) / self.dt
        # self.accel = (current_vel - self.vel) / self.dt
        # self.vel = current_vel
        # self.position = current_position

        # loss filter
        # # 计算当前速度
        # current_vel = (current_position - self.position) / self.dt
        # # 平滑速度
        # self.vel = self.alpha * current_vel + (1 - self.alpha) * self.vel
        # # 计算加速度
        # current_accel = (current_vel - self.vel) / self.dt
        # # 平滑加速度
        # self.accel = self.alpha * current_accel + (1 - self.alpha) * self.accel
        # # 更新位置
        # self.position = current_position

        # loss filter
        # # 计算当前速度
        # current_vel = (current_position - self.position) / self.dt
        # # 平滑速度
        # current_vel = self.alpha * current_vel + (1 - self.alpha) * self.vel
        # # 计算加速度
        # current_accel = (current_vel - self.vel) / self.dt
        # # 平滑加速度
        # self.accel = self.alpha * current_accel + (1 - self.alpha) * self.accel
        # self.vel = current_vel
        # # 更新位置
        # self.position = current_position

        # MovingAverageFilter2D
        # position_value = current_position - self.position
        # self.filter.add_value(position_value[0, 0], position_value[1, 0])
        # filtered_x, filtered_y = self.filter.get_filtered_value()
        # vx = filtered_x / self.dt
        # vy = filtered_y / self.dt
        # current_vel = np.array([[vx], [vy]])
        # self.accel = (current_vel - self.vel) / self.dt
        # self.vel = current_vel
        # self.position = current_position

        # # double filter
        # # 计算当前速度
        # current_vel = (current_position - self.position) / self.dt
        # # 平滑速度
        # current_vel = self.alpha * current_vel + (1 - self.alpha) * self.vel
        # self.filter.add_value(current_vel[0, 0], current_vel[1, 0])
        # filtered_x, filtered_y = self.filter.get_filtered_value()
        # current_vel = np.array([[filtered_x], [filtered_y]])
        # # 计算加速度
        # current_accel = (current_vel - self.vel) / self.dt
        # # 平滑加速度
        # self.accel = self.alpha * current_accel + (1 - self.alpha) * self.accel
        # self.vel = current_vel
        # # 更新位置
        # self.position = current_position

    def return_position(self):
        return self.position

    def return_vel(self):
        return self.vel

    def return_accel(self):
        return self.accel

    def return_current(self):
        return self.current

    def dead_zone(self, input):
        total_force = np.linalg.norm(input)

        if np.any(self.vel == 0):
            if total_force <= self.max_limit:
                # 如果合力小于或等于限制值，直接将输入设为零
                self.valid_input = np.zeros_like(input)
            else:
                # Step 3: 超出限制时，调整合力
                adjusted_force = total_force - self.max_limit  # 调整后的合力
                scale = adjusted_force / total_force  # 计算缩放比例
                self.valid_input = input * scale  # 调整输入分量
        elif np.any(self.vel != 0):
            if total_force <= self.min_limit:
                # 如果合力小于或等于限制值，直接将输入设为零
                self.valid_input = np.zeros_like(input)
            else:
                # Step 3: 超出限制时，调整合力
                adjusted_force = total_force - self.min_limit  # 调整后的合力
                scale = adjusted_force / total_force  # 计算缩放比例
                self.valid_input = input * scale  # 调整输入分量
        self.dead_input_log.append(self.valid_input)

    def delay_input(self, ulog):
        if len(self.input_log) < self.delay_time:
            self.valid_input = np.array([[0], [0]])
        else:
            self.valid_input = ulog[-self.delay_time]
        self.delay_input_log.append(self.valid_input)

    def dynamic_position(self, input, dt):
        self.input_log.append(input)
        self.dead_zone(input)
        self.delay_input(self.dead_input_log)
        self.accel = (self.valid_input - self.resistance * self.vel) / self.mass
        # self.vel = self.vel + self.accel * dt
        # self.position = self.position + self.vel * dt

    def smith_com(self, input, dt, target_position, target_vel):
        accel = (input - self.resistance * self.vel) / self.mass
        vel = self.vel + accel * dt
        position = self.position + vel * dt
        # # smc stc
        # s = 10 * (target_position - position)
        # ffstc
        s = 10 * (position - target_position)
        return s