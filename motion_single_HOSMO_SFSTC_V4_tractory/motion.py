import numpy as np
import pickle
import model
import scipy.sparse as sparse
# import osqp
import scipy.linalg as sl
import math

from simulation import theta

# sx = 0.0  # start x position [m]
# sy = 0.0  # start y position [m]
# syaw = np.deg2rad(0.0)  # start yaw angle [rad]
# sv = 0.001  # start speed [m/s]
# sa = 0.0  # start accel [m/ss]
#
# gx = 0.0/100  # goal x position [m]
# gy = 4.0/100  # goal y position [m]
# gyaw = np.deg2rad(180.0)  # goal yaw angle [rad]
# gv = 0.001  # goal speed [m/s]
# ga = 0.0  # goal accel [m/ss]
#
# max_accel = 0.004  # max accel [m/ss]
# max_jerk = 0.002  # max jerk [m/sss]
# dt = 0.05  # interval time [s]
#
# ppc_lamda = np.diag([20, 20])
# ppc_gamma = np.diag([10, 10])
# ppc_upsilon = np.diag([20, 20])
# gain = 40

# 设置速度参数
alpha = np.radians(0)
beta = np.radians(0)
miu_0 = 4 * np.pi * 10 ** -7
frequency = 5  # 频率，单位：Hz


# 磁场幅值
amplitude = 0.0005 # 单位 T

r = 20        # 半径 (m)
# v_max = 2.5    # 恒定速度 (m/s)
v_max = 2.5    # 恒定速度 (m/s)
omega = v_max / r  # 角速度 (rad/s)
T = 2 * np.pi / omega  # 运动周期 (s)

# =====================
# 生成轨迹数据
# =====================
t_start = 0         # 起始时间 (s)
t_end = T           # 结束时间 (s) - 完整周期
dt = 0.035           # 时间步长 (s)
dt_ms = 35

det_L_augumented_standard = 1e-23
last_current_error_robust_weight = 0.95
Ks = np.zeros((4, 1))

magnetic_model = model.magnetic_motion_model(dt=dt, position=[[8], [0]], vel=[[0.0001],[0.0001]], theta = np.pi/2, current_limit = 10)

def generate_follow_path():
    global magnetic_models
    # magnetic_model.ref_time, magnetic_model.ref_x, magnetic_model.ref_y, magnetic_model.ref_vx, magnetic_model.ref_vy, magnetic_model.ref_ax, magnetic_model.ref_ay = QuinticPolynomial.quintic_polynomials_planner(
    #     sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt)

    # 轨迹跟踪
    global r
    magnetic_model.ref_time = np.arange(t_start, t_end + dt, dt)
    magnetic_model.ref_x = r * np.cos(omega * magnetic_model.ref_time)
    magnetic_model.ref_y = r * np.sin(omega * magnetic_model.ref_time)
    magnetic_model.ref_vx = -r * omega * np.sin(omega * magnetic_model.ref_time)
    magnetic_model.ref_vy = r * omega * np.cos(omega * magnetic_model.ref_time)
    magnetic_model.ref_ax = -r * omega ** 2 * np.cos(omega * magnetic_model.ref_time)
    magnetic_model.ref_ay = -r * omega ** 2 * np.sin(omega * magnetic_model.ref_time)
    magnetic_model.ref_angles = 0.2 * magnetic_model.ref_time   # 直接计算各时间点角度

    # 组合成路径数组 (N×2的矩阵)
    magnetic_model.ref_path = np.column_stack((magnetic_model.ref_x, magnetic_model.ref_y))
    magnetic_model.ref_velpath = np.column_stack((magnetic_model.ref_vx, magnetic_model.ref_vy))
    magnetic_model.ref_accelpath = np.column_stack((magnetic_model.ref_ax, magnetic_model.ref_ay))

    # # 路近点跟踪
    # global r
    # magnetic_model.ref_time = np.arange(t_start, t_end + dt, dt)
    # # 组合成路径数组 (N×2的矩阵)
    # N = len(magnetic_model.ref_time)  # 总点数
    # n_quarter = N // 2  # 每段 1/4 时间的点数
    #
    # magnetic_model.ref_x_1 = r * np.cos(np.pi/2)
    # magnetic_model.ref_x_2 = r * np.cos(np.pi)
    # magnetic_model.ref_y_1 = r * np.sin(np.pi/2)
    # magnetic_model.ref_y_2 = r * np.sin(np.pi)
    # magnetic_model.ref_vx = np.zeros(N)
    # magnetic_model.ref_vy = np.zeros(N)
    # magnetic_model.ref_ax = np.zeros(N)
    # magnetic_model.ref_ay = np.zeros(N)
    #
    # # 构造 ref_x
    # magnetic_model.ref_x = np.zeros(N)
    # magnetic_model.ref_x[:n_quarter] = magnetic_model.ref_x_1  # 前 1/4 时间
    # magnetic_model.ref_x[n_quarter:2 * n_quarter] = magnetic_model.ref_x_2  # 接下来的 1/4
    # magnetic_model.ref_y = np.zeros(N)
    # magnetic_model.ref_y[:n_quarter] = magnetic_model.ref_y_1
    # magnetic_model.ref_y[n_quarter:2 * n_quarter] = magnetic_model.ref_y_2
    #
    # magnetic_model.ref_path = np.column_stack((magnetic_model.ref_x, magnetic_model.ref_y))
    # magnetic_model.ref_velpath = np.column_stack((magnetic_model.ref_vx, magnetic_model.ref_vy))
    # magnetic_model.ref_accelpath = np.column_stack((magnetic_model.ref_ax, magnetic_model.ref_ay))


def return_current_position():
    global magnetic_model
    return magnetic_model.return_position()

def update_status(current_positon):
    global magnetic_model
    magnetic_model.update_status(current_positon)

# clf_initialization
K1 = np.array([[1, 0, 1, 0],
               [0, 1, 0, 1]])
Q = np.eye(4)
epsilon = 1
c3 = 1

import cv2
import numpy as np

kf = cv2.KalmanFilter(4, 2)  # 4 维状态（x, y, vx, vy），2 维观测（x, y）
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

def calculate_angle(A, B):
    delta_y = B[1] - A[1]
    delta_x = B[0] - A[0]
    # 这边计算角度的没懂，不过应该是微分法，里头应该存了上一时刻的位置和本时刻位置
    angle = np.arctan2(delta_y, delta_x)  # 计算角度（弧度）
    # if (B[0] - A[0]) >= 0:
    #     angle = np.arctan2(delta_y, delta_x)  # 计算角度（弧度）
    # else:
    #     angle = np.arctan2(delta_y, delta_x) + np.pi # 计算角度（弧度）
    # 这个是防止打转
    theta = angle if angle >= 0 else angle + 2 * np.pi
    # # 保证是正数
    return theta

cnt = 0
def adjust_angle():
    # 小作弊的用法，用于直接设定初始位移角度
    global cnt
    magnetic_model.theta = calculate_angle(magnetic_model.return_position().reshape(-1), magnetic_model.ref_path[0])
    magnetic_model.vel = np.array([[0.001*np.cos(magnetic_model.theta)], [0.001*np.sin(magnetic_model.theta)]])
    u_x, u_y, u_z = np.sin(alpha), -np.cos(alpha), 0  # 假设 x 方向单位向量
    v_x, v_y, v_z = np.cos(alpha) * np.cos(magnetic_model.theta), np.sin(alpha) * np.cos(magnetic_model.theta), -np.sin(magnetic_model.theta)  # 转向
    cHx = amplitude * (u_x * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_x * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    cHz = amplitude * (u_y * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_y * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    cHy = amplitude * (u_z * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_z * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    # 生成对应的电流
    IHx = cHx / ((4 / 5) ** (3 / 2) * miu_0 * 297 / 0.236)
    IHz = cHz / ((4 / 5) ** (3 / 2) * miu_0 * 202 / 0.162)
    IHy = cHy / ((4 / 5) ** (3 / 2) * miu_0 * 129 / 0.1)

    # cnt += 1
    return IHx, IHy, IHz

part2 = 0
def motion_control(i_motion, dt):
    global magnetic_model,part2,beita,cost_state,L_state,exceed_flag_state,Ks,last_current_error_robust

    if i_motion < len(magnetic_model.ref_time):
        time, xd, yd, vxd, vyd, axd, ayd = magnetic_model.ref_time[i_motion], magnetic_model.ref_x[i_motion], magnetic_model.ref_y[i_motion], magnetic_model.ref_vx[i_motion], magnetic_model.ref_vy[i_motion], magnetic_model.ref_ax[i_motion], magnetic_model.ref_ay[i_motion]
        beita = np.zeros((4, 1))
        if time < 0.07:
            last_current_error_robust = np.zeros((4, 1))

        if i_motion == 0:
            kf.statePost = np.array([[magnetic_model.position[0,0]], [magnetic_model.position[1,0]], [magnetic_model.vel[0,0]], [magnetic_model.vel[1,0]]], dtype=np.float32)
        else:
            kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                                  [0, 1, 0, dt],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
            # 预测
            prediction = kf.predict()
            # 更新
            kf.correct(magnetic_model.return_position().reshape(-1).astype(np.float32))
            # 获取估计的 (x, y) 和速度 (vx, vy)
            estimated_x, estimated_y = kf.statePost[0, 0], kf.statePost[1, 0]
            estimated_vx, estimated_vy = kf.statePost[2, 0], kf.statePost[3, 0]
            magnetic_model.vel = np.array([[estimated_vx], [estimated_vy]])
            # ZLF,这个地方ysh觉得有问题，因为卡尔曼是迭代的过程，但是这套设备对于磁体位置的矫正速度很快，直接认为磁场方向和磁体方向一致
            # magnetic_model.theta = math.atan2(estimated_vy, estimated_vx)
            if magnetic_model.theta < 0:
                magnetic_model.theta += 2 * np.pi

        magnetic_model.error = np.array(magnetic_model.return_position()) - np.array([[xd], [yd]])
        error = magnetic_model.error
        dot_error = np.array([[vxd], [vyd]]) - np.array(magnetic_model.return_vel())
        ddot_error = np.array([[axd], [ayd]]) - np.array(magnetic_model.return_accel())

        # 微分平坦状态量构建
        current_state = np.array([error, dot_error])
        dot_current_state = np.array([dot_error, ddot_error])
        current_state = current_state.reshape(-1, 1)
        dot_current_state = dot_current_state.reshape(-1, 1)

        ############################################################################################################################################################
        # 控制器修改只修改######内的内容

        # 轨迹跟踪：

        # # original controller
        # sliding = dot_error + 0.5 * error
        # part2 = part2 + 0.1 * np.sign(sliding) * dt
        # input = magnetic_model.mass * np.array(
        #     [[axd], [ayd]]) + magnetic_model.resistance * magnetic_model.vel + 1 * sliding + 1 * np.abs(sliding) ** (
        #                     1 / 2) * np.sign(sliding) + part2

        sliding = dot_error + 0.1 * error
        input = 0.1 * np.sign(sliding) - np.array([[axd], [ayd]]) - 0.1 * error
        omega_vel_current = np.linalg.norm(magnetic_model.vel)
        trans_matrix = np.array([[np.cos(magnetic_model.theta), -omega_vel_current * np.sin(magnetic_model.theta)]
                               , [np.sin(magnetic_model.theta), omega_vel_current * np.cos(magnetic_model.theta)]])
        input_real = np.linalg.inv(trans_matrix) @ input
        avv_velocity = input_real[0, 0]
        vel_theta = input_real[1, 0]

        #
        # print(f"线加速度: {avv_velocity}, 角速度: {vel_theta}")
        ############################################################################################################################################################

        # 这边有问题
        # magnetic_model.input_log.append([input])  # 记录每一时刻的理论输入
        # magnetic_model.dynamic_position(input, dt)
        # 后面的包括了史密斯预估器，将F结算为角速度再到磁场旋转速度
        # u_ff = magnetic_model.smith_com(input, dt, np.array([[xd], [yd]]), np.array([[vxd], [vyd]]))
        # 其实史密斯预估器这边也不建议更改，还是保留
        # input = input + u_ff
        # magnetic_model.dynamic_position(input, dt)
        # 这边有问题

        # omega应该是指磁体要滚得多块，这边对应磁体边长上一点的线速度
        # (np.linalg.norm)用来求取二范数，这边由于vel里头是velx和vely，所以取二范数
        omega_vel = omega_vel_current + avv_velocity * dt
        print(f"avv_velocity:{avv_velocity}, ideal_vel:{np.linalg.norm([[vxd], [vyd]])}", )
        # 将线速度转化为角速度，除以2pi*r，这边的r应该是2mm的意思，在更换磁体的时候需要修改
        # print("orignal omgea: ", omega_vel)
        # 这边有一个速度的限幅

        if np.linalg.norm(magnetic_model.error) < 4:
            magnetic_model.close_flag = 1
            print(f"it's close")
        elif (magnetic_model.close_flag) == 1 and (np.linalg.norm(magnetic_model.error) > 6):
            magnetic_model.close_flag = 0
            print(f"it's not close")

        if magnetic_model.close_flag == 1:
            omega_vel = np.clip(omega_vel, -0.25, 0.25)
            # omega_vel = omega_vel * np.exp(np.linalg.norm(magnetic_model.error) - 6)
        else:
            omega_vel = np.clip(omega_vel, -0.3, 0.3)

        # omega_vel = np.clip(omega_vel, -0.35, 0.35)

        print("omega： ", omega_vel)
        # 当前的theta
        # print("theta： ", np.degrees(magnetic_model.theta))
        # 这边处理theta
        # magnetic_model.theta = magnetic_model.theta + vel_theta * dt
        magnetic_model.theta = calculate_angle(magnetic_model.return_position().reshape(-1),
                                               magnetic_model.ref_path[i_motion])
        # print("modify theta：", np.degrees(magnetic_model.theta))

        # # 不同角度的转动
        # magnetic_model.theta = 1/6 * np.pi
        # omega_vel = 10 / (2 * np.pi * 2)

        # 角度是需要矫正的，需要增加pi/2
        theta_applied = magnetic_model.theta - 1/2 * np.pi

        # # 用于绘制理论位置
        # add = np.array([omega_vel * (2 * np.pi * 2) * np.cos(magnetic_model.theta) * dt, omega_vel * (2 * np.pi * 2) *
        #                 np.sin(magnetic_model.theta) * dt])
        # magnetic_model.ideal_position = np.array(magnetic_model.return_position()) + add

        u_x, u_y, u_z = np.sin(alpha), -np.cos(alpha), 0  # 假设 x 方向单位向量
        v_x, v_y, v_z = np.cos(alpha) * np.cos(theta_applied), np.sin(alpha) * np.cos(theta_applied), -np.sin(theta_applied)  # 转向

        cHx = amplitude * (u_x * np.cos(2 * np.pi * omega_vel * magnetic_model.ref_time[i_motion]) - v_x * np.sin(2 * np.pi * omega_vel * magnetic_model.ref_time[i_motion]))
        cHz = amplitude * (u_y * np.cos(2 * np.pi * omega_vel * magnetic_model.ref_time[i_motion]) - v_y * np.sin(2 * np.pi * omega_vel * magnetic_model.ref_time[i_motion]))
        cHy = amplitude * (u_z * np.cos(2 * np.pi * omega_vel * magnetic_model.ref_time[i_motion]) - v_z * np.sin(2 * np.pi * omega_vel * magnetic_model.ref_time[i_motion]))

        # 生成对应的电流
        IHx = cHx / ((4 / 5) ** (3 / 2) * miu_0 * 297 / 0.236)
        IHz = cHz / ((4 / 5) ** (3 / 2) * miu_0 * 202 / 0.162)
        IHy = cHy / ((4 / 5) ** (3 / 2) * miu_0 * 129 / 0.1)
        # print("Ideal Current: ", IHx, IHy, IHz

        # # 进行限幅
        # IH = jkm
        # current_input = [max(min(i, magnetic_model.current_limit), -magnetic_model.current_limit) for i in IH]

        magnetic_model.state_flat_log.append(current_state)
        magnetic_model.dot_state_flat_log.append(dot_current_state)
        magnetic_model.input_log.append(input)  # 记录每一时刻的加上smith补偿的理论输入
        magnetic_model.current_log.append([IHx, IHy, IHz])
        magnetic_model.error_log.append(error)
        magnetic_model.path_log.append(magnetic_model.return_position())
        magnetic_model.vel_log.append(magnetic_model.return_vel())

        # # 在 UI 线程中更新 QTextEdit 内容
        # str_1 = "target position: " + str(np.array([[xd], [yd]]).reshape(-1).tolist()) + "\r\n"
        # str_2 = "current position: " + str(np.array(magnetic_model.return_position()).reshape(-1).tolist()) + "\r\n"
        # str_3 = "current input: " + str(current_input) + "\r\n"
        # str_4 = "error: " + str(error.reshape(-1)) + "\r\n"
        #
        # text_edit.setPlainText(str_1)
        # text_edit.insertPlainText(str_2)
        # text_edit.insertPlainText(str_3)
        # text_edit.insertPlainText(str_4)

    else:
        IHx = 0
        IHy = 0
        IHz = 0
    return IHx, IHy, IHz

def motion_log():
    # 将列表打包成一个字典或列表
    data = {
        "time": magnetic_model.ref_time,
        "ref_path": magnetic_model.ref_path,
        "input": magnetic_model.input_log,
        "path": magnetic_model.path_log,
        "velocity": magnetic_model.vel_log,
        "error": magnetic_model.error_log,
        "current": magnetic_model.current_log,
        "state_flat": magnetic_model.state_flat_log,
        "dot_state_flat": magnetic_model.dot_state_flat_log

        # "input": magnetic_model.input_log,
        # "input_smith": magnetic_model.input_smith_log,
        # "valid_input": magnetic_model.delay_input_log,
        # "theta": magnetic_model.theta_log
        # "L_state": magnetic_model.L_state_log,"cost_state": magnetic_model.cost_state_log,
        # "exceed_flag_state": magnetic_model.exceed_flag_state_log,
        # "current_error_robust_state": magnetic_model.current_error_robust_log,
        # "det_L_augumented_state": magnetic_model.det_L_augumented_log,
        # "Ks_state": magnetic_model.Ks_log,

    }

    # 保存到一个 pkl 文件
    with open("ffstc_data.pkl", "wb") as f:
        pickle.dump(data, f)

    # 创建一个新的Excel工作簿
    # workbook = openpyxl.Workbook()
    # # 选择要操作的工作表
    # sheet = workbook.active
    # draw.draw_log(magnetic_model.ref_time, magnetic_model.ref_x, magnetic_model.ref_y, magnetic_model.ref_v, magnetic_model.ref_yaw, magnetic_model, magnetic_ppc)