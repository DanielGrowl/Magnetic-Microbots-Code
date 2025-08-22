import numpy as np
import pickle
import model

from hermite_trajectory_gen import (
    generate_follow_path_from_theta_autotime,
    generate_circle_trajectory,
    generate_sine_trajectory,
    generate_square_trajectory
)

# 设置速度参数
alpha = np.radians(0)
beta = np.radians(0)
miu_0 = 4 * np.pi * 10 ** -7
frequency = 5  # 频率，单位：Hz

# 磁场幅值
amplitude = 0.0005 # 单位 T

# =====================
# 生成轨迹数据
# =====================
t_com = 0        # 用于延时补偿的
dt = 0.035           # 时间步长 (s)
dt_ms = 35

# 初始的最近的跟踪点
selected_index = 0
# 一定时间后改变min_track_distance
tracking_frame_counter = 0
min_track_distance_warmup_duration = int(2.0 / dt)  # 初始宽容时间（例如2秒）

det_L_augumented_standard = 1e-23
last_current_error_robust_weight = 0.95
Ks = np.zeros((4, 1))

magnetic_model = model.magnetic_motion_model(dt=dt, position=[[8], [0]], vel=[[0.0001],[0.0001]], theta = np.pi/2, current_limit = 10)

def generate_follow_path(trajectory_type='circle', **kwargs):
    global magnetic_model

    if trajectory_type == 'circle':
        path, vel, accel, ref_time = generate_circle_trajectory()
    elif trajectory_type == 'sine':
        path, vel, accel, ref_time = generate_sine_trajectory()
    elif trajectory_type == 'square':
        path, vel, accel, ref_time = generate_square_trajectory()
    elif trajectory_type == 'hermite':
        x, y, theta = kwargs['x'], kwargs['y'], kwargs['theta']
        ref_time = np.arange(0, 6.0, 0.035)  # 或者你自己传进来
        path, vel, accel, ref_time = generate_follow_path_from_theta_autotime(x, y, theta)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    # 保存 ref_time 及参考轨迹
    magnetic_model.ref_time = ref_time
    magnetic_model.ref_x, magnetic_model.ref_y = path[:, 0], path[:, 1]
    magnetic_model.ref_vx, magnetic_model.ref_vy = vel[:, 0], vel[:, 1]
    magnetic_model.ref_ax, magnetic_model.ref_ay = accel[:, 0], accel[:, 1]
    magnetic_model.ref_path = path
    magnetic_model.ref_velpath = vel
    magnetic_model.ref_accelpath = accel

def return_current_position():
    global magnetic_model
    return magnetic_model.return_position()

def update_status(current_positon):
    global magnetic_model
    magnetic_model.update_status(current_positon)

def get_selected_index():
    global selected_index
    return selected_index

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
    # 0峰的
    angle = np.arctan2(delta_y, delta_x)  # 计算角度（弧度）
    theta = angle if angle >= 0 else angle + 2 * np.pi

    # # YSH的
    # # 这边计算角度的没懂，不过应该是微分法，里头应该存了上一时刻的位置和本时刻位置
    # angle = np.arctan2(delta_y, delta_x)  # 计算角度（弧度）
    # if (B[0] - A[0]) >= 0:
    #     angle = np.arctan2(delta_y, delta_x)  # 计算角度（弧度）
    # else:
    #     angle = np.arctan2(delta_y, delta_x) + np.pi # 计算角度（弧度）
    # 这个是防止打转
    # theta = angle if angle >= 0 else angle + 2 * np.pi
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
    global selected_index, tracking_frame_counter, min_track_distance_warmup_duration

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

        # 这边用于跟踪最近的目标点，胡萝卜算法 =================================================================================
        # 初始几秒使用较大 min_track_distance ==============================================================================
        if tracking_frame_counter < min_track_distance_warmup_duration:
            min_track_distance = 4.0  # 初始宽松距离（单位 mm）
        else:
            min_track_distance = 2.0  # 后续精细跟踪
        # 当前的点
        current_position = np.array(magnetic_model.return_position())
        error = magnetic_model.error
        current_vel = np.array(magnetic_model.return_vel())
        # 遍历从 i_motion 开始的后续点
        for k in range(selected_index, len(magnetic_model.ref_path)):
            dist_k = np.linalg.norm(magnetic_model.ref_path[k] - current_position.T)
            if dist_k >= min_track_distance:
                selected_index = k
                tracking_frame_counter += 1
                break  # 找到满足距离的最前点即退出
        # 使用新的目标点
        xd, yd = magnetic_model.ref_path[selected_index]
        vxd, vyd = magnetic_model.ref_velpath[selected_index]
        axd, ayd = magnetic_model.ref_accelpath[selected_index]
        # 更新误差计算
        resized_error = np.array([[xd], [yd]]) - current_position
        resized_dot_error = np.array([[vxd], [vyd]]) - current_vel
        resized_error_norm = np.linalg.norm(resized_error)

        # 计算 error、dot_error、ddot_error（全部基于 selected_index）
        target_pos = np.array([[xd], [yd]])
        target_vel = np.array([[vxd], [vyd]])
        target_acc = np.array([[axd], [ayd]])

        error = current_position - target_pos
        dot_error = target_vel - current_vel
        ddot_error = target_acc - np.array(magnetic_model.return_accel())

        # 微分平坦状态量构建
        current_state = np.array([error, dot_error])
        dot_current_state = np.array([dot_error, ddot_error])
        current_state = current_state.reshape(-1, 1)
        dot_current_state = dot_current_state.reshape(-1, 1)

        # # 计算偏差：在ref_path中找到离current_position最近的点，然后计算二范数作为登记的error，这个error只是用来计算偏离轨迹的程度
        # === 在划定的search_range参考轨迹中查找最近点 =======================================================================
        # search_range = 20
        # start_index = max(0, i_motion - search_range)
        # end_index = min(len(magnetic_model.ref_path), i_motion + search_range + 1)
        # local_points = magnetic_model.ref_path[start_index:end_index].T  # 形状 (2, N)
        # differences = local_points - current_position  # 广播相减，形状 (2, N)
        # squared_distances = np.sum(differences ** 2, axis=0)
        # closest_index = np.argmin(squared_distances)
        # closest_point = magnetic_model.ref_path[start_index + closest_index]
        # closest_vel = magnetic_model.ref_velpath[start_index + closest_index]
        # resized_error = closest_point - current_position.T
        # resized_error_norm = np.linalg.norm(resized_error)
        # resized_dot_error = closest_vel - current_vel.T

        # === 在整个参考轨迹中查找最近点 =====================================================================================
        local_points = magnetic_model.ref_path.T  # 形状 (2, N)
        differences = local_points - current_position  # 广播相减，形状 (2, N)
        squared_distances = np.sum(differences ** 2, axis=0)
        closest_index = np.argmin(squared_distances)
        closest_point = magnetic_model.ref_path[closest_index]
        closest_vel = magnetic_model.ref_velpath[closest_index]
        resized_error = closest_point - current_position.T
        # resized_error_norm = np.linalg.norm(resized_error)
        resized_dot_error = closest_vel - current_vel.T
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

        # print(f"sliding: {sliding}, input: {input}, error: {error}")
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
        print(f"avv_velocity:{avv_velocity * dt}, cur_vel:{omega_vel_current}, omega_vel:{omega_vel}")
        # 将线速度转化为角速度，除以2pi*r，这边的r应该是2mm的意思，在更换磁体的时候需要修改
        # print("orignal omgea: ", omega_vel)
        # 这边有一个速度的限幅

        if np.linalg.norm(magnetic_model.error) < 4:
            magnetic_model.close_flag = 1
            # print(f"it's close")
        elif (magnetic_model.close_flag) == 1 and (np.linalg.norm(magnetic_model.error) > 6):
            magnetic_model.close_flag = 0
            # print(f"it's not close")

        omega_vel = 0.4

        # # 圆专用：
        # N = len(magnetic_model.ref_time)
        # # if 1.5 * N/14 <= i_motion <= (4.5*N/14) or (11*N/14) <= i_motion <= (14*N/14) :
        # if i_motion <= N / 4:
        #     omega_vel = np.clip(omega_vel, -0.3, 0.3)
        # # elif (3*N / 14) <= i_motion <= (4.5 * N / 14):
        # #         omega_vel = np.clip(omega_vel, -0.38, 0.38)
        # # elif i_motion <= (N/14):
        # #     omega_vel = np.clip(omega_vel, -0.45, 0.45)
        # else :
        #     omega_vel = np.clip(omega_vel, -0.3, 0.3)

        # # 正弦轨线专用：
        # N = len(magnetic_model.ref_time)
        # # if 1.5 * N/14 <= i_motion <= (4.5*N/14) or (11*N/14) <= i_motion <= (14*N/14) :
        # if 1.5 * N / 14 <= i_motion <= (4.5 * N / 14):
        #     omega_vel = np.clip(omega_vel, -0.24, 0.24)
        # # elif (3*N / 14) <= i_motion <= (4.5 * N / 14):
        # #         omega_vel = np.clip(omega_vel, -0.38, 0.38)
        # # elif i_motion <= (N/14):
        # #     omega_vel = np.clip(omega_vel, -0.45, 0.45)
        # else :
        #     omega_vel = np.clip(omega_vel, -0.26, 0.26)

        # # square专用：
        # N = len(magnetic_model.ref_time)
        # # if 1.5 * N/14 <= i_motion <= (4.5*N/14) or (11*N/14) <= i_motion <= (14*N/14) :
        # if 0.9 * N / 4 <= i_motion <= 3.9 * N / 4:
        #     omega_vel = np.clip(omega_vel, -0.55, 0.55)
        # # elif (3*N / 14) <= i_motion <= (4.5 * N / 14):
        # #         omega_vel = np.clip(omega_vel, -0.38, 0.38)
        # # elif i_motion <= (N/14):
        # #     omega_vel = np.clip(omega_vel, -0.45, 0.45)
        # elif 3.9 * N / 4 <= i_motion:
        #     omega_vel = np.clip(omega_vel, -0.6, 0.6)
        # else :
        #     omega_vel = np.clip(omega_vel, -0.58, 0.58)

        print("omega： ", omega_vel)
        # 当前的theta
        # print("theta： ", np.degrees(magnetic_model.theta))
        # 这边处理theta
        # magnetic_model.theta = magnetic_model.theta + vel_theta * dt
        magnetic_model.theta = calculate_angle(magnetic_model.return_position().reshape(-1),
                                               magnetic_model.ref_path[selected_index])
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

        # 偏差
        magnetic_model.resized_error_log.append(resized_error)
        magnetic_model.resized_vel_error_log.append(resized_dot_error)


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
        "dot_state_flat": magnetic_model.dot_state_flat_log,
        "resized_error": magnetic_model.resized_error_log,
        "resized_vel_error": magnetic_model.resized_vel_error_log,
        "theta_deg":magnetic_model.theta_deg_log

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