import numpy as np
import time
# import osqp
import matplotlib.pyplot as plt
import argparse
import scipy.sparse as sparse
import os
import scipy.linalg as sl
import math

parser = argparse.ArgumentParser()
parser.add_argument('--sim_time', type=int, default=40)
parser.add_argument('--dt', type=float, default=0.035)
args = parser.parse_args()
sim_time = args.sim_time
dt = args.dt

Steps = int(sim_time / dt)

# clf_initialization
K1 = np.array([[1, 0, 1, 0],
               [0, 1, 0, 1]])
Q = np.eye(4)
epsilon = 1
c3 = 1

Ft = lambda t: np.array([10 * np.cos(0.2 * t), 10 * np.sin(0.2 * t)])
dFt = lambda t: np.array([-10 * 0.2 * np.sin(0.2 * t), 10 * 0.2 * np.cos(0.2 * t)])
ddFt = lambda t: np.array([-10 * 0.04 * np.cos(0.2 * t), 10 * 0.2 * 0.2 * -1 * np.sin(0.2 * t)])

position_log = []
vel_log = []
theta_log = []
freq_log = []

ref_pos_log = []
ref_vel_log = []
ref_theta_log = []
ref_freq_log = []

mul_log = []
U_log = []

position =  np.array([8, 0])
vel =  np.array([0.0001, 0.0001])
accel = np.array([0, 0])
theta = math.atan2(vel[1], vel[0])  # 计算弧度
if theta < 0:
    theta += 2 * math.pi  # 转换到 0 到 2π

def main():
    global ref_pos_log, ref_vel_log, ref_theta_log, theta, position, vel, accel, ref_freq_log, position_log, vel_log, theta_log, freq_log, mul_log, U_log
    save_dir = 'simulation_path_jpg'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for iters in np.arange(0, sim_time, dt):

        F_d = Ft(iters)
        dF_d = dFt(iters)
        ddF_d = ddFt(iters)
        thetad = math.atan2(dF_d[1], dF_d[0])  # 计算弧度
        if thetad < 0:
            thetad += 2 * math.pi  # 转换到 0 到 2π

        ref_pos_log.append(F_d)
        ref_vel_log.append(dF_d)
        ref_theta_log.append(thetad)

        mul, B = qp_res_clf_disturb(position, vel, theta, F_d, dF_d, ddF_d)

        position, vel, accel, U, theta = bicycle_dynamics(position, vel, accel, ddF_d, B, mul)

        U_log.append(U)
        position_log.append(position)
        vel_log.append(vel)
        mul_log.append(mul)
        theta_log.append(theta)

        print('t_one_iters:', iters)

    position_log = np.stack(position_log)
    vel_log = np.stack(vel_log)
    theta_log = np.stack(theta_log)

    ref_pos_log = np.stack(ref_pos_log)
    ref_vel_log = np.stack(ref_vel_log)
    ref_theta_log = np.stack(ref_theta_log)

    U_log = np.stack(U_log)
    mul_log = np.stack(mul_log)

    tout = np.linspace(0, sim_time, Steps)
    tout1 = tout[1:]

    plt.figure()
    plt.plot(position_log[:, 0], position_log[:, 1], label='true')
    plt.plot(ref_pos_log[:, 0], ref_pos_log[:, 1], label='ref')
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.legend()
    plt.savefig(save_dir + "/twoD.png")

    # plt.figure()
    # plt.plot(tout1, position_log[: -1, 0], label='position_x')
    # plt.plot(tout1, ref_pos_log[:, 0], label='ref_x')
    # plt.xlabel("t / s")
    # plt.ylabel("position / m")
    # plt.legend()
    # plt.savefig(save_dir + '/position_x')
    #
    # plt.figure()
    # plt.plot(tout1, position_log[: -1, 1], label='position_y')
    # plt.plot(tout1, ref_pos_log[:, 1], label='ref_y')
    # plt.xlabel("t / s")
    # plt.ylabel("position / m")
    # plt.legend()
    # plt.savefig(save_dir + '/position_y')
    #
    # plt.figure()
    # plt.plot(tout1, position_log[0: -1, 0] - ref_pos_log[:, 0], label='error_x')
    # plt.plot(tout1, position_log[0: -1, 1] - ref_pos_log[:, 1], label='error_y')
    # plt.xlabel("t / s")
    # plt.ylabel("error / m")
    # plt.legend()
    # plt.savefig(save_dir + "/error_x_y")

    # plt.figure()
    # plt.plot(tout1, dF_log[:, 0], label='vx')
    # plt.plot(tout1, ref_dF_log[:, 0], label='ref_vx')
    # plt.xlabel("t / s")
    # plt.ylabel("velocity / m/s")
    # plt.title('vx')
    # plt.legend()
    # plt.savefig(save_dir + '/vx')
    #
    # plt.figure()
    # plt.plot(tout1, freq_log, label='freq')
    # plt.plot(tout1, ffreq_log, label='ffreq')
    # plt.xlabel("t / s")
    # plt.ylabel("freq / m/s")
    # plt.title('freq')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(tout1, dF_log[:, 1], label='vy')
    # plt.plot(tout1, ref_dF_log[:, 1], label='ref_vy')
    # plt.xlabel("t / s")
    # plt.ylabel("velocity / m/s")
    # plt.title('vy')
    # plt.legend()
    # plt.savefig(save_dir + '/vy')
    #
    # plt.figure()
    # plt.plot(tout1, dF_log[:, 0] - ref_dF_log[:, 0], label='error_vx')
    # plt.plot(tout1, dF_log[:, 1] - ref_dF_log[:, 1], label='error_vy')
    # plt.xlabel("t / s")
    # plt.ylabel("error / m/s")
    # plt.legend()
    # plt.savefig(save_dir + "/error_vx_vy")
    #
    # plt.figure()
    # plt.plot(tout1, U_log[:, 0], label='U[0]_eta')
    # plt.plot(tout1, U_log[:, 1], label='U[1]_w')
    # plt.xlabel("t / s")
    # plt.ylabel("m/s**2_or_rad/s")
    # plt.legend()
    # plt.savefig(save_dir + "/U")
    #
    # plt.figure()
    # plt.plot(tout1, mul_log[:, 0], label='mul[0]')
    # # plt.plot(tout1, mul_star_log[:, 0], label='mul_star[0]')
    # plt.xlabel("t / s")
    # plt.ylabel("m/s**2")
    # plt.legend()
    # plt.savefig(save_dir + "/mul_mulstar0")
    #
    # plt.figure()
    # plt.plot(tout1, mul_log[:, 1], label='mul[1]')
    # # plt.plot(tout1, mul_star_log[:, 1], label='mul_star[1]')
    # plt.xlabel("t / s")
    # plt.ylabel("m/s**2")
    # plt.legend()
    # plt.savefig(save_dir + "/mul_mulstar1")
    #
    # plt.figure()
    # plt.plot(tout1, mul_u_log[:], label='mul_u')
    # plt.xlabel("t / s")
    # plt.ylabel("a")
    # plt.legend()
    # plt.savefig(save_dir + "/mul_u")
    #
    # plt.figure()
    # plt.plot(tout, theta_log[:], label='theta')
    # plt.plot(tout, thetad_log[:], label='thetad')
    # plt.xlabel("t / s")
    # plt.ylabel("angle / rad")
    # plt.legend()
    # plt.savefig(save_dir + '/theta')
    #
    # plt.figure()
    # plt.plot(tout, theta_log[:] - thetad_log[:], label='error_theta')
    # plt.xlabel("t / s")
    # plt.ylabel("angle / rad")
    # plt.legend()
    # plt.savefig(save_dir + '/error_theta')



def bicycle_dynamics(F, dF, ddF, ddF_d, B, mul):
    U = np.linalg.inv(B) @ (ddF_d + mul)
    ddF = B @ U
    dF = ddF * dt + dF
    F = dF * dt + F
    theta = math.atan2(dF[1], dF[0])  # 计算弧度
    if theta < 0:
        theta += 2 * math.pi  # 转换到 0 到 2π
    return F, dF, ddF, U, theta


def qp_res_clf_disturb(F, dF, theta, F_d, dF_d, ddF_d):
    # clf
    e1 = F - F_d
    e2 = dF - dF_d
    e = np.concatenate((e1, e2))

    a1 = np.zeros((2, 2))
    b1 = np.eye(2)
    E1 = np.concatenate((a1, b1), axis=1)
    E2 = np.concatenate((a1, a1), axis=1)
    E = np.concatenate((E1, E2), axis=0)
    J = np.concatenate((a1, b1), axis=0)
    A_cl = E - J @ K1
    P = sl.solve_continuous_are(A_cl, np.zeros((4, 4), dtype=np.float32), Q, np.eye(4, dtype=np.float32))

    P_1 = np.eye(2) / epsilon
    P_2 = np.zeros((2, 2))
    P_3 = np.eye(2)
    P_11 = np.concatenate((P_1, P_2), axis=1)
    P_12 = np.concatenate((P_2, P_3), axis=1)
    P_ = np.concatenate((P_11, P_12), axis=0)
    P_epsilon = P_ @ P @ P_

    vel = np.linalg.norm(dF)

    B = np.zeros((2, 2))
    B[0, 0] = np.cos(theta)
    B[0, 1] = -vel * np.sin(theta)
    B[1, 0] = np.sin(theta)
    B[1, 1] = vel * np.cos(theta)


    l = e.T @ P_epsilon @ J
    x = np.array([e.T @ (E.T @ P_epsilon + P_epsilon @ E) @ e])
    V_epsilon = e.T @ P_epsilon @ e
    up = -(c3 / epsilon * V_epsilon + x)

    low = np.ones(up.shape) * np.inf * -1

    p_matrix = np.eye(2)
    p_matrix = sparse.csc_matrix(p_matrix)
    q_matrix = np.zeros((2, 1))
    # q_matrix = sparse.csc_matrix(q_matrix)

    A = 2 * l
    A = sparse.csc_matrix(A)

    prob = osqp.OSQP()
    prob.setup(P=p_matrix, q=q_matrix, A=A, l=low, u=up, verbose=False)
    res = prob.solve()

    if res.x[0] is None or res.x[1] is None or np.isnan(res.x).any():
        mul = np.ones(2)
        print("QP_clf_Failed!")
    else:
        mul = res.x[0: 2]

    return mul, B



if __name__ == '__main__':
    main()
    plt.show()
