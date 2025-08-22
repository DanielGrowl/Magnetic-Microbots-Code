import math

import matplotlib.pyplot as plt
import numpy as np
import os

save_path = "./pic"

if not os.path.exists(save_path):
    os.makedirs(save_path)


def draw_log(ref_time, ref_x, ref_y, ref_vx, ref_vy, model):
    print("draw log picture")
    xd_axis = [log for log in ref_x]
    yd_axis = [log for log in ref_y]
    x_axis = [log[0, 0] for log in model.path_log]
    y_axis = [log[1, 0] for log in model.path_log]

    fig = plt.figure(figsize=(12, 10))
    plt.plot(xd_axis, yd_axis, "r")
    plt.plot(x_axis, y_axis, "b")
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # 设置x轴和y轴为相同尺度
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(save_path + "/path")

    # 创建一个新的图形
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(ref_time, xd_axis, label='xd', color='r')
    ax1.plot(ref_time, x_axis, label='x', color='b')
    ax1.set_title('X Position')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(ref_time, yd_axis, label='yd', color='r')
    ax2.plot(ref_time, y_axis, label='y', color='b')
    ax2.set_title('Y Position')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Position')
    ax2.legend()
    ax2.grid(True)
    plt.savefig(save_path + "/div_path")

    x_current = [log[0, 0] for log in model.current_log]
    y_current = [log[1, 0] for log in model.current_log]
    # 创建一个新的图形
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
    ax3.plot(ref_time, x_current, label='x_current', color='b')
    ax3.set_title('X Current')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Current')
    ax3.legend()
    ax3.grid(True)

    ax4.plot(ref_time, y_current, label='y_current', color='b')
    ax4.set_title('Y Current')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('Current')
    ax4.legend()
    ax4.grid(True)
    # 设置相同的 y 轴范围
    ymin = min(min(x_current), min(y_current))
    ymax = max(max(x_current), max(y_current))
    ax3.set_ylim(ymin, ymax)
    ax4.set_ylim(ymin, ymax)

    # 从 current_log 提取 x 轴记录（对应二维向量的第一个分量）
    x_error = [log[0, 0] for log in model.error_log]
    y_error = [log[1, 0] for log in model.error_log]
    # limit_up = [log * ppc.limit_up for log in model.pt_log]
    # limit_down = [log * -ppc.limit_down for log in model.pt_log]

    # 创建一个新的图形
    fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(12, 10))
    ax5.plot(ref_time, x_error, label='x_error', color='b')
    ax5.set_title('X Error')
    ax5.set_xlabel('Time (t)')
    ax5.set_ylabel('Error')
    ax5.legend()
    ax5.grid(True)

    ax6.plot(ref_time, y_error, label='y_error', color='b')
    ax6.set_title('Y Error')
    ax6.set_xlabel('Time (t)')
    ax6.set_ylabel('Error')
    ax6.legend()
    ax6.grid(True)
    # 设置相同的 y 轴范围
    ymin = min(min(x_error), min(y_error))
    ymax = max(max(x_error), max(y_error))
    ax5.set_ylim(ymin, ymax)
    ax6.set_ylim(ymin, ymax)
    plt.savefig(save_path + "/error")

    vxd_axis = [log for log in ref_vx]
    vyd_axis = [log for log in ref_vy]
    x_vel = [log[0, 0] for log in model.vel_log]
    y_vel = [log[1, 0] for log in model.vel_log]
    # 创建一个新的图形
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
    ax3.plot(ref_time, x_vel, label='vx', color='b')
    ax3.plot(ref_time, vxd_axis, label='vxd', color='r')
    ax3.set_title('X vel')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('vel')
    ax3.legend()
    ax3.grid(True)

    ax4.plot(ref_time, y_vel, label='vy', color='b')
    ax4.plot(ref_time, vyd_axis, label='vyd', color='r')
    ax4.set_title('Y vel')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('vel')
    ax4.legend()
    ax4.grid(True)

    # 设置相同的 y 轴范围
    # 计算所有数据的全局最小值和最大值
    ymin = min(min(x_vel), min(vxd_axis), min(y_vel), min(vyd_axis))
    ymax = max(max(x_vel), max(vxd_axis), max(y_vel), max(vyd_axis))

    # 设置相同的 y 轴范围
    ax3.set_ylim(ymin, ymax)
    ax4.set_ylim(ymin, ymax)
    plt.savefig(save_path + "/vel")

    fn_x = [log[0, 0] for log in model.fn_log]
    fn_y = [log[1, 0] for log in model.fn_log]
    fx_x = [log[0, 0] for log in model.fitting_log]
    fx_y = [log[0, 0] for log in model.fitting_log]
    # 创建一个新的图形
    fig6, (ax11, ax12) = plt.subplots(2, 1, figsize=(12, 10))
    ax11.plot(ref_time, fn_x, label='fn_x', color="r")
    ax11.plot(ref_time, fx_x, label='hat_fn_x', color="b")
    ax11.set_title('fx_x')
    ax11.set_xlabel('Time (t)')
    ax11.set_ylabel('fx_x')
    ax11.legend()
    ax11.grid(True)

    ax12.plot(ref_time, fn_y, label='fn_y', color="r")
    ax12.plot(ref_time, fx_y, label='hat_fn_y',color="b")
    ax12.set_title('fx_y')
    ax12.set_xlabel('Time (t)')
    ax12.set_ylabel('fx_y')
    ax12.legend()
    ax12.grid(True)

    # 显示图形
    plt.show()
