import sys
import time
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import *
from Widgets_window import Ui_Form
import cv2
import numpy as np
import UartSerial
import PowerSupply
import Drive
import motion
import threading
import drawpath
from collections import deque
from utils_tracking import get_min_area_rect_from_defective_contours, has_valid_path, transform_local_batch_to_image,\
    compute_heading_angle_from_rect, draw_heading_arrow, draw_velocity_arrow, transform_local_spots_and_angles_to_global

port_state = False
motion_state = False
camera_state = False
video_state = True
i_motion = 0
pts = deque(maxlen=1240000)
circle_state = False
circle_position = (0, 0)
transform_state = False
transform_mile = 10
min_area = 3500 #   用于磁体检测
max_area = 4500 #   用于磁体检测
area_thresh_min = 100000   #   用于缺陷矩形检测
area_thresh_max = 200000   #   用于缺陷矩形检测
Camera_index = 0

# 这边定义了串口通讯的东西，应该是初始化的部分
uart = UartSerial.UartSerial()
power = PowerSupply.PowerSupply()
magnetic_drive = Drive.Drive(uart, power)

T = 2 * np.pi / 0.2  # 运动周期 (s)

# =====================
# 生成轨迹数据
# =====================
t_start = 0  # 起始时间 (s)
t_end = T  # 结束时间 (s) - 完整周期
# t_com = T/12        # 用于延时补偿的,只有正弦用到了
t_com = 0       # 用于延时补偿的
dt = 0.035  # 时间步长 (s)
dt_ms = 35
ref_time = np.arange(t_start, t_end + dt + t_com, dt)

# 创建一个空白图像（用于绘制路径）
global path_image
width, height = 1920, 1080  # 这边和摄像头保持一致
path_image = np.zeros((height, width, 3), dtype=np.uint8)
path_image = cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB)

# 用于获得磁体的前进方向的
prev_center = None  # 初始化
# 用于实现停车的路径规划的标志位
first_time_detected = True

class WinForm(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.comboBox_baud.addItem("921600", "921600")
        self.comboBox_baud.addItem("460800", "460800")
        self.comboBox_baud.addItem("230400", "230400")
        self.comboBox_baud.addItem("1000000", "1000000")
        self.comboBox_baud.addItem("38400", "38400")
        self.comboBox_baud.addItem("19200", "19200")
        self.comboBox_baud.addItem("9600", "9600")
        self.comboBox_baud.addItem("4800", "4800")
        self.comboBox_baud.setCurrentIndex(3)
        # 初始化槽函数
        self.slot_init()
        # 扫描串口
        self.refresh_serial_ports()
        # 初始化摄像头线程
        self.thread_camera = threading.Thread(target=self.detect)

    def slot_init(self):
        pass
        # 串口按钮
        self.button_open_serial.clicked.connect(self.serial_open_off)
        # 关闭串口按钮
        self.button_refresh_serial.clicked.connect(self.refresh_serial_ports)
        # 退出
        self.button_exit.clicked.connect(app.quit)

        # 摄像头函数
        self.button_camera.clicked.connect(self.camera_event)
        # path motion
        self.button_motion.clicked.connect(self.motion_event)

        # up
        self.button_up.clicked.connect(magnetic_drive.motion_up)
        # down
        self.button_down.clicked.connect(magnetic_drive.motion_down)
        # left
        self.button_left.clicked.connect(magnetic_drive.motion_left)
        # right
        self.button_right.clicked.connect(magnetic_drive.motion_right)
        # stop
        self.button_stop.clicked.connect(magnetic_drive.motion_stop)

    # 串口检测
    def refresh_serial_ports(self):
        _ports = uart.get_all_port()
        # print(_ports)
        self.comboBox_port.clear()
        if len(_ports) == 0:
            self.comboBox_port.addItem('')
        else:
            for item in _ports:
                self.comboBox_port.addItem(item)

    def serial_open_off(self):
        global port_state
        str = self.button_open_serial.text()
        #   这个命令存储了选定的是哪个串口以及是哪个波特率
        port_name = self.comboBox_port.currentText()
        baud_rate = int(self.comboBox_baud.currentText())

        # 这边要改一下，if的判断要改成判断是否使能成功的
        if str == '关闭串口':
            # 关闭电源
            ret = magnetic_drive.uninit_power()
            # port_close也是0f写的，具体的点进去看就行
            uart.CloseSerialPort()
            if ret:
                self.button_open_serial.setText('打开串口')
                port_state = False
            else:
                print("Close Uart Fail!!")

        # 这边也要改一下，if的判断要改成判断是否使能成功的
        if str == '打开串口':
            if uart.is_port_open():  # port_name, baud_rate):
                # 这边决定打开的是哪个串口和波特率，也就是在这一步把port_name和baud_rate传递给uarserial中的port的
                ret = uart.try_port_open(port_name, baud_rate)
                if ret:
                    self.button_open_serial.setText('关闭串口')
                    # 初始化电流
                    magnetic_drive.init_power()
                    port_state = True

    def camera_event(self):
        global camera_state
        str = self.button_camera.text()
        if str == '打开相机':
            # start the thread
            if self.thread_camera.is_alive():
                camera_state = True
            else:
                camera_state = True
                self.thread_camera.start()
            self.button_camera.setText('关闭相机')
        elif str == '关闭相机':
            camera_state = False
            self.button_camera.setText('打开相机')

    def motion_event(self):
        global i_motion, motion_state, path_image
        if not motion_state:
            print("start motion")
            motion_state = True
            path_image = drawpath.draw_path(path_image, motion.magnetic_model.ref_x, motion.magnetic_model.ref_y,
                                            circle_position)
            i_motion = 0
            self.button_motion.setText('stop motion')
        else:
            print("stop motion")
            motion_state = False
            self.button_motion.setText('start motion')
            motion.motion_log()

    def detect(self):
        global circle_position, circle_state, camera_state, i_motion, path_image
        global transform_state, transform_mile, circle_state, circle_position
        global prev_center, first_time_detected
####################################################################################
        # # 路径生成
        # # motion.generate_follow_path('circle')
        # # motion.generate_follow_path('sine')
        # # motion.generate_follow_path('square')
        #
        # # 这边用目标点定义一个hermite曲线
        # # x = [  ]  # 关键点的 x
        # # y = [...]  # 关键点的 y
        # # theta = [...]  # 关键点对应的面积方向角（单位：弧度）
        # r = 25
        # x = [r, 0, -r, 0, r]
        # y = [0, r, 0, -r, 0]
        # # theta = [np.pi / 2, np.pi, 3 * np.pi / 2, 0]
        # theta = [ np.pi / 2, np.pi / 2, - np.pi, - np.pi / 2, 0]
        # motion.generate_follow_path('hermite', x=x, y=y, theta=theta)
#####################################################################################

        cap = cv2.VideoCapture(Camera_index)
        # 设置摄像头的帧率为60fps 设置摄像头的分辨率为1080p（1920x1080）
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        print('HEIGHT:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WIDTH:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('FPS:', round(cap.get(cv2.CAP_PROP_FPS)))

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_output = cv2.VideoWriter('output.avi', fourcc, 30.0, (1920, 1080))
        video_output_notrackingpoint = cv2.VideoWriter('output_notrackingpoint.avi', fourcc, 30.0, (1920, 1080))
        if not video_output.isOpened():
            print("错误：VideoWriter 未打开！请检查编码器或路径。")
            exit()
        if not video_output_notrackingpoint.isOpened():
            print("错误：VideoWriter 未打开！请检查编码器或路径。")
            exit()

        # 计算帧率
        time_mark = time.perf_counter()

        while True:
            if not camera_state:
                break

            # 计算帧率
            #   这一块说实话没能看懂为什么要出deltaT
            deltaT = (time.perf_counter() - time_mark) * 1000
            print('time:%.2f ms' % deltaT)
            time_mark = time.perf_counter()

            # 读取图像
            ret, cv_img = cap.read()
            cv_img = cv2.flip(cv_img, 1)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)  # 二值化
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 卷积核
            eroded = cv2.erode(binary, kernel, iterations=2)  # 腐蚀
            contours, hierarchy = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 边缘检测

            # # 添加面积输出,用于划定area的最大最小值
            # print("=== 所有轮廓面积统计 ===")
            # for i, cnt in enumerate(contours):
            #     area = cv2.contourArea(cnt)
            #     print(f"轮廓 {i}: 面积 = {area:.2f}")

            if contours:
                # === 1. 找最大圆（已有代码）===============================================================================
                if not circle_state:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # 画出该轮廓
                    # cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    circle_position = [int(x), int(y)]
                    circle_radius = int(radius)
                    print("img center: ", circle_position)
                    circle_state = True
                else:
                    cv2.circle(cv_img, circle_position, 3, (0, 0, 255), -1)

                # === 2. 新增：识别含缺口的外接矩形（可包围所有非圆目标）=========================================================
                # 检测旋转贴合矩形
                box, center, angle = get_min_area_rect_from_defective_contours(contours, area_thresh_min,
                                                                               area_thresh_max, auto_flip=False)

                if box is not None:
                    # 画出停车场的外界矩形
                    cv2.drawContours(cv_img, [box], 0, (0, 255, 255), 2)
                    cv2.putText(cv_img, "Parking Lot", (center[0], center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # === Step 1: 定义局部泊车点和角度 =====================================================================
                    # # 2*2 的磁柱
                    # local_spots = [[-31, 18], [-31, 9], [-16.2423, 19.243], [-10.7573, 9.258],
                    #                [17.2321, 19.7321], [8.3398, 10.2321], [42, 9], [30, 9]]  # 单位：mm
                    # local_theta = [np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4, 2 * np.pi / 3, 2 * np.pi / 3,
                    #                np.pi, np.pi]
                    # 3*6 的磁柱
                    local_spots = [[-28, -20], [-28, -11], [-12, -21], [-3, -9],
                                   [20, -20], [16, -7], [46, -6], [34, -6]]  # 单位：mm
                    local_theta = [-np.pi / 2, -np.pi / 2, -np.pi / 3, -np.pi / 3, -3 * np.pi / 4, -3 * np.pi / 4,
                                   np.pi, np.pi]
                    # === Step 2: 转换为图像像素坐标和全局角度 ===============================================================
                    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7]
                    selected_local_spots = [local_spots[i] for i in selected_indices]
                    selected_local_theta = [local_theta[i] for i in selected_indices]
                    selected_pixels, selected_global_theta = transform_local_spots_and_angles_to_global(
                        selected_local_spots, selected_local_theta, center, angle_deg=angle, scale=10)
                    # # === Step 2.5: 打印图像坐标和目标角度（单位：°）===
                    # for i, ((px, py), theta_deg) in enumerate(zip(selected_pixels, selected_global_theta)):
                    #     label = f"{'T' if i % 2 == 0 else 'P'}{i // 2 + 1}"
                    #     print(f"[{label}] Pixel: ({px}, {py}), Global Orientation: {np.degrees(theta_deg):.1f}°")
                    # === Step 3: 转换为磁体局部坐标（单位 mm）===
                    target_poses = []  # 存储位置（单位 mm）
                    target_orientations_deg = []
                    target_orientations_rad = []
                    for i, (px, py) in enumerate(selected_pixels):
                        tx = (px - circle_position[0]) / 10
                        ty = -(py - circle_position[1]) / 10
                        target_poses.append((tx, ty))
                    # === Step 4: 可视化泊车点 ===
                    for i, pt in enumerate(selected_pixels):
                        cv2.circle(cv_img, pt, 10, (255, 0, 0), -1)
                        label_type = "T" if i % 2 == 0 else "P"
                        label_index = i // 2 + 1
                        label_text = f"{label_type}{label_index}"
                        cv2.putText(cv_img, label_text, (pt[0] + 5, pt[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # === 3. 磁体部分的检测（已有）=============================================================================
                filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
                for c in filtered_contours:
                    #     # 遍历所有轮廓，计算并输出面积
                    #     for i, c in enumerate(contours):
                    #         area = cv2.contourArea(c)
                    #         print(f"轮廓 {i} 的面积: {area}")
                    cv2.drawContours(cv_img, [c], -1, (255, 0, 0), 2)  # 蓝色表示符合条件的轮廓
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    robot_center = (int(x), int(y), deltaT / 1000)

                    # 航向角估计 + 速度方向判断
                    theta_rad, theta_deg, box, center, velocity_theta = compute_heading_angle_from_rect(c, prev_center,
                                                                                        flip_by_velocity=True)
                    # 绘制航向箭头
                    draw_heading_arrow(cv_img, center[0], center[1], theta_rad, length=60)
                    # === 3. 画出速度方向箭头（红）===
                    if prev_center is not None:
                        dx = center[0] - prev_center[0]
                        dy = center[1] - prev_center[1]
                        # draw_velocity_arrow(cv_img, center[0], center[1], dx, dy, scale=30.0, color=(0, 0, 255))
                        print(f"diretion theta:{theta_deg}, velocity theta: {velocity_theta}, (dx,dy): {dx,dy}")
                    # 更新前一帧中心
                    prev_center = center

                    # cv的坐标系是以左上角为原点，所以x方向一致，但是y方向相反，这边转化为圆心坐标系
                    motion.update_status(
                        np.array([[(x - circle_position[0]) / 10], [ - (y - circle_position[1]) / 10]]))
                    error = motion.magnetic_model.error

                    # =================================================================================================
                    # 初次识别后生成Hermite轨迹
                    # =================================================================================================
                    if first_time_detected and 'target_poses' in locals() and len(target_poses) > 0:
                        try:
                            current_pos = motion.magnetic_model.return_position().reshape(-1)
                            current_theta = -np.pi / 2  # 可根据需要修改为真实角度
                            # 只选用部分目标点(一组两个，对应上面的local_point，分别为T1,P1;T2,P2;T3,P3;T4,P4)
                            selected_indices_for_path = [6, 7]
                            selected_targets = [target_poses[i] for i in selected_indices_for_path]
                            x = [current_pos[0]] + [p[0] for p in selected_targets]
                            y = [current_pos[1]] + [p[1] for p in selected_targets]
                            theta = [current_theta] + [selected_global_theta[i] for i in
                                                       selected_indices_for_path]
                            motion.generate_follow_path('hermite', x=x, y=y, theta=theta)
                            print(f"Hermite trajectory generated through selected points: {selected_targets}")
                            print(f"Hermite trajectory generated through selected theta: {np.degrees(theta)}")
                            first_time_detected = False
                        except Exception as e:
                            print(f"[Error] Failed to generate Hermite path: {e}")

                    if motion_state:
                        pts.append(robot_center)
                        motion.magnetic_model.theta_deg_log.append(theta_deg)

                # draw path
                if circle_state:
                    # 绘制轮廓范围
                    cv2.circle(cv_img, circle_position, circle_radius, (0, 255, 0), thickness=10)
                    # 绘制坐标轴
                    cv2.line(cv_img, (circle_position[0]-10, circle_position[1]),   # x坐标轴
                             (circle_position[0]+100, circle_position[1]), [255, 0, 255], 2)
                    cv2.line(cv_img, (circle_position[0], circle_position[1] + 10), # y坐标轴
                             (circle_position[0], circle_position[1] - 100), [255, 0, 0], 2)
                    # 绘制起始点
                    if has_valid_path(motion.magnetic_model):
                        cv2.circle(cv_img, np.array([int(circle_position[0] + 10 * motion.magnetic_model.ref_x[0]),
                                                     int(circle_position[1] - 10 * motion.magnetic_model.ref_y[0])]), 4,
                                   (0, 0, 255), 4)

                if motion_state:
                    # # 将跟踪的点绘制出来,这边要小心检测不到会出问题，不过不想改了，乏了，之后再看
                    # if motion_state:
                    #     selected_idx = motion.get_selected_index()
                    #     if 0 <= selected_idx < len(motion.magnetic_model.ref_x):
                    #         cv2.circle(cv_img, np.array(
                    #             [circle_position[0] + 10 * int(motion.magnetic_model.ref_x[selected_idx]),
                    #              circle_position[1] - 10 * int(motion.magnetic_model.ref_y[selected_idx])]),
                    #                    4, (0, 0, 0), -1)
                    for i in range(1, len(pts)):
                        if pts[i - 1][:2] is None or pts[i][:2] is None:
                            continue
                        cv2.line(cv_img, pts[i - 1][:2], pts[i][:2], (0, 0, 255), thickness=1)

            # 录像
            if video_state:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                path_image = cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB)
                mix_img = cv2.addWeighted(cv_img, 1, path_image, 1, 0)
                video_output.write(mix_img)  # 保存视频

            # 让界面显示图像
            resized_img = cv2.resize(mix_img, (640, 360))
            frame = QImage(resized_img, 640, 360, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.label_show_camera.setPixmap(pix)

            if motion_state:
                #   这边是要改的，要同时控制六个线圈，得加上IHx1, IHy1, IHz1，IHx2, IHy2, IHz2
                IHx, IHy, IHz = motion.motion_control(i_motion, deltaT / 1000)
                # print("Applied Current: ", IHx, IHy, IHz)
                i_motion = i_motion + 1
            else:
                # IHx, IHy, IHz = motion.adjust_angle()  # 这边是那个小作弊的方式
                IHx, IHy, IHz = 0, 0, 0

            # 驱动电流
            # 驱动部分的修改主要集中在这部分和Drive.py部分
            if port_state:  # and motion_state:
                # 这边也是要改的，VHx决定了电压的方向，从而产生正负的电流，设置的时候应该设置的都是绝对值，
                # 但对于有驱动的情况，无需这样设置，直接用占空比产生负电
                if IHx > 10:
                    IHx = 10
                elif IHx < -10:
                    IHx = -10
                if IHy > 10:
                    IHy = 10
                elif IHy < -10:
                    IHy = -10
                if IHz > 10:
                    IHz = 10
                elif IHz < -10:
                    IHz = -10
                magnetic_drive.set_currents(IHx, IHx, IHy, IHy, IHz, IHz)

        cap.release()
        # 录像
        if video_state:
            video_output.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = WinForm()
    form.show()
    sys.exit(app.exec())
