import math
import time
import numpy as np
import cv2
from PyQt6 import QtCore
# import torch
import queue
import motion
from collections import deque

# from ultralytics import YOLO
# # Load a model
# model = YOLO('best_n.pt')
# classes = ["Magnetic", "Vessel"]

# def target_yolo_detect(frame):
#     global boundary_init, boundary_center
#     # using yolov8
#     results = model(frame, verbose=False)  # detect
#     annotated_frame = results[0].plot()  # 得到有标注的frame
#     boxes = results[0].boxes  # Boxes object for bbox outputs
#     class_ids = boxes.cls.cpu().numpy().astype(int)  # 转为int类型数组
#     xywh = boxes.xywh.cpu().numpy().astype(int)  # 转为int类型数组
#     # if len(class_ids) > 0:
#     #     for i in range(0, len(class_ids)):
#     #         # print(class_ids[i], xywh[i])
#     #         # 更新机器人位置
#     #         if class_ids[i] == 0:
#     #             magnetic_model.update_position(xywh[i][0], xywh[i][1])
#     #             magnetic_model.update_path()
#     #             # dynamic_model.update_vel()
#     #             # dynamic_model.update_vir_position()
#     #         # 更新边界中心
#     #         if not boundary_init and class_ids[i] == 1:
#     #             boundary_center = xywh[i]
#     #             for i in range(0, len(t)):
#     #                 get_target_position(t[i], boundary_center[0], boundary_center[1])
#     #                 get_target_velocity(t[i])
#     #
#     #             print(target_vel)
#     #             boundary_init = True
#     return annotated_frame



def hough_circles(img):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊平滑图像，减少噪声
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 使用霍夫圆变换检测图像中的圆
    circles = cv2.HoughCircles(
        gray_blurred,  # 输入图像
        cv2.HOUGH_GRADIENT,  # 使用霍夫梯度法检测圆
        dp=1,  # 累加器分辨率（图像分辨率/累加器分辨率）
        minDist=200,  # 圆心之间的最小距离
        param1=50,  # canny 边缘检测高阈值
        param2=50,  # 检测圆的累加器阈值（越小越容易检测更多圆）
        minRadius=500,  # 最小半径
        maxRadius=600  # 最大半径
    )

    # 如果检测到圆
    if circles is not None:
        global circle_position
        # 将圆的坐标转换为整数
        # print(circles)
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            # 获取圆心的 x, y 坐标，和半径
            x, y, r = circle
            # 在原图上画出圆
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            # 在圆心处画一个点
            cv2.circle(img, (x, y), 2, (0, 0, 255), 2)

            # YSH新增的，为了路径点跟踪的标识，但后来发现没用
            cv2.circle(img, (x + r, y), 4, (0, 0, 255), 2)
            cv2.circle(img, (x - r, y), 4, (0, 0, 255), 2)
            cv2.circle(img, (x, y + r), 4, (0, 0, 255), 2)
            cv2.circle(img, (x, y - r), 4, (0, 0, 255), 2)
            circle_position = [x, y]
            print("circle_position", circle_position)

        return True
    return False


    # return img


def threshold_detect(img):
    global transform_state, transform_mile, circle_state, circle_position
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用阈值法进行二值化分割
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((8, 8), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 添加腐蚀操作
    # kernel = np.ones((5, 5), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=2)

    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if circle_state:
        # 遍历所有检测到的轮廓
        for contour in contours:
            # 计算轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # x 坐标
                cy = int(M["m01"] / M["m00"])  # y 坐标
                if not transform_state:
                    # print("pixelllll: ", cx, cy, w, h)
                    if 20 < w < 30 and 20 < h < 30:
                        cx = (cx - circle_position[0])
                        cy = (circle_position[1] - cy)
                        print(cx, cy)
                        cr = math.sqrt(cx ** 2 + cy ** 2)
                        print(cr)
                        if cr < 250:
                            transform_mile = w / 3.15  # 2 是磁体的宽度大小  用来换算的 更换磁体要改变
                            transform_state = True
                elif transform_state:
                    if 20 < w < 30 and 20 < h < 30:

                        cx = (cx - circle_position[0]) / transform_mile
                        cy = (circle_position[1] - cy) / transform_mile
                        cr = math.sqrt(cx ** 2 + cy ** 2)
                        # print("pixel: ", cx, cy, w, h)
                        # print("cr: ", cr)
                        if cr < 70:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # w = w / transform_mile
                            # h = h / transform_mile
                            # print(f"检测到物体的质心坐标: ({cx}, {cy})")
                            # print(f"检测到物体的外接矩形: w={w}, h={h}")
                            motion.update_status(np.array([[cx/1000], [cy/1000]]))
                            # print("VideoThread", time.time(), "[", motion.return_current_position()[0, 0], ", ",
                            #       motion.return_current_position()[1, 0], "]")

    return img

pts = deque(maxlen=1240000)
circle_state = False
circle_position = (0, 0)
transform_state = False
transform_mile= 10
min_area = 400
max_area = 500

class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        global circle_position, circle_state
        global transform_state, transform_mile, circle_state, circle_position
        cap = cv2.VideoCapture(0)

        # 录像
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1920, 1080))

        # 设置摄像头的帧率为60fps 设置摄像头的分辨率为1080p（1920x1080）
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        print('HEIGHT:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WIDTH:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('FPS:', round(cap.get(cv2.CAP_PROP_FPS)))

        # 计算帧率
        time_mark = time.perf_counter()

        while self._run_flag:
            # 计算帧率
            deltaT = (time.perf_counter() - time_mark) * 1000
            print('time:%.2f ms' % deltaT)
            time_mark = time.perf_counter()

            # 读取图像
            ret, cv_img = cap.read()
            cv_img = cv2.flip(cv_img, 1)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)  # 二值化
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 卷积核
            eroded = cv2.erode(binary, kernel, iterations=2)  # 腐蚀
            contours, hierarchy = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 边缘检测

            # cv2.drawContours(cv_img, contours, -1, (0, 0, 255), 1)  # 画出检测的轮廓
            # # 遍历所有轮廓
            # for contour in contours:
            #      area = cv2.contourArea(contour)  # 计算轮廓面积
            #      print(f"轮廓面积: {area}")

            if contours:
                # 找到面积最大的轮廓
                if not circle_state:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # 画出该轮廓
                    # cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    circle_position = [int(x), int(y)]
                    cv2.circle(cv_img, circle_position, 3, (0, 0, 255), -1)
                    print("img center: ", circle_position)
                    circle_state = True
                else:
                    cv2.circle(cv_img, circle_position, 3, (0, 0, 255), -1)

                filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
                for c in filtered_contours:
                    cv2.drawContours(cv_img, [c], -1, (255, 0, 0), 2)  # 蓝色表示符合条件的轮廓
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    robot_center = (int(x), int(y), deltaT/1000)
                    # motion.update_status(np.array([[x / 10], [y / 10]]))
                    motion.update_status(np.array([[(x-circle_position[0]) / 10], [(circle_position[1]-y) / 10]]))
                    # print("robot_center: ", robot_center)
                    print("detect position: ", motion.return_current_position().reshape(-1))
                    pts.append(robot_center)

                    # # 计算速度
                    # if len(pts) >= 2:
                    #     x1, y1, t1 = pts[-2]  # 倒数第二个数据点
                    #     x2, y2, t2 = pts[-1]  # 最新数据点
                    #
                    #     x1 = x1/10
                    #     y1 = y1 / 10
                    #     x2 = x2 / 10
                    #     y2 = y2 / 10
                    #
                    #     # 避免时间间隔为 0（除零错误）
                    #     dt = t2 - t1
                    #     if dt > 0:
                    #         vx = (x2 - x1) / dt
                    #         vy = (y2 - y1) / dt
                    #     else:
                    #         vx, vy = 0, 0  # 如果时间间隔太小，速度设为 0
                    # else:
                    #     vx, vy = 0, 0  # 第一次运行时没有历史数据，速度设为 0
                    # print("detect vel: ", vx/10, vy/10)
                    #
                    # # 计算加速度
                    # if len(pts) >= 3:
                    #     x0, y0, t0 = pts[-3]  # 倒数第三个数据点
                    #     x0 = x0/10
                    #     y0 = y0/10
                    #     dt1 = t1 - t0  # 时间间隔 1
                    #     dt2 = t2 - t1  # 时间间隔 2
                    #
                    #     if dt1 > 0 and dt2 > 0:
                    #         vx1 = (x1 - x0) / dt1
                    #         vy1 = (y1 - y0) / dt1
                    #         ax = (vx - vx1) / dt2
                    #         ay = (vy - vy1) / dt2
                    #     else:
                    #         ax, ay = 0, 0
                    # else:
                    #     ax, ay = 0, 0  # 还没有足够的数据点，设加速度为 0
                    # print("detect accel: ", ax/10, ay/10)

                    # x, y, width, height = cv2.boundingRect(c)  # 获取外接矩形（x, y, 宽度, 高度）
                    # print(f"轮廓位置: ({x}, {y}), 宽度: {width}, 高度: {height}")

                # draw path
                for i in range(1, len(pts)):
                    if pts[i - 1][:2] is None or pts[i][:2] is None:
                        continue
                    cv2.line(cv_img, pts[i - 1][:2], pts[i][:2], (0, 0, 255), thickness=1)

            # annotated_frame = threshold_detect(cv_img)
            # hough_circles(annotated_frame)

            # if not circle_state:
            #     if hough_circles(annotated_frame):
            #         circle_state = True
            # else:
            #     cv2.circle(annotated_frame, (circle_position[0], circle_position[1]), 2, (0, 0, 255), 2)

            # 录像
            self.out.write(cv_img)  # 保存视频

            # if ret:
            resized_img = cv2.resize(cv_img, (640, 360))
            self.change_pixmap_signal.emit(resized_img)

        cap.release()
        # 录像
        self.out.release()
        cv2.destroyAllWindows()

    def stop(self):
        # Sets run flag to False and waits for thread to finish
        self._run_flag = False
        self.wait()