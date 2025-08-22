import numpy as np
import motion
import pickle
import model
import cv2

color = (255, 255, 255)

def draw_path(path_image, ref_x, ref_y, circle_position):
    ref_x_length = len(ref_x)
    ref_y_length = len(ref_y)
    if ref_x_length == ref_y_length:
        for i in range(1, ref_x_length):
            x1, y1 = int(10 * ref_x[i - 1]) + circle_position[0], - int(10 * ref_y[i - 1]) + circle_position[1] # 添加原点偏移
            x2, y2 = int(10 * ref_x[i]) + circle_position[0], - int(10 * ref_y[i]) + circle_position[1]
            cv2.line(path_image, (x1, y1), (x2, y2), color, thickness=2)
    else:
        print(f"路径的x和y值不匹配")
    cv2.imwrite("path_image.png", path_image)
    return path_image
