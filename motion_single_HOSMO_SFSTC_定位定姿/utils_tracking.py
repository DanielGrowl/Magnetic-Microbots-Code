import cv2
import numpy as np

# 用于防止航向角跳变的全局变量
last_theta_rad = None

# 20250717：velocity_theta调好了，不过direction_theta应该是由于正方形难以分别长短变的缘故，一直失败

def angle_diff(a, b):
    """计算两个角度之间的差值，结果在 [-π, π]"""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def compute_heading_angle_from_rect(contour, prev_center=None, flip_by_velocity=False):
    """
    使用最小外接矩形估计航向角（长边方向 + 90°），
    可选择性结合速度方向判断是否需要反转方向（加180°），
    同时加入平滑滤波避免跳变。
    """
    global last_theta_rad
    velocity_theta = None
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(int)
    (cx, cy), (w, h), angle = rect
    # === 1. 矫正角度：仅长边方向加 0° ===
    if w < h:
        angle += 90
    angle += 90 # 转为垂直于长轴的
    theta_deg = angle
    theta_rad = np.radians(angle)  # 模拟磁体滚动方向
    center = (int(cx), int(cy))
    # # === 2. 航向角归一化到 [-π, π] ===
    # theta_rad = (theta_rad + np.pi) % (2 * np.pi) - np.pi
    # === 3. 使用速度方向修正180度（角度制） ===
    # 这边问题在于opencv以我画面上的原点向左的方向为0°，所以这边的角度需要转180°
    if flip_by_velocity and prev_center is not None:
        dx = center[0] - prev_center[0]
        dy = center[1] - prev_center[1]
        speed_norm = np.hypot(dx, dy)
        if speed_norm > 1e-2:
            # v_vec = np.array([dx, dy])
            # v_unit = v_vec / (np.linalg.norm(v_vec) + 1e-6)
            # theta_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
            # dot_product = np.clip(np.dot(theta_unit, v_unit), -1.0, 1.0)
            # delta_deg = np.degrees(np.arccos(dot_product))
            # cross_product = theta_unit[0] * v_unit[1] - theta_unit[1] * v_unit[0]
            # # if delta_deg > 90:
            # #     theta_rad = (theta_rad + np.pi) % (2 * np.pi) - np.pi
            # # theta_deg = (np.degrees(theta_rad)) % 360
            velocity_theta = np.degrees(np.arctan2(dy, dx))
            delta_deg = abs(velocity_theta - angle)
            if abs(delta_deg) > 90:
                # theta_rad = (theta_rad + np.pi) % (2 * np.pi) - np.pi
                theta_rad = (theta_rad + np.pi)
                theta_deg = (np.degrees(theta_rad)) % 360
            else:
                theta_rad = theta_rad
                theta_deg = (np.degrees(theta_rad)) % 360
        else:
            velocity_theta = None
    else:
        velocity_theta = None
    # === 4. 平滑处理：避免角度跳变 ===
    if last_theta_rad is not None:
        delta = angle_diff(theta_rad, last_theta_rad)
        if abs(delta) > np.pi / 2:
            theta_rad = last_theta_rad
    last_theta_rad = theta_rad
    # # === 5. 输出角度（0° 向右，90° 向上） ===
    # theta_deg = (np.degrees(theta_rad)) % 360  # 将270°转为90°
    return theta_rad, theta_deg, box, center, velocity_theta


def draw_heading_arrow(img, x, y, theta_rad, length=50, color=(0, 255, 255)):
    """
    在图像 img 上以 (x, y) 为起点，按角度 theta_rad 绘制箭头表示朝向。
    """
    tip_x = int(x + length * np.cos(theta_rad))
    tip_y = int(y + length * np.sin(theta_rad))
    cv2.arrowedLine(img, (x, y), (tip_x, tip_y), color, 3, tipLength=0.3)


def draw_velocity_arrow(img, x, y, dx, dy, scale=5.0, color=(0, 0, 255), thickness=2):
    """
    在图像 img 上以 (x, y) 为起点，按 (dx, dy) 绘制放大后的速度方向箭头。

    参数:
        x, y      : 起点坐标
        dx, dy    : 速度向量（将被缩放）
        scale     : 放大倍数，用于使箭头更明显
        color     : 颜色 (B, G, R)
        thickness : 线宽
    """
    norm = np.hypot(dx, dy)
    if norm < 1e-3:
        return  # 速度太小不画
    dx /= norm
    dy /= norm
    tip_x = int(x + scale * dx)
    tip_y = int(y + scale * dy)
    cv2.arrowedLine(img, (x, y), (tip_x, tip_y), color, thickness, tipLength=0.2)

def get_bounding_rect_from_defective_rect(contours, area_thresh_min=1000, area_thresh_max=1e6):
    """
    根据面积范围，从多个轮廓中找出合适的（破损）矩形区域，并计算整体外接矩形。
    :param contours: OpenCV findContours 的输出
    :param area_thresh_min: 最小面积阈值
    :param area_thresh_max: 最大面积阈值
    :return: (x, y, w, h) 或 None
    """
    if not contours:
        return None
    # 过滤面积在合理范围内的轮廓
    valid_contours = [cnt for cnt in contours if area_thresh_min < cv2.contourArea(cnt) < area_thresh_max]
    if len(valid_contours) == 0:
        return None
    # 合并所有轮廓点
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    return (x, y, w, h)

def get_min_area_rect_from_defective_contours(contours, area_thresh_min=1000, area_thresh_max=1e6, auto_flip=False):
    """
    获取缺陷区域的最小外接矩形并自动补角（如果启用）。
    """
    if not contours:
        return None, None, None
    valid_contours = [cnt for cnt in contours if area_thresh_min < cv2.contourArea(cnt) < area_thresh_max]
    if len(valid_contours) == 0:
        return None, None, None
    all_points = np.vstack(valid_contours)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    center = tuple(map(int, rect[0]))
    w, h = rect[1]
    angle = rect[2]
    # 保证 angle 表示长边方向
    if w < h:
        angle += 90
    # === 自动补角（180° - angle）===
    if auto_flip:
        angle = 180 - angle
    angle %= 180  # 保证在 [0, 180) 范围
    # print(f"[角度调试] OpenCV原始角: {rect[2]:.2f}°, 修正角: {angle:.2f}° (auto_flip={auto_flip})")
    return box, center, angle

def transform_local_batch_to_image(local_points, center, angle_deg, scale=10):
    """
    将一组局部坐标（相对于旋转矩形中心，单位：mm）转换为图像坐标（单位：像素）。

    参数:
        local_points : (N, 2) 的列表或数组，表示多个局部坐标点（单位 mm）
        center       : (cx, cy) 图像坐标系下的矩形中心点（单位 px）
        angle_deg    : cv2.minAreaRect 返回的角度（单位°）
        scale        : mm → 像素的比例系数

    返回:
        global_points : (N, 2) 的整数数组，每行是图像坐标系下的像素点
    """
    theta = np.radians(angle_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    local_points = np.array(local_points) * scale  # (N, 2) 乘以缩放系数
    global_points = np.dot(local_points, R.T) + np.array(center)
    return global_points.astype(int)

import numpy as np

def transform_local_spots_and_angles_to_global(local_spots, local_theta, center, angle_deg, scale=10):
    """
    将局部定义的泊车点及其角度转换为图像坐标系下的像素位置和全局角度。
    参数:
        local_spots: List[Tuple[float, float]]，局部坐标系下的泊车点位置（单位 mm）
        local_theta: List[float]，局部泊车点的朝向角（单位 rad）
        center: Tuple[int, int]，旋转矩形中心在图像中的位置（像素）
        angle_deg: float，检测到的旋转矩形的角度（单位度）
        scale: float，单位 mm → 像素的缩放比例
    返回:
        global_pixels: List[Tuple[int, int]]，图像中对应的像素点
        global_theta: List[float]，全局朝向角（单位 rad）
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    global_pixels = []
    global_theta = []
    for (lx, ly), theta in zip(local_spots, local_theta):
        # 位置转换（旋转 + 平移）
        x_img = int(center[0] + scale * (lx * cos_a - ly * sin_a))
        y_img = int(center[1] + scale * (lx * sin_a + ly * cos_a))
        global_pixels.append((x_img, y_img))
        # 朝向角转换
        global_theta.append(theta + angle_rad)
    return global_pixels, global_theta

def has_valid_path(model):
    """
    检查给定的 magnetic_model 是否具有有效轨迹数据（ref_x、ref_y 均非空且长度一致）。
    """
    return (
        hasattr(model, 'ref_x') and hasattr(model, 'ref_y')
        and isinstance(model.ref_x, (list, np.ndarray))
        and isinstance(model.ref_y, (list, np.ndarray))
        and len(model.ref_x) > 0 and len(model.ref_y) > 0
        and len(model.ref_x) == len(model.ref_y)
    )
