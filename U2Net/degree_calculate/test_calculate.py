'''
这里是给定先验信息
1中心点 2最小点 3最小值 4最大点 5最大值
这里指针给的是两个点，但实际用的时候返回的是指向针尖的向量
'''

import cv2
import numpy as np
import math


def getRotateAngle(x1, y1, x2, y2):
    epsilon = 1.0e-6
    dist = math.sqrt(x1 * x1 + y1 * y1)
    x1 /= dist
    y1 /= dist
    dist = math.sqrt(x2 * x2 + y2 * y2)
    x2 /= dist
    y2 /= dist

    dot = x1 * x2 + y1 * y2
    if math.fabs(dot - 1.0) <= epsilon:
        angle = 0.0
    elif math.fabs(dot + 1.0) <= epsilon:
        angle = np.pi
    else:
        angle = np.arccos(dot)
        cross = x1 * y2 - x2 * y1
        if cross < 0:
            angle = 2 * np.pi - angle
    angle = angle * 180.0 / np.pi
    # angle = 360 - angle
    return angle

if __name__ == '__main__':
    #image = cv2.imread('meter3.jpg')
    # h, w = image.shape[:2]  # 895 1191
    min_point = (1161, 398)  # (185, 426)
    min_num = 0
    max_point = (1508, 397)  # (1044, 408)
    max_num = 160
    center_point = (1322, 465)  # (601, 530)
    meter_first = (282, 266)  # (281, 264)
    meter_last = (534, 499)  # (537, 494)

    x1 = min_point[0] - center_point[0]
    y1 = min_point[1] - center_point[1]
    x2 = max_point[0] - center_point[0]
    y2 = max_point[1] - center_point[1]
    # x_point = meter_first[0] - meter_last[0]
    # y_point = meter_first[1] - meter_last[1]
    x_point = -69
    y_point = -136

    angle_max = getRotateAngle(x1, y1, x2, y2)
    angle_point = getRotateAngle(x1, y1, x_point, y_point)

    degree = min_num + (max_num - min_num) * (angle_point / angle_max)
    degree = int(degree + 0.5)
    print('angle_max:', angle_max)
    print('angle_point:', angle_point)
    print('meter_degree:', degree)
