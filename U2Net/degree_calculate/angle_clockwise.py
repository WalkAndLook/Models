'''
给定两个向量，顺时针计算两个向量之间的夹角
getRotateAngle输入的分别是两个向量的x值y值
'''

import math
import numpy as np

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
    angle = 360 - angle
    return angle

if __name__ == '__main__':
    h = 895
    min_point = (131, h - 394)  # (185, 426)
    max_point = (115, h - 143)  # (1044, 408)
    center_point = (261, h - 255)  # (601, 530)
    x1 = center_point[0] - min_point[0]
    y1 = center_point[1] - min_point[1]
    x2 = center_point[0] - max_point[0]
    y2 = center_point[1] - max_point[1]
    angle = getRotateAngle(x1, y1, x2, y2)
    print('angle:', angle)