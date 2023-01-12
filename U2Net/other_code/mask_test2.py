#从标记的mask来计算刀闸的开合


import cv2
import numpy
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

def fitline( mask, image):
    gray_pointer = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts_pointer = cv2.findContours(gray_pointer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_pointer = cnts_pointer[0] if len(cnts_pointer) == 2 else cnts_pointer[1]

    points_pointer = []
    pointer_k = []
    pointer_b = []
    for j in range(len(cnts_pointer)):
        try:
            rect = cv2.minAreaRect(cnts_pointer[j])
            box_origin = np.int0(cv2.boxPoints(rect))
            print("box_origin", box_origin)
            cv2.polylines(mask, [box_origin], True, (0, 0, 255), 1)  # pic
            cv2.polylines(image, [box_origin], True, (0, 0, 255), 1)  # pic
            output = cv2.fitLine(box_origin, 2, 0, 0.001, 0.001)

            # 经验证在直线为垂直的时候，output[0]不会为0，会得到无限接近0的k值,
            # 需要注意的是当直线垂直时求得的斜率k非常大,但是不影响得到直线方程
            pointer_k = output[1] / output[0]
            pointer_k = round(pointer_k[0], 2)
            b = output[3] - pointer_k * output[2]
            b = round(b[0], 2)

            # pointer_k.append(k)
            # pointer_b.append(b)

            x1 = 1  # 随机选取直线上的一点
            x2 = mask.shape[0]
            y1 = int(pointer_k * x1 + b)
            y2 = int(pointer_k * x2 + b)
            # 拟合指针所在的直线
            cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            points_pointer.append(box_origin.tolist())
        except:
            continue

    # return points_pointer, pointer_k
    return pointer_k, mask, image

def get_center(mask,image):
    gray_mask =cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours =cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    M =cv2.moments(contours[0])
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    cv2.drawContours(image, contours, 0, (0,255,0)) #绘制轮廓
    cv2.circle(image, (center_x, center_y), 2, (0,255,0),-1)  # 绘制中心点
    return center_x,center_y


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/STFANGSO.TTF", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)




mask =cv2.imread('data/data2/train/20220426_475.png',0) #(644, 1044, 3)
image =cv2.imread('data/data2/train/20220426_475.jpg')
h,w =image.shape[0:2]
#返回值为1的下标
#itemindex = numpy.argwhere(mask1 == 1)
#print(itemindex)



#刀闸直线拟合
mask_daozha = np.zeros((h,w,3),dtype='uint8')
condition = mask ==1
mask_daozha[condition] = (0,0,255)
#cv2.imwrite(r"temp/{}.png".format(count), mask_left_daozha)
left_kine_k,line_left,image = fitline(mask_daozha,image)

#右边脚的中心点
mask_youdian =np.zeros((h,w,3),dtype='uint8')
condition =mask ==2
mask_youdian[condition] =(0,255,0)
right_x,right_y =get_center(mask_youdian,image)

#左边脚的中心点
mask_zuodian =np.zeros((h,w,3),dtype='uint8')
condition =mask ==3
mask_zuodian[condition] =(0,255,0)
left_x,left_y =get_center(mask_zuodian,image)

#计算两个点之间的斜率
point_k =(left_y -right_y)/(left_x -right_x)
#拟合两个点之间的直线
cv2.line(image, (right_x, right_y), (left_x, left_y), (0, 255, 0), 1)

degree = -(left_kine_k - point_k) / (1 + left_kine_k*point_k)
degree = math.atan(degree)*180.0/math.pi
if degree<0:
    degree = degree+180

degree = round(degree,2)

if degree>6.00 and degree<174.00:
    state="刀闸开"
else:
    state = "刀闸合"

print("degree:",degree)
img = cv2ImgAddText(image, state +":水平角度"+ str(degree)+"度", 10, 10, textColor=(255, 0, 0), textSize=20)


cv2.imwrite(r"result/2222.png",img)