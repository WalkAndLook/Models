import math
import os
import cv2
import numpy
import torch
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from models.net import U2NET

class Tester(object):

    def __init__(self, is_cuda=True):
        self.net = U2NET(3, 3)  #输出3个mask
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.device =torch.device('cpu')
        self.net.load_state_dict(torch.load('weight/net_dian0001.pt', map_location='cpu'), False)

        print("init segment")

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module

        self.net.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, image,count):
        h,w = image.shape[0:2]
        image = self.square_picture(image, 416) #(416, 416, 3)

        image_tensor = self.to_tensor(image.copy()).to(self.device)
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        print("d0:",d0.shape) #d0: torch.Size([1, 3, 416, 416])

        mask = d0.squeeze(0).cpu().numpy()
        print("mask:",mask.shape) #mask: (3, 416, 416)

        daozha_mask = self.binary_image(mask[0])
        print("daozha_mask\n") #(416, 416)
        print(daozha_mask.shape)
        youdian_mask = self.binary_image(mask[1])
        zuodian_mask = self.binary_image(mask[2])

        # 刀闸直线拟合
        mask_daozha = np.zeros((416,416,3),dtype='uint8')
        condition = daozha_mask ==1
        mask_daozha[condition] = (0,0,255)
        #cv2.imwrite(r"temp/{}.png".format(count), mask_left_daozha)
        left_kine_k,line_left,image = self.fitline(mask_daozha,image)

        # 右边脚的中心点
        mask_youdian = np.zeros((416,416,3), dtype='uint8')
        condition = youdian_mask == 1
        mask_youdian[condition] = (0, 255, 0)
        right_x, right_y = self.get_center(mask_youdian, image)

        # 左边脚的中心点
        mask_zuodian = np.zeros((416,416,3), dtype='uint8')
        condition = zuodian_mask == 1
        mask_zuodian[condition] = (0, 255, 0)
        left_x, left_y = self.get_center(mask_zuodian, image)

        # 计算两个点之间的斜率
        point_k = (left_y - right_y) / (left_x - right_x)
        # 拟合两个点之间的直线
        cv2.line(image, (right_x, right_y), (left_x, left_y), (0, 255, 0), 1)

        degree = (left_kine_k - point_k) / (1 + left_kine_k * point_k)
        degree = math.atan(degree) * 180.0 / math.pi
        if degree < 0:
            degree = degree + 180
        elif degree>90 and degree<=180:
            degree = 180-degree
        else:
            pass


        degree = round(degree, 2)

        if degree > 4.00 :
            state = "刀闸开"
        else:
            state = "刀闸合"

        if w>=h:
            fx = 416/w
            h_aug = int(h*fx)
            h_d = int((416-h_aug)/2)
            img = image[h_d:h_d+h_aug,:]  # 获取416*416中的原图
            img = cv2.resize(img,(w,h))
        else:
            fy = 416/h
            w_aug = int(w*fy)
            w_d = int((416-w_aug)/2)
            img = image[:, w_d:w_d+w_aug]
            img = cv2.resize(img,(w,h))

        #cv2.putText(img, state +": level"+ str(degree), (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        img = self.cv2ImgAddText(img, state +":水平角度"+ str(degree)+"度", 10, 10, textColor=(255, 0, 0), textSize=20)

        print("degree",degree)
        image_save_path =r"result/dian_zhadao0002"
        if not os.path.isdir(image_save_path):
            os.mkdir(image_save_path)
        cv2.imwrite(os.path.join(image_save_path,"{}.png".format(count)),img)
        # cv2.imshow('image', img)nic
        # cv2.waitKey()

    def fitline(self,mask,image):

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
                print("box_origin",box_origin)
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
        return pointer_k,mask,image

    def get_center(self,mask, image):
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        M = cv2.moments(contours[0])
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        cv2.drawContours(image, contours, 0, (0, 0, 255))  # 绘制轮廓
        cv2.circle(image, (center_x, center_y), 2, (0, 255, 0), -1)  # 绘制中心点
        return center_x, center_y

    def binary_image(self, image):
        condition = image > 0.5
        image[condition] = 1
        image[~condition] = 0
        # image = self.corrosion(image,7)
        image = self.corrosion(image,5)
        # image = self.corrosion(image,3)
        return image

    def corrosion(self, image,kernel_size):
        """
        腐蚀操作
        :param image:
        :return:
        """

        kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
        image = cv2.erode(image, kernel)
        return image


    @staticmethod
    def to_tensor(image):
        image = torch.tensor(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def square_picture(image, image_size):
        """
        任意图片正方形中心化
        :param image: 图片
        :param image_size: 输出图片的尺寸
        :return: 输出图片
        """
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = image_size / max_len
        fy = image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, _ = image.shape
        background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
        background[:, :, :] = 127
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        return background

    def cv2ImgAddText(self,img, text, left, top, textColor=(0, 255, 0), textSize=20):
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




if __name__ == '__main__':
    import time
    tester = Tester()
    root = r'data/val'
    count = 0
    for image_name in os.listdir(root):
        if ".jpg" in image_name:
            path = f'{root}/{image_name}'
            image = cv2.imread(path)
            t0 = time.time()
            tester(image,count)
            t1 = time.time()
            print("time:",t1-t0)
            count+=1

