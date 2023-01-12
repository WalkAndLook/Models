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
        self.net = U2NET(3, 2)
        self.device = torch.device('cuda:1' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.net.load_state_dict(torch.load('weight/net_daozha.pt', map_location='cpu'), False)

        print("init segment")

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module

        self.net.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, image,count):
        h,w = image.shape[0:2]
        # 图片放缩到416*416 是32的倍数
        image = self.square_picture(image, 416)
        # 更改图片通道顺序
        image_tensor = self.to_tensor(image.copy()).to(self.device)
        # 送入模型
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        # d0是模型最后输出点的6合1图片
        print("d0:",d0.shape) #d0: torch.Size([1, 2, 416, 416])

        # 降维 torch-4变成了numpy-3
        mask = d0.squeeze(0).cpu().numpy()
        print("mask:",mask.shape) #mask: (2, 416, 416)

        # 提取分刀闸mask
        daozha_mask0 = self.binary_image(mask[0])
        # 提取合刀闸mask
        daozha_mask1 = self.binary_image(mask[1])
        print("daozha_mask1\n")  # (416, 416)
        print(daozha_mask1.shape)

        # mask0用来展示图片测试出来的mask，3维图像
        mask0 = np.zeros((416,416,3),dtype='uint8')
        condition = daozha_mask0 ==1
        # 分别在原图和mask上把图画出来 红色
        mask0[condition] = (0,0,255)
        image[condition] = (0,0,255)

        # mask1用来展示图片测试出来的mask，3维图像
        mask1 = np.zeros((416, 416, 3), dtype='uint8')
        condition = daozha_mask1 == 1
        # 分别在原图和mask上把图画出来 绿色
        mask1[condition] = (0, 255, 0)
        image[condition] = (0, 255, 0)

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

        image_save_path = r"result/daozha"
        if not os.path.isdir(image_save_path):
            os.mkdir(image_save_path)
        cv2.imwrite(os.path.join(image_save_path, "{}.png".format(count)), img)
        # cv2.imshow('image', img)
        # cv2.waitKey()

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
        # 互换通道位置，添加新的通道
        # torch.Tensor.permute  # eg:x.permute(2, 0, 1)  # 更改顺序
        # torch.squeeze()  # 降维
        # torch.unsqueeze()  # 增维
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
    root = r'data/data3/val'
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

