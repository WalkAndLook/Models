# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd
from PIL import Image


#classes = ["light", "post"]  # 改成自己的类别
'''
classes = [
"yibiao", #仪表
"character", #数字
"qrcode",    #二维码
"00000000",  #压板分
"00000001",  #压板合
"00010000",  #防误压板分
"00010001",  #防误压板合
"00020000",  #水平字的装置灯灭
"00020001",  #水平字的装置灯亮
"00029999",  #水平字的装置灯负样本
"00030000",  #装置高亮字灭
"00030001",  #装置高亮字亮
"00039999",  #装置高亮字负样本
"00040000",  #把手上
"00040001",  #把手右上
"00040002",  #把手右
"00040003",  #把手右下
"00040004",  #把手下
"00040005",  #把手左下
"00040006",  #把手左
"00040007",  #把手左上
"00050000",  #空气开关关
"00050001",  #空气开关开
"00060000",  #指示灯灭
"00060001",  #指示灯亮
"00070000",  #带垂直字的装置灯灭
"00070001",  #带垂直字的装置灯亮
"00090000",  #印刷体数字0
"00090001",  #印刷体数字1
"00090002",  #印刷体数字2
"00090003",  #印刷体数字3
"00090004",  #印刷体数字4
"00090005",  #印刷体数字5
"00090006",  #印刷体数字6
"00090007",  #印刷体数字7
"00090008",  #印刷体数字8
"00090009",  #印刷体数字9
"00100000",  #断路器开关分
"00100001",  #断路器开关合
"00110000",  #线路刀闸分
"00110001",  #线路刀闸合
"00120000",  #接地刀闸分
"00120001",  #接地刀闸合
"00130000",  #SF6仪表指针末端
"00140000",  #避雷器表指针末端
"00150000",  #表计指针末端(短)
"00150001",  #表计指针末端(长)
"00160000",  #弹簧储能开关(箭头)分
"00160001",  #弹簧储能开关(箭头)合
"00170000",  #弹簧储能开关(字)分
"00170001",  #弹簧储能开关(字)合
"00190000",  #CT液位    电流   舍弃 合并为00300000
"00210000",  #开关柜状态灯分
"00210001",  #开关柜状态灯合
"00220000",  #配电房开关指示器分
"00220001",  #配电房开关指示器合
"00220002",  #配电房开关指示器接地
"00300000",  #液位  （电容器液位、CT液位、PT液位统一标成00300000
"10000001",  #部件表面油污
"10010001",  #地面油污
"10020001",  #金属锈蚀
"10030000",  #表计正常
"10030001",  #表计表盘破损
"10030002",  #表计外壳破损
"10030003",  #表计表盘模糊
"10040000",  #呼吸器硅胶正常
"10040001",  #呼吸器硅胶变色
"10040002",  #呼吸器硅胶筒玻璃破损
"10040003",  #呼吸器油封玻璃破损
"10050000",  #箱门闭合正常
"10050001",  #箱门闭合异常
"10060000",  #绝缘子正常
"10060001",  #绝缘子破裂
"10070001",  #挂空悬浮物
"10080001",  #鸟巢
"10090001",  #盖板破损
"10100000",  #安全帽正常
"10100001",  #未穿安全帽
"10110000",  #工装正常
"10110001",  #未穿工装
"10120001",  #吸烟
"10130000",  #油位状态_油封正常
"10130001",  #油位状态_油封异常
"10140000",  #表计读数异常
"10150000",  #越线闯入
"10160000",  #火灾烟雾
"10170000",  #室内地面积水
"10170001",  #墙面漏水
"10170002",  #屋顶漏水
"10180000"   #小动物闯入
]
'''
classes =["yibiao"]
abs_path = os.getcwd()
print(abs_path)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open(r'./Annatotions/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(r'./labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    # 这里对不标准的xml文件（没有size字段）做了特殊处理，打开对应的图片，获取h, w
    if size == None:
        print('{}no_exist'.format(image_id))
        img = Image.open('VOCdata/images/' + image_id + '.jpg')
        w, h = img.size  # 大小/尺寸
        print('{}.xml  w{} h{}'.format(image_id, w, h))
    elif int(size.find('width').text) == 0 or int(size.find('height').text) == 0:
        print("123_123:  ", in_file)
        print('{}h_w = 0'.format(image_id))
        img = Image.open('VOCdata/images/' + image_id + '.jpg')
        w, h = img.size  # 大小/尺寸
        print('{}.xml  w{} h{}'.format(image_id, w, h))
    else:
        w = int(size.find('width').text)
        h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
sets = ['train', 'val', 'test']
#代码开始的地方
for image_set in sets:
    if not os.path.exists(r'./labels/'):
        os.makedirs(r'./labels/')
    image_ids = open(r'./ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open(r'%s.txt' % (image_set), 'w')
    # 这行路径不需更改，这是相对路径
    for image_id in image_ids:
        list_file.write('mydata_yibiao2/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
