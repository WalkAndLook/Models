'''
代码实现：
划分数据集为训练集，测试集，验证集，生成txt文件
txt文件里保存图片的路径
'''


import os
import random

#划分数据集为训练集，测试集，验证集
trainval_percent = 0.9
train_percent = 0.9
imgfilepath = r'./images'
savepath = r'./mydata_yibioa/images/'
total_img = os.listdir(imgfilepath)

num = len(total_img)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(r'trainval.txt', 'w')
ftest = open(r'test.txt', 'w')
ftrain = open(r'train.txt', 'w')
fval = open(r'val.txt', 'w')

for i in list:
    name = savepath+total_img[i] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()