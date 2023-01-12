import os
import random

#划分数据集为训练集，测试集，验证集
trainval_percent = 0.95
train_percent = 0.8
xmlfilepath = r'Annatotions'
txtsavepath = r'ImageSets'  # 存放划分的数据集的对应名称，文件名不含后缀名
if not os.path.exists(txtsavepath):
    os.mkdir(txtsavepath)

total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(r'./ImageSets/trainval.txt', 'w')
ftest = open(r'./ImageSets/test.txt', 'w')
ftrain = open(r'./ImageSets/train.txt', 'w')
fval = open(r'./ImageSets/val.txt', 'w')

for i in list:
    # name 去掉后缀名
    name = total_xml[i][:-4] + '\n'
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