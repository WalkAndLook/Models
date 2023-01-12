'''
代码作用：
因为只选择了图片中的仪表，其他图片删除掉，所以对应的txt文件也只保存仪表对应的标签
也就是在标签中选择仪表对应的标签
'''


import os
import cv2
from shutil import move
input_img_path = r'./images/'
input_label_path = r'./labels_tmp'
out_labels = r'./labels'

#print(os.listdir(input_img_path))
for img_file_name in os.listdir(input_img_path):
    label_file_name = os.path.splitext(img_file_name)[0] + ".txt"
    label_file_path = os.path.join(input_label_path,label_file_name)
    move(label_file_path,out_labels)
