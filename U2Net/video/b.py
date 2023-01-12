#读取视频中的图片
import cv2
import os

'''
in_path =r'E:\Work\视频\附件2-一键顺控基准视频\附件2-一键顺控基准视频'
for file in os.listdir(in_path):
    if ('.mp4' in file):
        mp4_path =os.path.join(in_path,file[:-4])
        if not os.path.isdir(mp4_path):
            os.mkdir(mp4_path)
        videoCapture = cv2.VideoCapture()
        videoCapture.open(os.path.join(in_path,file))
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        # fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
        print("fps=", fps, "frames=", frames)
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            out_path =os.path.join(mp4_path,str(i)+'.jpg')
            cv2.imwrite(out_path, frame)
    else:
        pass
'''



videoCapture = cv2.VideoCapture()
videoCapture.open('0001_normal.mp4')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
#fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
print("fps=",fps,"frames=",frames)
for i in range(int(frames)):
    ret,frame = videoCapture.read()
    cv2.imwrite("video/pictures0001/%d.jpg"%i,frame)
