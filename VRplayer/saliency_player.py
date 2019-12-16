from video import *
from collections import deque  # 双端队列
from pylab import mpl
from prediction import Predict
import matplotlib.pyplot as plt  # plt 用于显示图片
import imutils
import pandas as pd
import numpy as np
import math
import csv
import cv2
import os
# 设置字体（不然显示不出中文）
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取坐标

# 户号
no_user = 1

# 初始化路径
file_path = Video().dataset_path
video_name = Video().video_name
file_name = 'user_' + "%03d" % no_user + '_' + video_name + '.csv'
file_in = os.path.join(file_path, file_name)

class Player:
    def __init__(self, path_in):
        self.file_in = path_in

        # 初始化
        self.video = Video()
        self.config = Configuration()
        self.sliding_windows = int(self.video.video_fps * self.config.training_time)
        self.predict_windows = int(self.video.video_fps * self.config.predict_time)
        #file = pd.read_csv(self.file_in, usecols=[0, 1, 2])
        #df = pd.DataFrame(file)
        self.real_coordinates = deque()
        self.pre_coordinates = deque()
        self.objects = deque()
        #self.play(df)
        self.play()
        print(self.objects)

    def play(self):
        #数据部分
        #training_set = deque(maxlen=self.sliding_windows)
        # 视频部分
        video_in = self.video.video_path
        camera = cv2.VideoCapture(video_in)
        # 用BackgroundSubtractorKNN构建背景模型
        bs = cv2.createBackgroundSubtractorKNN()
        cv2.namedWindow("surveillance", 0)
        frame_no = 1
        while True:
            #frame_no = index + 1
            #current_coordinates = [row['no.frames'], row['yaw'], row['pitch']]
            #current_x, current_y = self.trans_coordinates(current_coordinates[1], current_coordinates[2])
            #self.real_coordinates.append(current_coordinates)
            #print("当前帧为", frame_no, "坐标为", current_coordinates)
            # 视频
            success, frame = camera.read()
            if not success:
                break
            fgmask = bs.apply(frame)
            # 每一帧的对象入队
            objects_in_frame = []
            objects_in_frame.append(frame_no)
            # 阈值化
            th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
            # 通过对前景掩模进行膨胀和腐蚀处理，相当于进行闭运算
            # 开运算：先腐蚀再膨胀
            # 闭运算：先膨胀再腐蚀
            th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
            # 轮廓提取
            image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            counter = 0
            for c in contours:
                # 对每一个轮廓，如果面积大于阈值500
                if cv2.contourArea(c) > 5000:
                    # 绘制外包矩形 x,y为左上顶点坐标；w,h为矩形宽高
                    (x, y, w, h) = cv2.boundingRect(c)
                    # 外包矩形中点
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    counter += 1
                    # 判断是否在框中
                    objects_in_frame.append((x, y, w, h))
            frame_no += 1
            #绘制当前坐标点
            # 每一帧的对象入队
            self.objects.append(objects_in_frame)
            print(objects_in_frame)
            # 限制显示窗口大小
            frame = imutils.resize(frame, width=1000)
            cv2.imshow("surveillance", frame)
            #out.write(frame)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
        # out.release()
        camera.release()

    def csv_write(self):
        line = ['frame']
        csv_name = 'saliency_data/' + self.video.video_name + ".csv"
        out = open(csv_name, 'w', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(line)
        n = 0
        print("写入数据中...")
        while n < self.video.video_frames:
            csv_write.writerow(self.objects[n])
            n = n + 1
        print("写入成功！")






if __name__ == '__main__':
    # PredictPolicy.LSRpolicy
    a = Player(file_in)
    a.csv_write()

