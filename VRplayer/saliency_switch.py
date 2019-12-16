from video import *
from collections import deque  # 双端队列
from pylab import mpl
from prediction import Predict
import matplotlib.pyplot as plt  # plt 用于显示图片
import imutils
import pandas as pd
import numpy as np
import accuracy
import math
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help="m (or nothing) for meanShift and c for camshift")
args = vars(parser.parse_args())
font = cv2.FONT_HERSHEY_SIMPLEX
# 设置字体（不然显示不出中文）
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取视频
video_path = "video"

# 读取坐标
file_path = "player"
file_name = "user_001_Clash_of_Clans_360.csv"
file_in = os.path.join(file_path, file_name)

# 追踪状态：0：无显著，无追踪；1：有显著，无追踪；2：有显著，有追踪；3：无显著，有追踪
tracking_state = False

class Player:
    def __init__(self, path_in):
        self.file_in = path_in

        # 初始化
        self.video = Video()
        self.config = Configuration()
        self.sliding_windows = int(self.video.video_fps * self.config.training_time)
        self.predict_windows = int(self.video.video_fps * self.config.predict_time)
        file = pd.read_csv(self.file_in, usecols=[0, 1, 2])
        df = pd.DataFrame(file)
        self.real_coordinates = deque()
        self.pre_coordinates = deque()
        self.objects = deque()
        self.play(df)
        print(self.objects)
        self.accuracy = accuracy.Accuracy(self.real_coordinates, self.pre_coordinates).execute()
        self.tile_accuracy = accuracy.TileAccuracy(self.real_coordinates, self.pre_coordinates).execute()

    def play(self, df):
        # 数据部分
        #global tracking_state
        training_set = deque(maxlen=self.sliding_windows)
        # 视频部分
        video_name = self.video.video_name + '.mp4'  # 文件名.MP4
        video_in = os.path.join(video_path, video_name)
        camera = cv2.VideoCapture(video_in)
        # 用BackgroundSubtractorKNN构建背景模型
        bs = cv2.createBackgroundSubtractorKNN()
        cv2.namedWindow("surveillance")
        frames = 1
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
        self.width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.max_area = int(self.width * self.height * 0.8)
        self.min_area = int(self.width * self.height * 0.005)



        for index, row in df.iterrows():
            frame_no = index + 1
            current_coordinates = [row['no.frames'], row['yaw'], row['pitch']]
            current_x, current_y = self.trans_coordinates(current_coordinates[1], current_coordinates[2])
            self.real_coordinates.append(current_coordinates)
            print("当前帧为", frame_no, "坐标为", current_coordinates)
            # 视频
            success, frame = camera.read()
            if not success:
                break
            fgmask = bs.apply(frame)
            # 每一帧的对象入队
            objects_in_frame = deque()
            objects_in_frame.append(frames)
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
            token = False

            for c in contours:
                # 对每一个轮廓，如果面积大于阈值500
                if cv2.contourArea(c) > self.min_area:# and cv2.contourArea(c) < self.max_area:
                    # 绘制外包矩形 x,y为左上顶点坐标；w,h为矩形宽高
                    (x, y, w, h) = cv2.boundingRect(c)
                    # 外包矩形中点
                    center_x = int((2 * x + w) / 2)
                    center_y = int((2 * y + h) / 2)
                    cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    counter += 1
                    # 判断是否在框中
                    if self.is_area(current_x, current_y, x, y, w, h):
                        # 进入有显著情况
                        token = True
                    objects_in_frame.append((x, y, w, h))

            frames += 1
            # 绘制当前坐标点
            cv2.circle(frame, (current_x, current_y), 4, (0, 0, 255), -1)
            if frame_no <= self.video.video_frames - self.predict_windows:
                next_frame_no = float(frame_no + self.predict_windows)
                training_set.append(current_coordinates)
                sample = np.asarray(training_set, dtype=float)
                train_yaw = sample[:, [0, 1]]
                train_pitch = sample[:, [0, 2]]
                if token:
                    next_yaw = Predict(train_yaw, next_frame_no).LR()
                    next_pitch = Predict(train_pitch, next_frame_no).LR()
                else:
                    next_yaw = Predict(train_yaw, next_frame_no).LSR()
                    next_pitch = Predict(train_pitch, next_frame_no).LSR()

                predict_coordinates = [next_frame_no, next_yaw, next_pitch]
                # predict_coordinates = self.predict_policy.predict(training_set, next_frame_no)
                self.pre_coordinates.append(predict_coordinates)
            if self.pre_coordinates[0][0] <= frame_no:
                i = frame_no - self.predict_windows - 1
                pre_coord = self.pre_coordinates[i]
                print("预测坐标", pre_coord[0], "坐标为", pre_coord)
                predict_x, predict_y = self.trans_coordinates(pre_coord[1], pre_coord[2])
                cv2.circle(frame, (predict_x, predict_y), 4, (255, 0, 0), -1)
                # 每一帧的对象入队

            # cv2.imshow("surveillance", dilated)
            # 限制显示窗口大小
            frame = imutils.resize(frame, width=1000)
            cv2.imshow("surveillance", frame)
            # out.write(frame)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
        # out.release()
        camera.release()

    def trans_coordinates(self, yaw, pitch):
        trans_yaw = int(((yaw + 180) / 360) * self.width)
        trans_pitch = int((abs(pitch - 90) / 180) * self.height)
        return trans_yaw, trans_pitch

    def is_area(self, current_x, current_y, x, y, w, h):
        if current_x >= x and current_x <= x + w and current_y >= y and current_y <= y + h:
            return True
        else:
            return False

    def draw(self):
        plt.figure(1)
        yaw = plt.subplot(2, 1, 1)
        pitch = plt.subplot(2, 1, 2)
        colors = ['black', 'red']  # 设置颜色
        label = ['实际点', '预测点']  # 标签题目

        plt.sca(yaw)  # 选中yaw图
        plt.title("yaw的运动轨迹")
        plt.xlim(0, self.video.video_frames)
        plt.ylim(-180, 180)
        real_frame = np.asarray(self.real_coordinates)[:, 0]
        real_yaw = np.asarray(self.real_coordinates)[:, 1]
        plt.scatter(real_frame, real_yaw, s=1, color=colors[0], label=label[0])
        pre_frame = np.asarray(self.pre_coordinates)[:, 0]
        pre_yaw = np.asarray(self.pre_coordinates)[:, 1]
        plt.scatter(pre_frame, pre_yaw, s=1, color=colors[1], label=label[1])  # alpha=：设置透明度（0-1）
        plt.xlabel("帧")
        plt.ylabel("角度")
        plt.legend(loc="best")

        plt.sca(pitch)  # 选中pitch图
        plt.title("pitch的运动轨迹")
        plt.xlim(0, self.video.video_frames)
        plt.ylim(-90, 90)
        real_frame = np.asarray(self.real_coordinates)[:, 0]
        real_pitch = np.asarray(self.real_coordinates)[:, 2]
        plt.scatter(real_frame, real_pitch, s=1, color=colors[0], label=label[0])
        pre_frame = np.asarray(self.pre_coordinates)[:, 0]
        pre_pitch = np.asarray(self.pre_coordinates)[:, 2]
        plt.scatter(pre_frame, pre_pitch, s=1, color=colors[1], label=label[1])
        plt.xlabel("帧")
        plt.ylabel("角度")
        plt.legend(loc="best")
        plt.show()


def center(points):
    """计算矩阵的质心"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


class Pedestrian():
    """Pedestrian class
    each pedestrian is composed of a ROI, an ID and a Kalman filter
    so we create a Pedestrian class to hold the object state
    """
    def __init__(self, frame, track_window):
        """init the pedestrian object with track window coordinates"""
        # set up the roi
        global tracking_state
        self.tracker = cv2.TrackerKCF_create()
        ok = self.tracker.init(frame, track_window)
        if ok is True:
            tracking_state = True
        else:
            tracking_state = False
            cv2.putText(frame, "Tracking init failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    def __del__(self):
        print("Pedestrian destroyed")

    def update(self, frame):
        # print "updating %d " % self.id
        ok, bbox = self.tracker.update(frame)
        if ok is True:
            print(bbox)
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            return True, (x, y, w, h)
        else:
            return False, (0, 0, 0, 0)



if __name__ == '__main__':
    # PredictPolicy.LSRpolicy
    a = Player(file_in)
    a.draw()

