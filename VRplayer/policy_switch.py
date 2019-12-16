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
tolerance_threshold = 3
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
        global tracking_state
        tracker = None
        predict = True
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
        self.max_area = int(self.width * self.height * 0.4)
        self.min_area = int(self.width * self.height * 0.005)



        for index, row in df.iterrows():
            frame_no = index + 1
            current_coordinates = [row['no.frames'], row['yaw'], row['pitch']]
            # 转换为平面坐标
            next_frame_no = None
            current_x, current_y = self.trans_coordinates(current_coordinates[1], current_coordinates[2])
            self.real_coordinates.append(current_coordinates)
            #print("当前帧为", frame_no, "坐标为", current_coordinates)


            #寻找当前时间点的预测点

            # 视频
            success, frame = camera.read()
            if not success:
                break
            fgmask = bs.apply(frame)
            # 每一帧的对象入队
            objects_in_frame = deque()
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
            #开始预测决策
            saliency_state = False

            for c in contours:
                # 对每个轮廓c,绘制外包矩形 x,y为左上顶点坐标；w,h为矩形宽高
                (x, y, w, h) = cv2.boundingRect(c)
                S = w * h
                # 如果外包矩形符合要求，那么他就是合法的显著性对象
                if self.min_area <= S <= self.max_area:
                    (x, y, w, h) = cv2.boundingRect(c)
                    # 外包矩形中点
                    center_x = int((2 * x + w) / 2)
                    center_y = int((2 * y + h) / 2)
                    cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    counter += 1
                    # 判断预测点是否在框中
                    # 这里暂定
                    if predict is True:
                        if self.is_area(current_x, current_y, x, y, w, h):
                            # 进入有显著情况
                            saliency_state = True
                        objects_in_frame.append((x, y, w, h))
                    else:
                        print('预测结束！')
            # 显著性对象入队
            self.objects.append(objects_in_frame)
            if frame_no <= self.video.video_frames - self.predict_windows:
                next_frame_no = float(frame_no + self.predict_windows)
                training_set.append(current_coordinates)
                sample = np.asarray(training_set, dtype=float)
                train_yaw = sample[:, [0, 1]]
                train_pitch = sample[:, [0, 2]]
                if saliency_state is True:
                    next_yaw = Predict(train_yaw, next_frame_no).WLR()
                    next_pitch = Predict(train_pitch, next_frame_no).WLR()
                else:
                    next_yaw = Predict(train_yaw, next_frame_no).TLP()
                    next_pitch = Predict(train_pitch, next_frame_no).TLP()
                #后面删掉，这里只是TLP的预测点！
                predict_coordinates = [next_frame_no, next_yaw, next_pitch]
                self.pre_coordinates.append(predict_coordinates)
                #next_x, next_y = self.trans_coordinates(next_yaw, next_pitch)
            # 后面再思考这里如何进行停止预测，这里改141行也要改


            #当前时间实际点
            cv2.circle(frame, (current_x, current_y), 4, (0, 0, 255), -1)
            print(self.pre_coordinates[0][0])

            if self.pre_coordinates[0][0] <= frame_no:
                i = frame_no - self.predict_windows - 1
                pre_coord = self.pre_coordinates[i]

                predict_x, predict_y = self.trans_coordinates(pre_coord[1], pre_coord[2])
                cv2.circle(frame, (predict_x, predict_y), 4, (0, 255, 0), -1)
            else:
                pass


            if predict is True:
                # 当前时间TLP预测点
                cv2.circle(frame, (self.trans_coordinates(predict_coordinates[1], predict_coordinates[2])), 4, (255, 255, 0), -1)
                # 当前时间预测点位置



            frame = imutils.resize(frame, width=1000)
            cv2.imshow("surveillance", frame)
            #frames += 1
            # out.write(frame)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
        # out.release()
        print(self.pre_coordinates)
        camera.release()

    def trans_coordinates(self, yaw, pitch):
        x = int(((yaw + 180) / 360) * self.width)
        y = int(((90 - pitch) / 180) * self.height)
        return x, y

    def trans_sphere(self, x, y):
        yaw = x / self.width * 360 - 180
        pitch = 90 - y / self.height * 180
        return yaw, pitch

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
        print(len(pre_pitch), len(pre_yaw))
        plt.show()

def cal_center(point):
    center_x = (2 * point[0] + point[2])
    center_y = (2 * point[1] + point[2])
    return [center_x, center_y]



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
        #定长训练集，训练集长度在config文件里
        self.history = deque(maxlen=int(Video().video_fps * Configuration().training_time))
        #Pedestrian(frame, (min_x, min_y, min_w, min_h))
        #(x, y, w, h) = elem
        #center_x = int((2 * x + w) / 2)
        #center_y = int((2 * y + h) / 2)

        center = cal_center(track_window)

        self.history.append(center)
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
            center = cal_center((x, y, w, h))
            self.history.append(center)
            return True, (x, y, w, h)
        else:
            return False, (0, 0, 0, 0)

    def is_closer(self, comparison):
        global tolerance_threshold
        length = len(self.history)
        tracking_center = self.history[length - 1]
        player_center = comparison[length - 1]
        current_distance = math.sqrt(pow((tracking_center[0] - player_center[0]), 2) + pow((tracking_center[1] - player_center[1]), 2))
        counter = 0
        for i in range(len(self.history) - 1):
            tracking_center = self.history[length - i - 2]
            player_center = comparison[length - i - 2]
            past_distance = math.sqrt(pow((tracking_center[0] - player_center[0]), 2) + pow((tracking_center[1] - player_center[1]), 2))
            if past_distance <= current_distance:
                counter += 1
            else:
                break
        if counter >= tolerance_threshold:
            return True
        else:
            return False



    def predict(self):
        train_x = deque()
        train_y = deque()
        for i in range(len(self.history)):
            train_x.append([i, self.history[i][0]])
            train_y.append([i, self.history[i][1]])
        next = float(len(self.history) + int(Video().video_fps * Configuration().predict_time))
        next_x = Predict(train_x, next).LR()
        next_y = Predict(train_y, next).LR()
        return [next_x, next_y]




if __name__ == '__main__':
    # PredictPolicy.LSRpolicy
    a = Player(file_in)
    a.draw()

