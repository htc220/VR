from video import *
from collections import deque  # 双端队列
from pylab import mpl
import matplotlib.pyplot as plt  # plt 用于显示图片
import predictpolicy as pp
import pandas as pd
import numpy as np
import accuracy
import math
import os

# 设置字体（不然显示不出中文）
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
out_path = "images"

# 路径
file_path = "player"
file_name = "user_003_Clash_of_Clans_360.csv"
file_in = os.path.join(file_path, file_name)
# 预设参数：



class HMDplayer:
    def __init__(self, path_in, policy):
        # 接口
        self.file_in = path_in
        self.predict_policy = policy()
        # 初始化
        self.video = Video()
        self.config = Configuration()
        self.sliding_windows = int(self.video.video_fps * self.config.training_time)
        self.predict_windows = int(self.video.video_fps * self.config.predict_time)
        file = pd.read_csv(self.file_in, usecols=[0, 1, 2])
        df = pd.DataFrame(file)
        self.real_coordinates = deque()
        self.pre_coordinates = deque()
        # 速度
        self.rate = deque()
        # 开始播放
        self.play(df)
        self.accuracy = accuracy.Accuracy(self.real_coordinates, self.pre_coordinates).execute()
        self.accuracy_pro = accuracy.AccuracyPro(self.real_coordinates, self.pre_coordinates).execute()
        self.yaw_mean, self.pitch_mean = accuracy.Mean(self.real_coordinates, self.pre_coordinates).execute()
        self.yaw_mean_pro, self.pitch_mean_pro = accuracy.MeanPro(self.real_coordinates, self.pre_coordinates).execute()
        self.tile_accuracy = accuracy.TileAccuracy(self.real_coordinates, self.pre_coordinates).execute()

    def play(self, df):
        training_set = deque(maxlen=self.sliding_windows)
        past_coordinates =[]
        for index, row in df.iterrows():
            frame = index + 1
            current_coordinates = [row['no.frames'], row['yaw'], row['pitch']]   # 当前坐标
            self.real_coordinates.append(current_coordinates)
            print("当前帧为", frame, "坐标为", current_coordinates)
            if frame <= self.video.video_frames - self.predict_windows:
                next_frame = float(frame + self.predict_windows)
                training_set.append(current_coordinates)
                predict_coordinates = self.predict_policy.predict(training_set, next_frame)
                print("预测坐标", next_frame, "坐标为", predict_coordinates)
                self.pre_coordinates.append(predict_coordinates)
            if frame == 1:
                past_coordinates = current_coordinates
            elif frame > 1:
                rate_yaw = self.velocity(current_coordinates[1], past_coordinates[1])
                rate_pitch = self.velocity(current_coordinates[2], past_coordinates[2])
                current_rate = [frame, rate_yaw, rate_pitch]  # 当前速度
                print("当前速度", frame, "坐标为", current_rate)
                self.rate.append(current_rate)
                past_coordinates = current_coordinates
        print(self.rate)

    '''
    def velocity(self, raw_present, raw_past):
        present = math.radians(raw_present)
        past = math.radians(raw_past)
        present_d1_t = math.sin(present)
        present_d2_t = math.cos(present)
        present_t = math.degrees(math.atan(present_d1_t / present_d2_t))
        past_d1_t = math.sin(past)
        past_d2_t = math.cos(past)
        past_t = math.degrees(math.atan(past_d1_t / past_d2_t))
        result = (present_t - past_t) * self.video.video_fps
        return result
    '''


    def velocity(self, raw_present, raw_past):
        result = (raw_present - raw_past) * self.video.video_fps
        return result


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

    def yaw_speed(self):
        plt.figure(1)
        yaw = plt.subplot(2, 1, 1)
        yaw_speed = plt.subplot(2, 1, 2)
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

        plt.sca(yaw_speed)  # 选中pitch图
        plt.title("yaw的速度轨迹")
        plt.xlim(0, self.video.video_frames)
        #plt.ylim(-180, 180)
        real_frame = np.asarray(self.rate)[:, 0]
        speed = np.asarray(self.rate)[:, 1]
        plt.bar(real_frame, speed, color=colors[0], label=label[0])
        plt.xlabel("帧")
        plt.ylabel("速度")
        plt.legend(loc="best")

        plt.show()




if __name__ == '__main__':
    # PredictPolicy.LSRpolicy
    a = HMDplayer(file_in, pp.LSRpolicy)
    a.yaw_speed()
