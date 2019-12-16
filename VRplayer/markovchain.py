from collections import deque  # 双端队列
from transformtile import Transform_tile
import matplotlib.pyplot as plt  # plt 用于显示图片
from video import *
import numpy as np
import pandas as pd
import os
import math

out_path = 'markovchain'


class MC:
    def __init__(self):
        # 视频信息初始化
        self.video = Video()
        self.configure = Configuration()
        self.tf = Transform_tile()
        self.video_name = self.video.video_name
        self.train_set_path = self.video.train_set_path
        self.predict_time = self.configure.predict_time
        self.frames = self.video.video_frames
        self.fps = self.video.video_fps
        self.predict_windows = int(self.predict_time * self.fps)
        self.max_x = self.video.max_x
        self.max_y = self.video.max_y
        # 算出单个tile的角度跨度
        self.tile_width = 360 / self.max_x
        self.tile_height = 180 / self.max_y
        # 计算相关的数据
        self.length = self.frames - self.predict_windows
        self.tile_num = self.max_x * self.max_y
        self.markovchain = np.zeros((self.length, self.tile_num, self.tile_num), dtype=np.float)
        # 开始生成马尔科夫链
        self.init_MC()
        self.save_MC()

    # 马尔科夫链初始化
    def init_MC(self):
        files = os.listdir(self.train_set_path)
        for file in files:
            print('正在读取：', file, '...')
            csv = os.path.join(self.train_set_path, file)
            self.read_csv(csv)
        for i in range(len(self.markovchain)):
            for j in range(len(self.markovchain[i])):
                summation = sum(self.markovchain[i][j])
                if summation != 0:
                    self.markovchain[i][j] /= summation
        print('马尔科夫链初始化成功！')



    def read_csv(self, file_in):
        df = pd.read_csv(file_in, usecols=[0, 1, 2])
        data = np.array(df)
        for i in range(self.length):
            #current_frame = data[i][0]
            current_yaw = data[i][1]
            current_pitch = data[i][2]
            current_tile_no = self.tf.transform_coordinates(current_yaw, current_pitch)
            #next_frame = data[i + self.predict_windows][0]
            next_yaw = data[i + self.predict_windows][1]
            next_pitch = data[i + self.predict_windows][2]
            next_tile_no = self.tf.transform_coordinates(next_yaw, next_pitch)
            self.markovchain[i][current_tile_no - 1][next_tile_no - 1] += 1

    def save_MC(self):
        print('正在保存马尔可夫链...')
        np_name = self.video_name + '_' + str(self.predict_time) + 's'
        path = os.path.join(out_path, np_name)
        np.save(path, self.markovchain)
        print('马尔可夫链保存成功！')

    def play(self):
        for i in range(len(self.markovchain)):
            frame = self.markovchain[i]
            plt.imshow(frame, interpolation='nearest', cmap=plt.cm.hot, vmin=0, vmax=1, extent=(0, self.video.max_x, 0, self.video.max_y))





if __name__ == '__main__':
    MC()



    '''多文件生成马尔科夫链
    init = MC()
    init.init_MC()
    result = init.markovchain
    np.savetxt('test.txt', result[1], fmt='%0.2f')
    print(result[1])
    print(np.sum(result[1]), np.max(result[1]), np.where(result[1]==np.max(result[1])))
    '''



    '''单文件马尔科夫链
    init = MC()
    path = Video().dataset_path
    file = 'user_001_video_001.csv'
    path_in = os.path.join(path, file)
    print(path_in)
    init.read_csv(path_in)
    result = init.markovchain
    np.savetxt('test.txt', result[1], fmt='%d')
    '''



    '''坐标转换示例
    yaw = 180
    pitch = 90
    # yaw = yaw % 180
    yaw = (yaw + 180) % 360 - 180
    a = MC()
    x = 22.5
    y = 180
    print(yaw, pitch)
    print(x, y)
    print('----------------------------')
    print('等矩形坐标', a.to_coordinates(yaw, pitch))
    print('tile号', a.to_tile(x, y))
    '''