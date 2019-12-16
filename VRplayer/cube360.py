from collections import deque  # 双端队列
from transformtile import Transform_tile
from video import *
import numpy as np
import pandas as pd
import os
import math

out_path = 'cube360'


class Load_data:
    def __init__(self):
        # 视频信息初始化
        self.video = Video()
        self.configure = Configuration()
        self.tf = Transform_tile()
        self.video_name = self.video.video_name
        self.train_set_path = self.video.train_set_path
        self.frames = self.video.video_frames
        self.sum_users = len(os.listdir(self.video.train_set_path))
        self.data = np.empty(shape=[self.frames, self.sum_users])
        print(self.data)
        self.read_csv('train_set/frame/video_005/user_001_video_005.csv')



    # 马尔科夫链初始化
    def init_list(self):
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
                else:
                    self.markovchain[i][j][j] = 1
        print('马尔科夫链初始化成功！')



    def read_csv(self, file_in):
        df = pd.read_csv(file_in, usecols=[1, 2])
        data = np.array(df)
        self.data = np.append(self.data, data, axis=0)
        print(data)

    def save_MC(self):
        print('正在保存马尔可夫链...')
        np_name = self.video_name + '_' + str(self.predict_time) + 's'
        path = os.path.join(out_path, np_name)
        np.save(path, self.markovchain)
        print('马尔可夫链保存成功！')





if __name__ == '__main__':
    Load_data()



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