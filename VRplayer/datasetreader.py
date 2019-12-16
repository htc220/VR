# -*-  coding:utf-8 -*-
from transformtile import Tile_counter as Tile
import numpy as np
from video import Video
import pandas as pd  # csv、numpy处理数据的包
import os
import glob

# 初始化数据
class Datasetreader(Video):
    def __init__(self):
        Video.__init__(self)

    # 转换成numpy数组
    def npCreater(self, yaw, pitch):
        tile_arr = Tile().counter(yaw, pitch)
        tile_np = np.zeros((1, self.total_tile_num), dtype=np.int)  # 构建一个一维数组
        for i in tile_arr:
            tile_np[0][i-1] += 1
        return tile_np
    # 返回一个所有数据的数组

    def csvReader(self, file_in):
        numpy = np.zeros((self.video_frames, self.total_tile_num), dtype=np.int)
        file = pd.read_csv(file_in)
        df = pd.DataFrame(file)
        for i in range(self.video_frames):
            document = df[i:i + 1]
            yaw = document['yaw'][i]
            pitch = document['pitch'][i]
            temp = self.npCreater(yaw, pitch)  # 创建一个临时一维数组
            numpy[i] = temp
        return numpy

    # 加法器，遍历数据集，求和（返回的是int类型）
    def adder(self):
        summation = np.zeros((self.video_frames, self.total_tile_num), dtype=np.int)
        root_dir = self.dataset_path
        array_list = os.listdir(root_dir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(array_list)):
            print("正在读取用户", i + 1, "的数据...")
            file_in = os.path.join(root_dir, array_list[i])
            if os.path.isfile(file_in):
                temp = self.csvReader(file_in)
                print("文件：", file_in, "......读取完毕！")
                summation = np.add(summation, temp)
            else:
                print("ERROR！路径存在问题！")
        array_list.clear()
        return summation
        # result = summation.astype('float64')

    # 用户人数统计
    def userCounter(self):
        path_name = os.path.join(self.dataset_path,  '*.csv')
        path_file_number = glob.glob(path_name)  # 获取当前文件夹下个数
        return len(path_file_number)

    # 概率统计（返回的是概率）
    def analyst(self):
        summation = self.adder()
        user_num = self.userCounter()
        result = summation.astype('float64')
        result /= user_num
        return result

if __name__ == '__main__':
    a = Datasetreader()
    print(a.npCreater(2.347, 6.026))
    np.savetxt('out.txt', Datasetreader().analyst(), fmt="%f", delimiter=" ")
    # print(Frame().analyst())
