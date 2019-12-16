import matplotlib.pyplot as plt # plt 用于显示图片
from pylab import mpl
from video import Video
from datasetreader import Datasetreader
import numpy as np
import cv2 as cv
import os
# 设置字体（不然显示不出中文）
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 缓冲区路径
cache_path = "E:\\My_projects/VRplayer/cache"

class Client:
    def __init__(self):
        self.video = Video()
        print("初始化缓冲区...")
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        self.cache = cache_path
        print("缓冲区初始化成功！")
        self.size = (640, 480)

    def iniCache(self):
        print("初始化路径...")
        dir_path = os.path.join(self.cache, "Thermal_map")
        file = os.path.join(dir_path, self.video.video_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if not os.path.exists(file):
            os.mkdir(file)
        print("初始化成功！")
        dr = Datasetreader()
        numpy = dr.analyst()
        for i in range(0, self.video.video_frames):
            print("正在生成第", i + 1, "张图......")
            temp = numpy[i]
            frame = np.reshape(temp, (self.video.max_y, self.video.max_x))
            plt.imshow(frame, interpolation='nearest', cmap=plt.cm.hot, vmin=0, vmax=1, extent=(0, self.video.max_x, 0, self.video.max_y))  # cmap=plt.cm.gray绘制黑白图像
            # plt.colorbar()
            image = str(i + 1) + ".png"
            image_dir = os.path.join(file, image)
            plt.savefig(image_dir)
            print("图片生成成功！")
            plt.close()
        return file
    #  视频生成器
    def videoMaker(self, path):  # path:热力图路径
        print("初始化视频生成器路径...")
        dir_path = os.path.join(self.cache, 'video')
        file_path = os.path.join(dir_path, self.video.video_name)
        if not os.path.exists(path):
            print("初始化失败！没有图片路径！")
            os._exit(1)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        print("初始化视频生成器成功！")
        video_path = os.path.join(file_path, self.video.video_name + '.avi')
        video = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'DIVX'), self.video.video_fps, self.size)
        print("视频生成中...")
        for i in range(0, self.video.video_frames):  # 修改： 基于总帧数的生成方式
            img_name = str(i + 1) + ".png"
            img_dir = os.path.join(path, img_name)
            img = cv.imread(img_dir, 1)
            video.write(img)
        video.release()
        cv.destroyAllWindows()
        print("视频生成成功！")
        return video_path
    # 播放器初始化
    def iniPlayer(self):
        print("初始化播放器...")
        path = self.iniCache()
        video_path = self.videoMaker(path)
        print("播放器初始化成功！")
        return video_path
    # 播放功能
    def play(self, path):
        cap = cv.VideoCapture(path)
        dt = int(1 / self.video.video_fps * 1000)
        n = 1
        print("开始播放，按下q提前结束播放...")
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("视频播放完毕！")
                break
            cv.imshow('video', frame)
            if cv.waitKey(dt) == ord('q'):
                break
            n += 1
        cap.release()
        cv.destroyAllWindows()

    def autoplay(self):
        video = self.iniPlayer()
        self.play(video)



if __name__ == '__main__':
    a = Client()
    #a.autoplay()

# 手动播放
    #path = a.iniCache()
    #a.videoMaker("E:\My_projects\VRplayer\cache\Thermal_map\Roller_coaster_360")
    a.play("E:\\My_projects/VRplayer/cache/video/Roller_coaster_360/Roller_coaster_360.avi")



