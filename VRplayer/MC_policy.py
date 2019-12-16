from collections import deque  # 双端队列
from video import *
import numpy as np
from transformtile import Transform_tile, Tile_to_coordinates

class MCpolicy:
    def __init__(self):
        self.name = "MC"
        self.video = Video()
        self.configure = Configuration()
        self.tf = Transform_tile()
        self.t_c = Tile_to_coordinates()
        self.video_name = self.video.video_name
        self.predict_time = self.configure.predict_time
        np_name = self.video_name + '_' + str(self.predict_time) + 's.npy'
        np_path = os.path.join('markovchain', np_name)
        if os.path.exists(np_path):
            self.markovchain = np.load(np_path)
        else:
            print('马尔科夫链不存在！请生成！')
            os._exit(0)

    def predict(self, current_coordinates):
        current_frame = int(current_coordinates[0])
        current_yaw = current_coordinates[1]
        current_pitch = current_coordinates[2]
        current_tile_no = self.tf.transform_coordinates(current_yaw, current_pitch)
        current_markovchain = self.markovchain[current_frame - 1][current_tile_no - 1]

        # 为了防止有重复最大值，这里有待改进
        max_tile_no = np.argmax(current_markovchain) + 1  # 索引和tile_id相差1
        next_yaw, next_pitch = self.t_c.cal_coordinates(max_tile_no)
        print(current_tile_no, max_tile_no)



if __name__ == '__main__':
    a = np.array([2.0, 3, 2.0])
    print(np.argmax(a))
    MCpolicy().predict(a)
