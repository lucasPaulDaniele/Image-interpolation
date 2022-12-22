import os
import random
import numpy as np

from cv2 import imread, IMREAD_GRAYSCALE
from glob import glob
from tensorflow.keras.utils import Sequence

random.seed()

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size, im_size, frame_dist=4, n_past=1, n_future=1, videos_dir="./youtube", data_aug=False, shuffle=False, type_='train'):
        'Initialization'
        self.im_size = im_size
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.frame_dist = frame_dist
        self.videos_dir = videos_dir
        self.type_ = type_
        self.n_past = n_past
        self.n_future = n_future
        self.im_list_paths = self.get_im_paths()
        print(len(self.im_list_paths))
        self.on_epoch_end()

    def __len__(self):
        return len(self.im_list_paths) // self.batch_size
      
    def __getitem__(self, index):
        X, y = [], []
        for _ in range(self.batch_size):
            im_path = self.im_list_paths[index]
            frame_num = self.get_frame_num(im_path)
        
            X_temp, y_temp = self.get_X_y_temp(im_path, frame_num)
            X.append(X_temp)
            y.append(y_temp)        
        #
        X , y = np.array(X).astype("float32"), np.array(y).astype("float32")
        #
        return self.format_output(X, y)

    def get_frame_num(self, im_path):
      return int(os.path.splitext(im_path)[0].split('_')[1])

    def get_path(self, im_path, index):
      return im_path.replace(str(self.get_frame_num(im_path)), str(index))

    def get_im_paths(self):
      return [f for f in glob(f'{self.videos_dir}/{self.type_}/**/*.jpg', recursive=True)]

    def load_image(self, im_path, index):
      return imread(self.get_path(im_path, index), IMREAD_GRAYSCALE) if os.path.exists(self.get_path(im_path, index)) else np.zeros((128, 128))

    def get_X_y_temp(self, im_path, frame_num):
      past_indexes = list(range(frame_num-self.n_past*self.frame_dist-self.frame_dist, frame_num-self.frame_dist, self.frame_dist))
      future_indexes = list(range(frame_num+self.frame_dist, frame_num+(self.n_future+1)*self.frame_dist, self.frame_dist))
      X = np.array([self.load_image(im_path, index) for index in past_indexes + future_indexes])
      y = self.load_image(im_path, frame_num-(self.frame_dist))
      return np.rollaxis(X, 0, 3).astype("float32")/255., y.reshape((*y.shape, 1)).astype("float32")/255.

class DataGeneratorDoubleInput(DataGenerator):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def format_output(self, X, y):
    return [X[:, :, :, 0].reshape((self.batch_size,128,128,1)), X[:, :, :, 1].reshape((self.batch_size,128,128,1))], y.reshape((self.batch_size,128,128,1))

class DataGeneratorDoubleInputDict(DataGenerator):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def format_output(self, X, y):
    return {"x0": X[:, :, :, 0].reshape((self.batch_size,128,128,1)), "x1": X[:, :, :, 1].reshape((self.batch_size,128,128,1))}, y.reshape((self.batch_size,128,128,1))

class DataGeneratorSingleInput(DataGenerator):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def format_output(self, X, y):
    return np.array(X).astype("float32"), np.array(y).astype("float32")