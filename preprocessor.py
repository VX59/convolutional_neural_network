import os
import glob
from PIL import Image, ImageFilter,ImageEnhance, ImageOps
import shutil
from tqdm import tqdm
import os.path as path
import numpy as np
import tensorflow as tf
import tempfile
import random

# our dataset contains 40 thousand images 800 * 50
# labels are derived from the folder names and correspond to the part
# this program compresses and labels training images

from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN)

class input_pipeline(object):
    def __init__(self, SCALE, classes):

        self.SCALE = SCALE
        self.classes = classes
        self.batch_size = 16

        self.working_dir = "/Users/deros/Downloads/dataset/"
        self.dir_list = os.listdir(self.working_dir)  
        self.dir_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    def encode_pixels(self, image):
        WIDTH, HEIGHT = image.size
        data = list(image.getdata())
        data = [data[offset:offset+WIDTH] for offset in range(0,WIDTH*HEIGHT, WIDTH)]
        data = np.array(data)
        return data

    def select_data(self, group_size = 800,
                          start_group = 0):
        raw_x = []
        raw_y = []

        for i in range(start_group*group_size, 
                       (start_group+self.classes)*group_size,
                       group_size):

            for j in tqdm (range(int(group_size)), desc=f"subset {i}..."):    
                a = i+j # specific index
                if j < group_size:
                    file = self.dir_list[a]
                    filepath = self.working_dir+'/'+file

                    image = Image.open(filepath).convert('L')
                    image.thumbnail((self.SCALE,self.SCALE))
                    image = ImageOps.invert(image)
                    #image = image.filter(EMBOSS)
                    image = self.encode_pixels(image)

                    raw_x.append(image)
                    raw_y.append(int(i/group_size) - start_group)

        x_count = [[] for _ in range(self.classes)]
        y_count = [[] for _ in range(self.classes)]
        
        print(y_count)
        print(x_count)

        a = 0
        for y in raw_y:
            for i in range(0, self.classes):
                if y == i:
                    x_count[i].append(raw_x[a])
                    y_count[i].append(y)
                    break
            a += 1

        x_count = np.array(x_count)
        y_count = np.array(y_count)

        print(y_count.shape)
        print(y_count)

        # now we have separated our x y into respective classses, we need to partition into train and test
        split = int(y_count.shape[1] * 0.2)

        train_x_sep = np.array([x[split:] for x in x_count])
        test_x_sep = np.array([x[:split] for x in x_count])

        print(train_x_sep.shape)
        print(test_x_sep.shape)
        train_y_sep = np.array([y[split:] for y in y_count])
        test_y_sep = np.array([y[:split] for y in y_count])

        print(train_y_sep.shape)
        print(test_y_sep.shape)

        train_x = np.reshape(train_x_sep, (train_x_sep.shape[0] * train_x_sep.shape[1], self.SCALE, self.SCALE))
        test_x = np.reshape(test_x_sep, (test_x_sep.shape[0] * test_x_sep.shape[1], self.SCALE, self.SCALE))

        train_y = np.reshape(train_y_sep, (train_y_sep.shape[0] * train_y_sep.shape[1]))
        test_y = np.reshape(test_y_sep, (test_y_sep.shape[0] * test_y_sep.shape[1]))
    
        print("len train,", np.array(train_x).shape,
                            np.array(train_y).shape,
              "len test, ", np.array(test_x).shape,
                            np.array(test_y).shape)

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)) 
        print('created train dataset')
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        print('created test dataset')

        cardinality = int(tf.data.experimental.cardinality(train_ds))

        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(buffer_size=cardinality)

        cardinality = int(tf.data.experimental.cardinality(test_ds))

        test_ds = test_ds.cache()
        test_ds = test_ds.shuffle(buffer_size=cardinality)

        return train_ds, test_ds

    def create_sample(self):
        index = random.randint(0,(self.classes * 800))
        print(index)
        label = int(index / 800)
        file = self.dir_list[index]
        filepath = self.working_dir+'/'+file

        image = Image.open(filepath).convert('L')
        image.thumbnail((self.SCALE,self.SCALE))
        image = ImageOps.invert(image)
        data = self.encode_pixels(image)
        sample = tf.constant(data, dtype=tf.float32)
        sample = tf.reshape(sample, [1, self.SCALE, self.SCALE, 1])

        return sample, label, image

        # add data sharding function to save the dataset
        # this way we dont have to run the batchin fuction every time we train a sorter