import os
import glob
from PIL import Image, ImageFilter,ImageEnhance
import shutil
from tqdm import tqdm
import os.path as path
import numpy as np
import tensorflow as tf

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
        self.batch_size = 32
        self.raw_x = []
        self.raw_y = []

        self.working_dir = "/home/rsenic/dataset/"
        self.dir_list = os.listdir(self.working_dir)  
        self.dir_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    def encode_pixels(self, image):
        WIDTH, HEIGHT = image.size
        data = list(image.getdata())
        data = [data[offset:offset+WIDTH] for offset in range(0,WIDTH*HEIGHT, WIDTH)]
        data = np.array(data)
        return data

    def select_data(self, skip = 8,
                          max_limit = 800,
                          group_size = 800,
                          start_group = 0):

        for i in range(start_group*group_size, 
                       (start_group*group_size)+(self.classes*group_size),
                       group_size):

            for j in tqdm (range(int(group_size)), desc="Loading..."):    
                a = i+j # specific index
                if j % skip == 0 and j < max_limit:
                    file = self.dir_list[a]
                    filepath = self.working_dir+'/'+file

                    image = Image.open(filepath).convert('L')
                    image.thumbnail((self.SCALE,self.SCALE))
                    image = image.filter(EMBOSS)

                    # encode image
                    encoded_image = self.encode_pixels(image)

                    self.raw_x.append(encoded_image)
                    self.raw_y.append(int(i/group_size))
        
        raw_ds = tf.data.Dataset.from_tensor_slices((self.raw_x, self.raw_y))

        cardinality = int(tf.data.experimental.cardinality(raw_ds))

        raw_ds.cache()
        raw_ds.shuffle(buffer_size=cardinality)
        raw_ds.batch(self.batch_size)
        raw_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        test_split = int(cardinality * 0.2)

        self.test_ds = raw_ds.take(test_split)

        train_ds = raw_ds.skip(test_split)

        cardinality = int(tf.data.experimental.cardinality(train_ds))
        val_split = int(cardinality * 0.2)

        self.train_ds = train_ds.skip(val_split)
        self.val_ds = train_ds.take(val_split)

        print('created dataset')

        return self.train_ds, self.val_ds, self.test_ds
