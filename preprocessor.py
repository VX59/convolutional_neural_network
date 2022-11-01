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

SCALE = 28

class input_pipeline(object):
    def __init__(self):

        self.SCALE = 28
        self.batch_size = 32
        self.raw_list = []

        self.working_dir = "data/"
        self.dir_list = os.listdir(self.working_dir)  
        self.dir_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    def encode_pixels(self, image):
        WIDTH, HEIGHT = image.size
        data = list(image.getdata())
        data = [data[offset:offset+WIDTH] for offset in range(0,WIDTH*HEIGHT, WIDTH)]
        data = np.array(data)

    def select_data(self, skip = 8,
                          max_limit = 800,
                          group_size = 800,
                          start_group = 4,
                          classes = 4):

        for i in range(start_group*group_size, 
                       (start_group*group_size)+(classes*group_size),
                       group_size):

            for j in tqdm (range(int(group_size)), desc="Loading..."):    
                a = i+j # specific index
                if j % skip == 0 and j < max_limit:
                    file = self.dir_list[a]
                    filepath = self.working_dir+'/'+file

                    image = Image.open(filepath).convert('L')
                    image.thumbnail((SCALE,SCALE))
                    image = image.filter(EMBOSS)

                    # encode image
                    encoded_image = self.encode_pixels(image)

                    self.raw_list.append((encoded_image, int(i/group_size)))

        raw_tnsr = tf.constant(self.raw_list,dtype=tf.float16, name='raw dataset')
        raw_tnsr = tf.reshape(raw_tnsr,[len(self.raw_list), 1, self.SCALE, self.SCALE, 1])
        
        raw_ds = tf.data.Dataset.from_tensor_slices(raw_tnsr)

        cardinality = tf.data.experimental.cardinality(raw_ds)

        raw_ds.cache()
        raw_ds.shuffle(buffer_size=cardinality)
        raw_ds.batch(self.batch_size)
        raw_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        test_split = int(cardinality * 0.2)

        test_ds = raw_ds.take(test_split)

        train_ds = raw_ds.skip(test_split)

        val_split = int(tf.data.experimental.cardinality(train_ds) * 0.2)

        train_ds = train_ds.skip(val_split)
        val_ds = train_ds.take(val_split)

        print('created dataset')
