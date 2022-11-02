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

        self.working_dir = "/Users/deros/Downloads/dataset"
        self.dir_list = os.listdir(self.working_dir)  
        self.dir_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    def encode_pixels(self, image):
        WIDTH, HEIGHT = image.size
        data = list(image.getdata())
        data = [data[offset:offset+WIDTH] for offset in range(0,WIDTH*HEIGHT, WIDTH)]
        data = np.array(data)
        return data

    def select_data(self, skip = 1,
                          max_limit = 800,
                          group_size = 800,
                          start_group = 0):
        raw_x = []
        raw_y = []
        for i in range(start_group*group_size, 
                       (start_group+self.classes)*group_size,
                       group_size):

            for j in tqdm (range(int(group_size)), desc="Loading..."):    
                a = i+j # specific index
                if j % skip == 0 and j < max_limit:
                    file = self.dir_list[a]
                    filepath = self.working_dir+'/'+file

                    image = Image.open(filepath).convert('L')
                    image.thumbnail((self.SCALE,self.SCALE))
                    image = image.filter(EMBOSS)
                    image = self.encode_pixels(image)

                    raw_x.append(image)
                    raw_y.append(int(i/group_size) - start_group)
        
        raw_ds = tf.data.Dataset.from_tensor_slices((raw_x, raw_y))

        cardinality = int(tf.data.experimental.cardinality(raw_ds))

        raw_ds.cache()
        #raw_ds.shuffle(buffer_size=cardinality)
        raw_ds.batch(self.batch_size)
        raw_ds.prefetch(buffer_size=cardinality)
        
        test_split = int(cardinality * 0.2)

        self.test_ds = raw_ds.take(test_split)
        self.train_ds = raw_ds.skip(test_split)

        print('created dataset')

        return self.train_ds, self.test_ds