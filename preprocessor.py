import os
from PIL import Image, ImageFilter,ImageEnhance, ImageOps
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tempfile
import random
import platform
import math

# our dataset contains 40 thousand images 800 * 50
# labels are derived from the folder names and correspond to the part
# this program compresses and labels training images

from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN)

class preprocessor(object):
    def __init__(self, SCALE, classes,kfolds,group_start):
        self.kfolds = kfolds
        self.SCALE = SCALE
        self.classes = classes
        self.batch_size = 32
        self.filter = ''
        self.invert = False
        self.group_start = group_start
        os_type = platform.system()
        if os_type == "Windows": path = "/Users/deros/Downloads/dataset/"
        else: path = "/home/rsenic/dataset/"
        self.working_dir = path
        self.dir_list = os.listdir(self.working_dir)  
        self.dir_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    def encode_pixels(self, image):
        WIDTH, HEIGHT = image.size
        data = list(image.getdata())
        data = [data[offset:offset+WIDTH] for offset in range(0,WIDTH*HEIGHT, WIDTH)]
        data = np.array(data)
        return data

    def img_filter(self,image):
        if(self.filter == "BLUR"): image = image.filter(BLUR)
        if(self.filter == "CONTOUR"): image = image.filter(CONTOUR)
        if(self.filter == "DETAIL"): image = image.filter(DETAIL)
        if(self.filter == "EDGE_ENHANCE"): image = image.filter(EDGE_ENHANCE)
        if(self.filter == "EDGE_ENHANCE_MORE"): image = image.filter(EDGE_ENHANCE_MORE)
        if(self.filter == "EMBOSS"): image = image.filter(EMBOSS)
        if(self.filter == "FIND_EDGES"): image = image.filter(FIND_EDGES)
        if(self.filter == "SMOOTH"): image = image.filter(SMOOTH)
        if(self.filter == "SMOOTH_MORE"): image = image.filter(SMOOTH_MORE)
        if(self.filter == "SHARPEN"): image = image.filter(SHARPEN)
        return image
    def select_data(self, group_size = 800,
                          kfold=False,
                          get_test=False):
        raw_x = []
        raw_y = []

        for i in range(self.group_start*group_size, 
                       (self.group_start+self.classes)*group_size,
                       group_size):

            for j in tqdm (range(int(group_size)), desc=f"subset {int(i/800)+1}/{self.classes} "):    
                a = i+j # specific index
                if j < group_size:
                    file = self.dir_list[a]
                    filepath = self.working_dir+'/'+file

                    image = Image.open(filepath).convert('L')
                    if(self.invert): image = ImageOps.invert(image)
                    image = self.img_filter(image)
                    image.thumbnail((self.SCALE,self.SCALE))

                    image = self.encode_pixels(image)

                    raw_x.append(image)
                    raw_y.append(int(i/group_size) - self.group_start)

        x_count = [[] for _ in range(self.classes)]
        y_count = [[] for _ in range(self.classes)]

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

        # now we have separated our x y into respective classses, we need to partition into train and test
        split = int(y_count.shape[1] * 0.2)

        train_x_sep = np.array([x[split:] for x in x_count])
        test_x_sep = np.array([x[:split] for x in x_count])

        train_y_sep = np.array([y[split:] for y in y_count])
        test_y_sep = np.array([y[:split] for y in y_count])

        test_x = np.reshape(test_x_sep, (test_x_sep.shape[0] * test_x_sep.shape[1], self.SCALE, self.SCALE))
        test_y = np.reshape(test_y_sep, (test_y_sep.shape[0] * test_y_sep.shape[1]))
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        print('created test dataset')

        self.train_ds_path = os.path.join(tempfile.gettempdir(),"tf_train_ds")
        self.test_ds_path = os.path.join(tempfile.gettempdir(),"tf_test_ds")

        cardinality = int(tf.data.experimental.cardinality(test_ds))

        test_ds = test_ds.cache()
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.shuffle(buffer_size=cardinality)

        if not os.path.isdir("tf_test_ds"):
            os.mkdir("tf_test_ds")
        tf.data.experimental.save(test_ds, self.test_ds_path)

        if not kfold:
            train_x = np.reshape(train_x_sep, (train_x_sep.shape[0] * train_x_sep.shape[1], self.SCALE, self.SCALE))
            train_y = np.reshape(train_y_sep, (train_y_sep.shape[0] * train_y_sep.shape[1]))
            train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
            cardinality = int(tf.data.experimental.cardinality(train_ds))
            train_ds = train_ds.cache()
            train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            train_ds = train_ds.shuffle(buffer_size=cardinality)
            tf.data.experimental.save(train_ds, self.train_ds_path)

            print("len train,", np.array(train_x).shape,
                                np.array(train_y).shape,
                  "len test, ", np.array(test_x).shape,
                                np.array(test_y).shape)
            return train_ds, test_ds
        else: 
            folded_training_datasets = self.chop_and_fold(train_x_sep, train_y_sep)
            return folded_training_datasets, test_ds

   # this is intended for k fold implementation
   # we need to partition the training data into k random size subsets, a separate classifier will be trained on each subseet
    def chop_and_fold(self, train_x_sep, train_y_sep):

        # generate random partition ratios
        alpha = 1.0
        partition_ratios = []
        group_size = self.kfolds/2
        for k in range(int(group_size)):
            end = alpha / group_size - 0.05
            theta = random.random()
            while not (0.05 < theta and theta < end):
                theta = random.random()
            iota = alpha / group_size
            iota -= theta
            partition_ratios.append(theta)
            partition_ratios.append(iota)

        print(partition_ratios)
        print(np.sum(partition_ratios)) # should be 1.0

        # with the partitions we can split up the datasets into respective folds

        folded_data = []    # shape should be (k folds, c classes, x images (ragged) , w, h)
        start = 0
        total = 0
        for k in range(self.kfolds):
            end = start + math.ceil(partition_ratios[k] * len(train_x_sep[1]))
            print("start: ", start, "   end: ", end)
            total += len(train_x_sep[1][start:end])

            fold_data_acm = np.array([])
            fold_label_acm = np.array([])
            # create class folds
            for c in range(self.classes):
                fold_x = np.array(train_x_sep[c][start:end])
                fold_y = np.array(train_y_sep[c][start:end])
                if c == 0: 
                    fold_data_acm = fold_x
                    fold_label_acm = fold_y
                else: 
                    fold_data_acm = np.concatenate((fold_data_acm, fold_x))
                    fold_label_acm = np.concatenate((fold_label_acm, fold_y))
                print(np.array(fold_data_acm).shape, "  ", np.array(fold_label_acm).shape)

            fold = (fold_data_acm, fold_label_acm)
            folded_data.append(fold)
            start = end

        folded_tensors = []
        for k in range(self.kfolds):
            fold = folded_data[k]
            fold_ds = tf.data.Dataset.from_tensor_slices(fold)
            fold_ds = fold_ds.cache()
            fold_ds = fold_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            cardinality = int(tf.data.experimental.cardinality(fold_ds))
            fold_ds = fold_ds.shuffle(buffer_size=cardinality)
            folded_tensors.append(fold_ds)
            path = "tf_fold_ds"
            if not os.path.isdir(path):
                os.mkdir(path)
            tf.data.experimental.save(fold_ds, path)

        return folded_tensors
          
    def create_sample_from_tensor(self,preview=False):
        sampleset = list(tf.data.experimental.load( os.path.join(tempfile.gettempdir(),"tf_test_ds")).as_numpy_iterator())
        sample = sampleset[random.randint(0,len(sampleset)-1)]
        data,label = sample
        arr_data = np.array(data, dtype=np.uint8)
        print(arr_data)
        print(arr_data.shape)
        image = Image.fromarray(arr_data)
        if(self.invert): image = ImageOps.invert(image.convert('L'))
        image.thumbnail((self.SCALE,self.SCALE))
        image = self.img_filter(image)
        image.resize((self.SCALE, self.SCALE))
        if preview: return image
        else:
            sample = tf.constant(data, dtype=tf.float32)
            sample = tf.reshape(sample, [1, self.SCALE, self.SCALE, 1])
            return sample, label

    def create_sample_from_path(self,preview=False):
        index = random.randint(0,(self.classes * 800))
        print(index)
        label = int(index / 800)
        file = self.dir_list[index]
        filepath = self.working_dir+'/'+file

        image = Image.open(filepath).convert('L')
        image.thumbnail((self.SCALE,self.SCALE))
        if(self.invert): image = ImageOps.invert(image)
        image = self.img_filter(image)
        data = self.encode_pixels(image)
        if(preview): return image
        else: 
            sample = tf.constant(data, dtype=tf.float32)
            sample = tf.reshape(sample, [1, self.SCALE, self.SCALE, 1])
            return sample,label