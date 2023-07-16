import os
from PIL import Image, ImageFilter,ImageEnhance, ImageOps
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tempfile
import random
import platform
import math
from itertools import chain
import IPython.display as display

from PIL import ImageFilter
class preprocessor(object):
    def __init__(self, SCALE, classes, folds):
        os_type = platform.system()
        self.SCALE = SCALE
        self.folds = folds
        if os_type == "Windows": path = "/Users/deros/Downloads/dataset/"
        else: path = "/home/rsenic/Downloads/archive(5)/Cropped Images/"
        self.working_dir = path
        self.classes = os.listdir(self.working_dir)
        if classes != 0: self.classes = self.classes[:classes]
        self.class_num = len(self.classes)

    def encode_pixels(self, image):
        image = image.resize((self.SCALE,self.SCALE))
        data = list(image.getdata())
        data = [data[offset:offset+self.SCALE] for offset in range(0,self.SCALE*self.SCALE, self.SCALE)]
        return np.array(data, dtype='int16')

    def select_data(self, kfold=False,):
        raw_dataset = []
        print(list(self.classes))
        for key, name in enumerate(self.classes,self.class_num):
            path = self.working_dir + name
            files = os.listdir(path)
            image_data = []
            for file in files:
                filepath = path + '/' + file
                image = Image.open(filepath).convert('L')
                image = ImageOps.invert(image)
                image.thumbnail((self.SCALE,self.SCALE))
                image = self.encode_pixels(image)
                image_data.append((image,key))
            raw_dataset.append(image_data)

        # now we have separated our x y into respective classses, we need to partition into train and test
        train_sep = []
        test_sep = []
        for i in raw_dataset:
            split = int(len(i) * 0.2)
            [train_sep.append(x) for x in i[split:]]
            [test_sep.append(x) for x in i[:split]]

        train_x = np.array([data for data,key in train_sep])
        train_y = np.array([key for data,key in train_sep])
        test_x = np.array([data for data,key in test_sep])
        test_y = np.array([key for data,key in test_sep])

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
            train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
            print("created train dataset")
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
            folded_training_datasets = self.chop_and_fold(train_x,train_y)
            return folded_training_datasets, test_ds

   # this is intended for k fold implementation
   # we need to partition the training data into k random size subsets, a separate classifier will be trained on each subset
    def chop_and_fold(self, train_x_sep, train_y_sep):
        folds=self.folds
        # generate random partition ratios
        alpha = 1.0
        partition_ratios = []
        group_size = folds/2
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
        for k in range(folds):
            end = start + math.ceil(partition_ratios[k] * len(train_x_sep[1]))
            print("start: ", start, "   end: ", end)
            total += len(train_x_sep[1][start:end])

            fold_data_acm = np.array([])
            fold_label_acm = np.array([])
            # create class folds
            for c in range(self.class_num):
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
        for k in range(folds):
            fold = folded_data[k]
            print(type(fold))
            print(np.array(fold[0]))
            print(np.array(fold[1]))

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
        print(total*folds)

        return folded_tensors


