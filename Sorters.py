import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
import os.path as path
import random
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from tqdm import tqdm
from six.moves.urllib.request import urlopen
from six import BytesIO
import random
import shutil
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import time

from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN)

from neural_models import *

class Sorter_Framework(object):
    def __init__(self,input_size,dimension='',class_num=0,name='',working_dir='dataset/'):
        self.working_dir_list = os.listdir(working_dir)
        self.working_dir_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.input_size = input_size
        self.name = name
        self.class_num = class_num
        self.dimension = dimension
        self.prefix = dimension+'_sorter/'
        if path.isdir(self.prefix) == False: os.mkdir(self.prefix)
        self.target_train = self.prefix+'train'
        self.target_train_synth = self.prefix+'train_synth'
        self.data_dir_train = self.target_train+'_data/'
        self.data_synth_dir_train = self.target_train_synth+'_data'

        self.target_test=self.prefix+'test'
        self.data_dir_test = self.target_test+'_data/'

        self.test_dataset_raw = []

        self.checkpoint_path = self.prefix+"model_training/"+self.name+".ckpt"
    def rename_model(self,name):
        self.name = name

    def load_data(self,folds=10):

        self.raw_x = []
        self.raw_y = []

        fold_partitions = []


        for fold in range(folds):
            rand = random.randint(int(len(self.raw_x/folds),self.raw_x))
            fold_partitions 


        self.train_dataset = []
        self.train_labelset = []
        log = self.log_list_train
        # encode training data
        self.encode_pixels(log, self.data_dir_train, self.raw_x, self.raw_y, fold=True)

        # wrap the folds in another array
        for i in range(self.folds):
            self.train_dataset.append(self.raw_train_data[i])
            self.train_labelset.append(self.raw_train_labels[i])

        for i in range(self.folds):
            print(self.train_dataset[i])
            self.train_dataset[i] = np.array(self.train_dataset[i])
            self.train_labelset[i] = np.array(self.train_labelset[i])
            self.train_dataset[i] = tf.constant(self.train_dataset[i],dtype=tf.float16, name='train images')
            self.train_labelset[i] = tf.constant(self.train_labelset[i],dtype=tf.float16, name='train labels')
            self.train_dataset[i] = tf.reshape(self.train_dataset[i],[self.training_partition_size,1,self.input_size,self.input_size,1])
        # encode test data

        log = self.log_list_test
        self.encode_pixels(log,self.data_dir_test,self.raw_test_data,self.raw_test_labels)

        test_images = np.array(self.raw_test_data)
        test_labels = np.array(self.raw_test_labels)
        self.test_images = tf.constant(test_images,dtype=tf.float16, name='test images')

        print(len(self.log_list_test),'     ',self.test_images.shape)

        self.test_labels = tf.constant(test_labels,dtype=tf.int8, name='test labels')
        self.test_images = tf.reshape(test_images,[len(self.log_list_test),1,self.input_size,self.input_size,1])

    def preprocess_data(self,mode,working_dir='dataset/',alpha_split=2,beta_split=800,offset=0,group_size=800,start=0,groups=0,filters=EMBOSS):
        if path.isdir(mode+'_data/'):   shutil.rmtree(mode+'_data/')
        if path.isfile(mode+'_log.txt'): os.remove(mode+'_log.txt')
        if path.isfile(mode+'_labels.txt'): os.remove(mode+'_labels.txt')
        os.mkdir(mode+'_data/')
        log =  open(mode+'_log.txt','w')
        dir_list = os.listdir(working_dir)  
        dir_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        print(len(dir_list))
        for i in range(start*group_size,(start*group_size)+(groups*group_size),group_size):
            for j in tqdm (range (int(group_size)), desc="Loading..."):    
                a = i+j # specific index
                j = j+offset
                if j % alpha_split == 0 and j < beta_split:
                    file = dir_list[a]
                    filepath = working_dir+'/'+file
                    image = Image.open(filepath).convert('L')
                    WIDTH, HEIGHT = image.size
                    image.thumbnail((self.input_size,self.input_size))
                    image.crop(((WIDTH- self.input_size) // 2,
                                 (HEIGHT - self.input_size) // 2,
                                 (WIDTH + self.input_size) // 2,
                                 (HEIGHT + self.input_size) // 2))
                    image = image.filter(filters)
                    image.save(mode+'_data/'+str(a)+'.png')
                    log.write(str(a)+'.png'+'\n')
        print('created '+mode+' set')

        self.samples = groups*(group_size/alpha_split)

    def make_labels(self,mode,groups=0,offset=0,feeder_mode=False):
        if feeder_mode == False:
            file = open(mode+'_labels.txt','w')

            for i in range(groups):
                for j in range(int((self.samples/groups)-offset)):
                    if j + 1 != groups*int((self.samples/groups)-offset): file.write(str(i)+'\n')
                    else: file.write(str(i))
            file.close()

        if feeder_mode == True:
            file = open(mode+'_labels.txt','w')
            print('samples : ',self.samples)
            for i in range(self.class_num):
                for j in range(int((self.samples/self.class_num)-offset*(groups/self.class_num))):
                    if j + 1 != self.class_num*int((self.samples/self.class_num)-offset*(groups/self.class_num)): file.write(str(i)+'\n')
                    else: file.write(str(i))
            file.close()

    def parse_data_logs(self,target):
        log_dir_synth_train = open(target+'_log.txt','r')
        log_list_synth_train = log_dir_synth_train.read()
        log_list_synth_train = log_list_synth_train[:-1]
        label = open(target+'_labels.txt','r')
        return log_list_synth_train.split('\n'), label.readlines()

    def train_logs(self):
        self.log_list_train, self.labelset_train = self.parse_data_logs(self.target_train)
        self.training_partition_size = int(len(self.log_list_train)/8)

    def train_synth_logs(self):
        self.log_list_synth_train,self.labelset_synth_train = self.parse_data_logs(self.target_train_synth)
        self.training_synth_partition_size = int(len(self.log_list_synth_train)/8)

    def test_logs(self):
        self.log_list_test, self.labelset_test = self.parse_data_logs(self.target_test)

    def load_neural_model(self,model,lr=1e-3):
        self.model = model

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    def load_weights(self):
        load_model(self.prefix+'/saved_models/MODEL_'+self.name+'.h5')
        self.model.load_weights(self.checkpoint_path)
    def train_model(self,persistance,kfold=False,k=0,):
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        def train(train_data,train_labels,validation_data,persistance=True):
            if persistance: self.model.load_weights(self.checkpoint_path)
            self.model.fit(train_data, 
            train_labels,  
            epochs=12,
            batch_size=16,
            validation_data=validation_data,
            callbacks=[cp_callback])  # Pass callback to training

            self.model.save(self.dimension+'_sorter/saved_models/MODEL_'+self.name+'.h5')

        if kfold == True:
            print('validation segment = ',k)
            validation_data = (self.train_dataset[k],self.train_labelset[k])
            k_fold_data = []
            k_fold_labels = []
            for j in range(len(self.train_dataset)):
                if j != k:
                    print(j)
                    k_fold_data.append(self.train_dataset[j])
                    k_fold_labels.append(self.train_labelset[j])
            for i in range(len(k_fold_data)):
                if i == 0: train(k_fold_data[i],k_fold_labels[i],validation_data=validation_data,persistance=persistance)
                else: train(k_fold_data[i],k_fold_labels[i],validation_data=validation_data)

            print("\n\n"+"completed training cycle"+"\n\n")
            test_loss, test_acc = self.model.evaluate(self.train_dataset[k], self.train_labelset, verbose=2)
            return test_acc

        print('test accuracy: ', test_acc)
    def synthesize_data(self,mode,step):
        if path.isdir(mode+'_data/'): shutil.rmtree(mode+'_data/')
        if path.isfile(mode+'_log.txt'): os.remove(mode+'_log.txt')

        os.mkdir(mode+'_data/')

        log =  open(mode+'_log.txt','a')

        for j in tqdm (range (len(self.log_list_train)), desc="Loading..."):
            tr = random.randint(-10,10)
            r = random.randint(-45,45)    
            a = j*self.folds
            file = self.log_list_train[j]
            filepath = self.data_dir_train+'/'+file
            image = Image.open(filepath)
            for p in range(4):
                for i in range(step*j,step*(j+1)):
                    image = image.rotate(r,translate=(tr,tr))
                    image.save(mode+'_data/'+str(a+i)+'.png')
                    log.write(str(a+i)+'.png'+'\n')
        
        print('created '+mode+' set')
    def make_prediction(self,test_data,test_labels):
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        return test_loss, test_acc