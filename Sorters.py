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
from preprocessor import input_pipeline

class Sorter_Framework(object):
    def __init__(self,input_size,dimension='',class_num=0,name=''):

        self.input_size = input_size
        self.name = name
        self.class_num = class_num
        self.dimension = dimension
        self.prefix = dimension+'_sorter/'
        self.checkpoint_path = self.prefix+"model_training/"+self.name+".ckpt"

    def rename_model(self,name):
        self.name = name

    def load_data(self,folds=10):

        input = input_pipeline(self.input_size, self.class_num)
        self.train_ds, self.val_ds, self.test_ds = input.select_data()

        self.train_x = []
        self.train_y = []
        for x, y in self.train_ds:
            self.train_x.append(x)

        for x, y in self.train_ds:
            self.train_y.append(y)

        self.test_x = []
        self.test_y = []
        for x, y in self.train_ds:
            self.test_x.append(x)

        for x, y in self.train_ds:
            self.test_y.append(y)

        print(self.train_x)


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
            batch_size=32,
            validation_data=validation_data,
            callbacks=[cp_callback])  # Pass callback to training

            self.model.save(self.dimension+'_sorter/saved_models/MODEL_'+self.name+'.h5')

        train(self.train_x,self.train_y,validation_data=self.val_ds,persistance=persistance)

        test_loss, test_acc = self.model.evaluate(self.test_x, self.test_y, verbose=2)
        print('test accuracy: ', test_acc)
        return test_acc


test_sorter = Sorter_Framework(28,'10x10', 10)
test_sorter.load_data()
test_sorter.load_neural_model(CNN_10x10)
test_sorter.train_model(False)