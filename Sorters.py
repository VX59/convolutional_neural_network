import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
import random
import shutil
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt

from neural_models import *
from preprocessor import input_pipeline

class Sorter_Framework(object):
    def __init__(self,input_size,class_num=0,name=''):

        self.input_size = input_size
        self.name = name
        self.class_num = class_num
        self.dimension = str(self.input_size) + 'x' + str(self.class_num)
        self.prefix = self.dimension + '_sorter/'
        self.checkpoint_path = self.prefix + "model_training/" + self.name + ".ckpt"

    def rename_model(self,name):
        self.name = name

    def split_data(self, dataset):
        x_data = []
        y_data = []
        for x, y in dataset:
            x_data.append(x)
            y_data.append(y)

        x_data = tf.constant(np.array(x_data), dtype=tf.float32)
        y_data = tf.constant(np.array(y_data), dtype=tf.int16)

        return (x_data, y_data)

    def load_data(self,folds=10):

        self.input = input_pipeline(self.input_size, self.class_num)

        self.train_ds, self.test_ds = self.input.select_data()

        self.train_x, self.train_y = self.split_data(self.train_ds)

        self.test_x, self.test_y = self.split_data(self.test_ds)

    def load_neural_model(self):

        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal",
                                           input_shape=(self.input_size,
                                           self.input_size, 1)),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1)
            ])

        self.CNN = tf.keras.Sequential(
        [   
            augmentation,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 4 ,activation='relu',padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, 4 ,activation='relu',padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, 4 ,activation='relu',padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(self.class_num, activation='softmax'),
        ])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            2e-3,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True)

        optimizer = tf.keras.optimizers.Adam(lr_schedule)

        self.CNN.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

    def load(self):
        load_model(self.prefix+'/saved_models/MODEL_'+self.name+'.h5')
        self.CNN.load_weights(self.checkpoint_path)

    def train_model(self,persistance,kfold=False,k=0):
        
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=12)

        def train(train_x, train_y, persistance=True):

            if persistance: 
                print("loading existing model...")
                self.load()

            history = self.CNN.fit(
            train_x,
            train_y,
            epochs=16,
            batch_size= self.input.batch_size,
            validation_split=0.2,
            callbacks=[cp_callback, early_stopping])  # Pass callback to training

            self.CNN.save(self.dimension+'_sorter/saved_models/MODEL_'+self.name+'.h5')

            return history 

        self.history = train(self.train_x, self.train_y, persistance=persistance)

        test_loss, test_acc = self.CNN.evaluate(self.test_x, self.test_y, verbose=2)
        print('test accuracy: ', test_acc, '\n', 'test loss', test_loss)
        return test_loss, test_acc

    def plot_training(self):
        print(self.history.history.keys())
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(self.history.history['loss'], label="loss")
        ax1.plot(self.history.history['accuracy'], label="acc")
        ax1.legend(['loss', 'accuracy'], loc='lower left')
        ax2.plot(self.history.history['val_loss'], label="val_loss")
        ax2.plot(self.history.history['val_accuracy'], label="val_acc")
        ax2.legend(['val loss', 'val accuracy'], loc='lower left')
        plt.show()

    def make_predictions(self):

        # get 20 random samples from the test dataset
        fig, subplots = plt.subplots(10)
        fig = plt.figure(figsize=(10,5))
        
        for x in range(10):
            sample = self.input.create_sample()
            print(sample)
            prediction = self.CNN.predict(sample)
            print(np.argmax(prediction))
            for i in subplots:
                i.bar(list(range(self.input.classes)), np.reshape(prediction), color="blue", width=0.4)
                i.imshow(sample, interpolation='none')

                i.xlabel("class")
                i.ylabel("probablility")
            
            plt.show()

test_sorter = Sorter_Framework(64, 5)
test_sorter.load_data()
test_sorter.load_neural_model()
test_sorter.train_model(False)
test_sorter.plot_training()

