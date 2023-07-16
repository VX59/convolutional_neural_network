import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
import matplotlib.pyplot as plt
from preprocessor import *
import shutil

class Sorter_Framework(object):
    def __init__(self,input_size,class_num,ppr:preprocessor,name=''):
        self.ppr:preprocessor = ppr
        self.input_size = input_size
        self.name = name
        self.epochs = 12
        self.batch_size = 32
        self.kfold = 4
        self.lr_decay = 0.95
        self.learning_rate = 1e-3
        self.validation_split = 0.35
        self.estop = 0.5
        self.class_num = class_num
        self.class_names = range(self.class_num)
        self.dimension = str(self.input_size) + 'x' + str(self.class_num)
        self.prefix = 'models/'+self.dimension + '_sorter/'
        self.checkpoint_path = self.prefix + "model_training/" + self.name + ".ckpt"
        if not os.path.isdir(self.prefix): os.mkdir(self.prefix)
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

    def load_data(self,kfold=True):
        if(self.ppr==None):
            print("please configure a preprocessor")
            return -1
        else: print(self.ppr)
        if kfold:
            if os.path.isdir("tf_fold_ds"):
                fold_datasets = np.array([])
                test_ds_path = os.path.join(tempfile.gettempdir(),"tf_test_ds")
                test_ds = tf.data.experimental.load(test_ds_path)
                list_dir = os.listdir("tf_fold_ds")
                for dspath in list_dir:
                    dataset = tf.data.experimental.load("tf_fold_ds/")
                    np.append(fold_datasets, dataset)
            else:
                fold_datasets, test_ds = self.ppr.select_data(kfold=True)
                self.fold_x = []
                self.fold_y = []
            
            for fold in fold_datasets:
                fold_x, fold_y = self.split_data(fold)
                self.fold_x.append(fold_x)
                self.fold_y.append(fold_y)
            self.test_x, self.test_y = self.split_data(test_ds)
        else:
            train_ds, test_ds = self.ppr.select_data()
            self.train_x, self.train_y = self.split_data(train_ds)
            self.test_x, self.test_y = self.split_data(test_ds)

    def load_neural_model(self, shake=0.1):

        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal",
                                           input_shape=(self.input_size, self.input_size, 1)),
                tf.keras.layers.RandomRotation(shake),
                tf.keras.layers.RandomZoom(shake)
            ])

        CNN = tf.keras.Sequential(
        [   
            augmentation,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(96, 6 ,activation='relu',padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, 6 ,activation='relu',padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(64, 6 ,activation='relu',padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(self.class_num),
        ])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate,
            decay_steps=1000,
            decay_rate=self.lr_decay,
            staircase=True)

        optimizer = tf.keras.optimizers.Adam(lr_schedule)

        CNN.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

        return CNN

    def load_weights(self):
        self.CNN = load_model(self.prefix+'/saved_models/MODEL_'+self.name+'.h5')
        self.CNN.load_weights(self.checkpoint_path)

    def load_ensemble(self):
        self.models = []
        for name in os.listdir(self.prefix+'saved_models/'):
            print(name)
            model = load_model(self.prefix+'saved_models/'+name)
            model.load_weights(self.checkpoint_path)
            self.models.append(model)

    def train_model(self, persistance,kfold=True):
        
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=int(self.epochs * self.estop))

        def train(model, train_x, train_y, i=0, persistance=persistance):
            if persistance: 
                print("loading existing model...")
                self.load() # fix

            history = model.fit(
            train_x,
            train_y,
            epochs=self.epochs,
            batch_size= self.batch_size,
            validation_split=self.validation_split,
            callbacks=[cp_callback, early_stopping])  # Pass callback to training

            model.save(self.prefix+f'saved_models/MODEL_{i}'+self.name+'.h5')

            return history 

        if kfold:
            self.models = []
            self.history_ensemble = []
            for k in range(len(self.fold_x)):
                print("fold: ", k)
                CNN = self.load_neural_model()
                for j in range(len(self.fold_x)):
                    if j == k:
                        history = train(CNN, self.fold_x[j], self.fold_y[j], i=k, persistance=persistance)
                        self.history_ensemble.append(history)
                self.models.append(CNN)
                test_loss, test_acc = CNN.evaluate(self.test_x, self.test_y, verbose=2)
                print('test accuracy: ', test_acc, '\n', 'test loss', test_loss)
                self.plot_training(self.history_ensemble[k], f"model_{k}")
                
        else:
            self.CNN = self.load_neural_model()
            self.history = train(self.CNN, self.train_x, self.train_y, persistance=persistance)
            test_loss, test_acc = self.CNN.evaluate(self.test_x, self.test_y, verbose=2)
            print('test accuracy: ', test_acc, '\n', 'test loss', test_loss)
            return test_loss, test_acc

    def plot_training(self, history_obj, name):
        plot_path = self.prefix+'training_plots/'
        if not os.path.isdir(plot_path): os.mkdir(plot_path)

        print(history_obj.history.keys())
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(history_obj.history['loss'], label="loss")
        ax1.plot(history_obj.history['accuracy'], label="acc")
        ax1.legend(['loss', 'accuracy'], loc='lower left')
        ax2.plot(history_obj.history['val_loss'], label="val_loss")
        ax2.plot(history_obj.history['val_accuracy'], label="val_acc")
        ax2.legend(['val loss', 'val accuracy'], loc='lower left')
        plt.savefig(plot_path+name+".png")

    def make_predictions_from_ensemble(self, samples):
        fig, subplot = plt.subplots(samples)
        for i in subplot:
            sample, label = self.ppr.create_sample_from_tensor()
            print('label: ', label)
            m = 0
            predictions_acm = []
            for model in self.models:
                prediction = model.predict(sample)
                if m == 0: predictions_acm = prediction
                else: predictions_acm = np.add(predictions_acm, prediction)
                m += 1

            
            prediction_avg = np.divide(predictions_acm, len(self.models))
            print(prediction_avg)
            print(prediction_avg.shape)
            predicted_label = np.argmax(prediction_avg)
            print(predicted_label)

            plt.grid(False)
            plt.xticks(range(self.class_num))
            plot = i.bar(self.class_names, prediction[0], color="#777777")
            plot[predicted_label].set_color('red')
            plot[label].set_color('blue')

        plt.savefig(self.prefix+'predictions.png')
        plt.show()
        return fig

    def make_predictions(self):

        fig, subplot = plt.subplots(10)
        for i in subplot:
            sample, label = self.ppr.create_sample_from_tensor()
            print('label: ', label)
            prediction = self.CNN.predict(sample)
            print(prediction)
            print(prediction.shape)
            
            predicted_label = np.argmax(prediction)
            print("prediction: ", predicted_label)
            if predicted_label == label:
                color = 'blue'
            else:
                color = 'red'

            plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                         prediction[0][predicted_label],
                                         self.class_names[label]),
                                         color=color)
            plt.grid(False)
            plt.xticks(range(self.class_num))
            plot = i.bar(self.class_names, prediction[0], color="#777777")
            plot[predicted_label].set_color('red')
            plot[label].set_color('blue')

def make_sorter(test_sorter,persistance=False):
    if os.path.isdir("tf_test_ds"): os.rmdir("tf_test_ds")
    if os.path.isdir("tf_train_ds"): os.rmdir("tf_train_ds")
    if os.path.isdir("tf_fold_ds"): shutil.rmtree("tf_fold_ds")
    test_sorter.load_data()
    test_sorter.train_model(persistance)

def get_predictions(test_sorter,samples):
    test_sorter.load_data()
    test_sorter.load_ensemble()
    fig = test_sorter.make_predictions_from_ensemble(samples)
    return fig