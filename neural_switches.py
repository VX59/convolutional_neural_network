import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm

intake_path = '8X4_sorter/saved_models/MODEL_intake.h5'
a = '2X2_sorter/saved_models/MODEL_sorter_a.h5'
b = '2X2_sorter/saved_models/MODEL_sorter_b.h5'
c = '2X2_sorter/saved_models/MODEL_sorter_c.h5'
d = '2X2_sorter/saved_models/MODEL_sorter_d.h5'

test_dataset_raw = []
target_test='8X2_sorter/test'
data_dir_test = target_test+'_data/'
log_dir_test = open(target_test+'_log.txt','r')
logs_test = log_dir_test.read()
logs_test = logs_test[:-1]
log_list_test = logs_test.split('\n')

SWITCH_labels = open('8X2_sorter/SWITCH_labels.txt','r')
SWITCH_labelset = SWITCH_labels.readlines()

print('extracting pixel values for modeling ... this might take a while')
for i in tqdm (range (len(log_list_test)), desc="Loading..."):
    file=log_list_test[i]
    filepath = data_dir_test+file
    image = Image.open(filepath)
    WIDTH, HEIGHT = image.size

    data = list(image.getdata())
    data = [data[offset:offset+WIDTH] for offset in range(0,WIDTH*HEIGHT, WIDTH)]
    data = np.array(data)
    test_dataset_raw.append(data)

input_size=124
test_images = np.array(test_dataset_raw)
test_images = tf.constant(test_images,dtype=tf.float16, name='test images')
test_images = tf.reshape(test_images,[len(log_list_test),1,input_size,input_size,1])

class binary_switch(object):    # 4X2
    def __init__(self,feeder_path,alpha_path,beta_path,target):
        self.feeder = load_model(feeder_path)
        self.alpha = load_model(alpha_path)
        self.beta = load_model(beta_path)

        log_dir_test = open(target+'_log.txt','r')
        logs_test = log_dir_test.read()
        logs_test = logs_test[:-1]
        self.log_list_test = logs_test.split('\n')

    def __call__(self,test_data,test_labels_0,test_labels_1):
        self.test_data = test_data
        self.test_labels_0 = test_labels_0
        self.test_labels_1 = test_labels_1
        prediction_out = 0

        img = self.test_data[0]
        img = (np.expand_dims(img,0))
        feeder_pred_raw = self.feeder.predict(img, verbose=0)
        feeder_pred = np.argmax(feeder_pred_raw)

        if feeder_pred == 0:
            alpha_pred_raw = self.alpha.predict(img,verbose=0)
            alpha_pred = np.argmax(alpha_pred_raw)
            if alpha_pred == 0: prediction_out = 0
            if alpha_pred == 1: prediction_out = 1

        if feeder_pred == 1:     
            beta_pred_raw = self.beta.predict(img, verbose=0)
            beta_pred = np.argmax(beta_pred_raw)
            if beta_pred == 0: prediction_out = 2
            if beta_pred == 1: prediction_out = 3
        return prediction_out

class binary_step_switch(object):
    def __init__(self,alpha_path,beta_path,charlie_path,target):
        self.alpha = load_model(alpha_path)
        self.beta = load_model(beta_path)
        self.charlie = load_model(charlie_path)

        log_dir_test = open(target+'_log.txt','r')
        logs_test = log_dir_test.read()
        logs_test = logs_test[:-1]
        self.log_list_test = logs_test.split('\n')

    def __call__(self,test_data,test_labels_0,test_labels_1):
        self.test_data = test_data
        self.test_labels_0 = test_labels_0
        self.test_labels_1 = test_labels_1
        prediction_out = 0
        img = self.test_data[0]
        img = (np.expand_dims(img,0))
        raw = self.alpha.predict(img, verbose=0)
        alpha_pred = np.argmax(raw)

        if alpha_pred == 0: prediction_out = 0
        if alpha_pred == 1:     
            raw = self.beta.predict(img, verbose=0)
            beta_pred = np.argmax(raw)
            if beta_pred == 0: prediction_out = 1
            if beta_pred == 1:
                raw = self.charlie.predict(img, verbose=0)
                charlie_pred = np.argmax(raw)
                if charlie_pred == 0: prediction_out = 2
                if charlie_pred == 1: prediction_out = 2


        return prediction_out

class quad_switch(object):  # 8X2
    def __init__(self,intake_path,a,b,c,d,target=''):
        log_dir_test = open(target+'_log.txt','r')
        logs_test = log_dir_test.read()
        logs_test = logs_test[:-1]
        self.log_list_test = logs_test.split('\n')

        self.INTAKE_SCORE = 0
        self.A_SCORE = 0
        self.B_SCORE = 0
        self.C_SCORE = 0
        self.D_SCORE = 0
        self.intake = load_model(intake_path)
        self.a = load_model(a)
        self.b = load_model(b)
        self.c = load_model(c)
        self.d = load_model(d)

    def __call__(self,test_data,test_labels_0,SWITCH_labelset):
        self.test_data = test_data
        self.test_labels_0 = test_labels_0
        self.SWITCH_labelset = SWITCH_labelset
        for j in tqdm (range (len(self.log_list_test)), desc="Loading..."):
            if j % 5 == 0:
                prediction_out = 0

                img = self.test_data[j]
                img = (np.expand_dims(img,0))
                intake_pred_raw = self.intake.predict(img, verbose=0)
                intake_pred = np.argmax(intake_pred_raw)
                if intake_pred == int(self.test_labels_0[j]): self.INTAKE_SCORE = self.INTAKE_SCORE + 1
                
                if intake_pred == 0:
                    raw = self.a.predict(img,verbose=0)
                    a_pred = np.argmax(raw)
                    if a_pred == 0: prediction_out = 0
                    if a_pred == 1: prediction_out = 1
                    if int(prediction_out) == int(SWITCH_labelset[j]): self.A_SCORE = self.A_SCORE +1
                
                if intake_pred == 1:
                    raw = self.b.predict(img,verbose=0)
                    b_pred = np.argmax(raw)
                    if b_pred == 0: prediction_out = 2
                    if b_pred == 1: prediction_out = 3
                    if int(prediction_out) == int(SWITCH_labelset[j]): self.B_SCORE = self.B_SCORE +1
                
                if intake_pred == 2:
                    raw = self.c.predict(img,verbose=0)
                    c_pred = np.argmax(raw)
                    if c_pred == 0: prediction_out = 4
                    if c_pred == 1: prediction_out = 5
                    if int(prediction_out) == int(SWITCH_labelset[j]): self.C_SCORE = self.C_SCORE +1

                if intake_pred == 3:
                    raw = self.d.predict(img,verbose=0)
                    d_pred = np.argmax(raw)
                    if d_pred == 0: prediction_out = 6
                    if d_pred == 1: prediction_out = 7
                    if int(prediction_out) == int(SWITCH_labelset[j]): self.D_SCORE = self.D_SCORE +1


            if j % int(len(self.log_list_test)/4) == 0 and j > 0:
                print('INTAKE SCORE : ',self.INTAKE_SCORE/j*5)
                print('A SCORE  : ',self.A_SCORE/(j)*20)
                print('B SCORE   : ',self.B_SCORE/(j)*20)
                print('average      : ',((self.INTAKE_SCORE/j*5)+(self.A_SCORE/(j)*20)+(self.B_SCORE/(j)*20)+(self.C_SCORE/(j)*20)+(self.D_SCORE/(j)*20))/5.0)
            if j + 1 == len(self.log_list_test):
                print('INTAKE SCORE : ',self.INTAKE_SCORE/j*5)
                print('A SCORE  : ',self.A_SCORE/(j)*20)
                print('B SCORE   : ',self.B_SCORE/(j)*20)
                print('average      : ',((self.INTAKE_SCORE/j*5)+(self.A_SCORE/(j)*20)+(self.B_SCORE/(j)*20)+(self.C_SCORE/(j)*20)+(self.D_SCORE/(j)*20))/5.0)


label = open('8X4_sorter/test_labels.txt','r')  # 4 classes
labelset_0 = label.readlines()
label = open('switch_labels.txt','r')           # 8 classes
SWITCH_labelset = label.readlines()

switch = quad_switch(intake_path,a,b,c,d,target_test)
switch(test_images,labelset_0,SWITCH_labelset)