import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import math

a = '4X4_sorter/saved_models/MODEL_a.h5'
b = '4X4_sorter/saved_models/MODEL_b.h5'
c = '4X4_sorter/saved_models/MODEL_c.h5'
d = '4X4_sorter/saved_models/MODEL_d.h5'
e = '4X4_sorter/saved_models/MODEL_e.h5'

test_dataset_raw = []
target_test='4X4_sorter/test'
data_dir_test = target_test+'_data/'
log_dir_test = open(target_test+'_log.txt','r')
logs_test = log_dir_test.read()
logs_test = logs_test[:-1]
log_list_test = logs_test.split('\n')

labels = open('4X4_sorter/test_labels.txt','r')
test_labels = labels.readlines()
print(len(test_labels))

print('extracting pixel values for modeling ... this might take a while')
for i in tqdm (range (len(log_list_test)),"Loading..."):
    file=log_list_test[i]
    filepath = data_dir_test+file
    image = Image.open(filepath)
    WIDTH, HEIGHT = image.size

    data = list(image.getdata())
    data = [data[offset:offset+WIDTH] for offset in range(0,WIDTH*HEIGHT, WIDTH)]
    data = np.array(data)
    test_dataset_raw.append(data)

print(len(test_dataset_raw))

input_size=100
test_images = np.array(test_dataset_raw)
test_images = tf.constant(test_images,dtype=tf.float16, name='test images')
test_images = tf.reshape(test_images,[len(log_list_test),1,input_size,input_size,1])

class Quad_ensemble(object):
    def __init__(self):
        self.a = load_model(a)
        self.b = load_model(b)
        self.c = load_model(c)
        self.d = load_model(d)
        self.predictions = []
        self.a_pred = []
        self.b_pred = []
        self.c_pred = []
        self.d_pred = []
        self.favored_model = -1

    def __call__(self,data,labels):
        self.labels = labels
        shard=1
        i=0
        for img in tqdm (data, desc="Loading..."):
            img = (np.expand_dims(img,0))
            out_a = np.argmax(self.a.predict(img,verbose=0))
            self.a_pred.append(out_a)
            out_b = np.argmax(self.b.predict(img,verbose=0))
            self.b_pred.append(out_b)
            out_c = np.argmax(self.c.predict(img,verbose=0))
            self.c_pred.append(out_c)
            out_d = np.argmax(self.d.predict(img,verbose=0))

            votes_0 = []
            votes_1 = []
            votes_2 = []
            votes_3 = []
            a=1
            b=1
            c=1
            d=1
            biases = [a,b,c,d]
            if self.favored_model > -1:
                biases[self.favored_model] *= 3
                print(self.favored_model,' ',biases)

            if out_a == 0: votes_0.append(range(a))
            if out_b == 0: votes_0.append(range(b))
            if out_c == 0: votes_0.append(range(c))
            if out_d == 0: votes_0.append(range(d))

            if out_a == 1: votes_1.append(range(a))
            if out_b == 1: votes_1.append(range(b))
            if out_c == 1: votes_1.append(range(c))
            if out_d == 1: votes_1.append(range(d))

            if out_a == 2: votes_2.append(range(a))
            if out_b == 2: votes_2.append(range(b))
            if out_c == 2: votes_2.append(range(c))
            if out_d == 2: votes_2.append(range(d))

            if out_a == 3: votes_3.append(range(a))
            if out_b == 3: votes_3.append(range(b))
            if out_c == 3: votes_3.append(range(c))
            if out_d == 3: votes_3.append(range(d))


            votes = (len(votes_0),len(votes_1),len(votes_2),len(votes_3))
            #print(votes)
            elected_out = np.argmax(votes)
            self.predictions.append(elected_out)
            #print(labels[i],' ',elected_out,' | ',out_a,' ',out_b,' ',out_c,)
            i = i+1

        SCORE = 0
        SCORE_A = 0
        SCORE_B = 0
        SCORE_C = 0
        SCORE_D = 0
        i=0
        for prediction in self.predictions:
            if int(self.a_pred[i]) == int(labels[i*shard]):
                SCORE_A = SCORE_A + 1
            if int(self.b_pred[i]) == int(labels[i*shard]):
                SCORE_B = SCORE_B + 1
            if int(self.c_pred[i]) == int(labels[i*shard]):
                SCORE_C = SCORE_C + 1
            if int(self.d_pred[i]) == int(labels[i*shard]):
                SCORE_D = SCORE_D + 1
            if int(prediction) == int(labels[i*shard]): 
                SCORE = SCORE + 1
            print(SCORE,' ',prediction,' ',labels[i*shard])
            
            i=i+1
        self.predictions = [] 
        self.accuracy = shard*SCORE/(len(labels))
        self.a_accuracy = shard*SCORE_A/(len(labels))
        self.b_accuracy = shard*SCORE_B/(len(labels))
        self.c_accuracy = shard*SCORE_C/(len(labels))
        self.d_accuracy = shard*SCORE_D/(len(labels))

        print(self.accuracy, '  a ',self.a_accuracy,'  b ',self.b_accuracy,'  c ',self.c_accuracy,'  d ',self.d_accuracy)
        input('continue')
    def calculate_favored_model(self):
        accs = [self.a_accuracy,self.b_accuracy,self.c_accuracy,self.d_accuracy]
        self.favored_model = np.argmax(accs)
        print('favored model ' , self.favored_model)
        input('continue')

ensemble = Quad_ensemble()