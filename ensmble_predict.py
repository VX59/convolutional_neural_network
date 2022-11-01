import numpy as np
import random as rand
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN)
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from keras.models import load_model

t0 = load_model('32X16_sorter/saved_models/MODEL_t0.h5')
t1 = load_model('32X16_sorter/saved_models/MODEL_t1.h5')
t2 = load_model('32X16_sorter/saved_models/MODEL_t2.h5')
t3 = load_model('32X16_sorter/saved_models/MODEL_t3.h5')

t4 = load_model('32X16_sorter/saved_models/MODEL_t4.h5')
t5 = load_model('32X16_sorter/saved_models/MODEL_t5.h5')
t6 = load_model('32X16_sorter/saved_models/MODEL_t6.h5')
t7 = load_model('32X16_sorter/saved_models/MODEL_t7.h5')

test_dataset_raw = []
target_test='real'
data_dir_test = target_test+'_data/'
log_dir_test = open(target_test+'_log.txt','r')
logs_test = log_dir_test.read()
logs_test = logs_test[:-1]
log_list_test = logs_test.split('\n')

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

input_size=48
test_images = np.array(test_dataset_raw)
test_images = tf.constant(test_images,dtype=tf.float16, name='test images')
test_images = tf.reshape(test_images,[len(log_list_test),1,input_size,input_size,1])

SCORE = 0

for i in tqdm (range (len(log_list_test)), desc="Loading..."):
    img = test_images[i]
    img = (np.expand_dims(img,0))
    t0_raw = t0.predict(img,verbose=0)
    t0_out = np.argmax(t0_raw)

    t1_raw = t1.predict(img,verbose=0)
    t1_out = np.argmax(t1_raw)

    t2_raw = t2.predict(img,verbose=0)
    t2_out = np.argmax(t2_raw)

    t3_raw = t3.predict(img,verbose=0)
    t3_out = np.argmax(t3_raw)

    t4_raw = t4.predict(img,verbose=0)
    t4_out = np.argmax(t4_raw)

    t5_raw = t5.predict(img,verbose=0)
    t5_out = np.argmax(t5_raw)

    t6_raw = t6.predict(img,verbose=0)
    t6_out = np.argmax(t6_raw)

    t7_raw = t7.predict(img,verbose=0)
    t7_out = np.argmax(t7_raw)

    team = [t0_out,t1_out,t2_out,t3_out,t4_out,t5_out,t6_out,t7_out]
    average = np.average(team)
    print(average)