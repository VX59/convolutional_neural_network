import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

model = load_model('saved_models/model.h5')
sample_label = 6
dataset_raw = []

data_dir = 'sample_data/'
log_dir = open('sample_log.txt','r')
logs = log_dir.read()

log_list = logs.split('\n')

print(len(log_list))

i=0
for file in log_list:
    filepath = data_dir+file
    image = Image.open(filepath)
    WIDTH, HEIGHT = image.size
    data = list(image.getdata())
    data = [data[offset:offset+WIDTH] for offset in range(0,WIDTH*HEIGHT, WIDTH)]
    data = np.array(data)
    dataset_raw.append(data)
    i = i+1

print(len(dataset_raw))

sample_images = np.array(dataset_raw)
sample_images = tf.constant(sample_images,dtype=tf.float16, name='sample images')
sample_images = tf.reshape(sample_images,[len(log_list),1,148,148,1])

def plot_value_array(i, predictions_array):
    score = 0
    plt.grid(False)
    plt.xticks(range(10))
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('blue')
    return score


classes = []
for i in range(10): classes.append(i)


score = 0
total_size = 0
for j in range(len(log_list)):
    img = sample_images[j]
    img = (np.expand_dims(img,0))
    file = log_list[j]
    filepath = data_dir+file
    image = Image.open(filepath)
    image.show()
    predictions_single = model.predict(img)

    print('PREDICTION : ',np.argmax(predictions_single[0]))

    score += plot_value_array(j, predictions_single[0])
    _ = plt.xticks(range(10), classes, rotation=45)
    total_size = total_size + 1
    image.close()
plt.show()

target='train_synth'

raw_labelset = []

def generate_labels():
    for i in range(len(log_list)):
        raw_labelset.append(sample_label)

generate_labels()

sample_labels = np.array(raw_labelset)
sample_labels = tf.constant(sample_labels,dtype=tf.float16, name='sample labels')

checkpoint_path = "model_training/A(0-3).ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
def train(train_data,train_labels,persistance=True):
    if persistance: model.load_weights(checkpoint_path)
    model.fit(train_data, 
            train_labels,  
            epochs=10,
            batch_size=128,
            callbacks=[cp_callback])  # Pass callback to training

    model.save('saved_models/model1.h5')
