import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import re
import string

from Helper import *
from neural_models import *
from preprocessor import input_pipeline
from Sorters import Sorter_Framework

# The Oracle is an intelligent data pipeline that labels and cleans data then feeds it to active learning models

class Oracle(object):
    def __init__(self):
        self.input = input_pipeline(36, 3)
        # get the dataset and split the train and test folders
        self.dataset = self.input.select_data()

        print("total samples: ", self.y_data.shape[0])

        samples = self.input.classes * 800
        self.val_split = int(0.1 * samples)
        self.test_split = int(0.2 * samples)
        self.train_split = int(0.7 * samples)

    def train_full_model(self, full_train_dataset, val_dataset, test_dataset, verbose=1):
    
        model = CNN
        model.compile(
            loss="SparseCategoricalCrossentropy",
            optimizer="Adam",
            metrics=[
                keras.metrics.SparseCategoricalCrossentropy(),
                keras.metrics.FalseNegatives(),
                keras.metrics.FalsePositives()
            ])

        history = model.fit(
            full_train_dataset.batch(32),
            epochs=20,
            validation_data=val_dataset,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, verbose=1),
                keras.callbacks.ModelCheckpoint(
                    "fullmodelcheckpoint.h5", verbose=1, save_best_only=True
                )])

        model = keras.models.load_model("fullmodelcheckpoint.h5")

        if verbose:
            Helper.plot_history(
                history.history["loss"],
                history.history["val_loss"],
                history.history["SparseCategoricalCrossentropy"],
                history.history["val_SparseCategoricalCrossentropy"])

            print("-"*100)
            print("test set evaluation: ",model.evaluate(test_dataset, verbose=0, return_dict=True))
            print("-"*100)

        return model

    def __call__(self,verbose=1):
        
        # partition the labels into positive / negative
        self.x_positives, self.y_positives = self.dataset[self.y_data == 1], self.labels[self.labels == 1]
        self.x_negatives, self.y_negatives = self.reviews[self.labels == 0], self.labels[self.labels == 0]

        # partition validation x and y values
        self.x_val, self.y_val = (
            tf.concat((self.x_positives[:self.val_split], self.x_negatives[:self.val_split]), 0),
            tf.concat((self.y_positives[:self.val_split], self.y_negatives[:self.val_split]), 0))

        # partition test x and y values
        self.x_test, self.y_test = (
            tf.concat((
                self.x_positives[self.val_split: self.val_split + self.test_split],
                self.x_negatives[self.val_split: self.val_split + self.test_split]),
                0),
            tf.concat((
                self.y_positives[self.val_split: self.val_split + self.test_split],
                self.y_negatives[self.val_split: self.val_split + self.test_split]),
                0))

        # partition train x and y values
        self.x_train, self.y_train = (
            tf.concat((
                self.x_positives[self.val_split + self.test_split : self.val_split + self.test_split + self.train_split],
                self.x_negatives[self.val_split + self.test_split : self.val_split + self.test_split + self.train_split]),
                0),
            tf.concat((
                self.y_positives[self.val_split + self.test_split : self.val_split + self.test_split + self.train_split],
                self.y_negatives[self.val_split + self.test_split : self.val_split + self.test_split + self.train_split]),
                0))

        # the remaining samples are stored seperately and labeled when required
        self.x_pool_pos, self.y_pool_pos = (
            self.x_positives[self.val_split + self.test_split + self.train_split :],
            self.y_positives[self.val_split + self.test_split + self.train_split :])

        self.x_pool_neg, self.y_pool_neg = (
            self.x_negatives[self.val_split + self.test_split + self.train_split :],
            self.y_negatives[self.val_split + self.test_split + self.train_split :])

        # create tensorflow datasets for faster prefetching and parallelization
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

        self.pool_neg = tf.data.Dataset.from_tensor_slices((self.x_pool_neg, self.y_pool_neg))
        self.pool_pos = tf.data.Dataset.from_tensor_slices((self.x_pool_pos, self.y_pool_pos))
        
        if verbose == 1:
            print(f"initial training set size: {len(self.train_dataset)}")
            print(f"validation set size: {len(self.val_dataset)}")
            print(f"test set size: {len(self.test_dataset)}")
            print(f"unlabeled negative pool: {len(self.pool_neg)}")
            print(f"unlabled positive pool: {len(self.pool_pos)}")


        self.vectorizer = tf.keras.layers.TextVectorization(
            3000, standardize=self.standardization, output_sequence_length=150)
        
        self.vectorizer.adapt(
            self.train_dataset.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).batch(256))

        self.train_dataset = self.train_dataset.map(
            self.vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        self.pool_neg = self.pool_neg.map(self.vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
        self.pool_pos = self.pool_pos.map(self.vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

        self.val_dataset = self.val_dataset.batch(256).map(
            self.vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

        self.test_dataset = self.test_dataset.batch(256).map(
            self.vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

        full_train_dataset = (
            self.train_dataset.concatenate(self.pool_pos)
            .concatenate(self.pool_neg)
            .cache()
            .shuffle(20000))

        self.train_full_model(full_train_dataset,
                               self.val_dataset,
                                self.test_dataset)

        