import tensorflow as tf

augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", input_shape=(36, 36, 1)),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1)
            ])

CNN = tf.keras.Sequential(
        [   
            augmentation,
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Reshape((1, 36, 36, 1)),
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64,(3,3) ,activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(3, activation="softmax"),
        ])