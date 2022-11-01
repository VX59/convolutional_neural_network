import tensorflow as tf

linear_model = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(1,(2,2),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,2,2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(1,(2,2),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax'),
        ])

deep_model_3 = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax'),
        ])

deep_model_4 = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16,(2,2),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax'),
        ])

deep_model_5 = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax'),
        ])

deep_model_6 = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(16,(2,2),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(712,activation='relu'),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax'),
        ])

deep_model_8 = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,5,5)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax'),
        ])

deep_model_16x16 = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(16,(2,2),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(16, activation='softmax'),
        ])

deep_model_16x2 = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32,(4,4),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,5,5)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(16,(2,2),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(160,activation='relu'),
            tf.keras.layers.Dense(784,activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax'),
        ])

deep_model_32x16 = tf.keras.Sequential(
        [   
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(24,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,4,4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'),
            tf.keras.layers.MaxPool3D(pool_size=(1,3,3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.ActivityRegularization(),
            tf.keras.layers.Dense(1200,activation='relu'),
            tf.keras.layers.Dense(600,activation='relu'),
            tf.keras.layers.Dense(100,activation='relu'),
            tf.keras.layers.Dense(16, activation='softmax'),

        ])