import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(13, 13), strides=(1, 1), activation='relu',input_shape = (32,32,1)),
])

model.summary()