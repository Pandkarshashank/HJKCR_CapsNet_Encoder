import tensorflow as tf
import pipeline
from config import Config
import numpy as np

class CapsuleNetwork(tf.keras.Model):
    def __init__(self, no_of_conv_kernels, no_of_primary_capsules, primary_capsule_vector, no_of_secondary_capsules, secondary_capsule_vector, r):
        super(CapsuleNetwork, self).__init__()
        self.no_of_conv_kernels = no_of_conv_kernels
        self.no_of_primary_capsules = no_of_primary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.secondary_capsule_vector = secondary_capsule_vector
        self.r = r
        
        
        with tf.name_scope("Variables") as scope:
            self.convolution = tf.keras.layers.Conv2D(self.no_of_conv_kernels, [9,9], strides=[1,1], name='ConvolutionLayer', activation='relu')
            self.primary_capsule = tf.keras.layers.Conv2D(self.no_of_primary_capsules * self.primary_capsule_vector, [9,9], strides=[2,2], name="PrimaryCapsule")
            self.w = tf.Variable(tf.random_normal_initializer()(shape=[1, 1152, self.no_of_secondary_capsules, self.secondary_capsule_vector, self.primary_capsule_vector]), dtype=tf.float32, name="PoseEstimation", trainable=True)
            self.attention_1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)
            self.attention_2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)
            self.dense_1 = tf.keras.layers.Dense(units = 512, activation='relu')
            self.dense_2 = tf.keras.layers.Dense(units = 1024, activation='relu')
            self.dense_3 = tf.keras.layers.Dense(units = 784, activation='sigmoid', dtype='float32')
        
    def build(self, input_shape):
        pass
        
    def squash(self, s):
        with tf.name_scope("SquashFunction") as scope:
            s_norm = tf.norm(s, axis=-1, keepdims=True)
            return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + Config.epsilon)
    
    @tf.function
    def call(self, inputs):
        input_x, y = inputs
        
        x = self.convolution(input_x) 
        x = self.primary_capsule(x) 
        
        with tf.name_scope("CapsuleFormation") as scope:
            u = tf.reshape(x, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 8)) 
            u = tf.expand_dims(u, axis=-2) 
            u = tf.expand_dims(u, axis=-1)
            u_hat = tf.matmul(self.w, u)
            u_hat = tf.squeeze(u_hat, [4]) 

        
        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros((input_x.shape[0], 1152, self.no_of_secondary_capsules, 1))
            for i in range(self.r): 
                c = tf.nn.softmax(b, axis=-2) 
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) 
                v = self.squash(s) 
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                b += agreement
                
        with tf.name_scope("Masking") as scope:
            y = tf.expand_dims(y, axis=-1) 
            y = tf.expand_dims(y, axis=1) 
            mask = tf.cast(y, dtype=tf.float32) 
            v_masked = tf.multiply(mask, v) 
            
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) 
            reconstructed_image = self.attention_1(v_)
            reconstructed_image = self.attention_1(reconstructed_image)
            reconstructed_image = self.dense_1(reconstructed_image) 
            reconstructed_image = self.dense_2(reconstructed_image) 
            reconstructed_image = self.dense_3(reconstructed_image)
        
        return v, reconstructed_image

    @tf.function
    def predict_capsule_output(self, inputs):
        x = self.convolution(inputs) 
        x = self.primary_capsule(x) 
        
        with tf.name_scope("CapsuleFormation") as scope:
            u = tf.reshape(x, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 8)) # u.shape: (None, 1152, 8)
            u = tf.expand_dims(u, axis=-2)
            u = tf.expand_dims(u, axis=-1) 
            u_hat = tf.matmul(self.w, u) 
            u_hat = tf.squeeze(u_hat, [4])

        
        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros((inputs.shape[0], 1152, self.no_of_secondary_capsules, 1)) 
            for i in range(self.r):
                c = tf.nn.softmax(b, axis=-2) 
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) 
                v = self.squash(s)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) 
                b += agreement
        return v

    @tf.function
    def regenerate_image(self, inputs):
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(inputs, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) 
            reconstructed_image = self.attention_1(v_)
            reconstructed_image = self.attention_1(reconstructed_image)
            reconstructed_image = self.dense_1(reconstructed_image) 
            reconstructed_image = self.dense_2(reconstructed_image) 
            reconstructed_image = self.dense_3(reconstructed_image) 
        return reconstructed_image
    
    @tf.function
    def safe_norm(self,v, axis=-1):
        v_ = tf.reduce_sum(tf.square(v), axis = axis, keepdims=True)
        return tf.sqrt(v_ + Config.epsilon)

    @tf.function
    def loss_function(self,v, reconstructed_image, y, y_image):
        prediction = self.safe_norm(v)
        prediction = tf.reshape(prediction, [-1, self.no_of_secondary_capsules])
        
        left_margin = tf.square(tf.maximum(0.0, Config.m_plus - prediction))
        right_margin = tf.square(tf.maximum(0.0, prediction - Config.m_minus))
        
        l = tf.add(y * left_margin, Config.lambda_ * (1.0 - y) * right_margin)
        
        margin_loss = tf.reduce_mean(tf.reduce_sum(l, axis=-1))
        
        y_image_flat = tf.reshape(y_image, [-1, 784])
        reconstruction_loss = tf.reduce_mean(tf.square(y_image_flat - reconstructed_image))
        
        loss = tf.add(margin_loss, Config.alpha * reconstruction_loss)
        
        return loss

    @tf.function
    def predict(self,model, x):
        pred = self.safe_norm(model.predict_capsule_output(x))
        pred = tf.squeeze(pred, [1])
        return np.argmax(pred, axis=1)[:,0]