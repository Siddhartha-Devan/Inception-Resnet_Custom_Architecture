import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model


class QuadDense(Layer):
    def __init__(self, units=64, activation = 'relu'):
        super(QuadDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        w1_init = tf.random_normal_initializer()
        self.w1 = tf.Variable(name = 'kernel_1', initial_value = w1_init(shape = (input_shape[-1], self.units), dtype = 'float32'))
        
        w2_init = tf.random_normal_initializer()
        self.w2 = tf.Variable(name = 'kernel_2', initial_value = w2_init(shape = (input_shape[-1], self.units), dtype = 'float32'))

        bias_init = tf.zeros_initializer()
        self.b = tf.Variable(name = 'bias', initial_value = bias_init(shape = (self.units,), dtype = 'float32'))


    def call(self, inputs):
        return self.activation((tf.matmul(tf.math.square(inputs), self.w1)+tf.matmul(inputs, self.w2)+self.b))

                             

class ResidualBlock(Model):
    def __init__(self, n_filters, kernel_size):
        super(ResidualBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters = n_filters, padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(kernel_size = kernel_size, filters = n_filters, padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.Activation('relu')

        self.conv_adjuster = tf.keras.layers.Conv2D(kernel_size= (1,1), filters = n_filters, padding = 'same')

        self.add = tf.keras.layers.Add()

    
    def call(self, inputs):
        print(inputs.shape)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        input_tensor = self.conv_adjuster(inputs)
        x = self.add([x, input_tensor])
        x = self.activation(x)

        return x



class InceptionBlock(Model):
    def __init__(self, n_filters):
        super(InceptionBlock, self).__init__()
        self.path1_conv1 = tf.keras.layers.Conv2D(kernel_size = (1,1), filters = n_filters, padding = 'same')
        self.path1_conv2 = tf.keras.layers.Conv2D(kernel_size= (3,3), filters = n_filters, padding = 'same')
        self.path1_bn = tf.keras.layers.BatchNormalization()


        self.path2_conv1 = tf.keras.layers.Conv2D(kernel_size = (1,1), filters = n_filters*2, padding = 'same')
        self.path2_conv2 = tf.keras.layers.Conv2D(kernel_size= (3,3), filters = n_filters*2, padding = 'same')
        self.path2_bn = tf.keras.layers.BatchNormalization()

        self.path3_conv1 = tf.keras.layers.Conv2D(kernel_size = (1,1), filters = n_filters//2, padding = 'same')
        self.path3_conv2 = tf.keras.layers.Conv2D(kernel_size= (5,5), filters = n_filters//2, padding = 'same')
        self.path3_bn = tf.keras.layers.BatchNormalization()

        self.path4_pool = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (1,1), padding='same')
        self.path4_conv1 = tf.keras.layers.Conv2D(kernel_size = (1,1), filters = n_filters, padding = 'same')
        self.path4_bn = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.Activation('relu')

        self.concat = tf.keras.layers.Concatenate()

    
    def call(self, inputs):
        x1 = self.path1_conv1(inputs)
        x1 = self.path1_conv2(x1)
        x1 = self.path1_bn(x1)
        x1 = self.activation(x1)

        x2 = self.path2_conv1(inputs)
        x2 = self.path2_conv2(x2)
        x2 = self.path2_bn(x2)
        x2 = self.activation(x2)

        x3 = self.path3_conv1(inputs)
        x3 = self.path3_conv2(x3)
        x3 = self.path3_bn(x3)
        x3 = self.activation(x3)

        x4 = self.path4_pool(inputs)
        x4 = self.path4_conv1(x4)
        x4 = self.path4_bn(x4)
        x4 = self.activation(x4)

        outputs = self.concat([x1, x2, x3, x4])

        return outputs