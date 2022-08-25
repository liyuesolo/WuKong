from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, BatchNormalization
import math
import numpy as np
import tensorflow as tf

class IdentityFeatures(tf.keras.Model):
    def __init__(self, num_input):
        super(IdentityFeatures, self).__init__()
        
        self.num_input = num_input
        self.num_output = num_input
    def call(self, x):
        return x

class ConcatSquareFeatures(tf.keras.Model):
    def __init__(self, num_input):
        super(ConcatSquareFeatures, self).__init__()
        
        self.num_input = num_input
        self.num_output = 2*num_input
    def call(self, x):
        out = tf.concat((x , x*x), axis=-1)
        return out

class ConcatSineFeatures(tf.keras.Model):
    def __init__(self, num_input):
        super(ConcatSineFeatures, self).__init__()
        
        self.num_input = num_input
        self.num_output = 2*num_input
    def call(self, x):
        out = tf.concat((x , tf.math.sin(x)), axis=-1)
        return out

class DenseSIRENModel(tf.keras.Model):
    def __init__(self, num_input, num_output, num_hidden, last_layer_init_scale, omega0):
        super(DenseSIRENModel, self).__init__()
        
        def sin_activation(x):
            return K.sin(x);
        def sin_activation_first_layer(x):
            return K.sin(omega0*x);
        
        regularizer = keras.regularizers.l1(0.0)
        
        first_layer_initializer = tf.keras.initializers.RandomUniform(minval=-1.0/num_input, maxval=1.0/num_input)
        weight_initializer = tf.keras.initializers.RandomUniform(minval=-np.sqrt(6 / num_hidden) / omega0, maxval=np.sqrt(6 / num_hidden) / omega0)
        weight_initializer_middle_layer = tf.keras.initializers.RandomUniform(minval=-np.sqrt(6 / num_hidden), maxval=np.sqrt(6 / num_hidden) )
        last_initializer = tf.keras.initializers.RandomUniform(minval=-np.sqrt(6 / num_hidden) * last_layer_init_scale, maxval=np.sqrt(6 / num_hidden) * last_layer_init_scale)
        k_sqrt_first = np.sqrt(1.0/num_input)
        k_sqrt_middle = np.sqrt(1.0/num_hidden)
        bias_initializer_first = tf.keras.initializers.RandomUniform(minval=-k_sqrt_first, maxval=k_sqrt_first)
        bias_initializer = tf.keras.initializers.RandomUniform(minval=-k_sqrt_middle, maxval=k_sqrt_middle)

        self.dense0 = Dense(num_hidden, activation=sin_activation_first_layer, kernel_initializer=first_layer_initializer, kernel_regularizer=regularizer, bias_initializer=bias_initializer_first)
        self.dense1 = Dense(num_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer, kernel_regularizer=regularizer,bias_initializer=bias_initializer)
        self.dense2 = Dense(num_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer, kernel_regularizer=regularizer,bias_initializer=bias_initializer)
        self.dense3 = Dense(num_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer, kernel_regularizer=regularizer,bias_initializer=bias_initializer)
        self.dense4 = Dense(num_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer, kernel_regularizer=regularizer,bias_initializer=bias_initializer)
        self.dense5 = Dense(num_output, activation='linear', use_bias=False, kernel_initializer=last_initializer)

    def call(self, inputs):
        l0 = self.dense0(inputs)
        l1 = self.dense1(l0)
        l2 = self.dense2(l1)
        l2_concate = Concatenate()([inputs, l2])
        l3 = self.dense3(l2_concate)
        l4 = self.dense4(l3)
        l4_concate = Concatenate()([l2_concate, l4])
        l5 = self.dense5(l4_concate)
        # l5 = self.dense5(l1)
        return l5

class DenseSoftplusModel(tf.keras.Model):
    def __init__(self, num_input, num_output, num_hidden, last_layer_init_scale, omega0):
        super(DenseSoftplusModel, self).__init__()
        
        self.dense0 = Dense(num_hidden, activation=tf.keras.activations.softplus)
        self.dense1 = Dense(num_hidden, activation=tf.keras.activations.softplus)
        self.dense2 = Dense(num_hidden, activation=tf.keras.activations.softplus)
        self.dense3 = Dense(num_hidden, activation=tf.keras.activations.softplus)
        self.dense4 = Dense(num_hidden, activation=tf.keras.activations.softplus)
        self.dense5 = Dense(num_output, activation='linear', use_bias=False)

    def call(self, inputs):
        l0 = self.dense0(inputs)
        l1 = self.dense1(l0)
        l2 = self.dense2(l1)
        l2_concate = Concatenate()([inputs, l2])
        l3 = self.dense3(l2_concate)
        l4 = self.dense4(l3)
        l4_concate = Concatenate()([l2_concate, l4])
        l5 = self.dense5(l4_concate)
        # l5 = self.dense5(l1)
        return l5

class ConstitutiveModel(tf.keras.Model):

    def __init__(self):
        super(ConstitutiveModel, self).__init__()
        
        self.useSiren = True
        num_hidden = 10 
        omega0 = 10

        self.num_hidden = num_hidden
        self.omega0 = omega0
        self.last_layer_init_scale = 1.0

        self.features = ConcatSquareFeatures(num_input=2)
        
        # self.model = DenseSoftplusModel(num_input=self.features.num_output, num_output=1, 
        #     num_hidden = num_hidden, last_layer_init_scale=self.last_layer_init_scale, 
        #     omega0 = omega0)
        self.model = DenseSIRENModel(num_input=self.features.num_output, num_output=1, 
            num_hidden = num_hidden, last_layer_init_scale=self.last_layer_init_scale, 
            omega0 = omega0)

    def call(self, inputs):
        
        x = tf.gather(inputs, 0, axis=1, batch_dims=0)
        y = tf.gather(inputs, 1, axis=1, batch_dims=0)
        
        nm11 = tf.stack( (x , y) , axis=1)

        n = self.features(nm11)
        elastic_potential = self.model(n)
        
        return elastic_potential
    
    def get_config(self):
        return {"omega0" : self.omega0}


class AnisotropicModel(tf.keras.Model):

    def __init__(self):
        super(AnisotropicModel, self).__init__()
        
        self.useSiren = True
        num_hidden = 10 
        omega0 = 30
        num_input = 3

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.omega0 = omega0
        self.last_layer_init_scale = 1.0

        self.features = ConcatSquareFeatures(num_input=self.num_input)
        # self.model = DenseSIRENModel(num_input=self.features.num_output, num_output=1, 
        #     num_hidden = num_hidden, last_layer_init_scale=self.last_layer_init_scale, 
        #     omega0 = omega0)
        self.model = DenseSoftplusModel(num_input=self.features.num_output, num_output=1, 
            num_hidden = num_hidden, last_layer_init_scale=self.last_layer_init_scale, 
            omega0 = omega0)

    def call(self, inputs):
        
        x = tf.gather(inputs, 0, axis=1, batch_dims=0)
        y = tf.gather(inputs, 1, axis=1, batch_dims=0)
        theta = tf.gather(inputs, 2, axis=1, batch_dims=0)
        
        
        nm11 = tf.stack( (x , y, theta) , axis=1)

        n = self.features(nm11)
        elastic_potential = self.model(n)
        
        return elastic_potential
    
    def get_config(self):
        return {"omega0" : self.omega0}