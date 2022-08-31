from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, BatchNormalization, Dropout
import math
import numpy as np
import tensorflow as tf

class FourierFeatures(tf.keras.Model):
    def __init__(self, sigma, num_input, num_samples, B = None, dtype_np = np.float32):
        super(FourierFeatures, self).__init__()
        
        self.num_input = num_input
        self.num_output = num_samples* num_input * 2
        if B is None:
            self.B = np.random.normal(0.0, 2.0 * math.pi * sigma, size=(num_input, num_samples* num_input)).astype(dtype_np)
        else:
            self.B = np.reshape(B, (num_input, num_samples* num_input)).astype(dtype_np)
        self.num_samples = num_samples
        self.sigma = sigma
    
    def get_config(self):
        return {"num_samples": self.num_samples, 
                "sigma": self.sigma,
                "B": self.B}

    def call(self, x):
        x01 = 0.5 * (x + 1.0)
        out = tf.concat((tf.math.sin(tf.matmul(x01, np.float32(self.B))), tf.math.cos(tf.matmul(x01, np.float32(self.B)))), axis=-1)
        # print('fourier')
        # print(out)
        return out


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

class DenseSwishModel(tf.keras.Model):
    def __init__(self, num_input, num_output, num_hidden, use_BN = True, use_dropout = True):
        super(DenseSwishModel, self).__init__()
        
        regularizer = keras.regularizers.l1(0.0)
        
        self.dense0 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense1 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense2 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense3 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense4 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense5 = Dense(num_output, activation='linear', use_bias=False)


    def call(self, inputs):
        l0 = self.dense0(inputs)
        l0_concate = Concatenate()([inputs, l0])
        # l1 = self.dense1(l0)
        l1 = self.dense1(l0_concate)
        l1_concate = Concatenate()([inputs, l1])
        # l2 = self.dense2(l1)
        l2 = self.dense2(l1_concate)
        # l3 = self.dense3(l2_concate)
        # l4 = self.dense4(l3)
        # l4_concate = Concatenate()([l2_concate, l4])
        # l5 = self.dense5(l4_concate)
        l5 = self.dense5(l2)
        return l5

class ConstitutiveModel(tf.keras.Model):

    def __init__(self, num_input, B = None):
        super(ConstitutiveModel, self).__init__()
        
        num_hidden = 30
        self.num_hidden = num_hidden

        self.features = FourierFeatures(num_input=num_input, num_samples=30, sigma=2.0, B = B, dtype_np=np.float32)
        
        self.model = DenseSwishModel(num_input=self.features.num_output, num_output=1, 
            num_hidden = num_hidden)
        
    def call(self, inputs):
        n = self.features(inputs)
        elastic_potential = self.model(n)
        return elastic_potential
    
    def get_config(self):
        return {"B" : self.features.B}

def loadSingleFamilyModel(num_params, B):
    inputS = Input(shape=(3 + num_params,),dtype=tf.float32, name="inputS")
    output = ConstitutiveModel(3 + num_params, B)(inputS)
    model = Model(inputS, output)
    return model

def buildSingleFamilyModel(num_params):
    inputS = Input(shape=(3 + num_params,),dtype=tf.float32, name="inputS")
    output = ConstitutiveModel(3 + num_params)(inputS)
    model = Model(inputS, output)
    return model

def buildSrainStressModel():
    inputS = Input(shape=(3,),dtype=tf.float32, name="inputS")
    output = ConstitutiveModel(3)(inputS)
    model = Model(inputS, output)
    return model