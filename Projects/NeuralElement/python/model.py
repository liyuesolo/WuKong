from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate

import math
import numpy as np
import tensorflow as tf

def buildQuadratic(n_inputs, n_outputs, data_type = tf.float32):
    inputS = Input(shape=(n_inputs,),dtype=data_type, name="inputS")
    num_hidden = 256
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(inputS)
    for _ in range(5):
        x = Dense(num_hidden, activation=tf.keras.activations.swish)(x)
    output = Dense(n_outputs, activation=tf.keras.activations.linear)(x)
    model = Model(inputS, output)
    return model