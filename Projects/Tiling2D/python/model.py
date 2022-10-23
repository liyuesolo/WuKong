from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, BatchNormalization, Dropout, Add
from tensorflow.keras import regularizers
import math
import numpy as np
import tensorflow as tf
from tf_siren import SIRENModel, SinusodialRepresentationDense
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
        x01 = 0.5 * (0.5 * x + 1.0)
        out = tf.concat((tf.math.sin(tf.matmul(x01, np.float32(self.B))), tf.math.cos(tf.matmul(x01, np.float32(self.B)))), axis=-1)
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
            return K.sin(x)
        def sin_activation_first_layer(x):
            return K.sin(omega0*x)
        
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
        self.dense5 = Dense(num_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer, kernel_regularizer=regularizer,bias_initializer=bias_initializer)
        self.dense6 = Dense(num_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer, kernel_regularizer=regularizer,bias_initializer=bias_initializer)
        self.dense_middle = Dense(num_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer, kernel_regularizer=regularizer,bias_initializer=bias_initializer)
        self.dense_last = Dense(num_output, activation=tf.keras.activations.softplus, kernel_initializer=last_initializer, kernel_regularizer=regularizer)
        # self.dense_last = Dense(num_output, activation="linear", kernel_initializer=last_initializer, kernel_regularizer=regularizer)
        

    def call(self, inputs):
        output = self.dense0(inputs)
        for i in range(12):
            output = self.dense_middle(output)
        # l1 = self.dense1(l0)
        # l2 = self.dense2(l1)
        # l3 = self.dense3(l2)
        # l4 = self.dense4(l3)
        output = self.dense_last(output)
        return output

class DenseSwishModel(tf.keras.Model):
    def __init__(self, num_input, num_output, num_hidden, use_BN = True, use_dropout = True):
        super(DenseSwishModel, self).__init__()
        
        regularizer = keras.regularizers.l1(0.0)
        self.dropout = Dropout(0.2)

        self.dense0 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense1 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense2 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense3 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense4 = Dense(num_hidden, activation=tf.keras.activations.swish)
        self.dense5 = Dense(num_output, activation=tf.keras.activations.softplus)


    def call(self, inputs):
        l0 = self.dense0(inputs)
        # l0 = self.dropout(l0)
        l1 = self.dense1(l0)
        # l1 = self.dropout(l1)
        l2 = self.dense2(l1)
        # l2 = self.dropout(l2)
        l3 = self.dense3(l2)
        # l3 = self.dropout(l3)
        l4 = self.dense4(l3)
        # l4 = self.dropout(l4)
        l5 = self.dense5(l4)
        return l5


class ConstitutiveModel(tf.keras.Model):

    def __init__(self, num_input = None):
        super(ConstitutiveModel, self).__init__()
        
        num_hidden = 128
        self.num_hidden = num_hidden

        # self.features = FourierFeatures(num_input=num_input, num_samples=30, sigma=2.0 = B, dtype_np=np.float32)
        self.features = IdentityFeatures(num_input=num_input)
        
        # self.model = DenseSwishModel(num_input=self.features.num_output, num_output=1, 
        #     num_hidden = num_hidden)
        
        self.model = DenseSIRENModel(num_input=self.features.num_output, num_output=1, 
            num_hidden = num_hidden, last_layer_init_scale=1.0, 
            omega0 = 100.0)
        
        # self.model = SIRENModel(units=num_hidden, final_units=1, final_activation='softplus',
        #            num_layers=32, w0=1.0, w0_initial=30.0)
        
        
    def call(self, inputs):
        n = self.features(inputs)
        elastic_potential = self.model(n)
        return elastic_potential
    
    
    def get_config(self):
        return 

def loadSingleFamilyModel(num_params):
    inputS = Input(shape=(4 + num_params,),dtype=tf.float32, name="inputS")
    output = ConstitutiveModel(4 + num_params)(inputS)
    model = Model(inputS, output)
    return model

def buildSingleFamilyModel(num_params):
    inputS = Input(shape=(4 + num_params,),dtype=tf.float32, name="inputS")
    output = ConstitutiveModel(4 + num_params)(inputS)
    model = Model(inputS, output)
    return model

def get_sub_tensor(dim, start, end):
	def f(x):
		if dim == 0:
			return x[start:end]
		if dim == 1:
			return x[:, start:end]
		if dim == 2:
			return x[:, :, start:end]
		if dim == 3:
			return x[:, :, :, start:end]
	return Lambda(f)
    
def buildSrainStressModel():
    inputS = Input(shape=(3,),dtype=tf.float32, name="inputS")
    output = ConstitutiveModel(3)(inputS)
    model = Model(inputS, output)
    return model

def buildConstitutiveModel(n_strain_entry):
    inputS = Input(shape=(n_strain_entry,),dtype=tf.float32, name="inputS")
    num_hidden = 256
    # x = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(inputS)
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(inputS)
    for _ in range(5):
        # x = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(x)
        x = Dense(num_hidden, activation=tf.keras.activations.swish)(x)
    # output = SinusodialRepresentationDense(1, w0=1.0, activation=tf.keras.activations.softplus)(x)
    output = Dense(1, activation=tf.keras.activations.softplus)(x)
    model = Model(inputS, output)
    return model

def buildSingleFamilyModelSeparateTilingParamsSwish(num_params, data_type=tf.float32):
    
    inputS = Input(shape=(3 + num_params,),dtype=data_type, name="inputS")
    tiling_params = get_sub_tensor(1, 0, num_params)(inputS)
    strain = get_sub_tensor(1, num_params, num_params + 3)(inputS)
    num_hidden = 256
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(tiling_params)
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(x)
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(x)
    y = Dense(num_hidden, activation=tf.keras.activations.swish)(strain)
    y = Dense(num_hidden, activation=tf.keras.activations.swish)(y)
    y = Dense(num_hidden, activation=tf.keras.activations.swish)(y)
    z = Concatenate()([x, y]) 
    for i in range(5):
        z = Dense(num_hidden, activation=tf.keras.activations.swish)(z)
    output = Dense(1, activation=tf.keras.activations.softplus)(z)
    

    model = Model(inputS, output)
    return model

def buildSingleFamilyModelSeparateTilingParams(num_params, data_type=tf.float32):
    
    inputS = Input(shape=(4 + num_params,),dtype=data_type, name="inputS")
    tiling_params = get_sub_tensor(1, 0, num_params)(inputS)
    strain = get_sub_tensor(1, num_params, num_params + 4)(inputS)


    num_hidden = 256
    x = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(tiling_params)
    x = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(x)
    # x = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(x)
    y = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(strain)
    y = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(y)
    # y = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(y)
    z = Concatenate()([x, y]) 
    for i in range(5):
        z = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(z)
    # output = SinusodialRepresentationDense(1, w0=1.0, activation=tf.keras.activations.softplus)(z)
    output = Dense(1, activation=tf.keras.activations.softplus)(z)
    

    model = Model(inputS, output)
    return model

def buildSingleFamilyModel3Strain(num_params, data_type=tf.float32):
    
    inputS = Input(shape=(3 + num_params,),dtype=data_type, name="inputS")
    tiling_params = get_sub_tensor(1, 0, num_params)(inputS)
    strain = get_sub_tensor(1, num_params, num_params + 3)(inputS)


    num_hidden = 256
    x = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(tiling_params)
    x = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(x)
    x = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(x)
    y = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(strain)
    y = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(y)
    y = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(y)
    z = Concatenate()([x, y]) 
    for i in range(5):
        z = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(z)
    output = SinusodialRepresentationDense(1, w0=1.0, activation=tf.keras.activations.softplus)(z)
    

    model = Model(inputS, output)
    return model

def buildSingleFamilyModelSeparateTilingParamsAux(num_params, data_type=tf.float32):
    num_hidden = 256
    
    inputS = Input(shape=(4 + num_params + 5,),dtype=data_type, name="inputS")
    # tiling_params = get_sub_tensor(1, 0, num_params)(inputS)
    # strain = get_sub_tensor(1, num_params, num_params + 4)(inputS)
    # aux = get_sub_tensor(1, num_params + 4, num_params + 9)(inputS)
    
    # batch_dim = tf.shape(strain)[0]
    
    # strain_tensor = tf.reshape(strain, (batch_dim, 2, 2))
    # s, u, v = tf.linalg.svd(strain_tensor)
    # v = tf.reshape(v, (batch_dim, 4))
    # u = tf.reshape(u, (batch_dim, 4))
    # strain = tf.concat((strain, tf.concat((s, tf.concat((u, v), axis=-1)), axis=-1)), axis=-1)
    
    # x = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(tiling_params)
    # x = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(x)
    # y = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(strain)
    # y = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(y)
    # z = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(aux)
    # z = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(z)
    # xy = Concatenate()([x, y])
    # xyz = Concatenate()([xy, z])
    # for i in range(5):
    #     xyz = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(xyz)
    # output = SinusodialRepresentationDense(1, w0=1.0, activation=tf.keras.activations.softplus)(z)
    # output = Dense(1, activation=tf.keras.activations.softplus)(z)
    
    x = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(inputS)
    for i in range(10):
        x = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(x)
    output = SinusodialRepresentationDense(1, w0=1.0, activation=tf.keras.activations.softplus)(x)
    model = Model(inputS, output)
    return model