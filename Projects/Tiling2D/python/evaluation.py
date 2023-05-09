import os
from functools import cmp_to_key
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import math
import numpy as np
import tensorflow as tf
from model import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import keras.backend as K
from Summary import *
import tensorflow_probability as tfp
import scipy
tf.keras.backend.set_floatx('float64')
n_input = 3

from Common import *

def relativeL2(y_true, y_pred):
    if (y_true.shape[1] > 1):
        stress_norm = tf.norm(y_true, ord='euclidean', axis=1)
        norm = tf.tile(tf.keras.backend.expand_dims(stress_norm, 1), tf.constant([1, 3]))
        y_true_normalized = tf.divide(y_true, norm)
        y_pred_normalized = tf.divide(y_pred, norm)
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
    else:
        y_true_normalized = tf.ones(y_true.shape, tf.float64)
        y_pred_normalized = tf.divide(y_pred + K.epsilon(), y_true + K.epsilon())
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
        
def loadDataSplitTest(n_tiling_params, filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    
    for line in open(filename).readlines():
        # item = [np.around(float(i), decimals=6) for i in line.strip().split(" ")[:]]
        item = [float(i) for i in line.strip().split(" ")[:]]
        item[0] = np.round(item[0], 8)
        if (ignore_unconverging_result):
            if (item[-1] > 1e-6 or math.isnan(item[-1])):
                continue
            if (item[-5] < 1e-5 or item[-5] > 10):
                continue
            # if (np.abs(item[-3] - 1.001) < 1e-6 or np.abs(item[-3] - 0.999) < 1e-6):
            #     continue
            # if (np.abs(item[-2] - 1.001) < 1e-6 or np.abs(item[-2] - 0.999) < 1e-6):
            #     continue
        data = item[0:n_tiling_params]
        for i in range(2):
            data.append(item[n_tiling_params+i])
        data.append(2.0 * item[n_tiling_params+2])
        
        label = item[n_tiling_params+3:n_tiling_params+7]
        
        all_data.append(data)
        all_label.append(label)
        
    print("#valid data:{}".format(len(all_data)))
    # exit(0)
    start = 0
    end = -1
    all_data = np.array(all_data[start:]).astype(np.float64)
    all_label = np.array(all_label[start:]).astype(np.float64) 
    
    # all_data = np.array(all_data).astype(np.float64)
    # all_label = np.array(all_label).astype(np.float64)
    
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    
    all_data = all_data[indices]
    all_label = all_label[indices]
    
    return all_data, all_label


loss_l2 = tf.keras.losses.MeanSquaredError()
loss_logl2 = tf.keras.losses.MeanSquaredLogarithmicError()
# loss_function = relativeL2

w_grad = tf.constant(1.0, dtype=tf.float64)
w_e = tf.constant(1.0, dtype=tf.float64)

def generator(train_data, train_label):    
    indices = np.arange(train_data.shape[0])
    while True:
        np.random.shuffle(indices)
        yield train_data[indices], train_label[indices]

@tf.function
def trainStep(n_tiling_params, opt, lambdas, sigmas, model, train_vars):
    # aux = tf.tile(tf.constant([[50, 14.2045, 50, 26, 0.48]]), tf.constant((lambdas.shape[0], 1), tf.int32))
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(train_vars)
        tape.watch(lambdas)
        # all_input = tf.concat((lambdas, aux), axis=-1)
        
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 3])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, psi)

        loss = grad_loss + e_loss
        
    dLdw = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(dLdw, train_vars))
    gradNorm = tf.math.sqrt(tf.reduce_sum([tf.reduce_sum(gi*gi) for gi in dLdw]))
    # gradNorm = -1.0
    
    del tape
    return grad_loss, e_loss, gradNorm

@tf.function
def trainStepBatch(n_tiling_params, lambdas, sigmas, model, train_vars):
    # aux = tf.tile(tf.constant([[50, 14.2045, 50, 26, 0.48]]), tf.constant((lambdas.shape[0], 1), tf.int32))
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(train_vars)
        tape.watch(lambdas)
        # all_input = tf.concat((lambdas, aux), axis=-1)
        
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, n_input])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, n_input])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, psi)

        loss = grad_loss + e_loss
        
    dLdw = tape.gradient(loss, train_vars)
    
    del tape
    return grad_loss, e_loss, dLdw

@tf.function
def testStep(n_tiling_params, lambdas, sigmas, model):
    
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, n_input])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, n_input])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, psi)
    del tape
    return grad_loss, e_loss, stress_pred, psi



def evaluate(IH):
    
    base_folder = "/home/yueli/Documents/ETH/SandwichStructure/"
    IH_string = "IH" + str(IH)
    if IH == 1:
        IH_string = "IH01"
    full_data = base_folder + "Server" + IH_string + "/all_data_"+IH_string+"_shuffled.txt"
    model, n_tiling_params, ti_default, bounds = loadModel(IH, use_double=True)

    data_all, label_all = loadDataSplitTest(n_tiling_params, full_data, False, True)
    five_percent = int(len(data_all) * 0.05)
    test_data = data_all[-five_percent:]
    test_label = label_all[-five_percent:]
    test_dataTF = tf.convert_to_tensor(test_data)
    test_labelTF = tf.convert_to_tensor(test_label)
    loss_g, loss_e, _, _ = testStep(n_tiling_params,test_dataTF, test_labelTF, model)

    print("test IH: {} loss gradient: {}, loss energy: {} ".format(IH, np.sqrt(loss_g) * 100.0, np.sqrt(loss_e) * 100.0))

    
if __name__ == "__main__":
    
    # for IH in [1, 21, 22, 28, 29, 50, 67]:
    for IH in [50]:
        evaluate(IH)
    