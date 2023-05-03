import os
from functools import cmp_to_key
from statistics import mode
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

use_double = False

if use_double:
    tf.keras.backend.set_floatx('float64')
    tf_data_type = tf.float64
else:
    tf.keras.backend.set_floatx('float32')
    tf_data_type = tf.float32


n_inputs = 12
n_outputs = 6

def l2Loss(y_true, y_pred):
    return K.mean(K.square((y_true - y_pred) * tf.constant(1.0, dtype=tf_data_type)))
    
def relativeL2(y_true, y_pred):
    # print(y_true)
    # print(y_pred)
    y_true_normalized = tf.ones(y_true.shape, dtype=tf_data_type)
    y_pred_normalized = tf.divide(y_pred, y_true)
    # print("y_pred_normalized", y_pred_normalized)
    return K.mean(K.square(y_true_normalized - y_pred_normalized)) * tf.constant(1.0, dtype=tf_data_type)

def loadDataSplitTest(filename, shuffle = True):
    all_data = []
    all_label = []
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")]

        data = item[0:n_inputs]
        label = item[n_inputs:]

        all_data.append(data)
        all_label.append(label)

    if use_double:
        all_data = np.array(all_data[0:]).astype(np.float64)
        all_label = np.array(all_label[0:]).astype(np.float64)
    else:
        all_data = np.array(all_data[0:]).astype(np.float32)
        all_label = np.array(all_label[0:]).astype(np.float32)
    
    print("#valid data: {}".format(len(all_data)))

    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    all_data = all_data[indices]
    all_label = all_label[indices]


    return all_data, all_label

def generator(train_data, train_label):    
    indices = np.arange(train_data.shape[0])
    while True:
        np.random.shuffle(indices)
        yield train_data[indices], train_label[indices]



@tf.function
def trainStep(opt, data, label, model, train_vars):

    with tf.GradientTape() as tape:
        tape.watch(train_vars)
        pred = model(data)
        loss = relativeL2(label, pred)
    
    dLdw = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(dLdw, train_vars))
    gradNorm = tf.math.sqrt(tf.reduce_sum([tf.reduce_sum(gi*gi) for gi in dLdw]))

    del tape
    return loss, gradNorm

@tf.function
def testStep(data, label, model):
    pred = model(data)
    loss = relativeL2(label, pred)
    return loss

def train(model_name, train_data, train_label, validation_data, validation_label):

    batch_size = np.minimum(60000, len(train_data))
    model = buildQuadratic(n_inputs, n_outputs)
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-4)
    max_iter = 80000

    current_dir = os.path.dirname(os.path.realpath(__file__))

    count = 0
    with open('counter.txt', 'r') as f:
        count = int(f.read().splitlines()[-1])
    f = open("counter.txt", "w+")
    f.write(str(count+1))
    f.close()

    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    val_dataTF = tf.convert_to_tensor(validation_data)
    val_labelTF = tf.convert_to_tensor(validation_label)

    print("batch_size: {}".format(batch_size))
    losses = [[], []]
    for iteration in range(max_iter):
        data, label = next(generator(train_data, train_label))
        if batch_size == -1:
            batch = 1
        else:
            batch = int(np.floor(len(data) / batch_size))

        train_loss_sum = 0.0
        g_norm_sum = 0.0
        for i in range(batch):
            mini_bacth_data = data[i * batch_size:(i+1) * batch_size]
            mini_bacth_label = label[i * batch_size:(i+1) * batch_size]

            dataTF = tf.convert_to_tensor(mini_bacth_data)
            labelTF = tf.convert_to_tensor(mini_bacth_label)
            
            train_loss, g_norm = trainStep(opt, dataTF, labelTF, model, train_vars)
            
            train_loss_sum += train_loss
            g_norm_sum += g_norm
        if (iteration == 0):
            g_norm0 = g_norm_sum

        validation_loss = testStep(val_dataTF, val_labelTF, model)
        
        losses[0].append(train_loss_sum)
        losses[1].append(validation_loss)
        if iteration % 1000 == 0:
            print("epoch: {}/{} train_loss: {} validation_loss:{} |g|: {}, |g_init|: {} ".format(iteration, max_iter, train_loss, \
                         validation_loss, \
                        g_norm_sum, g_norm0))   
        if iteration % 1000 == 0:
            model.save_weights(save_path + model_name + '.tf')
    model.save_weights(save_path + model_name + '.tf')

def validate(count, model_name, validation_data, validation_label):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildQuadratic(n_inputs, n_outputs)
    model.load_weights(save_path + model_name + '.tf')
    
    loss = testStep(validation_data, validation_label, model)
    test10 = model(validation_data[0:2])
    print(test10)
    print(validation_label[0:2])
    print("validation loss: {}".format(loss))

if __name__ == "__main__":
    data_file = "data.txt"

    data_all, label_all = loadDataSplitTest(data_file, shuffle=False)

    if len(data_all) < 100:
        five_percent = 1
    else:
        five_percent = int(len(data_all) * 0.05)

    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]
    model_name = "overfit_test"

    train(model_name, train_data, train_label, validation_data, validation_label)
    # validate(19, model_name, validation_data, validation_label)
    # validate(22, model_name, train_data, train_label)