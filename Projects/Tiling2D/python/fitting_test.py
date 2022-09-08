import os
from functools import cmp_to_key
from pyexpat import model
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
import keras.backend as K

def sin_activation(x):
    return K.sin(x)

def buildModel(num_params):
    num_hidden = 5
    inputS = Input(shape=(num_params,),dtype=tf.float32, name="inputS")
    x = Dense(num_hidden, activation='sigmoid')(inputS)
    # for i in range(5):
        # x = Dense(num_hidden, activation=tf.keras.activations.swish)(x)
    output = Dense(1, activation='linear')(x)
    model = Model(inputS, output)
    return model

def generateTestingData():
    
    def value(x):
        return np.power(x, 3) + x + 0.5
        # return x * x
    
    def grad(x):
        return 3.0 * np.power(x, 2) + 1.0
        # return 2.0 * x
    
    f = open("data.txt", "w+")

    for i in range(100):
        x = float(i) / float(100)
        # x = i
        f.write(str(x) + " " + str(value(x)) + " " + str(grad(x)) + "\n")
    f.close()

# generateTestingData()

def loadData(shuffle = True):
    filename = "data.txt"
    all_data = []
    all_label = [] 
    all_energy = []
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]] 
        data = [item[0]]
        label = [item[2], item[1]]
                    
        all_data.append(data)
        all_label.append(label)
    
    all_data = np.array(all_data).astype(np.float32)
    all_label = np.array(all_label).astype(np.float32)

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

l2Loss = tf.keras.losses.MeanSquaredError()

@tf.function
def trainStep(opt, lambdas, sigmas, model, train_vars):
    rest = tf.constant([0.0], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(lambdas)
        tape.watch(rest)
        value = model(lambdas)
        grad = tape.gradient(value, lambdas)
        grad_loss = tf.constant(1.0, dtype=tf.float32) * l2Loss(sigmas[:, 0:], grad)
        e_loss = tf.constant(0.0, dtype=tf.float32) * tf.cast(l2Loss(sigmas[:, -1], value), tf.float32)
        
        value_rest = model(rest)
        e_loss += tf.constant(1.0, dtype=tf.float32) * l2Loss(tf.constant(0.5, dtype=tf.float32), value_rest)
        loss = grad_loss + e_loss
    dLdw = tape.gradient(loss, train_vars)
    gradNorm = tf.math.sqrt(tf.reduce_sum([tf.reduce_sum(gi*gi) for gi in dLdw]))
    # print(gradNorm)
    opt.apply_gradients(zip(dLdw, train_vars))

    del tape
    return grad_loss, e_loss, gradNorm

def plotPotential(model_name):
    data_all, label_all = loadData(False)
    model = buildModel(1)
    model.load_weights(model_name + '.tf')
    value = model(tf.convert_to_tensor(data_all)).numpy()
    gt = label_all[:, -1]
    x = np.array(data_all)
    plt.plot(x, value, label = "prediction")
    plt.plot(x, gt, label = "GT")
    plt.legend(loc="upper right")
    plt.savefig("quadratic.png", dpi = 300)
    plt.close()

def train(model_name, train_data, train_label, validation_data, validation_label):
    model = buildModel(1)
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-2)
    max_iter = 10000
    g_norm0 = 0
    for iteration in range(max_iter):
        lambdas, sigmas = next(generator(train_data, train_label))
        
        lambdasTF = tf.convert_to_tensor(lambdas)
        sigmasTF = tf.convert_to_tensor(sigmas)
        train_loss_grad, train_loss_e, g_norm = trainStep(opt, lambdasTF, sigmasTF, model, train_vars)
        if iteration == 0:
            g_norm0 = g_norm
        print("iter: {}/{} train_loss_grad: {} train_loss e: {} |g|: {}, |g0| {}".format(iteration, max_iter, train_loss_grad, train_loss_e, g_norm, g_norm0))
    
    save_path = "./"
    model.save_weights(save_path + model_name + '.tf')

if __name__ == "__main__":
    generateTestingData()
    data_all, label_all = loadData()
    train_data = data_all
    validation_data = data_all
    train_label = label_all
    validation_label = label_all

    train("poly", train_data, train_label, validation_data, validation_label)
    plotPotential("poly")