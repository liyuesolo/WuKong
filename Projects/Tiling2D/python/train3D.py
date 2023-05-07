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

def relativeL2(y_true, y_pred):
    if (y_true.shape[1] > 1):
        stress_norm = tf.norm(y_true, ord='euclidean', axis=1)
        norm = tf.tile(tf.keras.backend.expand_dims(stress_norm, 1), tf.constant([1, 6]))
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
        
        if (ignore_unconverging_result):
            if (item[-1] > 1e-6 or math.isnan(item[-1])):
                continue
            
            
        data = item[0:n_tiling_params]
        for i in range(3):
            data.append(item[n_tiling_params+i])
        for i in range(3):
            data.append(2.0 * item[n_tiling_params+i+3])
        
        label = item[n_tiling_params+6:n_tiling_params+13]
        
        all_data.append(data)
        all_label.append(label)
        
    print("#valid data:{}".format(len(all_data)))
    # exit(0)
    start = 0
    end = -1
    all_data = np.array(all_data[start:end]).astype(np.float64)
    all_label = np.array(all_label[start:end]).astype(np.float64) 
    
    # all_data = np.array(all_data).astype(np.float64)
    # all_label = np.array(all_label).astype(np.float64)
    
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    
    all_data = all_data[indices]
    all_label = all_label[indices]
    
    return all_data, all_label


w_grad = tf.constant(1.0, dtype=tf.float64)
w_e = tf.constant(1.0, dtype=tf.float64)

def generator(train_data, train_label):    
    indices = np.arange(train_data.shape[0])
    while True:
        np.random.shuffle(indices)
        yield train_data[indices], train_label[indices]

@tf.function
def trainStep(n_tiling_params, opt, lambdas, sigmas, model, train_vars):
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(train_vars)
        tape.watch(lambdas)
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 6])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 6])
        
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
def testStep(n_tiling_params, lambdas, sigmas, model):
    
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 6])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 6])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, psi)
    del tape
    return grad_loss, e_loss, stress_pred, psi

def validate(n_tiling_params, count, model_name, validation_data, validation_label):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + model_name + '.tf')
    # model.save(save_path + model_name + '.h5')
    grad_loss, e_loss, sigma, energy = testStep(n_tiling_params,validation_data, validation_label, model)
    
    # plotPotentialClean(save_path, n_tiling_params, validation_data, validation_label, model)
    # plot(save_path + model_name + "_validation", sigma.numpy(), validation_label, False)

    print("validation loss grad: {} energy: {}".format(grad_loss, e_loss))

def train(n_tiling_params, model_name, train_data, train_label, validation_data, validation_label):
    batch_size = np.minimum(10000, len(train_data))
    print("batch size: {}".format(batch_size))
    # model = buildSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParamsSwish3D(n_tiling_params)
    
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-4)
    max_iter = 80000

    val_lambdasTF = tf.convert_to_tensor(validation_data)
    val_sigmasTF = tf.convert_to_tensor(validation_label)

    losses = [[], []]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # model.load_weights("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/python/Models/Toy/" + model_name + '.tf')
    count = 0
    with open('counter.txt', 'r') as f:
        count = int(f.read().splitlines()[-1])
    f = open("counter.txt", "w+")
    f.write(str(count+1))
    f.close()
    summary = Summary("./Logs/" + str(count) + "/")
    
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    g_norm0 = 0
    iter = 0
    log_txt = open(save_path + "/log.txt", "w+")
    for iteration in range(max_iter):
        lambdas, sigmas = next(generator(train_data, train_label))
        if batch_size == -1:
            batch = 1
        else:
            batch = int(np.floor(len(lambdas) / batch_size))
        
        train_loss_grad = 0.0
        train_loss_e = 0.0
        g_norm_sum = 0.0
        for i in range(batch):
            mini_bacth_lambdas = lambdas[i * batch_size:(i+1) * batch_size]
            mini_bacth_sigmas = sigmas[i * batch_size:(i+1) * batch_size]

            lambdasTF = tf.convert_to_tensor(mini_bacth_lambdas)
            sigmasTF = tf.convert_to_tensor(mini_bacth_sigmas)
            
            grad, e, g_norm = trainStep(n_tiling_params, opt, lambdasTF, sigmasTF, model, train_vars)
            
            train_loss_grad += grad
            train_loss_e += e
            g_norm_sum += g_norm
        if (iteration == 0):
            g_norm0 = g_norm_sum
        validation_loss_grad, validation_loss_e, _, _ = testStep(n_tiling_params, val_lambdasTF, val_sigmasTF, model)
        
        losses[0].append(train_loss_grad + train_loss_e)
        losses[1].append(validation_loss_grad + validation_loss_e)
        
        log_txt.write("iter " + str(iteration) + " " + str(train_loss_grad.numpy()) + " " + str(train_loss_e.numpy()) + " " + str(validation_loss_grad.numpy()) + " " + str(validation_loss_e.numpy()) + " " + str(g_norm_sum.numpy()) + "\n")
        print("epoch: {}/{} train_loss_grad: {} train_loss e: {}, validation_loss_grad:{} loss_e:{} |g|: {}, |g_init|: {} ".format(iteration, max_iter, train_loss_grad, train_loss_e, \
                         validation_loss_grad, validation_loss_e, \
                        g_norm_sum, g_norm0))
        summary.saveToTensorboard(train_loss_grad, train_loss_e, validation_loss_grad, validation_loss_e, iteration)
        if iteration % 500 ==0:
            model.save_weights(save_path + model_name + '.tf')

    
    
    model.save_weights(save_path + model_name + '.tf')
    # fourier_B = model.get_config()['layers'][-1]['config']['B']
    # np.reshape(fourier_B,-1).astype(float).tofile(os.path.join(save_path, model_name + "B.dat"))
    idx = [i for i in range(len(losses[0]))]
    plt.plot(idx, losses[0], label = "train_loss")
    plt.plot(idx, losses[1], label = "validation_loss")
    plt.legend(loc="upper left")
    plt.savefig(save_path + model_name + "_log.png", dpi = 300)
    plt.close()


if __name__ == "__main__":
    n_tiling_params = 2
    
    full_data = "/home/yueli/Documents/ETH/SandwichStructure/ServerToy3D/uniaxial_data_toy3D_shuffled.txt"  

    
    data_all, label_all = loadDataSplitTest(n_tiling_params, full_data, False, True)
    

    five_percent = int(len(data_all) * 0.05)
    if (len(data_all) < 100):
        five_percent = 1
    
    
    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]

    model_name = "Toy"
        
    train(n_tiling_params, model_name, 
        train_data, train_label, validation_data, validation_label)
    # validate(n_tiling_params, 334, 
    #     model_name, train_data, train_label)
    
    