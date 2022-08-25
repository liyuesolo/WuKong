
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

# def loadData(filename, shuffle = True):
#     all_data = []
#     all_label = [] 
#     all_energy = []
#     for line in open(filename).readlines():
#         item = [float(i) for i in line.strip().split(" ")[2:]]
#         data = [item[0], item[1]]
#         label = [item[2], item[3]]
#         all_energy.append(item[-1])
#         all_data.append(data)
#         all_label.append(label)
    
#     def numeric_compare(x, y):
#         return all_energy[x] - all_energy[y]
#     indices = [i for i in range(len(all_energy))]
#     indices = sorted(indices, key=cmp_to_key(numeric_compare))

#     all_data = np.array(all_data).astype(np.float32)
#     all_label = np.array(all_label).astype(np.float32)
#     all_energy = np.array(all_energy).astype(np.float32)
#     all_data = all_data[indices]
#     all_label = all_label[indices]
#     all_energy = all_energy[indices]

#     if (shuffle):
#         indices = np.arange(all_data.shape[0])
#         np.random.shuffle(indices)
#         all_data = all_data[indices]
#         all_label = all_label[indices]

#     return all_data, all_label, all_energy

def loadDataSplitTest(filename):
    all_data = []
    all_label = [] 
    all_energy = []
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[2:]]
        data = [item[0], item[1]]
        label = [item[2], item[3]]
        # label = [ item[2] + 5.0 * item[2] * (2.0 * np.random.rand() - 1),  
        #     item[3] + 5.0 * item[3] * (2.0 * np.random.rand() - 1)]
        all_energy.append(item[-1])
        all_data.append(data)
        all_label.append(label)
    
    all_data = np.array(all_data).astype(np.float32)
    all_label = np.array(all_label).astype(np.float32)
    all_energy = np.array(all_energy).astype(np.float32)

    return all_data, all_label, all_energy

data_all, label_all, energy_all = loadDataSplitTest("data.txt")

five_percent = int(len(data_all) * 0.05)

train_data =  data_all[:-five_percent]
train_label =  label_all[:-five_percent]

test_data = data_all[-five_percent:]
test_label = label_all[-five_percent:]
test_energy = energy_all[-five_percent:]

def numeric_compare(x, y):
    return test_energy[x] - test_energy[y]
indices = [i for i in range(len(test_energy))]
indices = sorted(indices, key=cmp_to_key(numeric_compare))
test_label = test_label[indices]
test_data = test_data[indices]
test_energy = test_energy[indices]

def generator():    
    indices = np.arange(train_data.shape[0])
    while True:
        np.random.shuffle(indices)
        yield train_data[indices], train_label[indices]

@tf.function
def trainStep(opt, lambdas, sigmas, model, train_vars):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(lambdas)
        elastic_potential = model(lambdas)
        dedlambda = tape.gradient(elastic_potential, lambdas)
        l2Loss = tf.keras.losses.MeanSquaredError()
        loss = l2Loss(dedlambda, sigmas)
    dLdw = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(dLdw, train_vars))
    del tape
    return loss

@tf.function
def testStep(lambdas, sigmas, model):
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        elastic_potential = model(lambdas)
        dedlambda = tape.gradient(elastic_potential, lambdas)
        l2Loss = tf.keras.losses.MeanSquaredError()
        loss = l2Loss(dedlambda, sigmas)
    return loss, dedlambda, elastic_potential
    
# @tf.function
def testGradientHessianStep(lambdas, sigmas, model):
    lambdasTF = tf.convert_to_tensor(lambdas)
    with tf.GradientTape() as t2:
        t2.watch(lambdasTF)
        with tf.GradientTape() as t1:
            t1.watch(lambdasTF)
            elastic_potential = model(lambdasTF)
            dedlambda = t1.gradient(elastic_potential, lambdasTF)
    d2edlambda2 = t2.jacobian(dedlambda, lambdasTF)
    print("gradient gt")
    print(sigmas[0])
    print("gradient")
    print(dedlambda.numpy())
    print("hessian gt")
    print([lambdas[0][0] + 2.0 * lambdas[0][1], 0, 0, 0])
    print("hessian predict")
    print(d2edlambda2.numpy().flatten())
    
    return dedlambda, d2edlambda2

def fdGradient(lambdas, model):
    lambdasTF = tf.convert_to_tensor(lambdas)
    with tf.GradientTape() as t1:
        t1.watch(lambdasTF)
        E0 = model(lambdasTF)
    dedlambda = t1.gradient(E0, lambdasTF)
    eps = 0.0001
    lambdas[0][0] = lambdas[0][0] + eps
    lambdasTF = tf.convert_to_tensor(lambdas)
    E1 = model(lambdasTF)
    lambdas[0][0] = lambdas[0][0] - 2.0 * eps
    lambdasTF = tf.convert_to_tensor(lambdas)
    E2 = model(lambdasTF)
    print((E1 - E2) / 2.0 / eps, dedlambda[0])


def train():
    
    inputS = Input(shape=(2,),dtype=tf.float32, name="inputS")
    output = ConstitutiveModel()(inputS)
    model = Model(inputS, output)
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-2)
    max_iter = 10000

    for iteration in range(max_iter):
        lambdas, sigmas = next(generator())
        lambdasTF = tf.convert_to_tensor(lambdas)
        sigmasTF = tf.convert_to_tensor(sigmas)

        loss = trainStep(opt, lambdasTF, sigmasTF, model, train_vars)
        print("iter: {}/{} loss: {}".format(iteration, max_iter, loss))
    
    model.save_weights('cons_model.tf')
    
    # test_data, test_label, test_energy = loadData("data.txt", False)
    test_error, dedlambda, elastic_potential = testStep(test_data, test_label, model)
    data_point = [i for i in range(len(test_data))]
    e_gt = [test_energy[i] for i in range(len(test_data))]
    e = [elastic_potential[i] for i in range(len(test_data))]
    # stress_gt = [test_label[i][0] for i in range(len(data_all))]
    # nu_gt = [test_label[i][1] for i in range(len(data_all))]
    # stress = dedlambda.numpy()[:, 0].reshape(-1)
    # nu = dedlambda.numpy()[:, 1].reshape(-1)
    # plt.plot(data_point, stress_gt, linewidth=1.5, label = "GT sigma")
    # plt.plot(data_point, stress, linewidth=1.5, label = "prediction sigma")
    # plt.plot(data_point, nu_gt, linewidth=1.5, label = "GT nu")
    # plt.plot(data_point, nu, linewidth=1.5, label = "prediction nu")
    plt.title("StVK test data")
    plt.plot(data_point, e, linewidth=1.5, label = "Energy")
    plt.plot(data_point, e_gt, linewidth=1.5, label = "GT Energy")
    plt.legend(loc="upper left")
    plt.savefig("learned.png", dpi = 300)
    plt.close()

def test():
    inputS = Input(shape=(2,),dtype=tf.float32, name="inputS")
    output = ConstitutiveModel()(inputS)
    model = Model(inputS, output)
    # model.load_weights('cons_model.tf')
    model.load_weights('cons_model.tf')
    # tf.keras.models.save_model(model, 'cons_model.h5')
    # test_data, test_label, test_energy = loadData("data.txt", False)
    # test_data = tf.convert_to_tensor(test_data)
    # test_label = tf.convert_to_tensor(test_label)
    # print(test_data[0:1, :].shape)
    test_error, dedlambda, elastic_potential = testStep(test_data, test_label, model)
    # test_error, dedlambda, elastic_potential = testStep(test_data, test_label, model)
    g, h = testGradientHessianStep(test_data[0:1, :], test_label, model)
    # fdGradient(test_data[0:1, :], model)
    print("test error: {}".format(test_error))
    data_point = [i for i in range(len(test_data))]
    e_gt = [test_energy[i] for i in range(len(test_data))]
    e = [elastic_potential[i] for i in range(len(test_data))]
    
    plt.title("StVK test data")
    plt.plot(data_point, e, linewidth=1.5, label = "Energy")
    plt.plot(data_point, e_gt, linewidth=1.5, label = "GT Energy")
    plt.legend(loc="upper left")
    plt.savefig("learned.png", dpi = 300)
    plt.close()


# def trainNewModel():
#     model = buildConstitutiveModel()
#     train_vars = model.trainable_variables
#     opt = Adam(learning_rate=1e-2)
#     max_iter = 2

#     for iteration in range(max_iter):
#         lambdas, sigmas = next(generator())
#         lambdasTF = tf.convert_to_tensor(lambdas)
#         sigmasTF = tf.convert_to_tensor(sigmas)

#         loss = trainStep(opt, lambdasTF, sigmasTF, model, train_vars)
#         print("iter: {}/{} loss: {}".format(iteration, max_iter, loss))
    
#     model.save('cons_model.h5')
#     model = load_model('cons_model.h5', custom_objects={'sin_activation': sin_activation, 
#         'sin_activation_first_layer' : sin_activation_first_layer})

if __name__ == "__main__":
    # trainNewModel()
    # train()
    test()