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
from sklearn.preprocessing import StandardScaler



def loadDataSplitTest(filename):
    all_data = []
    all_label = [] 
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        # if (item[-1] > 1e-5):
        #     continue
        data = [item[4], item[7], item[5]]
        label = [item[8], item[11], item[9]]

        all_data.append(data)
        all_label.append(label)

    
    # random5k = np.arange(len(all_data))
    # np.random.shuffle(random5k)
    # random5k = random5k[:5000]

    # small strain
    # all_data = np.array(all_data[3500:4000]).astype(np.float32)
    # all_label = np.array(all_label[3500:4000]).astype(np.float32)

    # large
    # all_data = np.array(all_data[6000:7000]).astype(np.float32)
    # all_label = np.array(all_label[6000:7000]).astype(np.float32)

    all_data = np.array(all_data).astype(np.float32)
    all_label = np.array(all_label).astype(np.float32)
    indices = np.arange(all_data.shape[0])
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
def trainStep(opt, lambdas, sigmas, model, train_vars):
    rest = tf.convert_to_tensor(np.array([[0.0, 0.0, 0.0]]).astype(np.float32))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(lambdas)
        elastic_potential = model(lambdas)
        tape.watch(rest)
        stress = tape.gradient(elastic_potential, lambdas)
        l2Loss = tf.keras.losses.MeanSquaredError()
        loss = l2Loss(stress, sigmas)
        
        elastic_potential_reset = model(rest)
        dirichlet_loss = l2Loss(elastic_potential_reset, tf.convert_to_tensor(np.tile(np.array([0.0]).astype(np.float32), (rest.shape[0], 1))))
        loss += tf.constant(1.0, dtype=tf.float32) * dirichlet_loss
        
    dLdw = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(dLdw, train_vars))
    del tape
    return loss

@tf.function
def testStep(lambdas, sigmas, model):
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        elastic_potential = model(lambdas, training=False)
        dedlambda = tape.gradient(elastic_potential, lambdas)
        l2Loss = tf.keras.losses.MeanSquaredError()
        loss = l2Loss(dedlambda, sigmas)
    return loss, dedlambda, elastic_potential

def test(test_data, test_label, scaler_data, scaler_label):
    model = buildSrainStressModel()
    model.load_weights('strain_stress_model.tf')
    test_loss, sigma, energy = testStep(test_data, test_label, model)
    
    # sigma = scaler_label.inverse_transform(sigma)
    # test_label = scaler_label.inverse_transform(test_label)
    data_point = [np.linalg.norm(test_data[i]) for i in range(len(test_data))]
    
    sigma_xx_gt = [test_label[i][0] for i in range(len(test_data))]
    sigma_xx = [sigma[i][0] for i in range(len(test_data))]

    sigma_yy_gt = [test_label[i][1] for i in range(len(test_data))]
    sigma_yy = [sigma[i][1] for i in range(len(test_data))]

    sigma_xy_gt = [test_label[i][2] for i in range(len(test_data))]
    sigma_xy = [sigma[i][2] for i in range(len(test_data))]
    
    plt.plot(data_point, sigma_xx, linewidth=1.0, label = "Sigma_xx")
    plt.plot(data_point, sigma_xx_gt, linewidth=1.0, label = "GT Sigma_xx")
    plt.legend(loc="upper left")
    plt.savefig("learned_sigma_xx.png", dpi = 300)
    plt.close()
    plt.plot(data_point, sigma_yy, linewidth=1.0, label = "Sigma_yy")
    plt.plot(data_point, sigma_yy_gt, linewidth=1.0, label = "GT Sigma_yy")
    plt.legend(loc="upper left")
    plt.savefig("learned_sigma_yy.png", dpi = 300)
    plt.close()
    plt.plot(data_point, sigma_xy, linewidth=1.0, label = "Sigma_xy")
    plt.plot(data_point, sigma_xy_gt, linewidth=1.0, label = "GT Sigma_xy")
    plt.legend(loc="upper left")
    plt.savefig("learned_sigma_xy.png", dpi = 300)
    plt.close()

    print(test_loss)

def train(train_data, train_label):
    
    model = buildSrainStressModel()
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-3)
    max_iter = 40000

    for iteration in range(max_iter):
        lambdas, sigmas = next(generator(train_data, train_label))
        lambdasTF = tf.convert_to_tensor(lambdas)
        sigmasTF = tf.convert_to_tensor(sigmas)
        
        loss = trainStep(opt, lambdasTF, sigmasTF, model, train_vars)
        print("iter: {}/{} loss: {}".format(iteration, max_iter, loss))
    
    model.save_weights('strain_stress_model.tf')

    

def testOrthogonal(test_data, test_label):
    
    test_dataTF = tf.convert_to_tensor(test_data)
    test_labelTF = tf.convert_to_tensor(test_label)
    model = buildSrainStressModel()
    model.load_weights('tiling_model.tf')
    
    test_loss, sigma, energy = testStep(test_dataTF, test_labelTF, model)
    print("test_lost", test_loss)
    return sigma
    
    


if __name__ == "__main__":
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialStrain/"
    data_all, label_all = loadDataSplitTest(data_folder + "training_data_test.txt")
    # data_folder = "combine.txt"
    # data_all, label_all = loadDataSplitTest(data_folder)

    normalize = False
    scaler_data = StandardScaler()
    scaler_label = StandardScaler()


    if normalize:
        data_all = scaler_data.fit_transform(data_all)
        label_all = scaler_label.fit_transform(label_all)

    five_percent = int(len(data_all) * 0.1)

    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]

    
    # test_energy = energy_all[-five_percent:]

    def numeric_compare(x, y):
        return np.linalg.norm(validation_data[x]) - np.linalg.norm(validation_data[y])
        # return test_label[x][0] - test_label[y][0]

    indices = [i for i in range(len(validation_data))]
    indices = sorted(indices, key=cmp_to_key(numeric_compare))
    validation_label = validation_label[indices]

    validation_data = validation_data[indices]

    # test_data = []
    # test_label = [] 
    # strain_percents = []
    # for line in open("strain_stress.txt").readlines():
    #     item = [float(i) for i in line.strip().split(" ")[:]]
    #     strain_percent = [item[-1]]
    #     # data = [0.122398, 0.5, 0.143395, 0.625, item[1], item[3], item[2]]
    #     # label = [item[4], item[6], item[5]]

    #     data = [item[0], item[1], item[2], item[3], item[4], item[7], item[5]]
    #     label = [item[8], item[11], item[9]]

    #     # data = [0.122398, 0.5, 0.143395, 0.625, -0.2, 0.0654496, 4.29659e-06]
    #     # label = [-0.985659, 1.64334e-06, 6.85241e-17]
        
    #     test_data.append(data)
    #     test_label.append(label)
    #     strain_percents.append(strain_percent)
    #     # break

    
    # test_data = np.array(test_data).astype(np.float32)
    # test_label = np.array(test_label).astype(np.float32)
    # if normalize:
    #     test_data = scaler_data.transform(test_data)
    #     test_label = scaler_label.transform(test_label)
    
    train(train_data, train_label)
    test(validation_data, validation_label, scaler_data, scaler_label)
    # sigma = testOrthogonal(test_data, test_label)

    
    # sigma = scaler_label.inverse_transform(sigma)
    # test_label = scaler_label.inverse_transform(test_label)
    # print(np.linalg.norm(test_label - sigma, 2)/float(len(test_label)))

    # for i in range(len(sigma)):
    #     print(sigma[i], " ", test_label[i], " ", strain_percents[i], " ", np.linalg.norm(test_label[i] - sigma[i]))
    # ortho_dir = np.array([-np.sin(0.0), np.cos(0.0)])
    # for i in range(len(sigma)):
    #     stress = np.array([[sigma[i][0], sigma[i][2]],[sigma[i][2], sigma[i][1]]])
    #     stress_gt = np.array([[test_label[i][0], test_label[i][2]],[test_label[i][2], test_label[i][1]]])
    #     test_dot = np.dot(stress, ortho_dir)
    #     gt_dot = np.dot(stress_gt, ortho_dir)
    #     print("dot test: {} dot gt: {}".format(test_dot, gt_dot))
    # test()