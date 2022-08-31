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
        if (item[-1] > 1e-5):
            continue
        data = [item[0], item[1], item[2], item[3], item[4], item[7], item[5]]
        label = [item[8], item[11], item[9]]
                
        all_data.append(data)
        all_label.append(label)

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
    rest = tf.convert_to_tensor(np.tile(np.array([0.0, 0.0, 0.0]).astype(np.float32), (lambdas.shape[0], 1)))
    tiling_param = lambdas[:, :4]
    rest_configuration = tf.concat((tiling_param, rest), 1)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(lambdas)
        elastic_potential = model(lambdas)
        tape.watch(rest_configuration)
        stress = tape.gradient(elastic_potential, lambdas)[:,4:]
        l2Loss = tf.keras.losses.MeanSquaredError()
        loss = l2Loss(stress, sigmas)
        
        elastic_potential_reset = model(rest_configuration)
        dirichlet_loss = l2Loss(elastic_potential_reset, tf.convert_to_tensor(np.tile(np.array([0.0]).astype(np.float32), (rest_configuration.shape[0], 1))))
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
        dedlambda = tape.gradient(elastic_potential, lambdas)[:,4:]
        l2Loss = tf.keras.losses.MeanSquaredError()
        loss = l2Loss(dedlambda, sigmas)
    return loss, dedlambda, elastic_potential

def test(test_data, test_label):
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights('tiling_model.tf')
    test_loss, sigma, energy = testStep(test_data, test_label, model)
    

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

    print("validation loss", test_loss)

def train(train_data, train_label, validation_data, validation_label):
    
    model = buildSingleFamilyModel(4)
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-4)
    max_iter = 40000

    val_lambdasTF = tf.convert_to_tensor(validation_data)
    val_sigmasTF = tf.convert_to_tensor(validation_label)

    for iteration in range(max_iter):
        lambdas, sigmas = next(generator(train_data, train_label))
        
        lambdasTF = tf.convert_to_tensor(lambdas)
        sigmasTF = tf.convert_to_tensor(sigmas)
        
        
        train_loss = trainStep(opt, lambdasTF, sigmasTF, model, train_vars)
        validation_loss, _, _ = testStep(val_lambdasTF, val_sigmasTF, model)
        print("iter: {}/{} train_loss: {} validation_loss:{} ".format(iteration, max_iter, train_loss, validation_loss))
    
    save_path = "./"
    model.save_weights(save_path + 'tiling_model.tf')
    fourier_B = model.get_config()['layers'][-1]['config']['B']
    np.reshape(fourier_B,-1).astype(float).tofile(os.path.join(save_path, "B.dat"))

def testOrthogonal(test_data, test_label):
    
    test_dataTF = tf.convert_to_tensor(test_data)
    test_labelTF = tf.convert_to_tensor(test_label)
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights('tiling_model.tf')
    
    test_loss, sigma, energy = testStep(test_dataTF, test_labelTF, model)
    print("test_lost", test_loss)
    return sigma
    
    
if __name__ == "__main__":
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialStrain/"

    data_all, label_all = loadDataSplitTest(data_folder + "training_data_test.txt")

    five_percent = int(len(data_all) * 0.05)

    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]
    

    def numeric_compare(x, y):
        return np.linalg.norm(validation_data[x]) - np.linalg.norm(validation_data[y])
        # return test_label[x][0] - test_label[y][0]

    indices = [i for i in range(len(validation_data))]
    indices = sorted(indices, key=cmp_to_key(numeric_compare))
    validation_label = validation_label[indices]

    validation_data = validation_data[indices]

    test_data = []
    test_label = [] 
    strain_percents = []
    for line in open("strain_stress.txt").readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        strain_percent = [item[-1]]

        data = [item[0], item[1], item[2], item[3], item[4], item[7], item[5]]
        label = [item[8], item[11], item[9]]
        
        test_data.append(data)
        test_label.append(label)
        strain_percents.append(strain_percent)

    test_data = np.array(test_data).astype(np.float32)
    test_label = np.array(test_label).astype(np.float32)

    
    train(train_data, train_label, validation_data, validation_label)
    test(validation_data, validation_label)
    sigma = testOrthogonal(test_data, test_label)


    for i in range(len(sigma)):
        print(sigma[i], " ", test_label[i], " ", strain_percents[i], " ", np.linalg.norm(test_label[i] - sigma[i]))
    ortho_dir = np.array([-np.sin(0.0), np.cos(0.0)])
    for i in range(len(sigma)):
        stress = np.array([[sigma[i][0], sigma[i][2]],[sigma[i][2], sigma[i][1]]])
        stress_gt = np.array([[test_label[i][0], test_label[i][2]],[test_label[i][2], test_label[i][1]]])
        test_dot = np.dot(stress, ortho_dir)
        gt_dot = np.dot(stress_gt, ortho_dir)
        print("dot test: {} dot gt: {}".format(test_dot, gt_dot))
    