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
from sklearn.preprocessing import MinMaxScaler


def loadDataSplitTest(filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        # if (ignore_unconverging_result):
        #     if (item[-1] > 1e-5):
        #         continue
        data = [item[0], item[1], item[2], item[3], item[4], item[7], item[5]]
        label = [item[8], item[11], item[9]]
                
        all_data.append(data)
        all_label.append(label)
    # all_data = all_data[3400:]
    # all_label = all_label[3400:]
    all_data = np.array(all_data).astype(np.float32)
    all_label = np.array(all_label).astype(np.float32)
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    all_data = all_data[indices]
    all_label = all_label[indices]

    return all_data, all_label

def loadUniaxialDataSplitTest(filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        # if (ignore_unconverging_result):
        #     if (item[-1] > 1e-5):
        #         continue
        # cauchy strain
        data = [item[0], item[1], item[2], item[3], item[6], item[9], item[7]]

        # green strain
        # data = [item[0], item[1], item[2], item[3], item[10], item[13], item[11]]
        
        label = [item[14], item[17], item[15]]
                
        all_data.append(data)
        all_label.append(label)
    
    all_data = np.array(all_data).astype(np.float32)
    all_label = np.array(all_label).astype(np.float32)
    
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    all_label = all_label[indices]

    return all_data, all_label

def loadBiaxialDataSplitTest(filename):
    all_data = []
    all_label = [] 
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        if (item[-1] > 1e-5):
            continue
        # cauchy strain
        # data = [item[0], item[1], item[2], item[3], item[7], item[10], item[8]]

        # green strain
        data = [item[0], item[1], item[2], item[3], item[11], item[14], item[12]]
        
        label = [item[15], item[18], item[16]]
                
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
        # rest_stress = tape.gradient(elastic_potential_reset, rest_configuration)[:,4:]
        # rest_stress_loss = l2Loss(rest_stress, rest)
        loss += tf.constant(1.0, dtype=tf.float32) * dirichlet_loss
        # loss += tf.constant(0.0, dtype=tf.float32) * rest_stress_loss
        
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

def plot(prefix, prediction, label):
    def cmp_sigma_xx(i, j):
        return label[i][0] - label[j][0]
    def cmp_sigma_xy(i, j):
        return label[i][2] - label[j][2]
    def cmp_sigma_yy(i, j):
        return label[i][1] - label[j][1]
        
    indices = [i for i in range(len(label))]
    data_point = [i for i in range(len(label))]

    indices = sorted(indices, key=cmp_to_key(cmp_sigma_xx))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_xx_gt = [sigma_gt_sorted[i][0] for i in range(len(label))]
    sigma_xx = [sigma_sorted[i][0] for i in range(len(label))]
    plt.plot(data_point, sigma_xx, linewidth=1.0, label = "Sigma_xx")
    plt.plot(data_point, sigma_xx_gt, linewidth=1.0, label = "GT Sigma_xx")
    plt.legend(loc="upper left")
    plt.savefig(prefix+"_learned_sigma_xx.png", dpi = 300)
    plt.close()
    
    indices = sorted(indices, key=cmp_to_key(cmp_sigma_yy))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_yy_gt = [sigma_gt_sorted[i][1] for i in range(len(label))]
    sigma_yy = [sigma_sorted[i][1] for i in range(len(label))]
    plt.plot(data_point, sigma_yy, linewidth=1.0, label = "Sigma_yy")
    plt.plot(data_point, sigma_yy_gt, linewidth=1.0, label = "GT Sigma_yy")
    plt.legend(loc="upper left")
    plt.savefig(prefix + "_learned_sigma_yy.png", dpi = 300)
    plt.close()

    indices = sorted(indices, key=cmp_to_key(cmp_sigma_xy))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_xy_gt = [sigma_gt_sorted[i][2] for i in range(len(label))]
    sigma_xy = [sigma_sorted[i][2] for i in range(len(label))]
    plt.plot(data_point, sigma_xy, linewidth=1.0, label = "Sigma_xy")
    plt.plot(data_point, sigma_xy_gt, linewidth=1.0, label = "GT Sigma_xy")
    plt.legend(loc="upper left")
    plt.savefig(prefix + "_learned_sigma_xy.png", dpi = 300)
    plt.close()

def plotPotentialPolar(model_name):
    uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialStrain_backup/training_data.txt"
    all_data, all_label = loadDataSplitTest(uniaxial_data, False, False)
    n_sp_per_strain = 100
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights(model_name + '.tf')
    test_loss, sigma, energy = testStep(all_data, all_label, model)
    theta = np.arange(0.0, np.pi, np.pi/float(n_sp_per_strain))
    # potential = energy.numpy()[3400:3500]
    potential = energy.numpy()[4000:4100]
    # potential = energy.numpy()[600:700]
    n_pt = len(energy)
    for i in range(n_sp_per_strain):
        theta = np.append(theta, theta[i] - np.pi)
        potential = np.append(potential, potential[i])
    theta = np.append(theta, theta[0])
    potential = np.append(potential, potential[0])
    plt.polar(theta, potential)
    plt.savefig(model_name + "10percent_energy.png", dpi = 300)
    plt.close()

def plotStressPolar(model_name):
    all_data, all_label = loadDataSplitTest(uniaxial_data, False, False)
    n_sp_per_strain = 100
    theta = np.arange(-np.pi, np.pi, 2.0 * np.pi/float(n_sp_per_strain))
    # theta = np.append(theta, theta[0])
    # potential = energy.numpy()[3400:3500]
    stress_norm = [np.linalg.norm(all_label[i]) for i in range(4000, 4100)]
    # potential = np.append(potential, potential[0])
    plt.polar(theta, stress_norm)
    plt.savefig(model_name + "stress_norm.png", dpi = 300)
    plt.close()

def plotPotential(model_name):
    uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialStrain_backup/training_data.txt"
    all_data, all_label = loadDataSplitTest(uniaxial_data, False, False)
    n_sp_per_strain = 100
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights(model_name + '.tf')
    test_loss, sigma, energy = testStep(all_data, all_label, model)
    potential = np.squeeze(energy.numpy()[3400:7400:n_sp_per_strain])
    strain = [i for i in range(len(potential))]
    plt.plot(strain, potential)
    plt.savefig(model_name + "strain_energy.png", dpi = 300)
    plt.close()



def validate(model_name, validation_data, validation_label):
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights(model_name + '.tf')
    test_loss, sigma, energy = testStep(validation_data, validation_label, model)
    
    plot(model_name + "_validation", sigma.numpy(), validation_label)

    print("validation loss", test_loss)

def train(model_name, train_data, train_label, validation_data, validation_label):
    save_path = "./"
    # B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    B = None
    model = buildSingleFamilyModel(4, B)
    
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-4)
    max_iter = 100000

    val_lambdasTF = tf.convert_to_tensor(validation_data)
    val_sigmasTF = tf.convert_to_tensor(validation_label)

    losses = [[], []]
    for iteration in range(max_iter):
        lambdas, sigmas = next(generator(train_data, train_label))
        
        lambdasTF = tf.convert_to_tensor(lambdas)
        sigmasTF = tf.convert_to_tensor(sigmas)
        
        
        train_loss = trainStep(opt, lambdasTF, sigmasTF, model, train_vars)
        validation_loss, _, _ = testStep(val_lambdasTF, val_sigmasTF, model)
        losses[0].append(train_loss)
        losses[1].append(validation_loss)
        print("iter: {}/{} train_loss: {} validation_loss:{} ".format(iteration, max_iter, train_loss, validation_loss))
    
    
    model.save_weights(save_path + model_name + '.tf')
    fourier_B = model.get_config()['layers'][-1]['config']['B']
    np.reshape(fourier_B,-1).astype(float).tofile(os.path.join(save_path, model_name + "B.dat"))
    idx = [i for i in range(len(losses[0]))]
    plt.plot(idx, losses[0], label = "train_loss")
    plt.plot(idx, losses[1], label = "validation_loss")
    plt.legend(loc="upper left")
    plt.savefig(model_name + "_log.png", dpi = 300)
    plt.close()

def test(model_name, test_data, test_label):
    
    test_dataTF = tf.convert_to_tensor(test_data)
    test_labelTF = tf.convert_to_tensor(test_label)
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, model_name+"B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights(model_name+'.tf')
    
    test_loss, sigma, energy = testStep(test_dataTF, test_labelTF, model)
    plot(model_name + "_test", sigma.numpy(), test_label)
    print("test_lost", test_loss)
    return sigma
    
    
if __name__ == "__main__":
    train_uniaxial = True
    train_both = False
    # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialStrain_backup/training_data.txt"
    uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data.txt"
    
    # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialDense/training_data.txt"
    biaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyBiaxialStrain/biaxial_training_data.txt"
    uni_test = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialStrain/training_data.txt"
    bi_test = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyBiaxialStrainTest/training_data.txt"
    
    if train_uniaxial:
        data_all, label_all = loadUniaxialDataSplitTest(uniaxial_data)
        # data_all, label_all = loadDataSplitTest(uniaxial_data)
    else:
        data_all, label_all = loadBiaxialDataSplitTest(biaxial_data)
    
    if train_both:
        uni_data, uni_label = loadUniaxialDataSplitTest(uniaxial_data)
        bi_data, bi_label = loadBiaxialDataSplitTest(biaxial_data)
        data_all = np.vstack((uni_data, bi_data))
        label_all = np.vstack((uni_label, bi_label))


    five_percent = int(len(data_all) * 0.05)

    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]

    test_data = []
    test_label = [] 
    if train_uniaxial:
        test_data, test_label = loadBiaxialDataSplitTest(bi_test)
    else:
        for line in open(uni_test).readlines():
            
            item = [float(i) for i in line.strip().split(" ")[:]]
            
            # data = [item[0], item[1], item[2], item[3], item[4], item[7], item[5]]
            # data = [item[0], item[1], item[2], item[3], item[4], item[7], item[5]]
            data = [item[0], item[1], item[2], item[3], item[10], item[13], item[11]]
            
            label = [item[14], item[17], item[15]]
            
            test_data.append(data)
            test_label.append(label)

    test_data = np.array(test_data).astype(np.float32)
    test_label = np.array(test_label).astype(np.float32)
    if train_uniaxial:
        model_name = "uniaxial"
    else:
        model_name = "biaxial"
    # if train_both:
    #     model_name = "full"
    train(model_name, train_data, train_label, validation_data, validation_label)
    validate(model_name, validation_data, validation_label)
    # sigma = test(model_name, test_data, test_label)
    # plotPotential(model_name)
    plotPotentialPolar(model_name)
    # plotStressPolar(model_name)


    # for i in range(len(sigma)):
    #     print(sigma[i], " ", test_label[i], " ", strain_percents[i], " ", np.linalg.norm(test_label[i] - sigma[i]))
    # ortho_dir = np.array([-np.sin(0.0), np.cos(0.0)])
    # for i in range(len(sigma)):
    #     stress = np.array([[sigma[i][0], sigma[i][2]],[sigma[i][2], sigma[i][1]]])
    #     stress_gt = np.array([[test_label[i][0], test_label[i][2]],[test_label[i][2], test_label[i][1]]])
    #     test_dot = np.dot(stresss, ortho_dir)
    #     gt_dot = np.dot(stress_gt, ortho_dir)
    #     print("dot test: {} dot gt: {}".format(test_dot, gt_dot))
    