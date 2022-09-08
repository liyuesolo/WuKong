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
import keras.backend as K

def relativeL2(y_true, y_pred):
    if (y_true.shape[1] > 1):
        stress_norm = tf.norm(y_true, ord='euclidean', axis=1)
        norm = tf.tile(tf.keras.backend.expand_dims(stress_norm, 1), tf.constant([1, 4]))
        y_true_normalized = tf.divide(y_true, norm + K.epsilon())
        y_pred_normalized = tf.divide(y_pred, norm + K.epsilon())
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
    else:
        y_true_normalized = tf.divide(y_true, y_true + K.epsilon())
        y_pred_normalized = tf.divide(y_pred, y_true + K.epsilon())
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
        

def absL2(y_true, y_pred):
    sigma_xx = K.mean(K.square((y_true[:, 0] - y_pred[:, 0])))
    sigma_yy = K.mean(K.square((y_true[:, 1] - y_pred[:, 1])))
    sigma_xy = K.mean(K.square((y_true[:, 2] - y_pred[:, 2])))
    sigma_yx = K.mean(K.square((y_true[:, 3] - y_pred[:, 3])))
    
    return sigma_xx + sigma_yy + sigma_xy + sigma_yx

def loadDataSplitTest(filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    all_energy = []
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        if (ignore_unconverging_result):
            if (item[-1] > 1e-5):
                continue
        data = [item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[6]]
        label = [item[7], item[8], item[9], item[9], item[10]]
                    
        all_data.append(data)
        all_label.append(label)
    
    # all_data = np.array(all_data[1150:1200]).astype(np.float32)
    # all_label = np.array(all_label[1150:1200]).astype(np.float32)
    all_data = np.array(all_data).astype(np.float32)
    all_label = np.array(all_label).astype(np.float32)
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    all_data = all_data[indices]
    all_label = all_label[indices]

    return all_data, all_label

loss_l2 = tf.keras.losses.MeanSquaredError()
loss_logl2 = tf.keras.losses.MeanSquaredLogarithmicError()
# loss_function = relativeL2

w_grad = tf.constant(1.0, dtype=tf.float32)
w_e = tf.constant(1.0, dtype=tf.float32)

def generator(train_data, train_label):    
    indices = np.arange(train_data.shape[0])
    while True:
        np.random.shuffle(indices)
        yield train_data[indices], train_label[indices]

@tf.function
def trainStep(opt, lambdas, sigmas, model, train_vars):
    rest = tf.convert_to_tensor(np.tile(np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32), (lambdas.shape[0], 1)))
    tiling_param = lambdas[:, :4]
    # print(tiling_param)
    # exit(0)
    rest_configuration = tf.concat((tiling_param, rest), 1)
    # print(rest_configuration)
    # exit(0)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(lambdas)
        tape.watch(rest_configuration)
        elastic_potential = model(lambdas)
        dedlambda = tape.gradient(elastic_potential, lambdas)
        batch_dim = elastic_potential.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 4])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, 4], [batch_dim, 4])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, elastic_potential)

        elastic_potential_reset = model(rest_configuration)
        dirichlet_loss = tf.cast(loss_l2(tf.convert_to_tensor(np.tile(np.array([0.0]).astype(np.float32), (rest_configuration.shape[0], 1))), elastic_potential_reset), tf.float32)
        rest_stress = tape.gradient(elastic_potential_reset, rest_configuration)[:,4:]
        rest_stress_loss = loss_l2(rest, rest_stress)
        e_loss += tf.constant(0.0, dtype=tf.float32) * dirichlet_loss
        grad_loss += tf.constant(0.0, dtype=tf.float32) * rest_stress_loss
        loss = grad_loss + e_loss
        
    dLdw = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(dLdw, train_vars))
    gradNorm = tf.math.sqrt(tf.reduce_sum([tf.reduce_sum(gi*gi) for gi in dLdw]))
    
    del tape
    return grad_loss, e_loss, gradNorm

@tf.function
def testStep(lambdas, sigmas, model):
    # print(lambdas)
    rest = tf.convert_to_tensor(np.tile(np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32), (lambdas.shape[0], 1)))
    tiling_param = lambdas[:, :4]
    rest_configuration = tf.concat((tiling_param, rest), 1)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(lambdas)
        tape.watch(rest_configuration)
        
        elastic_potential = model(lambdas)
        dedlambda = tape.gradient(elastic_potential, lambdas)
        batch_dim = elastic_potential.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 4])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, 4], [batch_dim, 4])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, elastic_potential)

        elastic_potential_reset = model(rest_configuration)
        dirichlet_loss = tf.cast(loss_l2(tf.convert_to_tensor(np.tile(np.array([0.0]).astype(np.float32), (rest_configuration.shape[0], 1))), elastic_potential_reset), tf.float32)
        rest_stress = tape.gradient(elastic_potential_reset, rest_configuration)[:,4:]
        rest_stress_loss = loss_l2(rest, rest_stress)
        e_loss += tf.constant(0.0, dtype=tf.float32) * dirichlet_loss
        grad_loss += tf.constant(0.0, dtype=tf.float32) * rest_stress_loss

    return grad_loss, e_loss, dedlambda, elastic_potential

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

def plotPotentialClean(model_name):
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    # load pretrained model
    model.load_weights(model_name + '.tf')
    # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/data_45_only_off_diagonal.txt"
    uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data.txt"
    # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data_with_strain.txt"
    tiling_params_and_strain, stress_and_potential = loadDataSplitTest(uniaxial_data, False, False)
    print("first print line 179")
    # print("input", tiling_params_and_strain)
    # print("supervision", stress_and_potential)

    # send tiling_params_and_strain for prediction
    grad_loss, e_loss, sigma, energy = testStep(tf.convert_to_tensor(tiling_params_and_strain), stress_and_potential, model)
    # sigma and energy are the stress and potential from the network
    print("l2 loss on the gradient is ", grad_loss)

    #here is the only the prediction part let's use this!
    print("print again the input")
    # print(tiling_params_and_strain)
    elastic_potential = model(tf.convert_to_tensor(tiling_params_and_strain), training = False)

    potential_gt = stress_and_potential[:, -1] # last entry is the potential
    # potential_pred = energy.numpy() # prediction 
    potential_pred = elastic_potential.numpy() #identical to above
    indices = [i for i in range(len(potential_gt))]
    
    def compare_energy(i, j):
        return potential_gt[i] - potential_gt[j]
    indices_sorted = sorted(indices, key=cmp_to_key(compare_energy))

    plt.plot(indices, potential_pred[indices_sorted], label = "prediction")
    plt.plot(indices, potential_gt[indices_sorted], label = "GT")
    plt.legend(loc="upper right")
    plt.savefig("strain_energy.png", dpi = 300)
    plt.close()


def plotPotential(model_name):
    uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data_with_strain.txt"
    all_data, all_label = loadDataSplitTest(uniaxial_data, False, False)
    
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights(model_name + '.tf')
    grad_loss, e_loss, sigma, energy = testStep(tf.convert_to_tensor(all_data), all_label, model)
    potential = energy.numpy()
    strain = [i for i in range(len(potential))]
    
    
    def cmp_strain_xx(i, j):
        return all_data[i][4+0] - all_data[j][4+0]
    def cmp_strain_xy(i, j):
        return all_data[i][4+2] - all_data[j][4+2]
    def cmp_strain_yy(i, j):
        return all_data[i][4+1] - all_data[j][4+1]
    
    # print(np.linalg.norm(sigma.numpy()[:, 0:1] - all_label[:, 0:1])/float(len(all_label)))
    # print("grad loss", grad_loss)
    loss = 0
    for i in range(len(all_label)):
        # print(sigma[i, 0] - all_label[i, 0])
        loss += np.power(sigma[i, 0] - all_label[i, 0], 2)
    loss = loss / float(len(label_all))
    print("loss", loss)
    # print("gt", all_label[:, 0:4])
    # print("prediction", sigma)

    strain_x = all_data[:, 4]
    # indices = sorted(strain, key=cmp_to_key(cmp_strain_xx))
    indices = [i for i in range(len(strain_x))]
    plt.plot(strain_x[indices], potential[indices], label = "prediction")
    # print(all_data)
    plt.plot(strain_x[indices], all_label[:,-1][indices], label = "GT")

    # error = []
    # error_abs = []
    # for i in range(len(potential)):
    #     error.append(np.abs(potential[i] - all_label[i, -1]) / np.abs(all_label[i, -1]))
    #     error_abs.append(np.abs(potential[i] - all_label[i, -1]))
        
    # plt.plot(strain, error, label = "rel error")
    # plt.plot(strain, error_abs, label = "abs error")
    plt.legend(loc="upper right")
    plt.savefig(model_name + "strain_energy.png", dpi = 300)
    plt.close()
    

def plotPotentialPolar(model_name):
    uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data3.txt"
    
    # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data_with_strain.txt"
    # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialStrain_backup/training_data.txt"
    all_data, all_label = loadDataSplitTest(uniaxial_data, False, False)
    n_sp_per_strain = 50
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights(model_name + '.tf')

    grad_loss, e_loss, sigma, energy = testStep(all_data, all_label, model)
    print("test loss ", grad_loss + e_loss)
    
    for j in range(len(all_label)//n_sp_per_strain):
        potential = energy.numpy()[j*n_sp_per_strain:(j+1)*n_sp_per_strain]
        potential_gt = all_label[j*n_sp_per_strain:(j+1)*n_sp_per_strain, -1]

        
        theta = np.arange(0.0, np.pi, np.pi/float(n_sp_per_strain))
        for i in range(n_sp_per_strain):
            theta = np.append(theta, theta[i] - np.pi)
            potential = np.append(potential, potential[i])
            potential_gt = np.append(potential_gt, potential_gt[i])

        theta = np.append(theta, theta[0])
        potential = np.append(potential, potential[0])
        potential_gt = np.append(potential_gt, potential_gt[0])

        plt.polar(theta, potential,label="prediction")
        plt.polar(theta, potential_gt,label="gt")
        plt.legend()
        plt.savefig(model_name + "_"+str(j)+".png", dpi = 300)
        plt.close()

    # stress_norm = []
    # stress_norm_gt = []
    # for j in range(n_sp_per_strain):
    #     stress_norm[j] = np.linalg.norm(sigma[i])
    #     stress_norm_gt[j] = np.linalg.norm(all_label[i*n_sp_per_strain+j, 0:4])


def validate(model_name, validation_data, validation_label):
    save_path = "./"
    B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    model = loadSingleFamilyModel(4, B)
    model.load_weights(model_name + '.tf')
    grad_loss, e_loss, sigma, energy = testStep(validation_data, validation_label, model)
    
    
    plot(model_name + "_validation", sigma.numpy(), validation_label)

    print("validation loss", grad_loss + e_loss)

def train(model_name, train_data, train_label, validation_data, validation_label):
    save_path = "./"
    # B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    B = None
    model = buildSingleFamilyModel(4, B)
    # model.load_weights(model_name + '.tf')
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-4)
    max_iter = 40000

    val_lambdasTF = tf.convert_to_tensor(validation_data)
    val_sigmasTF = tf.convert_to_tensor(validation_label)

    losses = [[], []]
    g_norm0 = 0
    for iteration in range(max_iter):
        lambdas, sigmas = next(generator(train_data, train_label))
        
        lambdasTF = tf.convert_to_tensor(lambdas)
        sigmasTF = tf.convert_to_tensor(sigmas)
        
        
        train_loss_grad, train_loss_e, g_norm = trainStep(opt, lambdasTF, sigmasTF, model, train_vars)
        if (iteration == 0):
            g_norm0 = g_norm
        validation_loss_grad, validation_loss_e, _, _ = testStep(val_lambdasTF, val_sigmasTF, model)
        losses[0].append(train_loss_grad + train_loss_e)
        losses[1].append(validation_loss_grad + validation_loss_e)
        print("iter: {}/{} train_loss_grad: {} train_loss e: {}, validation_loss_grad:{} loss_e:{} |g|: {}, |g_init|: {} ".format(iteration, max_iter, train_loss_grad, train_loss_e, \
                         validation_loss_grad, validation_loss_e, \
                        g_norm, g_norm0))
    
    
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
    
    uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data3.txt"
    # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data_with_strain.txt"
    # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/data_45_only_off_diagonal.txt"
    
    
    if train_uniaxial:
        data_all, label_all = loadDataSplitTest(uniaxial_data)
    


    five_percent = int(len(data_all) * 0.05)
    # five_percent = int(len(data_all) * 0.2)
    
    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]
    # train_data = data_all
    # validation_data = data_all
    # train_label = label_all
    # validation_label = label_all

    test_data = []
    test_label = [] 

    if train_uniaxial:
        model_name = "uniaxial"
    else:
        model_name = "biaxial"
    # if train_both:
        # model_name = "full"
    train(model_name, train_data, train_label, validation_data, validation_label)
    validate(model_name, validation_data, validation_label)
    # sigma = test(model_name, test_data, test_label)
    plotPotentialPolar(model_name)
    # plotPotentialClean(model_name)
    


    # for i in range(len(sigma)):
    #     print(sigma[i], " ", test_label[i], " ", strain_percents[i], " ", np.linalg.norm(test_label[i] - sigma[i]))
    # ortho_dir = np.array([-np.sin(0.0), np.cos(0.0)])
    # for i in range(len(sigma)):
    #     stress = np.array([[sigma[i][0], sigma[i][2]],[sigma[i][2], sigma[i][1]]])
    #     stress_gt = np.array([[test_label[i][0], test_label[i][2]],[test_label[i][2], test_label[i][1]]])
    #     test_dot = np.dot(stresss, ortho_dir)
    #     gt_dot = np.dot(stress_gt, ortho_dir)
    #     print("dot test: {} dot gt: {}".format(test_dot, gt_dot))
    