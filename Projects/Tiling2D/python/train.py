
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

tf.keras.backend.set_floatx('float32')

full_tensor = False
n_input = 3
if full_tensor:
    n_input = 4

def relativeL2(y_true, y_pred):
    # msle = tf.keras.losses.MeanSquaredLogarithmicError()
    if (y_true.shape[1] > 1):
        # loss = tf.constant(0.0, dtype=tf.float32)
        # for i in range(y_true.shape[1]):
        #     # y_true_normalized = tf.ones(y_true[:, i].shape)
        #     # y_pred_normalized = tf.divide(y_pred[:, i] + tf.constant(1e-5), y_true[:, i] + tf.constant(1e-5))
        #     # y_pred_normalized = tf.divide(y_pred[:, i] + K.epsilon(), y_true[:, i] + K.epsilon())    
        #     loss += K.mean(K.square(y_true[:, i] - y_pred[:, i]))
        # return loss
        stress_norm = tf.norm(y_true, ord='euclidean', axis=1)
        norm = tf.tile(tf.keras.backend.expand_dims(stress_norm, 1), tf.constant([1, n_input]))
        y_true_normalized = tf.divide(y_true, norm)
        y_pred_normalized = tf.divide(y_pred, norm)
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
        # exit(0)
    else:
        # y_true_normalized = tf.divide(y_true, y_true + K.epsilon())
        # y_pred_normalized = tf.divide(y_pred, y_true + K.epsilon())
        y_true_normalized = tf.ones(y_true.shape, dtype=tf.float32)
        y_pred_normalized = tf.divide(y_pred + K.epsilon(), y_true + K.epsilon())
        return K.mean(K.square((y_true_normalized - y_pred_normalized) * tf.constant(1.0, dtype=tf.float32)))

def loadDataSplitTest(filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")]
        if (ignore_unconverging_result):
            if (item[-1] > 1e-6 or math.isnan(item[-1])):
                continue
            # if (item[-5] < 1e-5 or item[-5] > 10):
            #     continue
        data = item[:2]
        if full_tensor:
            data.append(item[2])
            data.append(item[2])
        else:
            data.append(2.0 * item[2])

        if full_tensor:
            label = item[3:6]
            label.append(item[6])
            label.append(label[-1])
        else:
            label = item[3:7]
        
        all_data.append(data)
        all_label.append(label)
        
    
    all_data = np.array(all_data[:]).astype(np.float32)
    all_label = np.array(all_label[:]).astype(np.float32)
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

w_grad = tf.constant(1.0, dtype=tf.float32)
w_e = tf.constant(1.0, dtype=tf.float32)

@tf.function
def trainStep(opt, lambdas, sigmas, model, train_vars):
    
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(train_vars)
        tape.watch(lambdas)
        
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, n_input])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, 0], [batch_dim, n_input])
        
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
def testStep(lambdas, sigmas, model):
    
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, n_input])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, 0], [batch_dim, n_input])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, psi)
    del tape
    return grad_loss, e_loss, stress_pred, psi

@tf.function
def valueGradHessian(inputs, model):
    batch_dim = inputs.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            psi = model(inputs, training=False)
            dedlambda = tape.gradient(psi, inputs)
            
            stress = tf.slice(dedlambda, [0, 0], [batch_dim, n_input])
            
    C = tape_outer.batch_jacobian(stress, inputs)[:, :, :]
    del tape
    del tape_outer
    return psi, stress, C

def NHAutodiffTest(inputs, lambda_tf, mu_tf, data_type = tf.float32):
    strain = inputs
    batch_dim = strain.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(strain)
        with tf.GradientTape() as tape:
            tape.watch(strain)
            strain_xx = tf.gather(strain, [0], axis = 1)
            strain_yy = tf.gather(strain, [1], axis = 1)
            
            strain_xy = tf.constant(0.5, dtype=data_type) * tf.gather(strain, [2], axis = 1)
            strain_vec_reorder = tf.concat((strain_xx, strain_xy, strain_xy, strain_yy), axis=1)
            
            strain_tensor = tf.reshape(strain_vec_reorder, (batch_dim, 2, 2))    
                        
            righCauchy = tf.constant(2.0, dtype=data_type) * strain_tensor + tf.eye(2, batch_shape=[batch_dim], dtype=data_type)
            
            J = tf.math.sqrt(tf.linalg.det(righCauchy))
            
            I1 = tf.linalg.trace(righCauchy)
            C1 = tf.constant(0.5 * mu_tf, dtype=data_type)
            D1 = tf.constant(lambda_tf * 0.5, dtype=data_type)
            lnJ = tf.math.log(J)
            psi = C1 * (I1 - tf.constant(2.0, dtype=data_type) - tf.constant(2.0, dtype=data_type) * lnJ) + D1 * (lnJ*lnJ)
            
            stress = tape.gradient(psi, strain)
            # print(stress)
            # exit(0)
    C = tape_outer.batch_jacobian(stress, strain)
    print(C)
    exit(0)

@tf.function
def psiGradHessNH(strain, data_type = tf.float32):
    lambda_tf = 26.0 * 0.48 / (1.0 + 0.48) / (1.0 - 2.0 * 0.48)
    mu_tf = 26.0 / 2.0 / (1.0 + 0.48)
    youngsModulus = 26.0
    poissonsRatio = 0.48

    batch_dim = strain.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(strain)
        with tf.GradientTape() as tape:
            tape.watch(strain)
            
            strain_xx = tf.gather(strain, [0], axis = 1)
            strain_yy = tf.gather(strain, [1], axis = 1)
            
            strain_xy = tf.constant(0.5, dtype=data_type) * tf.gather(strain, [2], axis = 1)
            strain_vec_reorder = tf.concat((strain_xx, strain_xy, strain_xy, strain_yy), axis=1)
            
            strain_tensor = tf.reshape(strain_vec_reorder, (batch_dim, 2, 2))    
                        
            righCauchy = tf.constant(2.0, dtype=data_type) * strain_tensor + tf.eye(2, batch_shape=[batch_dim], dtype=data_type)
            
            J = tf.math.sqrt(tf.linalg.det(righCauchy))
            
            I1 = tf.linalg.trace(righCauchy)
            C1 = tf.constant(0.5 * mu_tf, dtype=data_type)
            D1 = tf.constant(lambda_tf * 0.5, dtype=data_type)
            lnJ = tf.math.log(J)
            psi = C1 * (I1 - tf.constant(2.0, dtype=data_type) - tf.constant(2.0, dtype=data_type) * lnJ) + D1 * (lnJ*lnJ)
            
            stress = tape.gradient(psi, strain)
            # print(stress)
            # exit(0)
    C = tape_outer.batch_jacobian(stress, strain)
    return psi, stress, C

def psiGradHessStVK(strain, data_type = tf.float32):
    lambda_tf = 26.0 * 0.48 / (1.0 + 0.48) / (1.0 - 2.0 * 0.48)
    mu_tf = 26.0 / 2.0 / (1.0 + 0.48)
    youngsModulus = 26.0
    poissonsRatio = 0.48

    # NHAutodiffTest(tf.gather(strain, [5], axis=0), lambda_tf, mu_tf)

    f = youngsModulus / (1 - poissonsRatio * poissonsRatio)
    stiffnessTensor = np.reshape(np.array([f, f * poissonsRatio, 0, f * poissonsRatio, f, 0, 0, 0, (1 - poissonsRatio) / 2 * f]), (3,3))
    print("isotropic C ", stiffnessTensor)
	

    batch_dim = strain.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(strain)
        with tf.GradientTape() as tape:
            tape.watch(strain)
            
            strain_xx = tf.gather(strain, [0], axis = 1)
            strain_yy = tf.gather(strain, [1], axis = 1)
            
            strain_xy = tf.constant(0.5, dtype=data_type) * tf.gather(strain, [2], axis = 1)
            strain_vec_reorder = tf.concat((strain_xx, strain_xy, strain_xy, strain_yy), axis=1)
            
            strain_tensor = tf.reshape(strain_vec_reorder, (batch_dim, 2, 2))    
            
            E2 = tf.matmul(strain_tensor, strain_tensor)
            psi = tf.constant(0.5, dtype=data_type) *tf.math.pow(tf.linalg.trace(strain_tensor), tf.constant(2.0, dtype=data_type)) + tf.linalg.trace(E2)

            stress = tape.gradient(psi, strain)
            # print(stress)
            # exit(0)
    C = tape_outer.batch_jacobian(stress, strain)
    return C, stress, psi

@tf.function
def computeDirectionalStiffnessNH(inputs, thetas):
    batch_dim = inputs.shape[0]
    thetas = tf.expand_dims(thetas, axis=1)

    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)

    
    psi, stress, C = psiGradHessNH(tf.convert_to_tensor(inputs))
   
    Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :])
    dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
    
    for i in range(1, C.shape[0]):
        
        Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
        dTSd = tf.concat((tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0), dTSd), 0)
        
    stiffness = tf.squeeze(tf.math.divide(tf.ones((batch_dim), dtype=tf.float32), tf.expand_dims(dTSd, axis=0)))
    
    return stiffness


@tf.function
def computeDirectionalStiffness(inputs, thetas, model):
    batch_dim = inputs.shape[0]
    thetas = tf.expand_dims(thetas, axis=1)

    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)

    
    psi, stress, C = valueGradHessian(inputs, model)
   
    Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :])
    dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
    
    for i in range(1, C.shape[0]):
        
        Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
        dTSd = tf.concat((tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0), dTSd), 0)
        
    stiffness = tf.squeeze(tf.math.divide(tf.ones((batch_dim), dtype=tf.float32), tf.expand_dims(dTSd, axis=0)))
    eng_strain = 0.1
    stiffness2 = tf.constant(2.0, dtype=tf.float32) * tf.math.divide(tf.squeeze(psi), tf.math.pow(tf.constant(eng_strain, dtype=tf.float32), tf.constant(2.0, dtype=tf.float32)) * tf.ones((batch_dim), dtype=tf.float32))
    
    return stiffness, stiffness2

def plot(prefix, prediction, label, gt_only = False):
    def cmp_sigma_xx(i, j):
        return label[i][0] - label[j][0]
    def cmp_sigma_xy(i, j):
        return label[i][2] - label[j][2]
    def cmp_sigma_yx(i, j):
        return label[i][3] - label[j][3]
    def cmp_sigma_yy(i, j):
        return label[i][1] - label[j][1]
        
    indices = [i for i in range(len(label))]
    data_point = [i for i in range(len(label))]
    
    indices = sorted(indices, key=cmp_to_key(cmp_sigma_xx))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_xx_gt = [sigma_gt_sorted[i][0] for i in range(len(label))]
    sigma_xx = [sigma_sorted[i][0] for i in range(len(label))]
    if not gt_only:
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
    if not gt_only:
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
    if not gt_only:
        plt.plot(data_point, sigma_xy, linewidth=1.0, label = "Sigma_xy")
    plt.plot(data_point, sigma_xy_gt, linewidth=1.0, label = "GT Sigma_xy")
    plt.legend(loc="upper left")
    plt.savefig(prefix + "_learned_sigma_xy.png", dpi = 300)
    plt.close()

    if full_tensor:
        indices = sorted(indices, key=cmp_to_key(cmp_sigma_yx))
        sigma_gt_sorted = label[indices]
        sigma_sorted = prediction[indices]
        sigma_xy_gt = [sigma_gt_sorted[i][3] for i in range(len(label))]
        sigma_xy = [sigma_sorted[i][3] for i in range(len(label))]
        if not gt_only:
            plt.plot(data_point, sigma_xy, linewidth=1.0, label = "Sigma_yx")
        plt.plot(data_point, sigma_xy_gt, linewidth=1.0, label = "GT Sigma_yx")
        plt.legend(loc="upper left")
        plt.savefig(prefix + "_learned_sigma_yx.png", dpi = 300)
        plt.close()

def plotPotentialClean(result_folder, tiling_params_and_strain, stress_and_potential, model, prefix = "strain_energy"):
    save_path = result_folder
    
    grad_loss, e_loss, sigma, energy = testStep(tf.convert_to_tensor(tiling_params_and_strain), stress_and_potential, model)
    # sigma and energy are the stress and potential from the network

    elastic_potential = model(tf.convert_to_tensor(tiling_params_and_strain), training = False)

    potential_gt = stress_and_potential[:, -1] # last entry is the potential
    # potential_pred = energy.numpy() # prediction 
    potential_pred = elastic_potential.numpy() #identical to above
    indices = [i for i in range(len(potential_gt))]
    
    def compare_energy(i, j):
        return potential_gt[i] - potential_gt[j]
    indices_sorted = sorted(indices, key=cmp_to_key(compare_energy))
    print(np.max(potential_gt))
    plt.plot(indices, potential_pred[indices_sorted], linewidth=0.8, label = "prediction")
    plt.plot(indices, potential_gt[indices_sorted], linewidth=0.8, label = "GT")
    plt.legend(loc="upper right")
    plt.savefig(save_path + prefix + ".png", dpi = 300)
    plt.close()

def toPolarData(half):
    full = half
    n_sp_theta = len(half)
    for i in range(n_sp_theta):
        full = np.append(full, full[i])
    full = np.append(full, full[0])
    return full

def optimizeStiffnessProfile(exp_id, model_name):
    filename = "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/homo_sample_theta_1.055.txt"
    all_data = []
    all_label = [] 
    thetas = []
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")]
    
        data = item[:2]
        if full_tensor:
            data.append(item[2])
            data.append(item[2])
        else:
            data.append(2.0 * item[2])

        if full_tensor:
            label = item[3:6]
            label.append(item[6])
            label.append(label[-1])
        else:
            label = item[3:7]
        thetas.append(item[-4])
        
        all_data.append(data)
        all_label.append(label)
    
    thetas = np.array(thetas[0:]).astype(np.float32)
    all_data = np.array(all_data[0:]).astype(np.float32)
    all_label = np.array(all_label[0:]).astype(np.float32) 

    # computeNHGT(tf.convert_to_tensor(all_data), tf.convert_to_tensor(thetas))
    

    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(exp_id) + "/")
    
    model = buildConstitutiveModel(n_input)
    model.load_weights(save_path + model_name + '.tf')
    
    stiffness, stiffness2 = computeDirectionalStiffness(tf.convert_to_tensor(all_data), tf.convert_to_tensor(thetas), model)
    stiffness = stiffness.numpy()
    stiffness2 = stiffness2.numpy()
    
    energy_pred, grad_pred, hess_pred = valueGradHessian(tf.convert_to_tensor(all_data), model)
    energy_pred = energy_pred.numpy()
    grad_pred = grad_pred.numpy()
    

    

    # plotPotentialClean("", all_data, all_label, model)
    # plot("", grad_pred, all_label)

    # print(np.linalg.norm(all_label[:, -1] - stiffness2)/np.linalg.norm(stiffness2))
    n_sp_theta = len(thetas)
    stiffness_gt_from_Psi = []
    stiffness_gt = computeDirectionalStiffnessNH(tf.convert_to_tensor(all_data), tf.convert_to_tensor(thetas))
    eng_strain = 0.1
    green = eng_strain + 0.5 * eng_strain * eng_strain
    stress_gt = []
    stress_pred = []

    for i in range(n_sp_theta):
        d = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
        stress_tensor = np.reshape(np.array(
                        [all_label[i][0], all_label[i][2],
                        all_label[i][2], all_label[i][1]]), 
                        (2, 2))
        stress_tensor_pred = np.reshape(np.array(
                        [grad_pred[i][0], grad_pred[i][2],
                        grad_pred[i][2], grad_pred[i][1]]), 
                        (2, 2))
        stress_d_pred = np.dot(d, np.matmul(stress_tensor_pred, d))
        stress_d = np.dot(d, np.matmul(stress_tensor, d))
        stress_gt.append(stress_d)
        stress_pred.append(stress_d_pred)

        strain_tensor = np.reshape(np.array(
                        [all_data[i][0], 0.5 * all_data[i][2],
                        0.5 * all_data[i][2], all_data[i][1]]), 
                        (2, 2))
        strain_d = np.dot(d, np.matmul(strain_tensor, d))
        # stiffness_gt.append(stress_d/strain_d)
        # stiffness_gt_from_Psi.append(2.0 * all_label[i, -1] / (np.power(green, 2.0)))

    # stiffness_gt = all_label[:, -1]
    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] - np.pi)
        stiffness = np.append(stiffness, stiffness[i])
        stiffness2 = np.append(stiffness2, stiffness2[i])
        stiffness_gt = np.append(stiffness_gt, stiffness_gt[i])
        # stiffness_gt_from_Psi = np.append(stiffness_gt_from_Psi, stiffness_gt_from_Psi[i])
    thetas = np.append(thetas, thetas[0])
    stiffness = np.append(stiffness, stiffness[0])
    stiffness2 = np.append(stiffness2, stiffness2[0])
    stiffness_gt = np.append(stiffness_gt, stiffness_gt[0])
    # stiffness_gt_from_Psi = np.append(stiffness_gt_from_Psi, stiffness_gt_from_Psi[0])

    

    energy_pred = toPolarData(energy_pred)
    energy_gt = all_label[:, -1]
    energy_gt = toPolarData(energy_gt)
    plt.polar(thetas, energy_pred, label = "energy_pred", linewidth=3.0)
    plt.polar(thetas, energy_gt, label = "energy_gt", linewidth=3.0)
    plt.legend(loc="upper left")
    plt.savefig(save_path+"energy_check.png", dpi=300)
    plt.close()

    stress_pred = toPolarData(stress_pred)
    stress_gt = toPolarData(stress_gt)
    plt.polar(thetas, stress_pred, label = "stress_pred", linewidth=3.0)
    plt.polar(thetas, stress_gt, label = "stress_gt", linewidth=3.0)
    plt.legend(loc="upper left")
    plt.savefig(save_path+"stress_check.png", dpi=300)
    plt.close()

    plt.polar(thetas, stiffness, label = "stiffness_pred", linewidth=3.0)
    # plt.polar(thetas, stiffness2, label = "2Psi/strain^2", linewidth=3.0)
    # plt.polar(thetas, stiffness_gt_from_Psi, label = "gt_from_Psi", linewidth=3.0)
    plt.polar(thetas, stiffness_gt, label = "stiffness_gt", linewidth=3.0)
    plt.legend(loc="upper left")
    plt.savefig(save_path+"hessian_check.png", dpi=300)
    plt.close()


    # plt.polar(thetas, stiffness, label = "tensor", linewidth=3.0)
    # # plt.polar(thetas, stiffness_gt_from_Psi, label = "gt_from_Psi", linewidth=3.0)
    # plt.polar(thetas, stiffness_gt, label = "stiffness_gt", linewidth=3.0)
    # # plt.polar(thetas, stiffness2, label = "2Psi/strain^2", linewidth=3.0)
    # plt.legend(loc="upper left")
    # plt.show()

def train(model_name, train_data, train_label, validation_data, validation_label):
    batch_size = np.minimum(60000, len(train_data))
    # print("batch size: {}".format(batch_size))
    # model = buildSingleFamilyModel(n_tiling_params)
    model = buildConstitutiveModel(n_input)
    # model = buildSingleFamilyModelSeparateTilingParamsAux(n_tiling_params)
    
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-4)
    max_iter = 80000

    val_lambdasTF = tf.convert_to_tensor(validation_data)
    val_sigmasTF = tf.convert_to_tensor(validation_label)

    losses = [[], []]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # model.load_weights("/home/yueli/Documents/ETH/WuKong/Projects/Tilisng2D/python/Models/67/" + model_name + '.tf')
    count = 0
    with open('counter.txt', 'r') as f:
        count = int(f.read().splitlines()[-1])
    f = open("counter.txt", "w+")
    f.write(str(count+1))
    f.close()
    
    
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    g_norm0 = 0
    iter = 0
    
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
            
            grad, e, g_norm = trainStep(opt, lambdasTF, sigmasTF, model, train_vars)
            
            train_loss_grad += grad
            train_loss_e += e
            g_norm_sum += g_norm
        if (iteration == 0):
            g_norm0 = g_norm_sum
        validation_loss_grad, validation_loss_e, _, _ = testStep(val_lambdasTF, val_sigmasTF, model)
        
        losses[0].append(train_loss_grad + train_loss_e)
        losses[1].append(validation_loss_grad + validation_loss_e)
        print("epoch: {}/{} train_loss_grad: {} train_loss e: {}, validation_loss_grad:{} loss_e:{} |g|: {}, |g_init|: {} ".format(iteration, max_iter, train_loss_grad, train_loss_e, \
                         validation_loss_grad, validation_loss_e, \
                        g_norm_sum, g_norm0))
        if iteration % 10000 == 0:
            model.save_weights(save_path + model_name + '.tf')
    model.save_weights(save_path + model_name + '.tf')
    return count

def validate(count, model_name, validation_data, validation_label):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildConstitutiveModel(n_input)
    model.load_weights(save_path + model_name + '.tf')
    # model.save(save_path + model_name + '.h5')
    grad_loss, e_loss, sigma, energy = testStep(validation_data, validation_label, model)
    
    plotPotentialClean(save_path, validation_data, validation_label, model)
    plot(save_path + model_name + "_validation", sigma.numpy(), validation_label, False)

    print("validation loss grad: {} energy: {}".format(grad_loss, e_loss)) 


def shuffleDataFromFile(prefix):
    data_file = "/home/yueli/Documents/ETH/SandwichStructure/Homo/data.txt"
    lines = []
    for line in open(prefix + "homo_uni_bi.txt").readlines():
        lines.append(line)
    indices = indices = np.arange(len(lines))
    np.random.shuffle(indices)
    f = open(prefix + "homo_uni_bi_shuffled.txt", "w+")
    for idx in indices:
        f.write(lines[idx])
    f.close()

if __name__ == "__main__":
    
    data_file = "/home/yueli/Documents/ETH/SandwichStructure/Homo/homo_uni_bi_shuffled.txt"
    # data_file = "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/homo_sample_theta_1.1.txt"
    
    data_all, label_all = loadDataSplitTest(data_file, shuffle=False, ignore_unconverging_result=True)

    five_percent = int(len(data_all) * 0.05)

    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]
    
    # trainNewModel()
    model_name = "homo_NH"
    # exp_id = train(model_name, 
    #     train_data, train_label, validation_data, validation_label)
    optimizeStiffnessProfile(371, model_name)
    # validate(330, 
    #     model_name, validation_data, validation_label)
    # validate(253, 
    #     model_name, train_data, train_label)