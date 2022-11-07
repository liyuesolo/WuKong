
from cProfile import label
from doctest import master
from email.policy import default
from linecache import getlines
import os
from functools import cmp_to_key
from pickletools import optimize
from pyexpat import model
from statistics import mode
from tkinter import constants
from turtle import right
from joblib import Parallel, delayed

from scipy.optimize import BFGS
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
from requests import options
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
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy
from scipy.optimize import check_grad
from tactile import IsohedralTiling, tiling_types, EdgeShape, mul, Point
import dearpygui.dearpygui as dpg

from Derivatives import *

@tf.function
def testStep(n_tiling_params, lambdas, model):
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(lambdas)
        with tf.GradientTape() as tape:
            tape.watch(lambdas)
            
            elastic_potential = model(lambdas, training=False)
            dedlambda = tape.gradient(elastic_potential, lambdas)
            batch_dim = elastic_potential.shape[0]
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    dstress_dp = tape_outer.batch_jacobian(stress, lambdas)[:, :, 0:n_tiling_params]
    del tape
    del tape_outer
    return dstress_dp, stress, de_dp, elastic_potential

@tf.function
def testStepd2edp2(n_tiling_params, lambdas, model):
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(lambdas)
        with tf.GradientTape() as tape:
            tape.watch(lambdas)
            elastic_potential = model(lambdas, training=False)
            dedlambda = tape.gradient(elastic_potential, lambdas)
            batch_dim = elastic_potential.shape[0]
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    d2edp2 = tape_outer.batch_jacobian(de_dp, lambdas)[:, :, 0:n_tiling_params]
    del tape
    del tape_outer
    return d2edp2, de_dp, elastic_potential


@tf.function
def valueGradHessian(n_tiling_params, inputs, model):
    batch_dim = inputs.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            psi = model(inputs, training=False)
            dedlambda = tape.gradient(psi, inputs)
            
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            
    C = tape_outer.batch_jacobian(stress, inputs)[:, :, n_tiling_params:]
    del tape
    del tape_outer
    return psi, stress, C

@tf.function
def computeDirectionalStiffness(n_tiling_params, inputs, thetas, model):
    batch_dim = inputs.shape[0]
    thetas = tf.expand_dims(thetas, axis=1)
    
    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)
    psi, stress, C = valueGradHessian(n_tiling_params, inputs, model)
    
    Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :])
    dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
    
    for i in range(1, C.shape[0]):
        
        Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
        dTSd = tf.concat((tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0), dTSd), 0)
        
    stiffness = tf.squeeze(tf.math.divide(tf.ones((batch_dim), dtype=tf.float64), tf.expand_dims(dTSd, axis=0)))
    # stiffness2 = tf.constant(2.0) * tf.math.divide(tf.squeeze(psi), tf.constant(0.1) * tf.ones((batch_dim)))
    
    return stiffness, stiffness

@tf.function
def objGradPsiSum(n_tiling_params, inputs, ti, model):
    batch_dim = int(inputs.shape[0] // 3)
    
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        ti_batch = tf.tile(tf.expand_dims(ti, 0), (batch_dim, 1))
        strain = tf.reshape(inputs, (batch_dim, 3))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)
        psi = tf.math.reduce_sum(psi, axis=0)
    grad = tape.gradient(psi, inputs)
    del tape
    return tf.squeeze(psi), tf.squeeze(grad)

@tf.function
def hessPsiSum(n_tiling_params, inputs, ti, model):
    batch_dim = int(inputs.shape[0] // 3)
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            ti_batch = tf.tile(tf.expand_dims(ti, 0), (batch_dim, 1))
            strain = tf.reshape(inputs, (batch_dim, 3))
            nn_inputs = tf.concat((ti_batch, strain), axis=1)
            psi = model(nn_inputs, training=False)
            psi = tf.math.reduce_sum(psi, axis=0)
        grad = tape.gradient(psi, inputs)
    hess = tape_outer.jacobian(grad, inputs)
    del tape
    del tape_outer
    return tf.squeeze(hess)

def computeUniaxialStrainThetaBatch(n_tiling_params, strain, 
    thetas, model, tiling_params, verbose = True):

    
    strain_init = []
    for theta in thetas:
        d = np.array([np.cos(theta), np.sin(theta)])
        strain_tensor_init = np.outer(d, d) * strain
        strain_init.append(np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]]))

    strain_init = np.array(strain_init).flatten()
    
    m = len(strain_init) // 3
    n = len(strain_init)
    A = np.zeros((m, n))
    lb = []
    ub = []

    for i in range(m):
        d = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
        A[i, i * 3:i * 3 + 3] = computedCdE(d)
        lb.append(strain)
        ub.append(strain)

    uniaxial_strain_constraint = LinearConstraint(A, lb, ub)

    def hessian(x):
        
        H = hessPsiSum(n_tiling_params, tf.convert_to_tensor(x), 
            tf.convert_to_tensor(tiling_params), model)
        H = H.numpy()
        return H

    def objAndEnergy(x):
        obj, grad = objGradPsiSum(n_tiling_params, tf.convert_to_tensor(x), 
            tf.convert_to_tensor(tiling_params), model)
        
        obj = obj.numpy()
        grad = grad.numpy().flatten()
        return obj, grad

    if verbose:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
            constraints=[uniaxial_strain_constraint],
            options={'disp' : True})
    else:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, 
            hess=hessian,
            constraints= [uniaxial_strain_constraint],
            options={'disp' : False})
    
    return np.reshape(result.x, (m, 3))



def optimizeUniaxialStrain():
    filename = "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/sample_theta_1.050000.txt"
    all_data = []
    all_label = [] 
    n_tiling_params = 2
    thetas = []
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")]
    
        data = item[0:n_tiling_params]
        for i in range(2):
            data.append(item[n_tiling_params+i])
        data.append(2.0 * item[n_tiling_params+2])
        thetas.append(item[-4])

        label = item[n_tiling_params+3:n_tiling_params+7]
        
        
        all_data.append(data)
        all_label.append(label)
    
    thetas = np.array(thetas[0:]).astype(np.float32)
    all_data = np.array(all_data[0:]).astype(np.float32)
    all_label = np.array(all_label[0:]).astype(np.float32) 

    n_tiling_params = 2
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # save_path = os.path.join(current_dir, 'Models/' + str(221) + "/")
    # model = buildSingleFamilyModel3Strain(n_tiling_params)
    save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH21" + '.tf')
    theta = 0.0
    strain_eng = 0.05
    strain_green = strain_eng + 0.5 * np.power(strain_eng, 2.0)

    tiling_params = np.array([0.104512, 0.65])
    
    strain_nn_opt = []
    thetas = np.arange(0.0, np.pi, np.pi/float(50.0))
    for theta in thetas:
        strain_nn_opt.append(optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain_green, tiling_params))
        # strain_nn_opt.append([0, 0, 0])
    error = []
    for i in range(len(strain_nn_opt)):
        error.append(np.linalg.norm(strain_nn_opt[i] - all_data[i][2:5]) / np.linalg.norm(all_data[i][2:5]) * 100.0)
    
    for i in range(len(strain_nn_opt)):
        thetas = np.append(thetas, thetas[i] - np.pi)
        error = np.append(error, error[i])

    thetas = np.append(thetas, thetas[0])
    error = np.append(error, error[0])
    print("max error: {}".format(np.max(error)))
    plt.polar(thetas, error, linewidth=3.0)
    # plt.show()
    plt.savefig(save_path + "error.png", dpi=300)


def toPolarData(half):
    full = half
    n_sp_theta = len(half)
    for i in range(n_sp_theta):
        full = np.append(full, full[i])
    full = np.append(full, full[0])
    return full



def optimizeStiffnessProfile():
    filename = "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/sample_theta_1.010000_2.txt"
    all_data = []
    all_label = [] 
    n_tiling_params = 2
    thetas = []
    stiffness_gt = []
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")]
    
        data = item[0:n_tiling_params]
        for i in range(2):
            data.append(item[n_tiling_params+i])
        data.append(2.0 * item[n_tiling_params+2]) 
        thetas.append(item[-4])

        
        label = item[n_tiling_params+3:n_tiling_params+6]
        label.append(item[-5])
        
        all_data.append(data)
        all_label.append(label)
        stiffness_gt.append(item[-6])
    
    thetas = np.array(thetas[0:]).astype(np.float32)
    all_data = np.array(all_data[0:]).astype(np.float32)
    all_label = np.array(all_label[0:]).astype(np.float32) 
    stiffness_gt = np.array(stiffness_gt).astype(np.float32)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
    
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH21" + '.tf')
    stiffness, stiffness2 = computeDirectionalStiffness(n_tiling_params, tf.convert_to_tensor(all_data), 
        tf.convert_to_tensor(thetas), model)
    stiffness = stiffness.numpy()
    stiffness2 = stiffness2.numpy()
    
    energy_pred, grad_pred, C = valueGradHessian(n_tiling_params, tf.convert_to_tensor(all_data), model)

    n_sp_theta = len(thetas)
    # stiffness_gt = all_label[:, -1]
    # stiffness_gt = []
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

    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] - np.pi)
        stiffness = np.append(stiffness, stiffness[i])
        stiffness2 = np.append(stiffness2, stiffness2[i])
        stiffness_gt = np.append(stiffness_gt, stiffness_gt[i])

    thetas = np.append(thetas, thetas[0])
    stiffness = np.append(stiffness, stiffness[0])
    stiffness2 = np.append(stiffness2, stiffness2[0])
    stiffness_gt = np.append(stiffness_gt, stiffness_gt[0])
    plt.polar(thetas, stiffness, label = "tensor", linewidth=3.0)
    # plt.polar(thetas, stiffness_gt, label = "stiffness_gt", linewidth=3.0)
    # plt.polar(thetas, stiffness2, label = "2Psi/strain^2", linewidth=3.0)
    plt.legend(loc="upper left")
    # plt.show()
    plt.savefig(save_path + "stiffness.png", dpi=300)
    plt.close()

    energy_pred = toPolarData(energy_pred)
    energy_gt = all_label[:, -1]
    energy_gt = toPolarData(energy_gt)
    plt.polar(thetas, energy_pred, label = "energy_pred", linewidth=3.0)
    plt.polar(thetas, energy_gt, label = "energy_gt", linewidth=3.0)
    plt.legend(loc="upper left")
    plt.savefig(save_path + "energy_check.png", dpi=300)
    plt.close()

    stress_pred = toPolarData(stress_pred)
    stress_gt = toPolarData(stress_gt)
    plt.polar(thetas, stress_pred, label = "stress_pred", linewidth=3.0)
    plt.polar(thetas, stress_gt, label = "stress_gt", linewidth=3.0)
    plt.legend(loc="upper left")
    plt.savefig(save_path + "stress_check.png", dpi=300)
    plt.close()

    plt.polar(thetas, stiffness, label = "stiffness_pred", linewidth=3.0)
    plt.polar(thetas, stiffness2, label = "2Psi/strain^2", linewidth=3.0)
    # plt.polar(thetas, stiffness_gt, label = "stiffness_gt", linewidth=3.0)
    plt.legend(loc="upper left")
    plt.savefig(save_path + "hessian_check.png", dpi=300)
    plt.close()


def loadModel(IH):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if IH == 50:
        save_path = os.path.join(current_dir, 'Models/' + str(332) + "/")
        n_tiling_params = 2
    elif IH == 21:
        save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
        n_tiling_params = 2
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH" + str(IH) + '.tf')
    return model

def computeStiffness():
    n_tiling_params = 2
    model = loadModel(50)
    n_sp_theta = 50
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    strain = 0.1
    # ti = np.array([0.15, 0.65]).astype(np.float64)
    ti = np.array([0.23, 0.55]).astype(np.float64)
    uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    uniaxial_strain = np.reshape(uniaxial_strain, (batch_dim, 3))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    stiffness, _ = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
                
    stiffness = stiffness.numpy()
    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] - np.pi)
        stiffness = np.append(stiffness, stiffness[i])
    thetas = np.append(thetas, thetas[0])
    stiffness = np.append(stiffness, stiffness[0])
    plt.polar(thetas, stiffness, label = "tensor", linewidth=3.0)
    plt.savefig("stiffness.png", dpi=300)
    plt.close()

def computeStiffnessBatch():
    n_tiling_params = 2
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
    save_path = os.path.join(current_dir, 'Models/' + str(332) + "/")
    img_folder = save_path + "/stiffness/"
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    # model.load_weights(save_path + "IH21" + '.tf')
    model.load_weights(save_path + "IH50" + '.tf')

    n_sp_theta = 50
    ti = [0.15, 0.65]
    n_sp_strain = 50
    n_sp_tiling = 10
    strain_range = [0.001, 0.2]
    dstrain = (strain_range[1] - strain_range[0]) / float(n_sp_strain)
    t1_range = [0.105, 0.295]
    # t2_range = [0.505, 0.795]
    t2_range = [0.255, 0.745]
    dt1 = (t1_range[1] - t1_range[0]) / float(n_sp_strain)
    dt2 = (t2_range[1] - t2_range[0]) / float(n_sp_strain)

    ti_cnt = 0
    for t1_iter in range(n_sp_tiling):
        t1 = t1_range[0] + dt1 * float(t1_iter)
        for t2_iter in range(n_sp_tiling):
            t2 = t2_range[0] + dt2 * float(t2_iter)
            os.mkdir(img_folder + str(ti_cnt))
            f = open(img_folder +  str(ti_cnt) + "/tiling_params.txt", "w+")
            f.write(str(t1) + " " + str(t2))
            
            for strain in [0.001, 0.01, 0.025, 0.05, 0.075, 0.1]:
            
                thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta)).astype(np.float32)

                strain_green = strain #+ 0.5 * np.power(strain, 2.0)
                uniaxial_strain = []
                error = []
                for theta in thetas:
                    uni_strain, err = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain_green, ti)
                    uniaxial_strain.append(uni_strain)
                    error.append(err)
                f.write(str(np.max(error)) + " " + str(np.min(error)) + " " + str(np.mean(error)) + "\n")
                ti_tile = np.tile([t1, t2], (len(thetas), 1))
                nn_inputs = np.hstack((ti_tile, uniaxial_strain))
                stiffness, _= computeDirectionalStiffness(n_tiling_params, tf.convert_to_tensor(nn_inputs), 
                    tf.convert_to_tensor(thetas), model)
                
                for i in range(n_sp_theta):
                    thetas= np.append(thetas, thetas[i] - np.pi)
                    stiffness = np.append(stiffness, stiffness[i])
                
                thetas = np.append(thetas, thetas[0])
                stiffness = np.append(stiffness, stiffness[0])
                plt.polar(thetas, stiffness, label = "stiffness_NN", linewidth=3.0)
                # plt.legend(loc="upper left")
                # plt.show()
                
                plt.savefig(img_folder + str(ti_cnt) + "/stiffness_strain_"+str(strain)+".png", dpi=300)
                
                plt.close()
            f.close()
            # for i in range(n_sp_strain):
            #     strain = strain_range[0] + float(i) * dstrain
                
            #     # strain = 0.5
            #     thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta)).astype(np.float32)

            #     strain_green = strain + 0.5 * np.power(strain, 2.0)
            #     uniaxial_strain = []
            #     for theta in thetas:
            #         uniaxial_strain.append(optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain_green, ti))
                
            #     ti_tile = np.tile([t1, t2], (len(thetas), 1))
            #     nn_inputs = np.hstack((ti_tile, uniaxial_strain))
            #     stiffness, _= computeDirectionalStiffness(n_tiling_params, tf.convert_to_tensor(nn_inputs), 
            #         tf.convert_to_tensor(thetas), model)
                
            #     for i in range(n_sp_theta):
            #         thetas= np.append(thetas, thetas[i] - np.pi)
            #         stiffness = np.append(stiffness, stiffness[i])
                
            #     thetas = np.append(thetas, thetas[0])
            #     stiffness = np.append(stiffness, stiffness[0])
            #     plt.polar(thetas, stiffness, label = "stiffness_NN", linewidth=3.0)
            #     # plt.legend(loc="upper left")
            #     # plt.show()
                
            #     plt.savefig(img_folder + str(ti_cnt) + "/stiffness_strain_"+str(strain)+".png", dpi=300)
            #     plt.close()
            ti_cnt += 1

def optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, 
    theta, strain, tiling_params, verbose = True):
    
    strain_init = np.array([0.105, 0.2, 0.01])

    d = np.array([np.cos(theta), np.sin(theta)])
    strain_tensor_init = np.outer(d, d) * strain
    strain_init = np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]])

    def constraint(x):
        strain_tensor = np.reshape([x[0], 0.5 * x[-1], 0.5 * x[-1], x[1]], (2, 2))
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
        c = dTEd - strain
        return c

    def hessian(x):
        model_input = tf.convert_to_tensor([np.hstack((tiling_params, x))])
        C = computeStiffnessTensor(n_tiling_params, model_input, model)
        H = C.numpy()
        # alpha = 1e-6
        # while not np.all(np.linalg.eigvals(H) > 0):
        #     H += np.diag(np.full(3,alpha))
        #     alpha *= 10.0
        # print(H[0])
        # exit(0)
        return H

    def objAndEnergy(x):
        model_input = tf.convert_to_tensor([np.hstack((np.hstack((tiling_params, x))))])
        _, stress, _, psi = testStep(n_tiling_params, model_input, model)
        
        obj = np.squeeze(psi.numpy()) 
        grad = stress.numpy().flatten()
        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    if verbose:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
            constraints={"fun": constraint, "type": "eq"},
            options={'disp' : True})
    else:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
            constraints={"fun": constraint, "type": "eq"},
            options={'disp' : False})
    
    opt_model_input = tf.convert_to_tensor([np.hstack((tiling_params, result.x))])
    
    d2Phi_dE2 = computeStiffnessTensor(n_tiling_params, opt_model_input, model)
    dCdE = computedCdE(d)
    d2Ldqdp = np.zeros((3 + 1, n_tiling_params))
    d2Ldqdp[:3, :] = computedStressdp(n_tiling_params, opt_model_input, model)
    d2Ldq2 = np.zeros((3 + 1, 3 + 1))
    d2Ldq2[:3, :3] = d2Phi_dE2
    d2Ldq2[:3, 3] = -dCdE
    d2Ldq2[3, :3] = -dCdE
    lu, piv = lu_factor(d2Ldq2)
    
    dqdp = lu_solve((lu, piv), -d2Ldqdp)

    
    return result.x, dqdp


@tf.function
def objGradStiffness(ti, uniaxial_strain, thetas, model):
    batch_dim = uniaxial_strain.shape[0]
    
    thetas = tf.expand_dims(thetas, axis=1)
    
    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)

    ti = tf.expand_dims(ti, 0)
    with tf.GradientTape(persistent=True) as tape_outer_outer:
        tape_outer_outer.watch(ti)
        tape_outer_outer.watch(uniaxial_strain)
        with tf.GradientTape() as tape_outer:
            tape_outer.watch(ti)
            tape_outer.watch(uniaxial_strain)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(ti)
                tape.watch(uniaxial_strain)
                ti_batch = tf.tile(ti, (batch_dim, 1))
                inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)
                psi = model(inputs, training=False)
                stress = tape.gradient(psi, uniaxial_strain)
        C = tape_outer.batch_jacobian(stress, uniaxial_strain)
        
        
        Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :])
        dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
        
        for i in range(1, C.shape[0]):
            
            Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
            dTSd = tf.concat((tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0), dTSd), 0)
            
        stiffness = tf.squeeze(tf.math.divide(tf.ones((batch_dim), dtype=tf.float64), tf.expand_dims(dTSd, axis=0)))
    
    grad = tape_outer_outer.jacobian(stiffness, ti)
    dOdE = tape_outer_outer.jacobian(stiffness, uniaxial_strain)
    del tape
    del tape_outer
    del tape_outer_outer
    return tf.squeeze(stiffness), tf.squeeze(grad), tf.squeeze(dOdE)

def stiffnessOptimizationSA():
    n_tiling_params = 2
    model = loadModel(50)
    n_sp_theta = 50
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    strain = 0.1
    # ti = np.array([0.23, 0.5]).astype(np.float64)
    ti = np.array([0.15, 0.65]).astype(np.float64)
    # uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    uniaxial_strain = []
    for theta in thetas:
        uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, ti, False)
        uniaxial_strain.append(uni_strain)
    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    uniaxial_strain = np.reshape(uniaxial_strain, (batch_dim, 3))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    stiffness, _ = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    stiffness = stiffness.numpy()
    idx = [0, 23, 49]
    # idx = [0, 5, 9]

    # stiffness_targets = stiffness[idx]
    stiffness_targets = np.array([0.98796223, 1.0263252,  1.15254089])
    # print(stiffness_targets)
    # exit(0)
    sample_points_theta = thetas[idx]

    
    bounds = []
    bounds.append([0.105, 0.295])
    bounds.append([0.255, 0.745])

    def objAndGradient(x):
        _uniaxial_strain = []
        dqdp = []
        for theta in thetas:
            uni_strain, dqidpi = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, x, False)
            _uniaxial_strain.append(uni_strain)
            dqdp.append(dqidpi)
        

        ti_TF = tf.convert_to_tensor(x)

        uniaxial_strain_TF = tf.convert_to_tensor(_uniaxial_strain)
        stiffness_current, stiffness_grad, dOdE = objGradStiffness( 
                                            ti_TF, uniaxial_strain_TF, 
                                            tf.convert_to_tensor(thetas), 
                                            model)
        
        stiffness_current = stiffness_current.numpy()[idx]
        stiffness_grad = stiffness_grad.numpy()[idx]
        dOdE = dOdE.numpy()[idx]

        obj = (np.dot(stiffness_current - stiffness_targets, np.transpose(stiffness_current - stiffness_targets)) * 0.5)
        
        grad = np.zeros((n_tiling_params))
        
        for i in range(3):
            
            grad += (stiffness_current[i] - stiffness_targets[i]) * stiffness_grad[i].flatten() + \
                (stiffness_current[i] - stiffness_targets[i]) * np.dot(dOdE[i][i], dqdp[i][:3, :]).flatten()
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    result = minimize(objAndGradient, ti, method='trust-constr', jac=True, options={'disp' : True}, bounds=bounds, tol=1e-7)
    

    uniaxial_strain_opt = []
    for theta in thetas:
        uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, result.x, False)
        uniaxial_strain_opt.append(uni_strain)

    uniaxial_strain_opt = np.reshape(uniaxial_strain_opt, (batch_dim, 3))
    nn_inputs = tf.convert_to_tensor(np.hstack((np.tile(result.x, (batch_dim, 1)), uniaxial_strain_opt)))
    stiffness_opt, _ = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    stiffness_opt = stiffness_opt.numpy()
    
    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] - np.pi)
        stiffness = np.append(stiffness, stiffness[i])
        stiffness_opt = np.append(stiffness_opt, stiffness_opt[i])
    thetas = np.append(thetas, thetas[0])
    stiffness = np.append(stiffness, stiffness[0])
    stiffness_opt = np.append(stiffness_opt, stiffness_opt[0])
    plt.polar(thetas, stiffness, label = "stiffness initial", linewidth=3.0)
    plt.polar(thetas, stiffness_opt, label = "stiffness optimized", linewidth=3.0)
    plt.scatter(sample_points_theta, stiffness_targets, s=4.0)
    plt.legend(loc='upper left')
    plt.savefig("stiffness_optimization.png", dpi=300)
    plt.close()
    print(result.x)

if __name__ == "__main__":
    # computeStiffness()
    stiffnessOptimizationSA()