
from cProfile import label
from doctest import master
from email.policy import default
from linecache import getlines
import os
from functools import cmp_to_key
from pickletools import optimize
from statistics import mode
from tkinter import constants
from turtle import right
from joblib import Parallel, delayed


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


def smoothL1loss(x, y, beta = 1e-3):
    if (np.abs(x-y) < beta):
        return 0.5 * np.power(x - y, 2) / beta
    else:
        return np.abs(x-y) - 0.5 * beta

def smoothL1Grad(x, y, beta = 1e-3):
    if (np.abs(x-y) < beta):
        return (x-y) / beta
    else:
        if (x-y) < 0:
            return -1.0
        return 1.0

def l2Loss(x, y):
    return 0.5 * np.power(x - y, 2)

def l2Grad(x, y):
    return (x - y)

@tf.function
def psiGradHessNH(strain, data_type = tf.float32):
    lambda_tf = 26.0 * 0.48 / (1.0 + 0.48) / (1.0 - 2.0 * 0.48)
    mu_tf = 26.0 / 2.0 / (1.0 + 0.48)
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
    C = tape_outer.batch_jacobian(stress, strain)
    return C, stress, psi


def optimizeUniaxialStrainNHSingleDirection(theta, strain):
    data_type = tf.float32
    d = np.array([np.cos(theta), np.sin(theta)])
    strain_tensor_init = np.outer(d, d) * strain
    strain_init = np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]])
    n = np.array([-np.sin(theta), np.cos(theta)])
    def constraint(x):
        strain_tensor = np.reshape([x[0], 0.5 * x[-1], 0.5 * x[-1], x[1]], (2, 2))
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
        c = dTEd - strain
        # print("c", c)
        return c

    def hessian(x):
        C, stress, psi = psiGradHessNH(tf.convert_to_tensor([x], dtype=data_type), data_type)
        H = C[0].numpy()
        alpha = 1e-6
        while not np.all(np.linalg.eigvals(H) > 0):
            H += np.diag(np.full(3,alpha))
            alpha *= 10.0
        # print(H)
        return H

    def objAndEnergy(x):
        C, stress, psi = psiGradHessNH(tf.convert_to_tensor([x], dtype=data_type), data_type)
        
        obj = np.squeeze(psi.numpy()) 
        grad = stress.numpy().flatten()
        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    # result = minimize(objAndEnergy, strain_init, method='SLSQP', jac=True, hess=hessian,
    #     constraints={"fun": constraint, "type": "eq"},
    #     options={'disp' : True})
    result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
        constraints={"fun": constraint, "type": "eq"},
        options={'disp' : True})
    
    strain_opt = result.x
    
    return result.x

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


# def computeDirectionalStiffness(n_tiling_params, inputs, thetas, model):
#     batch_dim = inputs.shape[0]
#     thetas = tf.expand_dims(thetas, axis=1)
#     # d = tf.expand_dims(tf.concat((tf.math.cos(thetas), tf.math.sin(thetas)), axis = 1), axis = 2)
#     # dT = tf.expand_dims(tf.concat((tf.math.cos(thetas), tf.math.sin(thetas)), axis = 1), axis = 1)
#     # ddT = tf.multiply(dT,d)
#     d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
#                         tf.math.sin(thetas) * tf.math.sin(thetas), 
#                         tf.constant(2.0) * tf.math.sin(thetas) * tf.math.cos(thetas)), 
#                         axis = 1)

#     # d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
#     #                     tf.math.sin(thetas) * tf.math.sin(thetas), 
#     #                     tf.math.sin(thetas) * tf.math.cos(thetas),
#     #                     tf.math.sin(thetas) * tf.math.cos(thetas)), 
#     #                     axis = 1)
#     batch_dim = inputs.shape[0]
#     with tf.GradientTape() as tape_outer:
#         tape_outer.watch(inputs)
#         with tf.GradientTape() as tape:
#             tape.watch(inputs)
            
#             strain_xy = tf.gather(inputs, [4], axis=1)
#             full_input = tf.concat((inputs, strain_xy), axis=1)
            
#             psi = model(full_input, training=False)
#             dedlambda = tape.gradient(psi, inputs)
#             stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
#     C = tape_outer.batch_jacobian(stress, inputs)[:, :, n_tiling_params:]
    
#     strain_unique = tf.slice(inputs, [0, n_tiling_params], [batch_dim, 3])
#     tiling = tf.slice(inputs, [0, 0], [batch_dim, n_tiling_params])

#     with tf.GradientTape() as tape_outer:
#         tape_outer.watch(strain_unique)
#         with tf.GradientTape() as tape:
#             tape.watch(strain_unique)
#             strain_xy = tf.gather(strain_unique, [2], axis=1)
#             full_input = tf.concat((strain_unique, strain_xy), axis=1)
#             full_input = tf.concat((tiling, full_input), axis=1)
#             # print(full_input)
        
#             psi = model(full_input)
#             stress = tape.gradient(psi, strain_unique)
            
#             # stress_tensor = tf.reshape(stress, (batch_dim, 2, 2))
#             # print(tf.linalg.inv(stress_tensor))
#             # stress = tf.gather(stress, [0, 3, 1], axis =1)
#             # print(stress)
#     # S = tape_outer.batch_jacobian(stress, inputs)[:, :, :, n_tiling_params:]
    
#     C = tape_outer.batch_jacobian(stress, strain_unique)[:, :, :]
#     # print(C[0, :, :])
#     # exit(0)
#     Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :])
#     dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
    
#     for i in range(1, C.shape[0]):
        
#         Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
#         dTSd = tf.concat((tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0), dTSd), 0)
        
#     stiffness = tf.squeeze(tf.math.divide(tf.ones((batch_dim)), tf.expand_dims(dTSd, axis=0)))
#     stiffness2 = tf.constant(2.0) * tf.math.divide(tf.squeeze(psi), tf.constant(0.1) * tf.ones((batch_dim)))
#     # stiffness2 = psi
#     del tape
#     del tape_outer
#     return stiffness, stiffness2
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

# @tf.function
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
        
    stiffness = tf.squeeze(tf.math.divide(tf.ones((batch_dim)), tf.expand_dims(dTSd, axis=0)))
    stiffness2 = tf.constant(2.0) * tf.math.divide(tf.squeeze(psi), tf.constant(0.1) * tf.ones((batch_dim)))
    
    return stiffness, stiffness2

def StVKdStressdStrainHessian():
    _i_var = np.zeros((4))
    energyhessian = np.zeros((4,4))
    _i_var[0] = 3
    _i_var[1] = 1
    _i_var[2] = 0
    _i_var[3] = 2
    energyhessian[0,0] = _i_var[0]
    energyhessian[1,0] = _i_var[1]
    energyhessian[2,0] = _i_var[2]
    energyhessian[3,0] = _i_var[2]
    energyhessian[0,1] = _i_var[1]
    energyhessian[1,1] = _i_var[0]
    energyhessian[2,1] = _i_var[2]
    energyhessian[3,1] = _i_var[2]
    energyhessian[0,2] = _i_var[2]
    energyhessian[1,2] = _i_var[2]
    energyhessian[2,2] = _i_var[2]
    energyhessian[3,2] = _i_var[3]
    energyhessian[0,3] = _i_var[2]
    energyhessian[1,3] = _i_var[2]
    energyhessian[2,3] = _i_var[3]
    energyhessian[3,3]= _i_var[2]
    return energyhessian


def computeDirectionalStiffnessStVK(n_tiling_params, inputs, thetas, model):

    batch_dim = inputs.shape[0]
    thetas = tf.expand_dims(thetas, axis=1)
   
    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas),
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)
    
    with tf.GradientTape() as tape_outer:
        strain_vec = tf.slice(inputs, [0, n_tiling_params], [batch_dim, 4])
        # strain_vec = tf.gather(strain, [0, 2, 3, 1], axis =1)

        tape_outer.watch(strain_vec)
        with tf.GradientTape() as tape:
            tape.watch(strain_vec)
            strain_vec_reorder = tf.gather(strain_vec, [0, 2, 3, 1], axis =1)
            strain = tf.reshape(strain_vec_reorder, (batch_dim, 2, 2))    
            E2 = tf.matmul(strain, strain)
            psi = tf.constant(0.5) *tf.math.pow(tf.linalg.trace(strain), tf.constant(2.0)) + tf.linalg.trace(E2)
            
            stress = tape.gradient(psi, strain_vec)
            
            # stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 4])
            # print(stress)
            # stress = tf.gather(stress, [0, 3, 1], axis =1)
            # print(stress)
            # stress = tf.reshape(stress, (batch_dim, 2, 2))
    # S = tape_outer.batch_jacobian(stress, inputs)[:, :, :, n_tiling_params:]
    
    C = tape_outer.batch_jacobian(stress, strain_vec)[:, :, :]
    
    print(StVKdStressdStrainHessian())
    print((C[0, :, :]))
    # exit(0)
    Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :])
    dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
    
    for i in range(1, C.shape[0]):
        
        Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
        dTSd = tf.concat((tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0), dTSd), 0)
        
    stiffness = tf.squeeze(tf.math.divide(tf.ones((batch_dim)), tf.expand_dims(dTSd, axis=0)))
    stiffness2 = tf.constant(2.0) * tf.math.divide(tf.squeeze(psi), tf.constant(0.01) * tf.ones((batch_dim)))

    del tape
    del tape_outer
    return stiffness, stiffness2

@tf.function
def computeStiffnessTensor(n_tiling_params, inputs, model):
    batch_dim = inputs.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            
            psi = model(inputs, training=False)
            dedlambda = tape.gradient(psi, inputs)
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
    C = tape_outer.batch_jacobian(stress, inputs)[:, :, n_tiling_params:]
    return tf.squeeze(C)

def optimizeUniaxialStrainNH():
    filename = "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/homo_sample_theta_1.1.txt"
    all_data = []
    all_label = [] 
    
    thetas = []
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")]
    
        data = item[:2]
        data.append(2.0 * item[2])
            
        thetas.append(item[-4])
        
        label = item[3:7]
        
        
        all_data.append(data)
        all_label.append(label)
    
    thetas = np.array(thetas[0:]).astype(np.float32)
    all_data = np.array(all_data[0:]).astype(np.float32)
    all_label = np.array(all_label[0:]).astype(np.float32) 

    
    theta = 0.0
    strain_eng = 0.1
    strain_green = strain_eng + 0.5 * np.power(strain_eng, 2.0)
    
    strain_nn_opt = []
    thetas = np.arange(0.0, np.pi, np.pi/float(50.0))
    for theta in thetas:
        strain_nn_opt.append(optimizeUniaxialStrainNHSingleDirection(theta, strain_green))
        # strain_nn_opt.append([0, 0, 0])
    error = []
    for i in range(len(strain_nn_opt)):
        error.append(np.linalg.norm(strain_nn_opt[i] - all_data[i][:3]) / np.linalg.norm(all_data[i][:3]) * 100.0)
    
    for i in range(len(strain_nn_opt)):
        thetas = np.append(thetas, thetas[i] - np.pi)
        error = np.append(error, error[i])

    thetas = np.append(thetas, thetas[0])
    error = np.append(error, error[0])
    print("maximum error: {}%".format(np.max(error)))
    plt.polar(thetas, error, linewidth=3.0)
    # plt.show()
    plt.savefig("error.png", dpi=300)
    plt.close()



def optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, 
    theta, strain, tiling_params):
    
    strain_init = np.array([0.105, 0.2, 0.01])

    d = np.array([np.cos(theta), np.sin(theta)])
    strain_tensor_init = np.outer(d, d) * strain
    strain_init = np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]])
    n = np.array([-np.sin(theta), np.cos(theta)])

    def constraint(x):
        strain_tensor = np.reshape([x[0], 0.5 * x[-1], 0.5 * x[-1], x[1]], (2, 2))
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
        c = dTEd - strain
        return c

    def hessian(x):
        model_input = tf.convert_to_tensor([np.hstack((tiling_params, x))])
        C = computeStiffnessTensor(n_tiling_params, model_input, model)
        H = C.numpy()
        alpha = 1e-6
        while not np.all(np.linalg.eigvals(H) > 0):
            H += np.diag(np.full(3,alpha))
            alpha *= 10.0
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
    
    result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
        constraints={"fun": constraint, "type": "eq"},
        options={'disp' : True})
    # result = minimize(objAndEnergy, strain_init, method='SLSQP', jac=True, hess=hessian,
    #     constraints={"fun": constraint, "type": "eq"},
    #     options={'disp' : True})
    # print(result.method)
    strain_opt = result.x
    strain_tensor = np.reshape([strain_opt[0], 0.5 * strain_opt[-1], 0.5 * strain_opt[-1], strain_opt[1]], (2, 2))
    model_input = tf.convert_to_tensor([np.hstack((tiling_params, strain_opt))])
    destress_dp, stress, de_dp, psi = testStep(n_tiling_params, model_input, model)
    stress = stress.numpy().flatten()
    stress_tensor = np.reshape([stress[0], stress[-1], stress[-1], stress[1]], (2, 2))
    print(stress_tensor)
    print("stress orthogonal: {}".format(np.dot(n, np.dot(stress_tensor, np.transpose(n)))))
    # print(strain_tensor)
    # print(d)
    # print(np.dot(strain_tensor, np.transpose(d)))
    # print(np.dot(d, np.dot(strain_tensor, np.transpose(d))))
    # print(strain_opt, theta)
    # exit(0)
    return result.x
    # fdGradient(strain_init)
    exit(0)



def optimizeUniaxialStrainSingleDirection(model, n_tiling_params, theta, strain, tiling_params):
    
    strain_init = np.array([0.105, -0.1, 0.001])

    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-np.sin(theta), np.cos(theta)])
    ddT = np.outer(d, d)
    w_strain = 1000.0

    def objOnly(x):
        model_input = tf.convert_to_tensor([np.hstack((np.hstack((tiling_params, x)), x[-1]))])
        destress_dp, stress, de_dp, psi = testStep(n_tiling_params, model_input, model)
        strain_tensor = np.reshape([x[0], x[-1], x[-1], x[1]], (2, 2))
        
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
        
        obj = np.squeeze(psi.numpy()) + 0.5 * w_strain * np.power(dTEd - strain, 2)
        
        return obj
    
    def gradOnly(x):
        model_input = tf.convert_to_tensor([np.hstack((np.hstack((tiling_params, x)), x[-1]))])
        destress_dp, stress, de_dp, psi = testStep(n_tiling_params, model_input, model)
        strain_tensor = np.reshape([x[0], x[-1], x[-1], x[1]], (2, 2))
        
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
        ddE = w_strain * (d * strain_tensor * np.transpose(d) - strain) * ddT
        grad = stress.numpy().flatten()[:3] + [ddE[0][0], ddE[1][1], ddE[0][1]]
        return grad

    def fdGradient(x0):
        eps = 1e-4
        grad = gradOnly(x0)
        print(grad)
        E0 = objOnly(np.array([x0[0] - eps, x0[1], x0[2]]))
        E1 = objOnly(np.array([x0[0] + eps, x0[1], x0[2]]))
        fd_grad = []
        fd_grad.append((E1 - E0)/2.0/eps)
        E0 = objOnly(np.array([x0[0], x0[1] - eps, x0[2]]))
        E1 = objOnly(np.array([x0[0], x0[1] + eps, x0[2]]))
        fd_grad.append((E1 - E0)/2.0/eps)
        E0 = objOnly(np.array([x0[0], x0[1], x0[2] - eps]))
        E1 = objOnly(np.array([x0[0], x0[1], x0[2] + eps]))
        fd_grad.append((E1 - E0)/2.0/eps)
        print(fd_grad)

    def objAndEnergy(x):
        model_input = tf.convert_to_tensor([np.hstack((np.hstack((tiling_params, x)), x[-1]))])
        destress_dp, stress, de_dp, psi = testStep(n_tiling_params, model_input, model)
        strain_tensor = np.reshape([x[0], x[-1], x[-1], x[1]], (2, 2))
        
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
        
        obj = np.squeeze(psi.numpy()) + 0.5 * w_strain * np.power(dTEd - strain, 2)
        ddE = w_strain * (d * strain_tensor * np.transpose(d) - strain) * ddT
        grad = stress.numpy().flatten()[:3] + [ddE[0][0], ddE[1][1], ddE[0][1]]
        # print("obj: {} |grad|: {}".format(obj, grad))
        # exit(0)
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    result = minimize(objAndEnergy, strain_init, method='BFGS', jac=True, options={'disp' : True})
    strain_opt = result.x
    strain_tensor = np.reshape([strain_opt[0], strain_opt[-1], strain_opt[-1], strain_opt[1]], (2, 2))
    model_input = tf.convert_to_tensor([np.hstack((np.hstack((tiling_params, strain_opt)), strain_opt[-1]))])
    destress_dp, stress, de_dp, psi = testStep(n_tiling_params, model_input, model)
    stress = stress.numpy().flatten()
    stress_tensor = np.reshape([stress[0], stress[-2], stress[-1], stress[1]], (2, 2))
    print("stress orthogonal: {}".format(np.dot(n, np.dot(stress_tensor, np.transpose(n)))))
    # print(strain_tensor)
    # print(d)
    # print(np.dot(strain_tensor, np.transpose(d)))
    # print(np.dot(d, np.dot(strain_tensor, np.transpose(d))))
    print(strain_opt, theta)
    # exit(0)
    return result.x
    # fdGradient(strain_init)
    exit(0)

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

def plot(prefix, prediction, label, gt_only = False):
    # print(prediction)
    # print(label[0][0])
    # exit(0)
    def cmp_sigma_xx(i, j):
        return label[i][0] - label[j][0]
    def cmp_sigma_xy(i, j):
        return label[i][2] - label[j][2]
    def cmp_sigma_yy(i, j):
        return label[i][1] - label[j][1]
        
    indices = [i for i in range(len(label))]
    data_point = [i for i in range(len(label))]

    indices = sorted(indices, key=cmp_to_key(cmp_sigma_xx))
    print(indices)
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

def toPolarData(half):
    full = half
    n_sp_theta = len(half)
    for i in range(n_sp_theta):
        full = np.append(full, full[i])
    full = np.append(full, full[0])
    return full

def optimizeStiffnessProfile():
    filename = "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/sample_theta_1.050000_full.txt"
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
    stiffness, stiffness2 = computeDirectionalStiffness(n_tiling_params, tf.convert_to_tensor(all_data), tf.convert_to_tensor(thetas), model)
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
    # plt.polar(thetas, stiffness2, label = "2Psi/strain^2", linewidth=3.0)
    # plt.polar(thetas, stiffness_gt, label = "stiffness_gt", linewidth=3.0)
    # plt.legend(loc="upper left")
    plt.savefig(save_path + "stiffness.png", dpi=300)
    plt.close()


def optimizeStessProfile():
    n_tiling_params = 2
    bounds = []
    bounds.append([0.1, 0.2])
    bounds.append([0.5, 0.8])
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(52) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
    model.load_weights(save_path + "full40k" + '.tf')

    stress_target = np.array([0.12, 0.12, 0.00046986, 0.00049414])
    stress_target_current = np.array([0.14675923, 0.13248906, 0.00063026, 0.00063026])
    
    strain_target = np.array([0.1, 0.04, 0.002, 0.002])
    x0 = np.array([0.15, 0.55])
    model_input = tf.convert_to_tensor([np.hstack((x0, strain_target))])
    
    # x_opt = np.array([0.32896349, 0.24773578, 0.20972134, 0.12847836])
    def computeCurrentState(x):
        model_inputs = []
        n_pt = 10
        dx = (0.1 - 0.01) / float(10.0)
        strain = []
        for i in range(n_pt):
            eps_i = [float(i) * dx, 0.02, 0.002, 0.002]
            model_inputs.append(np.hstack((x, eps_i)))
            strain.append(eps_i)
        
        _, stress, _, psi = testStep(n_tiling_params, tf.convert_to_tensor(model_inputs), model)
        return strain, stress.numpy(), psi.numpy().flatten()
    
    strain, stress_init, psi = computeCurrentState(x0)
    
    strain_opt = [strain[4], strain[6]]
    stress_targets = [stress_init[4].copy(), stress_init[6].copy()]
    
    for i in range(len(stress_targets)):
        stress_targets[i][0] = 1.2 * stress_targets[i][0]
    
    stress_targets[0][0] = 0.065
    stress_targets[1][0] = 0.07
        
    # dstress_dp, stress, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    # print(stress, elastic_potential)
    # exit(0)
    def objAndGradientMatchX(x):
        model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        dstress_dp, stress, _, _ = testStep(n_tiling_params, model_input, model)
        dstress_dp = dstress_dp.numpy()
        stress = stress.numpy()
        
        obj = 0.0
        grad = np.zeros(n_tiling_params)
        for i in range(len(stress_targets)):
            obj += smoothL1loss(stress[i][0] / stress_targets[i][0], 1.0)
            # obj += 0.5 * np.power((stress[i][0] - stress_targets[i][0]) / stress_targets[i][0], 2)
            # grad += (stress[i][0] - stress_targets[i][0]) / stress_targets[i][0] * dstress_dp[i,0,:]
            grad += smoothL1Grad(stress[i][0] / stress_targets[i][0], 1.0) * dstress_dp[i,0,:]
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    def objAndGradient(x):
        model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        # model_input = tf.convert_to_tensor([np.hstack((x, strain_target))])
        dstress_dp, stress, _, _ = testStep(n_tiling_params, model_input, model)
        dstress_dp = dstress_dp.numpy()
        stress = stress.numpy()
        
        # obj = 0.0
        # grad = np.zeros(n_tiling_params)
        # for i in range(len(stress_targets)):
        #     obj += (np.dot(stress[i] - stress_target[i], np.transpose(stress[i] - stress_target[i])) * 0.5).flatten() / np.linalg.norm(stress_target[i])
        #     grad += np.dot(stress[i] - stress_target[i], dstress_dp).flatten() / np.linalg.norm(stress_target[i])
        
        obj = (np.dot(stress - stress_target, np.transpose(stress - stress_target)) * 0.5).flatten() / np.linalg.norm(stress_target)
        grad = np.dot(stress - stress_target, dstress_dp).flatten() / np.linalg.norm(stress_target)
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    
    
    result = minimize(objAndGradientMatchX, x0, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
    # result = minimize(objAndGradientMatchX, x0, method='BFGS', jac=True, options={'disp' : True})
    model_input = tf.convert_to_tensor(np.hstack((np.tile(result.x, (len(strain), 1)), strain)))
    dstress_dp, stress_new, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    print("Tiling params:", result.x)
    # strain_new, stress_new, psi_new = computeCurrentState(result.x)
    # print("stress_init", stress_init)
    # print("stress opt", stress_new)
    # print("stress targets", stress_targets)
    
    x_axis = []
    for i in range(len(strain)):
        x_axis.append(strain[i][0])
    
    strain_sp = []
    stress_sp = []

    for i in range(len(strain_opt)):
        strain_sp.append(strain_opt[i][0])
        stress_sp.append(stress_targets[i][0])
    
    init = []
    opt = []
    for i in range(len(strain)):
        init.append(stress_init[i][0])
        opt.append(stress_new[i][0])
    plt.plot(x_axis, init, label = "init")
    plt.plot(x_axis, opt, label = "optimized")
    plt.xlabel("strain_xx")
    plt.ylabel("stress_xx")
    plt.legend(loc="upper left")
    plt.scatter(strain_sp, stress_sp, s=4.0)
    plt.savefig(save_path + "/opt_stress.jpg", dpi=300)
    plt.close()




def optimizeDensityProfile():
    
    
    n_tiling_params = 2
    bounds = []
    bounds.append([0.1, 0.2])
    bounds.append([0.5, 0.8])
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(52) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
    model.load_weights(save_path + "full40k" + '.tf')

    stress_target = np.array([0.1, 0.15, 0.00046986, 0.00049414])
    stress_target_current = np.array([0.14675923, 0.13248906, 0.00063026, 0.00063026])
    energy_target = np.array([0.005])
    energy_target_current = 0.01058488
    strain_target = np.array([0.1, -0.154, 0.001, 0.001])
    x0 = np.array([0.18, 0.789])
    # x0 = np.array([0.169, 0.608])
    model_input = tf.convert_to_tensor([np.hstack((x0, strain_target))])
    
    # x_opt = np.array([0.32896349, 0.24773578, 0.20972134, 0.12847836])
    def computeCurrentState(x):
        model_inputs = []
        n_pt = 50
        dx = (0.1) / float(n_pt)
        strain = []
        for i in range(n_pt):
            eps_i = [0.0 + float(i) * dx, -0.2, 0.001, 0.001]
            model_inputs.append(np.hstack((x, eps_i)))
            strain.append(eps_i)
        
        _, stress, _, psi = testStep(n_tiling_params, tf.convert_to_tensor(model_inputs), model)
        return strain, stress, psi.numpy().flatten()
    
    strain, stress, psi = computeCurrentState(x0)
    # print(psi[10], psi[20], psi[30], psi[40])
    # exit(0)
    # strain_opt = strain[3:6]
    strain_opt = [strain[10], strain[20], strain[30], strain[40]]
    # psi_target = [0.0008, 0.0014, 0.0018]
    # psi_target = [0.0016, 0.0016, 0.0016, 0.0016]
    psi_target = [0.0015, 0.0015, 0.0015, 0.0015]
    # psi_target = [0.00508053, 0.00715527] # x0 = np.array([0.105, 0.55])
    # psi_target = [0.004102031, 0.0057697417] #x0 = np.array([0.125, 0.71])

    # print(np.hstack((np.tile(x0, (len(strain_opt), 1)), strain_opt)))
    # model_input = tf.convert_to_tensor(np.hstack((np.tile(x0, (len(strain_opt), 1)), strain_opt)))
    # dstress_dp, stress, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    # print(elastic_potential)
    # dstress_dp, stress, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    # print(stress, elastic_potential)
    # exit(0)

    def hessian(x):
        
        model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        d2edp2, dedp, e = testStepd2edp2(n_tiling_params, model_input, model)
        d2edp2 = d2edp2.numpy()
        dedp = dedp.numpy()
        e = e.numpy()
        H = np.zeros((n_tiling_params, n_tiling_params))
        for i in range(len(psi_target)):
            H += d2edp2[i, :, :] / psi_target[i] * l2Grad(e[i] / psi_target[i], 1.0)
            H += np.outer(dedp[i, :], dedp[i, :]) / psi_target[i] / psi_target[i]
        # alpha = 1e-6
        # while not np.all(np.linalg.eigvals(H) > 0):
        #     H += np.diag(np.full(n_tiling_params,alpha))
        #     alpha *= 10.0
        # print("Hessian PD", np.all(np.linalg.eigvals(H) > 0))
        # print("hessian", H)
        return H
    
    def objAndGradientEnergy(x):
        model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        # model_input = tf.convert_to_tensor([np.hstack((x, strain_target))])
        _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
        psi = psi.numpy()
        de_dp = de_dp.numpy()
        obj = 0.0
        grad = np.zeros(n_tiling_params)
        for i in range(len(psi_target)):
            obj += l2Loss(psi[i] / psi_target[i], 1.0)
            grad += l2Grad(psi[i] / psi_target[i], 1.0) * de_dp[i, :] / psi_target[i]
        
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    def objOnly(x):
        model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
        psi = psi.numpy()
        de_dp = de_dp.numpy()
        obj = 0.0
        for i in range(len(psi_target)):
            obj += l2Loss(psi[i] / psi_target[i], 1.0)    
        return obj
    
    def gradOnly(x):
        model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        
        _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
        psi = psi.numpy()
        de_dp = de_dp.numpy()
        
        grad = np.zeros(n_tiling_params)
        for i in range(len(psi_target)):
            grad += l2Grad(psi[i] / psi_target[i], 1.0) * de_dp[i, :] / psi_target[i]
        
        return grad

    def fdGradient():
        eps = 1e-4
        grad = gradOnly(x0)
        print(grad)
        E0 = objOnly(np.array([x0[0] - eps, x0[1]]))
        E1 = objOnly(np.array([x0[0] + eps, x0[1]]))
        fd_grad = []
        fd_grad.append((E1[0] - E0[0])/2.0/eps)
        E0 = objOnly(np.array([x0[0], x0[1] - eps]))
        E1 = objOnly(np.array([x0[0], x0[1] + eps]))
        fd_grad.append((E1[0] - E0[0])/2.0/eps)
        print(fd_grad)

    def fdHessian():
        eps = 1e-4
        H = hessian(x0)
        print(H)
        f0 = gradOnly(np.array([x0[0] - eps, x0[1]]))
        f1 = gradOnly(np.array([x0[0] + eps, x0[1]]))
        fd_hessian = []
        row_fd = (f1 - f0)/2.0/eps
        fd_hessian.append(row_fd)
        f0 = gradOnly(np.array([x0[0], x0[1] - eps]))
        f1 = gradOnly(np.array([x0[0], x0[1] + eps]))
        row_fd = (f1 - f0)/2.0/eps
        fd_hessian.append(row_fd)
        print(fd_hessian)
        


    def objEnergy(x):
        model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        # model_input = tf.convert_to_tensor([np.hstack((x, strain_target))])
        _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
        psi = psi.numpy()
        obj = 0.0
        grad = np.zeros(n_tiling_params)
        for i in range(len(psi_target)):
            obj += l2Loss(psi[i] / psi_target[i], 1.0)
        return obj
    

    # result = minimize(objAndGradientEnergy, x0, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
    result = minimize(objAndGradientEnergy, x0 ,method='trust-constr', jac=True, hess=hessian, options={'disp' : True}, bounds=bounds)
    # result = minimize(objAndGradientEnergy, x0 ,method='Newton-CG', jac=True, hess=hessian, options={'disp' : True, "xtol" : 1e-8})
    model_input = tf.convert_to_tensor([np.hstack((result.x, strain_target))])
    # dstress_dp, stress, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    print(result.x)

    strain_new, stress_new, psi_new = computeCurrentState(result.x)
    # print(psi)
    # print(psi_new)
    
    x_axis = []
    for i in range(len(strain)):
        x_axis.append(strain[i][0])
    strain_sp = [strain[10][0], strain[20][0], strain[30][0], strain[40][0]]
    
    plt.plot(x_axis, psi, label = "initial")
    plt.plot(x_axis, psi_new, label = "optimized")
    plt.xlabel("strain_xx")
    plt.ylabel("energy density")
    plt.legend(loc="upper left")
    plt.scatter(strain_sp, psi_target, s=4.0)
    plt.savefig(save_path + "/opt_energy.jpg", dpi=300)
    plt.close()


    # print(stress, elastic_potential)


def make_random_tiling(IH, params):
    # Construct a tiling
    # tiling = IsohedralTiling(random.choice(tiling_types))
    tiling = IsohedralTiling(IH)

    # Randomize the tiling vertex parameters
    ps = tiling.parameters
    for i in range(tiling.num_parameters):
        ps[i] = params[i]
    tiling.parameters = ps

    edges = []
    for shp in tiling.edge_shapes:
        ej = []
        ej.append(Point(0, 0))
        ej.append(Point(0.25, 0))
        ej.append(Point(0.75, 0))
        ej.append(Point(1, 0))
        if shp == EdgeShape.I:
            # Must be a straight line.
            pass
        elif shp == EdgeShape.J:
            # Anything works for J
            ej[2] = Point(1.0 - ej[1].x, ej[1].y)
            
        elif shp == EdgeShape.S:
            # 180-degree rotational symmetry
            ej[2] = Point(1.0 - ej[1].x, -ej[1].y)
            
        elif shp == EdgeShape.U:
            # Symmetry after reflecting/flipping across length.
            ej[1] = Point(ej[1].x, 0.0)
            ej[2] = Point(ej[2].x, 0.0)
            
        edges.append( ej )

    return tiling, edges
    
def getPolygons(IH, params):
    tx = 0
    ty = 0
    scale = 100
    tiling, edges = make_random_tiling(IH, params)
    

    ST = [scale, 0.0, tx, 0.0, scale, ty]
    polygons = []
    for i in tiling.fill_region_bounds( -5, -5, 10, 10 ):
        T = mul( ST, i.T )
        
        start = True
        polygon = []
        
        for si in tiling.shapes:
            
            S = mul( T, si.T )
            # Make the edge start at (0,0)
            seg = [ mul( S, Point(0., 0.)) ]

            if si.shape != EdgeShape.I:
                # The edge isn't just a straight line.
                ej = edges[ si.id ]
                seg.append( mul( S, ej[0] ) )
                seg.append( mul( S, ej[1] ) )

            # Make the edge end at (1,0)
            seg.append( mul( S, Point(1., 0.)) )

            if si.rev:
                seg.reverse()

            if start:
                start = False
                polygon.append([seg[0].x, seg[0].y])
            if len(seg) == 2:
                polygon.append([seg[1].x, seg[1].y])
            else:
                polygon.append([seg[1].x, seg[1].y])
                polygon.append([seg[2].x, seg[2].y])
                polygon.append([seg[3].x, seg[3].y])

        
        polygons.append(polygon)
    
    return polygons

def explorer():
    update = False
    IH = 21
    current_dir = os.path.dirname(os.path.realpath(__file__))

    if IH == 50:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(81) + "/")
        model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
        model.load_weights(save_path + "IH50" + '.tf')
        range_t0 = [0.1, 0.3]
        range_t1 = [0.25, 0.75]
        x0 = np.array([0.2, 0.5])
        params = [0.2, 0.5]
    elif IH == 21:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
        model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
        model.load_weights(save_path + "IH21" + '.tf')
        range_t0 = [0.1, 0.2]
        range_t1 = [0.5, 0.8]
        x0 = np.array([0.15, 0.55])
        params = [0.15, 0.55]
    
    polygons = getPolygons(IH, params)

    query_points = []
    range_strain = [-0.3, 0.5]
    n_sp_strain = 200
    delta_strain = (range_strain[1] - range_strain[0]) / float(n_sp_strain)
    strain_xx = []
    for i in range(n_sp_strain):
        strain = range_strain[1] - delta_strain * float(i)
        strain_xx.append(strain)
        query_points.append(np.hstack((x0, np.array([strain, 0.06, 0.0004]))))
        
    
    model_input = tf.convert_to_tensor(query_points)

    _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
    

    def t1Callback(sender):
        update = True
        for pt in query_points:
            pt[0] = dpg.get_value(sender)
        t1 = float(dpg.get_value(sender))
        
        polygons = getPolygons(IH, [t1, query_points[0][1]])
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        if update:
            dpg.set_value('plot', [strain_xx, psi])
            dpg.delete_item(item="draw")
            dpg.add_draw_layer(tag="draw", parent="structure")
            
            for polygon in polygons:
                for i in range(len(polygon) - 1):
                    vi, vj = polygon[i], polygon[i+1]
                    dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
            
            
        update = False 
        
    
    def t2Callback(sender):
        update = True
        for pt in query_points:
            pt[1] = dpg.get_value(sender)
        t2 = float(dpg.get_value(sender))
        
        polygons = getPolygons(IH, [query_points[0][0], t2])
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        if update:
            dpg.set_value('plot', [strain_xx, psi])
            dpg.delete_item(item="draw")
            dpg.add_draw_layer(tag="draw", parent="structure")
            for polygon in polygons:
                for i in range(len(polygon) - 1):
                    vi, vj = polygon[i], polygon[i+1]
                    dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
        update = False 
    def strainyyCallback(sender):
        for pt in query_points:
            pt[3] = dpg.get_value(sender)
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        dpg.set_value('plot', [strain_xx, psi])
    
    def strainxyCallback(sender):
        for pt in query_points:
            pt[4] = dpg.get_value(sender)
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        dpg.set_value('plot', [strain_xx, psi])

    dpg.create_context()
    dpg.create_viewport(title='Neural Constitutive Model',width=3700, height=2200)
    dpg.setup_dearpygui()
    dpg.set_global_font_scale(2.5)
    x = []
    y = []
    for polygon in polygons:
        for i in range(len(polygon) - 1):
            vi, vj = polygon[i], polygon[i+1]
            x.append(vi)
            y.append(vj)

    with dpg.window(label="Tiling Explorer"):
        with dpg.group(horizontal=True):
            dpg.add_text("Tiling Parameters")
            dpg.add_slider_float(default_value=0.5 * (range_t0[0] + range_t0[1]), min_value = range_t0[0], max_value=range_t0[1],width=200.0, height=500.0, callback= t1Callback)
            dpg.add_slider_float(default_value=0.5 * (range_t1[0] + range_t1[1]), min_value = range_t1[0], max_value=range_t1[1],width=200.0, height=500.0, callback= t2Callback)
            dpg.add_text("Strain yy Strain xy")
            dpg.add_slider_float(default_value=0.01, min_value = -0.3, max_value=0.5,width=200.0, height=500.0, callback= strainyyCallback)
            dpg.add_slider_float(default_value=0.001, min_value = -0.3, max_value=0.3,width=200.0, height=500.0, callback= strainxyCallback)
        with dpg.theme(tag="plot_theme"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (150, 255, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 10.0, category=dpg.mvThemeCat_Plots)
                
                # dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Diamond, category=dpg.mvThemeCat_Plots)
                # dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7, category=dpg.mvThemeCat_Plots)
        with dpg.group(horizontal=True):
            with dpg.plot(label="Line Series", height=2000, width=2600):
                # optionally create legend
                dpg.add_plot_legend()

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="strain_xx")
                dpg.add_plot_axis(dpg.mvYAxis, label="energy density", tag="y_axis")

                # series belong to a y axis
                dpg.add_line_series(strain_xx, psi, label="enery", parent="y_axis", tag="plot")
                dpg.bind_item_theme("plot", "plot_theme")
            with dpg.drawlist(width=1000, height=1000, tag="structure"):
                dpg.add_draw_layer(tag="draw", parent="structure")
                for polygon in polygons:
                    for i in range(len(polygon) - 1):
                        vi, vj = polygon[i], polygon[i+1]
                        dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
                
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    return

def stiffnessExplorer():
    update = False
    IH = 21
    current_dir = os.path.dirname(os.path.realpath(__file__))

    if IH == 50:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(81) + "/")
        model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
        model.load_weights(save_path + "IH50" + '.tf')
        range_t0 = [0.1, 0.3]
        range_t1 = [0.25, 0.75]
        x0 = np.array([0.2, 0.5])
        params = [0.2, 0.5]
    elif IH == 21:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
        model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
        model.load_weights(save_path + "IH21" + '.tf')
        range_t0 = [0.1, 0.2]
        range_t1 = [0.5, 0.8]
        x0 = np.array([0.15, 0.55])
        params = [0.15, 0.55]
    
    polygons = getPolygons(IH, params)

    query_points = []
    range_strain = [-0.3, 0.5]
    n_sp_strain = 200
    delta_strain = (range_strain[1] - range_strain[0]) / float(n_sp_strain)
    strain_xx = []
    for i in range(n_sp_strain):
        strain = range_strain[1] - delta_strain * float(i)
        strain_xx.append(strain)
        query_points.append(np.hstack((x0, np.array([strain, 0.06, 0.0004]))))
        
    
    model_input = tf.convert_to_tensor(query_points)

    _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
    

    def t1Callback(sender):
        update = True
        for pt in query_points:
            pt[0] = dpg.get_value(sender)
        t1 = float(dpg.get_value(sender))
        
        polygons = getPolygons(IH, [t1, query_points[0][1]])
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        if update:
            dpg.set_value('plot', [strain_xx, psi])
            dpg.delete_item(item="draw")
            dpg.add_draw_layer(tag="draw", parent="structure")
            
            for polygon in polygons:
                for i in range(len(polygon) - 1):
                    vi, vj = polygon[i], polygon[i+1]
                    dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
            
            
        update = False 
        
    
    def t2Callback(sender):
        update = True
        for pt in query_points:
            pt[1] = dpg.get_value(sender)
        t2 = float(dpg.get_value(sender))
        
        polygons = getPolygons(IH, [query_points[0][0], t2])
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        if update:
            dpg.set_value('plot', [strain_xx, psi])
            dpg.delete_item(item="draw")
            dpg.add_draw_layer(tag="draw", parent="structure")
            for polygon in polygons:
                for i in range(len(polygon) - 1):
                    vi, vj = polygon[i], polygon[i+1]
                    dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
        update = False 
    def strainyyCallback(sender):
        update = True
        for pt in query_points:
            pt[3] = dpg.get_value(sender)
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        dpg.set_value('plot', [strain_xx, psi])
        if update:
            plt.plot(strain_xx, psi)
            plt.show()
        update = False 
    
    def strainxyCallback(sender):
        for pt in query_points:
            pt[4] = dpg.get_value(sender)
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        # dpg.set_value('plot', [strain_xx, psi])

    dpg.create_context()
    dpg.create_viewport(title='Neural Constitutive Model',width=3700, height=2200)
    dpg.setup_dearpygui()
    dpg.set_global_font_scale(2.5)
    x = []
    y = []
    for polygon in polygons:
        for i in range(len(polygon) - 1):
            vi, vj = polygon[i], polygon[i+1]
            x.append(vi)
            y.append(vj)

    with dpg.window(label="Tiling Explorer"):
        with dpg.group(horizontal=True):
            dpg.add_text("Tiling Parameters")
            dpg.add_slider_float(default_value=0.5 * (range_t0[0] + range_t0[1]), min_value = range_t0[0], max_value=range_t0[1],width=200.0, height=500.0, callback= t1Callback)
            dpg.add_slider_float(default_value=0.5 * (range_t1[0] + range_t1[1]), min_value = range_t1[0], max_value=range_t1[1],width=200.0, height=500.0, callback= t2Callback)
            dpg.add_text("Strain Rate")
            dpg.add_slider_float(default_value=0.001, min_value = 0.15, max_value=0.5,width=200.0, height=500.0, callback= strainyyCallback)
        with dpg.theme(tag="plot_theme"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (150, 255, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 10.0, category=dpg.mvThemeCat_Plots)
                
            
        with dpg.group(horizontal=True):
            with dpg.plot(label="Line Series", height=2000, width=2600):
                # optionally create legend
                dpg.add_plot_legend()

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="strain_xx")
                dpg.add_plot_axis(dpg.mvYAxis, label="energy density", tag="y_axis")

                # series belong to a y axis
                dpg.add_line_series(strain_xx, psi, label="enery", parent="y_axis", tag="plot")
                dpg.bind_item_theme("plot", "plot_theme")
            with dpg.drawlist(width=1000, height=1000, tag="structure"):
                dpg.add_draw_layer(tag="draw", parent="structure")
                for polygon in polygons:
                    for i in range(len(polygon) - 1):
                        vi, vj = polygon[i], polygon[i+1]
                        dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
                
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    return

def sample():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(52) + "/")
    n_tiling_params = 2
    model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
    model.load_weights(save_path + "full40k" + '.tf')

    range_t0 = [0.1, 0.2]
    range_t1 = [0.5, 0.8]
    query_points = []
    n_sp_t0 = 10
    for j in range(n_sp_t0):
        dt0 = (range_t0[1] - range_t0[0]) / float(n_sp_t0)
        t0 = range_t0[0] + float(j) * dt0
        x0 = np.array([t0, 0.6])
        range_strain = [-0.3, 0.1]
        n_sp_strain = 20
        delta_strain = (range_strain[1] - range_strain[0]) / float(n_sp_strain)
        strain_xx = []
        for i in range(n_sp_strain):
            strain = range_strain[1] - delta_strain * float(i)
            strain_xx.append(strain)
            query_points.append(np.hstack((x0, np.array([strain, 0.06, 0.0004, 0.0004]))))
    
    model_input = tf.convert_to_tensor(query_points)

    _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
    for j in range(n_sp_t0):
        plt.plot(strain_xx, psi[j * 20:(j+1)*20])
    plt.savefig("energy.png", dpi=300)
    plt.close()

def sampleDir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(52) + "/")
    n_tiling_params = 2
    model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
    model.load_weights(save_path + "full40k" + '.tf')

    range_t0 = [0.1, 0.2]
    range_t1 = [0.5, 0.8]
    x0 = np.array([0.17, 0.55])
    query_points = []
    n_sp_t0 = 10
    dir = np.array([-0.006, 0.02])
    for j in range(n_sp_t0):
        x = x0 + dir * float(j)
        # range_strain = [-0.2, 0.05]
        range_strain = [0.01, 0.3]
        n_sp_strain = 20
        delta_strain = (range_strain[1] - range_strain[0]) / float(n_sp_strain)
        strain_xx = []
        for i in range(n_sp_strain):
            strain = range_strain[1] - delta_strain * float(i)
            strain_xx.append(strain)
            query_points.append(np.hstack((x, np.array([0.05, strain, 0.001, 0.001]))))
    
    model_input = tf.convert_to_tensor(query_points)

    _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
    for j in range(n_sp_t0):
        plt.plot(strain_xx, psi[j * 20:(j+1)*20], label=str(x0 + dir *float(j)))
    plt.title("0.05, " + str(range_strain) + " 0.001, 0.001")
    plt.legend(loc="upper left")
    plt.savefig("energy.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # sampleDir()
    # explorer()
    optimizeStiffnessProfile()
    # optimizeUniaxialStrain()
    # optimizeUniaxialStrainNH()
    # optimizeStessProfile()
    # optimizeDensityProfile()