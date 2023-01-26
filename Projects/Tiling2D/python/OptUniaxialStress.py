
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
from scipy.optimize import BFGS
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint

import time
from Derivatives import *
from Optimization import *

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
def objUniaxialStress(n_tiling_params, ti, uniaxial_strain, theta, model):
    batch_dim = uniaxial_strain.shape[0]
    thetas = tf.tile(theta, (uniaxial_strain.shape[0], 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    ti = tf.expand_dims(ti, 0)
    
    with tf.GradientTape() as tape:
        
        ti_batch = tf.tile(ti, (batch_dim, 1))
        inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)
        tape.watch(inputs)
        psi = model(inputs, training=False)
        dedlambda = tape.gradient(psi, inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))
        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.einsum("ij,ij->i",d, Sd)
    del tape
    return tf.squeeze(dTSd)

@tf.function
def objGradUniaxialStress(n_tiling_params, ti, uniaxial_strain, theta, model):
    batch_dim = uniaxial_strain.shape[0]
    thetas = tf.tile(theta, (uniaxial_strain.shape[0], 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    ti = tf.expand_dims(ti, 0)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(ti)
        tape.watch(uniaxial_strain)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)
        psi = model(inputs, training=False)
        dedlambda = tape.gradient(psi, inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))

        # s, u, v = tf.linalg.svd(stress_tensor, full_matrices=True, compute_uv=True)
        # s = tf.math.sqrt(s)
        # F = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
        # J = tf.linalg.det(F)
        # J =  tf.reshape(tf.tile(tf.expand_dims(J, 1), tf.constant([1, 4])), (batch_dim, 2, 2))
        
        # cauchy_stress = tf.divide(tf.linalg.matmul(F, tf.linalg.matmul(stress_tensor, F, transpose_b=True)), J)
        # Sd = tf.linalg.matvec(cauchy_stress, d)
        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
        # print(dTSd)
    grad = tape.jacobian(dTSd, ti)
    dOdE = tape.jacobian(dTSd, uniaxial_strain)
    del tape
    return tf.squeeze(dTSd), tf.squeeze(grad), tf.squeeze(dOdE)

@tf.function
def objGradUniaxialStressColocation(n_tiling_params, inputs, theta, model, stress_targets):
    stress_targets = tf.expand_dims(stress_targets, 1)
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    indices = np.arange(2, len(inputs), 1)

    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)

        dedlambda = tape.gradient(psi, nn_inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))
        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
        
        energy = tf.reduce_sum(tf.math.square(dTSd - stress_targets), axis=0)
        # energy += tf.cast(tf.reduce_sum(psi, axis=0), tf.float64)
    grad = tape.gradient(energy, inputs)
    del tape
    return tf.squeeze(energy), tf.squeeze(grad)

@tf.function
def hessUniaxialStressObjColocation(n_tiling_params, inputs, theta, model, stress_targets):
    stress_targets = tf.expand_dims(stress_targets, 1)
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    indices = np.arange(2, len(inputs), 1)

    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            ti = tf.gather(inputs, [0, 1], axis=0)
            strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
            ti = tf.expand_dims(ti, 0)
            ti_batch = tf.tile(ti, (batch_dim, 1))
            nn_inputs = tf.concat((ti_batch, strain), axis=1)
            psi = model(nn_inputs, training=False)

            dedlambda = tape.gradient(psi, nn_inputs)
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            stress_xx = tf.gather(stress, [0], axis = 1)
            stress_yy = tf.gather(stress, [1], axis = 1)
            stress_xy = tf.gather(stress, [2], axis = 1)
            stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
            stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))
            Sd = tf.linalg.matvec(stress_tensor, d)
            
            dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
            
            energy = tf.reduce_sum(tf.math.square(dTSd - stress_targets), axis=0)
        grad = tape.gradient(energy, inputs)
    hess = tape_outer.jacobian(grad, inputs)
    del tape
    del tape_outer
    return tf.squeeze(hess)

@tf.function
def objGradPhiColocation(n_tiling_params, inputs, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    indices = np.arange(2, len(inputs), 1)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)
        psi = tf.math.reduce_sum(psi, axis=0)
    grad = tape.gradient(psi, inputs)
    del tape
    return tf.squeeze(psi), tf.squeeze(grad)

@tf.function
def hessPhiColocation(n_tiling_params, inputs, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    indices = np.arange(2, len(inputs), 1)
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            ti = tf.gather(inputs, [0, 1], axis=0)
            strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
            ti = tf.expand_dims(ti, 0)
            ti_batch = tf.tile(ti, (batch_dim, 1))
            nn_inputs = tf.concat((ti_batch, strain), axis=1)
            psi = model(nn_inputs, training=False)
            psi = tf.math.reduce_sum(psi, axis=0)
        grad = tape.gradient(psi, inputs)
    hess = tape_outer.jacobian(grad, inputs)
    del tape
    del tape_outer
    return tf.squeeze(hess)


@tf.function
def objUniaxialStressColocation(n_tiling_params, inputs, theta, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        indices = tf.range(2, len(inputs), 1)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)

        dedlambda = tape.gradient(psi, nn_inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))

        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
    del tape
    return tf.squeeze(dTSd)

@tf.function
def gradUniaxialStressColocation(n_tiling_params, inputs, theta, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        indices = tf.range(2, len(inputs), 1)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)

        dedlambda = tape.gradient(psi, nn_inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))

        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
        # print(dTSd)
    grad = tape.jacobian(dTSd, inputs)
    del tape
    return tf.squeeze(grad)


@tf.function
def objUniaxialStressColocationNormal(n_tiling_params, inputs, theta, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((-tf.math.sin(thetas),
                        tf.math.cos(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        indices = tf.range(2, len(inputs), 1)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)

        dedlambda = tape.gradient(psi, nn_inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))
        
        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
    del tape
    return tf.squeeze(dTSd)

@tf.function
def objUniaxialStressColocationEV(n_tiling_params, inputs, theta, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((-tf.math.sin(thetas),
                        tf.math.cos(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        indices = tf.range(2, len(inputs), 1)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)

        dedlambda = tape.gradient(psi, nn_inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))
        e, v = tf.linalg.eigh(stress_tensor)
        
        v_dot_d = tf.expand_dims(tf.einsum("ij,ij->i",d, tf.squeeze(tf.gather(v, [0], axis=2))), 1)
        
    del tape
    return tf.squeeze(v_dot_d)

# @tf.function
def gradUniaxialStressColocationEV(n_tiling_params, inputs, theta, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((-tf.math.sin(thetas),
                        tf.math.cos(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        indices = tf.range(2, len(inputs), 1)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)

        dedlambda = tape.gradient(psi, nn_inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))
        e, v = tf.linalg.eigh(stress_tensor)
        
        v_dot_d = tf.expand_dims(tf.einsum("ij,ij->i",d, tf.squeeze(tf.gather(v, [0], axis=2))), 1)
    grad = tape.jacobian(v_dot_d, inputs)
    print(grad)
    exit(0)
    del tape
    return tf.squeeze(grad)

@tf.function
def objUniaxialStressColocationEnergy(n_tiling_params, inputs, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        indices = tf.range(2, len(inputs), 1)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)
    del tape
    return tf.squeeze(psi)

@tf.function
def gradUniaxialStressColocationNormal(n_tiling_params, inputs, theta, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((-tf.math.sin(thetas),
                        tf.math.cos(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        ti = tf.gather(inputs, [0, 1], axis=0)
        indices = tf.range(2, len(inputs), 1)
        strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
        ti = tf.expand_dims(ti, 0)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)

        dedlambda = tape.gradient(psi, nn_inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))

        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
        # print(dTSd)
    grad = tape.jacobian(dTSd, inputs)
    del tape
    return tf.squeeze(grad)

@tf.function
def hessUniaxialStressColocation(n_tiling_params, inputs, theta, model):
    batch_dim = int(len(inputs) - n_tiling_params) // 3
    thetas = tf.tile(theta, (batch_dim, 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            ti = tf.gather(inputs, [0, 1], axis=0)
            indices = tf.range(2, len(inputs), 1)
            strain = tf.reshape(tf.gather(inputs, indices, axis=0), (batch_dim, 3))
            ti = tf.expand_dims(ti, 0)
            ti_batch = tf.tile(ti, (batch_dim, 1))
            nn_inputs = tf.concat((ti_batch, strain), axis=1)
            psi = model(nn_inputs, training=False)

            dedlambda = tape.gradient(psi, nn_inputs)
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            stress_xx = tf.gather(stress, [0], axis = 1)
            stress_yy = tf.gather(stress, [1], axis = 1)
            stress_xy = tf.gather(stress, [2], axis = 1)
            stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
            stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))

            Sd = tf.linalg.matvec(stress_tensor, d)
            
            dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
        grad = tape.jacobian(dTSd, inputs)
    hess = tape_outer.jacobian(grad, inputs)
    del tape
    del tape_outer
    return tf.squeeze(hess)

@tf.function
def hessUniaxialStress(n_tiling_params, ti, uniaxial_strain, theta, model):
    batch_dim = uniaxial_strain.shape[0]
    thetas = tf.tile(theta, (uniaxial_strain.shape[0], 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    ti = tf.expand_dims(ti, 0)
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(ti)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ti)
            ti_batch = tf.tile(ti, (batch_dim, 1))
            inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)
            psi = model(inputs, training=False)
            dedlambda = tape.gradient(psi, inputs)
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            stress_xx = tf.gather(stress, [0], axis = 1)
            stress_yy = tf.gather(stress, [1], axis = 1)
            stress_xy = tf.gather(stress, [2], axis = 1)
            stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
            stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))
            Sd = tf.linalg.matvec(stress_tensor, d)
            
            dTSd = tf.einsum("ij,ij->i",d, Sd)
            
        grad = tape.gradient(dTSd, ti)
    hess = tape_outer.batch_jacobian(grad, ti)
    del tape
    del tape_outer
    return tf.squeeze(hess)

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

@tf.function
def computedPsidEEnergy(n_tiling_params, model_input, model):
    with tf.GradientTape() as tape:
        tape.watch(model_input)
        psi = model(model_input, training=False)
        dedlambda = tape.gradient(psi, model_input)
        batch_dim = psi.shape[0]
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
    del tape
    return tf.squeeze(stress)

@tf.function
def computedPsidEGrad(n_tiling_params, inputs, model):
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

@tf.function
def computedStressdp(n_tiling_params, opt_model_input, model):
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(opt_model_input)
        with tf.GradientTape() as tape:
            tape.watch(opt_model_input)
            
            elastic_potential = model(opt_model_input, training=False)
            dedlambda = tape.gradient(elastic_potential, opt_model_input)
            batch_dim = elastic_potential.shape[0]
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    dstress_dp = tape_outer.batch_jacobian(stress, opt_model_input)[:, :, 0:n_tiling_params]
    del tape
    del tape_outer
    return tf.squeeze(dstress_dp)



def optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, 
    theta, strain, tiling_params, verbose = True):
    
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
        ev_H = np.linalg.eigvals(H)
        min_ev = np.min(ev_H)
        if min_ev < 0.0:
            H += np.diag(np.full(len(x),min_ev + 1e-6))
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

    # stress = computedPsidEEnergy(n_tiling_params, opt_model_input, model)
    # stress = stress.numpy()
    # stress_tensor = np.reshape(np.hstack((stress[0], stress[2], stress[2], stress[1])), (2, 2))
    # print(np.dot(n, np.matmul(stress_tensor, n)))
    
    return result.x, dqdp

def optimizeUniaxialStrainSingleDirection(model, n_tiling_params, 
    theta, strain, tiling_params, verbose = True):
    
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
        ev_H = np.linalg.eigvals(H)
        min_ev = np.min(ev_H)
        if min_ev < 0.0:
            H += np.diag(np.full(len(x),min_ev + 1e-4))
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
    
    return result.x


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

def plotNNFDCurves():
    bounds = []
    IH = 21
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    theta = 0.0

    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti_init = np.array([0.115, 0.765])
        ti_NN = np.array([0.11611517, 0.6579828])
        ti_LBFGS = np.array([0.105, 0.79499])
        ti_MMA = np.array([0.101928, 0.509179])
        ti_GD = np.array([0.1, 0.54811])
        sample_idx = [2, 7, -1]
        theta = 0.0
    
    
    save_path = os.path.join(current_dir, 'Models/IH' + str(IH) + "/")
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH"+str(IH) + '.tf')

    
    strain_range = [-0.05, 0.1]
    strain_samples = np.arange(strain_range[0], strain_range[1], 0.01)
    for i in range(len(strain_samples)):
        strain = strain_samples[i]
        if strain < 0:
            strain_samples[i] = strain - 0.5 * strain * strain
        else:
            strain_samples[i] = strain + 0.5 * strain * strain
    
    def obj(ti):
        uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                    theta, strain_samples, 
                                    ti, model)
        ti_TF = tf.convert_to_tensor(ti)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        obj_init, _ , _ = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        obj_init = obj_init.numpy()
        return obj_init
    
    obj_NN_LBFGS = obj(ti_NN)
    obj_FD_LBFGS = obj(ti_LBFGS)
    obj_FD_MMA = obj(ti_MMA)
    obj_FD_GD = obj(ti_GD)
    obj_init = obj(ti_init)


    for i in range(len(strain_samples)):
        strain_samples[i] = strain_samples[i] * 100.0

    strain_points = strain_samples[sample_idx]
    stress_targets = [-0.00598749,  0.00477436,  0.04006726]
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(strain_samples, obj_init, label="initial guess", linewidth=2.0, zorder=0)
    plt.plot(strain_samples, obj_FD_GD, label="FD-PGD", linewidth=2.0, zorder=0, color = "#00ABBD")
    plt.plot(strain_samples, obj_FD_MMA, label="FD-MMA", linewidth=2.0, zorder=0, color = "#48D1CC")
    plt.plot(strain_samples, obj_FD_LBFGS, label="FD-LBFGS-B", linewidth=2.0, zorder=0, color = "#026E81")
    plt.plot(strain_samples, obj_NN_LBFGS, label="NN-LBFGS-B", linewidth=2.0, zorder=0, color = "#FF9933")
    plt.scatter(strain_points, stress_targets, marker='+', s=200.0, zorder=5, color = "red", label = "targets")
    plt.legend(loc="upper left")
    plt.savefig("NN_FD_comparison.png", dpi=300)
    os.system("convert NN_FD_comparison.png -trim NN_FD_comparison.png")
    plt.close()

def optimizeUniaxialStressSA():
    bounds = []
    IH = 21
    current_dir = os.path.dirname(os.path.realpath(__file__))
    plot_GT = False
    theta = 0.0

    strain_range = [-0.1, 0.0]
    n_sp_strain = 30
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))
    # strain_samples = np.arange(strain_range[0], strain_range[1], 0.01)
    for i in range(len(strain_samples)):
        strain = strain_samples[i]
        # if strain < 0:
        #     strain_samples[i] = strain - 0.5 * strain * strain
        # else:
        #     strain_samples[i] = strain + 0.5 * strain * strain

    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti0 = np.array([0.106, 0.65])
        # ti0 = np.array([0.115, 0.765])
        # ti0 = np.array([0.2, 0.7])
        sample_idx = [2, 7, -1]
        theta = 0.0
    elif IH == 50:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        # ti0 = np.array([0.2308, 0.5])
        # ti0 = np.array([0.2903, 0.6714])
        ti0 = np.array([0.23, 0.57])
        sample_idx = [2, 7, -1]
    elif IH == 67:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1])
        # ti0 = np.array([0.2308, 0.8969])
        ti0 = np.array([0.2131, 0.9464])
        sample_idx = [2, 7, -1]
        theta = 0.5 * np.pi
    elif IH == 28:
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        # ti0 = np.array([0.4528, 0.5])
        # ti0 = np.array([0.4, 0.8])
        sample_idx = [2, 7, -1]
        theta = 0.1 * np.pi
        ti0 = np.array([0.3, 0.5])
        ti0 = np.array([0.03411184, 0.37176683])
        # ti_target = np.array([0.2205, 0.6016])
        sample_idx = np.arange(0, n_sp_strain, 6)
        theta = 0.5 * np.pi
    elif IH == 1:
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        # ti0 = np.array([0.1224, 0.5, 0.1434, 0.625])
        # ti0 = np.array([0.1, 0.5, 0.13, 0.45])
        ti0 = np.array([0.11437572, 0.3939947,  0.06671481, 0.45272221])
        # ti_target = np.array([0.18, 0.5, 0.07, 0.59])
        # ti0 = np.array([ 0.12, 0.504, 0.1, 0.625])
        # sample_idx = [2, 8, -3]
        sample_idx = np.arange(0, n_sp_strain, 4)
        theta = 0.5 * np.pi

    # sample_idx = np.arange(0, n_sp_strain, 4)
    # theta = 0.5 * np.pi
    
    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)

    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    
    # uniaxial_strain = []
    # for strain in strain_samples:
    #     uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, ti0, False)
    #     uniaxial_strain.append(uni_strain)

    def computeTarget(ti):
        uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                    theta, strain_samples, 
                                    ti, model)
        ti_TF = tf.convert_to_tensor(ti)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        obj, _ , _ = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        obj = obj.numpy()
        return obj

    obj_init = computeTarget(ti0)
    # obj_target = computeTarget(ti_target)

    if plot_GT:
        info = ""
        for data in obj_init[sample_idx]:
            info += str(data) + ", "
        print(info)
        # exit(0)
    # stress_targets = [obj[2], obj[5], obj[-1]]
    if IH == 21:
        stress_targets = [-0.00598749,  0.00477436,  0.04006726]
        # stress_targets = [-0.00609539,  0.01901703,  0.08067064] #ti0 = np.array([0.115, 0.75]) green
    elif IH == 50:
        stress_targets = [-0.0001821,   0.00013866,  0.00098491] #ti0 = np.array([0.2903, 0.6714]) green
    elif IH == 67:
        stress_targets = [-0.00661061,  0.00590675,  0.01050055]
    elif IH == 28:
        stress_targets = [-0.00287036,  0.00225446,  0.01827932]
    # elif IH == 1:
    #     # stress_targets = [-0.12621401, -0.12035293, -0.11669242, -0.09796586, -0.01525695]
    #     stress_targets = obj_target[sample_idx]
    
    # stress_targets = obj_init[sample_idx]
    # stress_targets = [-0.265, -0.26, -0.25, -0.24, -0.21, -0.16,-0.1, -0.039]#IH01
    # stress_targets = [-0.309, -0.288, -0.258, -0.235, -0.222, -0.205, -0.148, -0.0532]#IH01
    # stress_targets = [-0.239, -0.23, -0.218, -0.209, -0.193, -0.147, -0.08714, -0.02885] #IH28
    def objAndGradient(x):
        uniaxial_strain = []
        dqdp = []
        for strain in strain_samples[sample_idx]:
            uni_strain, dqidpi = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, x, False)
            uniaxial_strain.append(uni_strain)
            dqdp.append(dqidpi)
        ti_TF = tf.convert_to_tensor(x)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        stress_d, stress_d_grad, dOdE = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        # print(stress_d_grad)
        # exit(0)
        stress_d = stress_d.numpy()
        stress_d_grad = stress_d_grad.numpy()
        dOdE = dOdE.numpy()
        
        # stress_current = stress_d[sample_idx]#np.array([stress_d[2], stress_d[5], stress_d[-1]])
        stress_current = stress_d
        obj = (np.dot(stress_current - stress_targets, np.transpose(stress_current - stress_targets)) * 0.5).flatten() 
        grad = np.zeros((n_tiling_params))
        
        for i in range(len(sample_idx)):
            grad += (stress_current[i] - stress_targets[i]) * stress_d_grad[i].flatten() + \
                (stress_current[i] - stress_targets[i]) * np.dot(dOdE[i][i], dqdp[i][:3, :]).flatten()
        # for i in range(3):
        #     grad += (stress_current[i] - stress_targets[i]) * stress_d_grad[sample_idx[i]].flatten() + \
        #         (stress_current[i] - stress_targets[i]) * np.dot(dOdE[sample_idx[i]][sample_idx[i]], dqdp[sample_idx[i]][:3, :]).flatten()
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    def fdGradient(x0):
        eps = 5e-4
        _, grad = objAndGradient(x0)
        print(grad)
        E0, _ = objAndGradient(np.array([x0[0] - eps, x0[1]]))
        E1, _ = objAndGradient(np.array([x0[0] + eps, x0[1]]))
        fd_grad = []
        fd_grad.append((E1 - E0)/2.0/eps)
        E0, _ = objAndGradient(np.array([x0[0], x0[1] - eps]))
        E1, _ = objAndGradient(np.array([x0[0], x0[1] + eps]))
        fd_grad.append((E1 - E0)/2.0/eps)
        print(grad)
        print(fd_grad)

    # fdGradient(ti0)
    # exit(0)
    if not plot_GT:
        # result = minimize(objAndGradient, ti0, method='trust-constr', jac=True, options={'disp' : True}, bounds=bounds)
        tic = time.perf_counter()
        result = minimize(objAndGradient, ti0, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
        toc = time.perf_counter()
        print(f"Optimization takes {toc - tic:0.6f} seconds")
        # uniaxial_strain_opt = []
        # for strain in strain_samples:
        #     uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, result.x, False)
        #     uniaxial_strain_opt.append(uni_strain)
        uniaxial_strain_opt = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                        theta, strain_samples, 
                        result.x, model)
        obj_opt, _, _ = objGradUniaxialStress(n_tiling_params, tf.convert_to_tensor(result.x), tf.convert_to_tensor(uniaxial_strain_opt), tf.constant([[theta]]), model)
        obj_opt = obj_opt.numpy()
        print(result.x)

    for i in range(len(strain_samples)):
        strain_samples[i] = strain_samples[i] * 100.0

    strain_points = strain_samples[sample_idx]#[strain_samples[2], strain_samples[5], strain_samples[-1]]
    plt.plot(strain_samples, obj_init, label="stress initial", linewidth=3.0, zorder=0)
    if not plot_GT:
        plt.plot(strain_samples, obj_opt, label = "stress optimized", linewidth=3.0, zorder=0)
    plt.scatter(strain_points, stress_targets, marker='+', s=200.0, label = "targets", c="red", zorder=5)
    plt.legend(loc="upper left")
    # plt.xlabel("strain")
    # plt.ylabel("stress")
    plt.savefig("uniaxial_stress_IH"+str(IH)+".png", dpi=300)
    plt.close()




def optimizeUniaxialStress():
    IH = 21
    bounds = []
    current_dir = os.path.dirname(os.path.realpath(__file__))

    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
    
        model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
        model.load_weights(save_path + "IH21" + '.tf')

        ti0 = np.array([0.105, 0.65])
    # ti0 = np.array([0.115, 0.75])
    theta = 0.0
    d = np.array([np.cos(theta), np.sin(theta)])
    strain_range = [-0.05, 0.1]
    n_sp_strain = 10
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))
    
    # uniaxial_strain = []
    # for strain in strain_samples:
    #     green = strain #+ 0.5 * strain * strain
    #     uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, green, ti0, False)
    #     uniaxial_strain.append(uni_strain)
    # ti_TF = tf.convert_to_tensor(ti0)
    # uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
    # obj_init, _ , _ = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
    # obj_init = obj_init.numpy()

    uniaxial_strain_batch = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                            theta, strain_samples, 
                            ti0, model)

    obj_init, _ , _ = objGradUniaxialStress(n_tiling_params, tf.convert_to_tensor(ti0), 
                                tf.convert_to_tensor(uniaxial_strain_batch), tf.constant([[theta]]), model)
    obj_init = obj_init.numpy()
    
    sample_points_indices = [2, 5, len(uniaxial_strain_batch)-1]
    # sample_points_indices = [2]
    strain_init = []
    # stress_targets = []
    for i in range(len(sample_points_indices)):
        strain_init.append(uniaxial_strain_batch[sample_points_indices[i]])
    strain_init = np.array(strain_init)
    
    # stress_targets = [obj[2], obj[5], obj[-1]]
    if IH == 21:
        stress_targets = np.array([-0.00683177, 0.02369076, 0.06924471]) #ti0 = np.array([0.115, 0.75])
    # stress_targets = np.array([-0.006, 0.03, 0.06]) #ti0 = np.array([0.115, 0.75])
    # stress_targets = [-0.03666779, 0.03493858, 4.71111735]
    
    design_variables = np.hstack((ti0, strain_init.flatten()))

    # obj_init = objUniaxialStressColocation(n_tiling_params, tf.convert_to_tensor(design_variables), tf.constant([[theta]]), model)
    # obj_init = obj_init.numpy()
    
    for i in range(len(design_variables) - n_tiling_params):
        bounds.append([-100.0, 100.0])

    def cons_f(x):
        c = objUniaxialStressColocation(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        return c.numpy().flatten()
    def cons_J(x):
        dc = gradUniaxialStressColocation(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        return dc.numpy()

    def cons_H(x, v):
        hess = hessUniaxialStressColocation(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        hess = hess.numpy()
        Hv = np.zeros((len(x), len(x)))
        for i in range(len(v)):
            Hv += hess[i] * v[i]
        return Hv

    
    def consFuncStressNormal(x):
        c = objUniaxialStressColocationNormal(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        return c.numpy().flatten()
    def consJacStressNormal(x):
        dc = gradUniaxialStressColocationNormal(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        return dc.numpy()
    
    A = np.zeros((n_tiling_params + len(stress_targets), len(design_variables)))
    lb = []
    ub = []

    for i in range(n_tiling_params):
        A[i][i] = 1.0
        lb.append(bounds[i][0])
        ub.append(bounds[i][1])
    for i in range(len(stress_targets)):
        A[n_tiling_params + i, n_tiling_params + i * 3: n_tiling_params + i * 3 + 3] = computedCdE(d)
        lb.append(strain_samples[sample_points_indices[i]])
        ub.append(strain_samples[sample_points_indices[i]])
    
    
    uniaxial_stress_constraint = NonlinearConstraint(cons_f, stress_targets, 
                                    stress_targets, jac=cons_J, hess=BFGS())
    
    normal_stress_constraint = NonlinearConstraint(consFuncStressNormal, 
                                    np.zeros(len(stress_targets)), 
                                    np.zeros(len(stress_targets)), 
                                    jac=consJacStressNormal, hess=BFGS())
    
    uniaxial_strain_constraint = LinearConstraint(A, lb, ub)


    def hessian(x):
        hess = hessPhiColocation(n_tiling_params, tf.convert_to_tensor(x), model)
        H = hess.numpy()        
        # alpha = 1e-6
        # while not np.all(np.linalg.eigvals(H) > 1e-8):
        #     H += np.diag(np.full(len(x),alpha))
        #     alpha *= 10.0
        return hess

    def objAndGradient(x):

        obj, grad = objGradPhiColocation(n_tiling_params, tf.convert_to_tensor(x), model)
        obj = obj.numpy()
        grad = grad.numpy().flatten()

        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    def fdGradient(x0):
        eps = 1e-4
        fd_grad = []
        _, grad = objAndGradient(x0)
        for i in range(len(x0)):
            x0[i] -= eps
            E0, _ = cons_f(x0)
            x0[i] += 2.0 * eps
            E1, _ = cons_f(x0)
            x0[i] -= eps
            fd_grad.append((E1 - E0)/2.0/eps)
        print(grad)
        print(fd_grad)

    def fdConstraint(x0):
        eps = 1e-4
        fd_Jacobian = []
        Jacobian = cons_J(x0)
        for i in range(len(x0)):
            x0[i] -= eps
            E0 = cons_f(x0)
            x0[i] += 2.0 * eps
            E1 = cons_f(x0)
            x0[i] -= eps
            fd_Jacobian.append((E1 - E0)/2.0/eps)
        
        print(Jacobian)
        print(fd_Jacobian)

    # fdConstraint(design_variables)
    # exit(0)
    # result = minimize(objAndGradient, ti0, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
    result = minimize(objAndGradient, design_variables, 
            method='trust-constr', jac=True, 
            hess=hessian,
            options={'disp' : True}, 
            # bounds=bounds,
            constraints = [uniaxial_strain_constraint, 
                            uniaxial_stress_constraint,
                            normal_stress_constraint])

    
    # obj_opt = objUniaxialStressColocation(n_tiling_params, tf.convert_to_tensor(result.x), tf.constant([[theta]]), model)
    # obj_opt = obj_opt.numpy()

    # uniaxial_strain = []
    # for strain in strain_samples:
    #     green = strain #+ 0.5 * strain * strain
    #     # uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, green, result.x[0:2], False)
    #     uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, green, ti0, False)
    #     uniaxial_strain.append(uni_strain)
    
    # for i in range(len(sample_points_indices)):
    #     uniaxial_strain[sample_points_indices[i]] = np.array(result.x[n_tiling_params + i * 3:n_tiling_params + i*3+3])
    
    
    # ti_TF = tf.convert_to_tensor(result.x[0:2])
    # ti_TF = tf.convert_to_tensor(ti0)
    # uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
    # obj_opt, _ , _ = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
    # obj_opt = obj_opt.numpy()

    uniaxial_strain_batch_opt = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
        theta, strain_samples, 
        result.x[0:2], model)

    obj_opt, _ , _ = objGradUniaxialStress(n_tiling_params, tf.convert_to_tensor(result.x[0:2]), 
                        tf.convert_to_tensor(uniaxial_strain_batch_opt), tf.constant([[theta]]), model)
    obj_opt = obj_opt.numpy()

    strain_points = []
    for idx in sample_points_indices:
        strain_points.append([strain_samples[idx] * 100.0])
    plt.plot(strain_samples, obj_init, label="stress initial", linewidth=3.0)
    plt.plot(strain_samples, obj_opt, label = "stress optimized", linewidth=3.0)
    # plt.plot(strain_samples, obj_init, label="opt batch")
    # plt.plot(strain_samples, obj_opt, label = "opt separate")
    plt.scatter(strain_points, stress_targets, s=6.0)
    plt.legend(loc="upper left")
    # plt.xlabel("strain")
    # plt.ylabel("stress")
    plt.savefig("uniaxial_stress_IH"+str(IH)+".png", dpi=300)
    plt.close()
    np.set_printoptions(suppress=True)
    print(result.x)
    print(objUniaxialStressColocationNormal(n_tiling_params, tf.convert_to_tensor(result.x), tf.constant([[theta]]), model))
    # print(A.dot(result.x))
    # for i in range(len(sample_points_indices)):
    #     x = uniaxial_strain_batch_opt[sample_points_indices[i]]
    #     strain_tensor = np.reshape([x[0], 0.5 * x[-1], 0.5 * x[-1], x[1]], (2, 2))
    #     dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
    #     print(x, dTEd)
        
        
def optimizeUniaxialStressConstraints():
    n_tiling_params = 2
    bounds = []
    bounds.append([0.105, 0.195])
    bounds.append([0.505, 0.795])
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH21" + '.tf')

    ti0 = np.array([0.15, 0.6])
    # ti0 = np.array([0.115, 0.75])
    theta = 0.0
    d = np.array([np.cos(theta), np.sin(theta)])
    strain_range = [-0.05, 0.1]
    n_sp_strain = 10
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))
    

    uniaxial_strain_batch = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                            theta, strain_samples, 
                            ti0, model)

    obj_init, _ , _ = objGradUniaxialStress(n_tiling_params, tf.convert_to_tensor(ti0), 
                                tf.convert_to_tensor(uniaxial_strain_batch), tf.constant([[theta]]), model)
    obj_init = obj_init.numpy()
    
    sample_points_indices = [2, 5, len(uniaxial_strain_batch)-1]
    
    strain_init = []
    # stress_targets = []
    for i in range(len(sample_points_indices)):
        strain_init.append(uniaxial_strain_batch[sample_points_indices[i]])
    strain_init = np.array(strain_init)
    
    # stress_targets = [obj[2], obj[5], obj[-1]]
    stress_targets = np.array([-0.00683177, 0.02369076, 0.06924471]) #ti0 = np.array([0.115, 0.75])
    # stress_targets = np.array([-0.006, 0.03, 0.06]) #ti0 = np.array([0.115, 0.75])
    # stress_targets = [-0.03666779, 0.03493858, 4.71111735]
    
    design_variables = np.hstack((ti0, strain_init.flatten()))


    
    def consFuncStressNormal(x):
        c = objUniaxialStressColocationNormal(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        return c.numpy().flatten()
    def consJacStressNormal(x):
        dc = gradUniaxialStressColocationNormal(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        return dc.numpy()

    def evCons(x):
        c = objUniaxialStressColocationEV(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        return c.numpy().flatten()

    def evConsJac(x):
        dc = gradUniaxialStressColocationEV(n_tiling_params, tf.convert_to_tensor(x), tf.constant([[theta]]), model)
        return dc.numpy()

    A = np.zeros((n_tiling_params + len(stress_targets), len(design_variables)))
    lb = []
    ub = []

    for i in range(n_tiling_params):
        A[i][i] = 1.0
        lb.append(bounds[i][0])
        ub.append(bounds[i][1])
    for i in range(len(stress_targets)):
        A[n_tiling_params + i, n_tiling_params + i * 3: n_tiling_params + i * 3 + 3] = computedCdE(d)
        lb.append(strain_samples[sample_points_indices[i]])
        ub.append(strain_samples[sample_points_indices[i]])
    
    
    
    normal_stress_constraint = NonlinearConstraint(consFuncStressNormal, 
                                    np.zeros(len(stress_targets)), 
                                    np.zeros(len(stress_targets)), 
                                    jac=consJacStressNormal, hess=BFGS())

    ev_constraint = NonlinearConstraint(evCons, 
                                    np.zeros(len(stress_targets)), 
                                    np.zeros(len(stress_targets)), 
                                    jac=evConsJac, hess=BFGS())
    
    uniaxial_strain_constraint = LinearConstraint(A, lb, ub)


    def hessian(x):
        hess = hessUniaxialStressObjColocation(n_tiling_params, 
                            tf.convert_to_tensor(x), 
                            tf.constant([[theta]]), model, 
                            tf.convert_to_tensor(stress_targets))
        H = hess.numpy()        
        # alpha = 1e-6
        # while not np.all(np.linalg.eigvals(H) > 1e-8):
        #     H += np.diag(np.full(len(x),alpha))
        #     alpha *= 10.0
        return hess

    def objAndGradient(x):

        obj, grad = objGradUniaxialStressColocation(n_tiling_params, 
                            tf.convert_to_tensor(x), 
                            tf.constant([[theta]]), model, 
                            tf.convert_to_tensor(stress_targets))
        obj = obj.numpy()
        grad = grad.numpy().flatten()

        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    result = minimize(objAndGradient, design_variables, 
            method='trust-constr', jac=True, 
            hess=hessian,
            options={'disp' : True}, 
            # bounds=bounds,
            constraints = [uniaxial_strain_constraint,ev_constraint,
                            normal_stress_constraint])

    uniaxial_strain_batch_opt = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
        theta, strain_samples, 
        result.x[0:2], model)

    obj_opt, _ , _ = objGradUniaxialStress(n_tiling_params, tf.convert_to_tensor(result.x[0:2]), 
                        tf.convert_to_tensor(uniaxial_strain_batch_opt), tf.constant([[theta]]), model)
    obj_opt = obj_opt.numpy()

    strain_points = []
    for idx in sample_points_indices:
        strain_points.append([strain_samples[idx]])
    plt.plot(strain_samples, obj_init, label="stress initial")
    plt.plot(strain_samples, obj_opt, label = "stress optimized")
    # plt.plot(strain_samples, obj_init, label="opt batch")
    # plt.plot(strain_samples, obj_opt, label = "opt separate")
    plt.scatter(strain_points, stress_targets, s=4.0)
    plt.legend(loc="upper left")
    plt.xlabel("strain")
    plt.ylabel("stress")
    plt.savefig("uniaxial_stress.png", dpi=300)
    plt.close()
    np.set_printoptions(suppress=True)
    print(result.x)
    print(objUniaxialStressColocationNormal(n_tiling_params, tf.convert_to_tensor(result.x), tf.constant([[theta]]), model))
    # print(A.dot(result.x))
    # for i in range(len(sample_points_indices)):
    #     x = uniaxial_strain_batch_opt[sample_points_indices[i]]
    #     strain_tensor = np.reshape([x[0], 0.5 * x[-1], 0.5 * x[-1], x[1]], (2, 2))
    #     dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
    #     print(x, dTEd)

def plotEnergyAlongDirection():
    n_tiling_params = 2
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # save_path = os.path.join(current_dir, 'Models/' + str(334) + "/")
    save_path = os.path.join(current_dir, 'Models/IH21/')
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH21" + '.tf')
    # cauchy = [-0.025-0.5*(-0.025 * -0.025), 0.025+ 0.5 * 0.025*0.025, 0.085 + 0.5 * 0.085*0.085]
    cauchy = [-0.025, 0.025, 0.085]
    strain_samples = []
    for strain in cauchy:
        if (strain < 0):
            strain_samples.append(strain - 0.5 * strain* strain)
        else:
            strain_samples.append(strain + 0.5 * strain* strain)
        
    
    # strain_samples = [-0.025, 0.025, 0.085]
    theta = 0.0
    # ti = np.array([0.139, 0.55]) # good
    ti = np.array([0.15, 0.55]) # good
    # uniaxial_strain_batch = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
    #                             theta, strain_samples, 
    #                             ti, model)
    # ti_TF = tf.convert_to_tensor(ti)
    # uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain_batch)
    
    # stress_d = objUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
    # print(stress_d)
    # exit(0)
    # test_dir = np.array([0.801549, 0.597929])
    # test_dir = np.array([0.130998, -0.0189072])
    test_dir = np.array([1.0, 0.0])
    # test_dir = np.array([0.195, 0.795]) - np.array([0.105, 0.505])
    # test_dir /= np.linalg.norm(test_dir)
    step = 50
    step_size = 1e-6
    steps = np.arange(-0.5 * float(step) * step_size, 0.5 * float(step) * step_size, step_size)
    
    theta = 0.0
    stress_targets = [-0.0062526045597, 0.0239273131735, 0.0717615511869]
    # stress_targets = [-0.00581623,  0.02364756,  0.06924607]
    obj = []
    x_axis = []
    cnt = 0
    for xi in steps:
        # cnt += 1
        # if (cnt > 50):
        #     break
        ti_step = ti + xi * test_dir
        x_axis.append(xi)
        uniaxial_strain_batch = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                theta, strain_samples, 
                                ti_step, model)
        # uniaxial_strain_batch = []
        # for strain in strain_samples:
        #     uni_strain = optimizeUniaxialStrainSingleDirection(model, 
        #                 n_tiling_params, theta, strain, ti_step, True)
        #     uniaxial_strain_batch.append(uni_strain)
        ti_TF = tf.convert_to_tensor(ti_step)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain_batch)
        # print("tiling params", ti_TF)
        # print("stress", uniaxial_strain_TF)
        stress_d = objUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        error = (stress_d.numpy().flatten() - stress_targets)
        # print(error)
        obj.append(0.5 * np.dot(error, error))
    print(obj)
    # print(steps.tolist())
    plt.plot(x_axis, obj, linewidth=3.0, label="energy_NN")
    plt.legend(loc="upper left")
    plt.savefig("energy_NN_large.png", dpi=300)
    

def loadModel(IH):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    bounds = []

    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti_default = np.array([0.1045, 0.65])
    elif IH == 22:
        n_tiling_params = 3
        # bounds.append([0.105, 0.195])
        # bounds.append([0.505, 0.795])
        bounds = [[0.1,0.3], [0.3, 0.7], [0.0, 0.3]]
        ti_default = np.array([0.2308, 0.5, 2253])
    elif IH == 29:
        n_tiling_params = 1
        bounds.append([0.005, 1.0])
        ti_default = np.array([0.3669])
    elif IH == 50:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti_default = np.array([0.2308, 0.5])
    elif IH == 67:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1]) 
        ti_default = np.array([0.2308, 0.8696])
    elif IH == 28:
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        ti_default = np.array([0.4528, 0.5])
    elif IH == 1:
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        ti_default = np.array([0.1224, 0.5, 0.1434, 0.625])
    
    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)

    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    return model, n_tiling_params, ti_default, bounds

def computeStressCurve(IH, ti):
    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    strain_range = [-0.1, 0.0]
    n_sp_strain = 30
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))

    
    theta = 0.5 * np.pi
    uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                theta, strain_samples, 
                                ti, model)
    ti_TF = tf.convert_to_tensor(ti)
    uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
    obj, _ , _ = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
    obj = obj.numpy()
    return obj

def teaserPlot():
    strain_range = [-0.1, 0.0]
    n_sp_strain = 30
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))
    sample_idx = np.arange(0, n_sp_strain, 4)

    # good
    stress_init = computeStressCurve(1, np.array([0.1224, 0.5, 0.1434, 0.625]))
    stress_opt1 = computeStressCurve(28, np.array([0.00930862, 0.67854564]))

    # good
    # stress_init = computeStressCurve(1, np.array([0.1224, 0.5, 0.1434, 0.625]))
    stress_opt2 = computeStressCurve(1, np.array([0.0667531,  0.65467978, 0.11134775, 0.65909504]))

    # good
    # stress_init = computeStressCurve(1, np.array([0.1224, 0.5, 0.1434, 0.625]))
    stress_opt3 = computeStressCurve(1, np.array([0.17194996, 0.42948579, 0.05507779, 0.66815703]))
    strain_samples =  strain_samples * 100.0
    plt.figure(figsize=(7, 5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(strain_samples, stress_init, label="stress initial", linewidth=3.0, zorder=0, color= "#A1C7E0")
    plt.plot(strain_samples, stress_opt1, label = "stress optimized part1", linewidth=3.0, zorder=0, color= "#00ABBD")
    plt.plot(strain_samples, stress_opt3, label = "stress optimized part2", linewidth=3.0, zorder=0, color= "#0099DD")
    plt.plot(strain_samples, stress_opt2, label = "stress optimized part3", linewidth=3.0, zorder=0, color= "#026B81")

    strain_points = strain_samples[sample_idx]
    # good for 1
    stress_targets1 = np.array([-0.09, -0.082, -0.074, -0.065, -0.055, -0.045, -0.03, -0.005])
    # good for 2
    stress_targets2 = [-0.265, -0.26, -0.255, -0.24, -0.21, -0.16,-0.1, -0.039]
    # good for 3
    stress_targets3 = np.array([-0.17, -0.162, -0.15, -0.134, -0.11, -0.083, -0.058, -0.03])
    
    plt.scatter(strain_points, stress_targets1, marker='+', s=100.0, c="#FF9933", zorder=5)
    plt.scatter(strain_points, stress_targets3, marker='+', s=100.0, c="#FF9933", zorder=5)
    plt.scatter(strain_points, stress_targets2, marker='+', label = "targets", s=100.0, c="#FF9933", zorder=5)

    # plt.scatter(strain_points, stress_targets1, marker='+', label = "targets part1", s=100.0, c="#00ABBD", zorder=5)
    # plt.scatter(strain_points, stress_targets3, marker='+', label = "targets part2", s=100.0, c="#0099DD", zorder=5)
    # plt.scatter(strain_points, stress_targets2, marker='+', label = "targets part3", s=100.0, c="#026B81", zorder=5)
    # plt.legend(loc="upper left")
    plt.legend()
    # plt.xlabel("strain")
    # plt.ylabel("stress")
    # axes = plt.axes()
    # axes.set_ylim([-0.3, 0])
    plt.savefig("uniaxial_stress_teaser.png", dpi=300)
    os.system("convert uniaxial_stress_teaser.png -trim uniaxial_stress_teaser.png")
    plt.close()

    targets = []
    # 0.11162167 0.52156324 IH28

def optimizeFromInitialGuess(model, ti, bounds, n_tiling_params, stress_targets):
    strain_range = [-0.1, 0.0]
    n_sp_strain = 30
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))
    sample_idx = np.arange(0, n_sp_strain, 4)
    theta = 0.5 * np.pi

    def objAndGradient(x):
        uniaxial_strain = []
        dqdp = []
        for strain in strain_samples[sample_idx]:
            uni_strain, dqidpi = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, x, False)
            uniaxial_strain.append(uni_strain)
            dqdp.append(dqidpi)
        ti_TF = tf.convert_to_tensor(x)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        stress_d, stress_d_grad, dOdE = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        
        stress_d = stress_d.numpy()
        stress_d_grad = stress_d_grad.numpy()
        dOdE = dOdE.numpy()
        
        # stress_current = stress_d[sample_idx]#np.array([stress_d[2], stress_d[5], stress_d[-1]])
        stress_current = stress_d
        obj = (np.dot(stress_current - stress_targets, np.transpose(stress_current - stress_targets)) * 0.5).flatten() 
        grad = np.zeros((n_tiling_params))
        
        for i in range(len(sample_idx)):
            grad += (stress_current[i] - stress_targets[i]) * stress_d_grad[i].flatten() + \
                (stress_current[i] - stress_targets[i]) * np.dot(dOdE[i][i], dqdp[i][:3, :]).flatten()
        
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    result = minimize(objAndGradient, ti, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
  
    return result.fun, result.x

def generateSamples(bounds, n_samples):
    n_tiling_params = len(bounds)
    random_samples = np.random.random_sample(n_samples * n_tiling_params)
    samples = []
    for i in range(n_samples):
        sample = np.zeros(n_tiling_params)
        for j in range(n_tiling_params):
            sample[j] = bounds[j][0] + (bounds[j][1] - bounds[j][0]) * random_samples[i * n_tiling_params + j]
        samples.append(sample)  
    return samples


def searchBestFitDifferentInitialGuess(IH):
    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    samples = generateSamples(bounds, 20)
    objectives = []
    # stress_targets = [-0.265, -0.26, -0.25, -0.24, -0.21, -0.16,-0.1, -0.039]
    # stress_targets = np.array([-0.17, -0.162, -0.15, -0.134, -0.11, -0.083, -0.058, -0.03])
    # stress_targets = [-0.309, -0.288, -0.258, -0.235, -0.222, -0.205, -0.148, -0.0532]
    stress_targets = [-0.265, -0.26, -0.255, -0.24, -0.21, -0.16,-0.1, -0.039]
    # stress_targets = np.array([-0.09, -0.082, -0.074, -0.065, -0.055, -0.045, -0.03, -0.005])
    results = []
    for sample in samples:  
        val, sol = optimizeFromInitialGuess(model, sample, bounds, n_tiling_params, stress_targets)
        objectives.append(val)
        results.append(sol)
    
    objectives = np.squeeze(np.array(objectives))

    min_value = 1e10
    min_idx = -1
    for i in range(len(objectives)):
        if objectives[i] < min_value:
            min_value = objectives[i]
            min_idx = i

    print(results[min_idx], objectives[min_idx], samples[min_idx])
def plotBatch():
    IH = 67
    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    samples = generateSamples(bounds, 10)
    samples.append(np.array([0.15, 0.84]))
    strain_range = [-0.1, 0.0]
    n_sp_strain = 30
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))
    sample_idx = np.arange(0, n_sp_strain, 4)
    theta = 0.5 * np.pi

    def computeTarget(ti):
        uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                    theta, strain_samples, 
                                    ti, model)
        ti_TF = tf.convert_to_tensor(ti)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        obj, _ , _ = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        obj = obj.numpy()
        return obj
    stresses = []
    cnt = 0
    for sample in samples:  
        stress = computeTarget(sample)
        stresses.append(stress)
        plt.plot(strain_samples, stress, label = str(cnt), linewidth=3.0, zorder=0)
        cnt += 1
    plt.legend()
    plt.savefig("batch.png", dpi=300)
    for i in range(len(samples)):
        print(i, samples[i])


if __name__ == "__main__":
    # optimizeUniaxialStressConstraints()
    # optimizeUniaxialStressSA()
    # plotEnergyAlongDirection()
    plotNNFDCurves()
    # teaserPlot()
    # searchBestFitDifferentInitialGuess(1)
    # plotBatch()