import math
import numpy as np
import tensorflow as tf

from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from Derivatives import *

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

def optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
    theta, strains, tiling_params, verbose = True):

    d = np.array([np.cos(theta), np.sin(theta)])
    strain_init = []
    for strain in strains:
        strain_tensor_init = np.outer(d, d) * strain
        strain_init.append(np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]]))

    strain_init = np.array(strain_init).flatten()
    
    m = len(strain_init) // 3
    n = len(strain_init)
    A = np.zeros((m, n))
    lb = []
    ub = []

    for i in range(m):
        A[i, i * 3:i * 3 + 3] = computedCdE(d)
        lb.append(strains[i])
        ub.append(strains[i])

    uniaxial_strain_constraint = LinearConstraint(A, lb, ub)

    def hessian(x):
        
        H = hessPsiSum(n_tiling_params, tf.convert_to_tensor(x), 
            tf.convert_to_tensor(tiling_params), model)
        H = H.numpy()
        
        ev_H = np.linalg.eigvals(H)
        min_ev = np.min(ev_H)
        if min_ev < 0.0:
            H += np.diag(np.full(len(x),min_ev + 1e-6))
        return H

    def objAndEnergy(x):
        obj, grad = objGradPsiSum(n_tiling_params, tf.convert_to_tensor(x), 
            tf.convert_to_tensor(tiling_params), model)
        
        obj = obj.numpy()
        grad = grad.numpy().flatten()
        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    if verbose:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True,
         hess=hessian,
            constraints=[uniaxial_strain_constraint],
            options={'disp' : True})
    else:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, 
            hess=hessian,
            constraints= [uniaxial_strain_constraint],
            options={'disp' : False})
    
    return np.reshape(result.x, (m, 3))

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
        
        ev_H = np.linalg.eigvals(H)
        min_ev = np.min(ev_H)
        if min_ev < 0.0:
            H += np.diag(np.full(len(x),min_ev + 1e-6))
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
    
    thetas = tf.expand_dims(thetas, axis=1)
    
    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)
    psi, stress, C = valueGradHessian(n_tiling_params, inputs, model)
    
    Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :])
    dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
    
    stiffness = tf.divide(tf.constant([1.0], dtype=tf.float64), dTSd)
    for i in range(1, C.shape[0]):
        
        Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
        dTSd = tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0)
        stiffness = tf.concat((stiffness, tf.divide(tf.constant([1.0], dtype=tf.float64), dTSd)), 0)
    return tf.squeeze(stiffness)