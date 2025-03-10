import math
import numpy as np
import tensorflow as tf

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
    del tape_outer
    del tape
    return tf.squeeze(C)


def computedCdE(d):
    _i_var = np.zeros(7)
    _i_var[0] = (d[1])*(d[0])
    _i_var[1] = (d[0])*(d[1])
    _i_var[2] = 0.5
    _i_var[3] = (_i_var[1])+(_i_var[0])
    _i_var[4] = (d[0])*(d[0])
    _i_var[5] = (d[1])*(d[1])
    _i_var[6] = (_i_var[3])*(_i_var[2])
    return np.array(_i_var[4:7])

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
    del tape
    del tape_outer
    return tf.squeeze(C)

@tf.function
def psiValueGradHessian(n_tiling_params, inputs, model):
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
def energyDensity(ti, uniaxial_strain, model):
    batch_dim = uniaxial_strain.shape[0]
    ti = tf.expand_dims(ti, 0)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(uniaxial_strain)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)
        psi = model(inputs, training=False)
    del tape
    return tf.squeeze(psi)

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
def objUniaxialStress(n_tiling_params, ti, uniaxial_strain, theta, model):
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

        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
    del tape
    return tf.squeeze(dTSd)
