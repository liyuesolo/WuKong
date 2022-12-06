import math
import numpy as np
import tensorflow as tf

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
    return tf.squeeze(C)