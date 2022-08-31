
import os
from functools import cmp_to_key
from statistics import mode

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
@tf.function
def testStep(model_inputs, model):
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(model_inputs)
        with tf.GradientTape() as tape:
            tape.watch(model_inputs)
            elastic_potential = model(model_inputs)
        dedx = tape.gradient(elastic_potential, model_inputs)
        stress = dedx[:, 4:]
    dstress_dp = tf.squeeze(tape_outer.jacobian(stress, model_inputs)[:, :, :, 0:4])
        
    return dstress_dp, stress

def optimizeTilingParametersSingleFamily():
    
    # bounds = Bounds([0.02, 0.02, 0.02, 0.02], [1.0, 1.0, 1.0, 1.0])
    n_tiling_params = 4
    bounds = []
    for i in range(n_tiling_params):
        bounds.append([0.02, 1.0])
    model = buildSingleFamilyModel(n_tiling_params)
    model.load_weights("tiling_model.tf")
    stress_target = np.array([32.0, 6.0, 10.0])
    strain_target = np.array([0.1, 0.0, 0.0])
    #16.27025032  5.96559334  9.81682301
    x0 = np.array([0.2, 0.2, 0.2, 0.2])
    # model_input = tf.convert_to_tensor([np.hstack((x0, strain_target))])
    uni_axial_strain = np.array([0.1, -0.00498983, 0.0329836])
    x_opt = np.array([0.32896349, 0.24773578, 0.20972134, 0.12847836])
    model_input = tf.convert_to_tensor([np.hstack((x_opt, uni_axial_strain))])
    dstress_dp, stress = testStep(model_input, model)
    print(stress)
    exit(0)
    def objAndGradient(x):
        model_input = tf.convert_to_tensor([np.hstack((x, strain_target))])
        dstress_dp, stress = testStep(model_input, model)
        obj = (np.dot(stress - stress_target, np.transpose(stress - stress_target)) * 0.5).flatten()
        grad = np.dot(stress - stress_target, dstress_dp).flatten()
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    result = minimize(objAndGradient, x0, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
    print(result.x)



if __name__ == "__main__":
    optimizeTilingParametersSingleFamily()