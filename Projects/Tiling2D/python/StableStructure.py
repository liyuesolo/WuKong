import os
from functools import cmp_to_key
from joblib import Parallel, delayed

from scipy.optimize import BFGS
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
from requests import options
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

from Derivatives import *
from Optimization import *

def CauchyToGreen(strain_samples):
    green_strain = strain_samples.copy()
    for i in range(len(strain_samples)):
        strain = strain_samples[i]
        if strain < 0:
            green_strain[i] = strain - 0.5 * strain * strain
        else:
            green_strain[i] = strain + 0.5 * strain * strain
    return green_strain

def checkEnergyHessianSingleStructure(ti, n_tiling_params, model):
    n_sp_theta = 25
    n_sp_strain = 25
    strain_max = 0.7
    strain_min = -0.2
    strain_rates = np.arange(strain_min, strain_max, (strain_max - strain_min) / float(n_sp_strain))
    
    strain_rates_green = strain_rates.copy()
    for i in range(len(strain_rates)):
        if strain_rates[i] < 0:
            strain_rates_green[i] = strain_rates[i] - 0.5 * strain_rates[i] * strain_rates[i]
        else:
            strain_rates_green[i] = strain_rates[i] + 0.5 * strain_rates[i] * strain_rates[i]

    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    strain_inputs = []
    for theta in thetas:
        uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                    theta, strain_rates_green, 
                                    ti, model)
        strain_inputs.append(uniaxial_strain)
    strain_inputs = np.reshape(np.array(strain_inputs), (n_sp_strain * n_sp_theta, 3))
    batch_dim = strain_inputs.shape[0]
    nn_inputs = tf.convert_to_tensor(np.hstack((np.tile(ti, (batch_dim, 1)), strain_inputs)))
    _, _, Hs = psiValueGradHessian(n_tiling_params, nn_inputs, model)
    Hs = Hs.numpy()
    cnt = 0
    smallest_evs = []
    npd_indices = []
    for i in range(len(Hs)):
        H = Hs[i]
        ev_H = np.linalg.eigvals(H)
        min_ev = np.min(ev_H)
        if min_ev < 1e-8:
            cnt += 1
            smallest_evs.append(min_ev)
            npd_indices.append(i)
    print("{}/{} d2Psi/dE2 is not PD".format(cnt, len(Hs)))
    print("eigen values")
    print(smallest_evs)
    print("strains")
    for idx in npd_indices:
        strain_idx = idx % n_sp_strain
        theta_idx = idx // n_sp_theta
        print(strain_inputs[idx], "theta", thetas[theta_idx], "strain rate cauchy", strain_rates[strain_idx])
    
def checkEnergyHessianSingleStructureDirection(ti, n_tiling_params, model, theta, strain_range):
   
    strain_max = strain_range[1]
    strain_min = strain_range[0]
    strain_rates = np.arange(strain_min, strain_max, 0.01)
    n_sp_strain = len(strain_rates)
    strain_rates_green = strain_rates.copy()
    for i in range(len(strain_rates)):
        if strain_rates[i] < 0:
            strain_rates_green[i] = strain_rates[i] - 0.5 * strain_rates[i] * strain_rates[i]
        else:
            strain_rates_green[i] = strain_rates[i] + 0.5 * strain_rates[i] * strain_rates[i]

    strain_inputs = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                            theta, strain_rates_green, 
                            ti, model)
    batch_dim = strain_inputs.shape[0]
    nn_inputs = tf.convert_to_tensor(np.hstack((np.tile(ti, (batch_dim, 1)), strain_inputs)))
    _, _, Hs = psiValueGradHessian(n_tiling_params, nn_inputs, model)
    Hs = Hs.numpy()
    cnt = 0
    smallest_evs = []
    npd_indices = []
    for i in range(len(Hs)):
        H = Hs[i]
        ev_H = np.linalg.eigvals(H)
        min_ev = np.min(ev_H)
        if min_ev < 1e-8:
            cnt += 1
            smallest_evs.append(min_ev)
            npd_indices.append(i)
    print("{}/{} d2Psi/dE2 is not PD".format(cnt, len(Hs)))
    if cnt  > 0:
        print("eigen values")
        print(smallest_evs)
        print("strains")
        for idx in npd_indices:
            strain_idx = idx % n_sp_strain
            print(strain_inputs[idx],  "strain rate cauchy", strain_rates[strain_idx])
    

def loadModel(IH):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    bounds = []
    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti_default = np.array([0.1045, 0.65])
    elif IH == 50:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
    elif IH == 67:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1]) 
    elif IH == 28:
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
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

    return model, n_tiling_params, ti_default,  bounds

def plotDirectionStiffness(ti, n_tiling_params, model, strain_cauchy):
    if strain_cauchy <  0:
        strain = strain_cauchy - 0.5 * strain_cauchy  * strain_cauchy
    else:
        strain = strain_cauchy + 0.5 * strain_cauchy  * strain_cauchy
    n_sp_theta = 50
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    
    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    # uniaxial_strain = np.reshape(uniaxial_strain, (batch_dim, 3))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    stiffness = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
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
    print(stiffness)
    # print(thetas)

def plotStrainStressCurve(ti, n_tiling_params, model, theta, strain_range):
    
    strain_samples = np.arange(strain_range[0], strain_range[1], 0.01)
    green_strain = CauchyToGreen(strain_samples)
    uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                theta, green_strain, 
                                ti, model)
    ti_TF = tf.convert_to_tensor(ti)
    uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
    stress, _ , _ = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
    stress = stress.numpy()
    psi = energyDensity(ti_TF, uniaxial_strain_TF, model)
    print(uniaxial_strain_TF)
    print(psi)
    plt.plot(strain_samples, stress, label="stress initial", linewidth=3.0, zorder=0)
    plt.savefig("uniaxial_stress.png", dpi=300)
    plt.close()

def findStableStructure():
    IH = 1
    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    ti = ti_default
    if IH == 1:
        ti = np.array([0.1224,  0.5254, 0.1433, 0.49])
        # ti = np.array([0.1224, 0.5, 0.1087, 0.5541])
    elif IH == 21:
        ti = np.array([0.115, 0.75])
    # checkEnergyHessianSingleStructure(ti_default, n_tiling_params, model)
    # checkEnergyHessianSingleStructureDirection(ti, n_tiling_params, model, 0.0, [-0.2, 0.8])
    # plotDirectionStiffness(ti, n_tiling_params, model, 0.67)
    plotStrainStressCurve(ti, n_tiling_params, model, 0.0, [-0.2, 0.8])
if __name__ == "__main__":
    findStableStructure()