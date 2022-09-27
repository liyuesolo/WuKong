
import os
from functools import cmp_to_key
from pickletools import optimize
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
def testStep(n_tiling_params, lambdas, model):
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(lambdas)
        with tf.GradientTape() as tape:
            tape.watch(lambdas)
            
            elastic_potential = model(lambdas)
            dedlambda = tape.gradient(elastic_potential, lambdas)
            batch_dim = elastic_potential.shape[0]
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 4])
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
   # tape_outer.batch_jacobian(stress, lambdas)[:, :, 0:n_tiling_params]
    dstress_dp = tf.squeeze(tape_outer.jacobian(stress, lambdas)[:, :, :, 0:n_tiling_params])   
    
    return dstress_dp, stress, de_dp, elastic_potential

def optimizeStrainProfile():
    n_tiling_params = 2
    bounds = []
    bounds.append([0.1, 0.2])
    bounds.append([0.5, 0.8])
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(20) + "/")
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
    
    # strain, stress_init, psi = computeCurrentState(x0)
    
    # strain_opt = strain[4:5]
    # stress_targets = stress_init[4:5]
    # stress_targets[0][0] = 0.1
    # for i in range(len(stress_targets)):
        # print(stress_targets[i][0])
        # stress_targets[i][0] += 0.2 * stress_targets[i][0]
        # print(stress_targets[i][0])
        # exit(0)

        
    # dstress_dp, stress, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    # print(stress, elastic_potential)
    # exit(0)
    def objAndGradientMatchX(x):
        model_input = tf.convert_to_tensor([np.hstack((x, strain_target))])
        dstress_dp, stress, _, _ = testStep(n_tiling_params, model_input, model)
        dstress_dp = dstress_dp.numpy()
        stress = stress.numpy()
        obj = 0.5 * np.power((stress[0] - strain_target[0]) / strain_target[0], 2)
        grad = (stress[0] - strain_target[0]) / strain_target[0] * dstress_dp[0, :]
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    def objAndGradient(x):
        # model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        model_input = tf.convert_to_tensor([np.hstack((x, strain_target))])
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
    model_input = tf.convert_to_tensor([np.hstack((result.x, strain_target))])
    dstress_dp, stress_new, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    print(result.x)
    # strain_new, stress_new, psi_new = computeCurrentState(result.x)
    print(stress_target_current)
    print(stress_new)
    print(stress_target)
    
    # x_axis = []
    # for i in range(len(strain)):
    #     x_axis.append(strain[i][0])
    # strain_sp = [strain[4][0], strain[5][0]]
    
    # plt.plot(x_axis, psi, label = "init")
    # plt.plot(x_axis, psi_new, label = "optimized")
    # plt.xlabel("strain_xx")
    # plt.ylabel("energy density")
    # plt.legend(loc="upper left")
    # plt.scatter(strain_sp, psi_target, s=4.0)
    # plt.savefig("./results/opt_energy.jpg", dpi=300)
    # plt.close()

def optimizeDensityProfile():
    
    
    n_tiling_params = 2
    bounds = []
    bounds.append([0.1, 0.2])
    bounds.append([0.5, 0.8])
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(20) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
    model.load_weights(save_path + "full40k" + '.tf')

    stress_target = np.array([0.1, 0.15, 0.00046986, 0.00049414])
    stress_target_current = np.array([0.14675923, 0.13248906, 0.00063026, 0.00063026])
    energy_target = np.array([0.005])
    energy_target_current = 0.01058488
    strain_target = np.array([0.1, 0.04, 0.002, 0.002])
    x0 = np.array([0.15, 0.65])
    model_input = tf.convert_to_tensor([np.hstack((x0, strain_target))])
    
    # x_opt = np.array([0.32896349, 0.24773578, 0.20972134, 0.12847836])
    def computeCurrentState(x):
        model_inputs = []
        n_pt = 10
        dx = (0.1 - 0.01) / float(10.0)
        strain = []
        for i in range(n_pt):
            eps_i = [float(i) * dx, 0.06, 0.0004, 0.0004]
            model_inputs.append(np.hstack((x, eps_i)))
            strain.append(eps_i)
        
        _, stress, _, psi = testStep(n_tiling_params, tf.convert_to_tensor(model_inputs), model)
        return strain, stress, psi.numpy().flatten()
    
    strain, stress, psi = computeCurrentState(x0)
    
    # strain_opt = strain[3:6]
    strain_opt = [strain[3], strain[5]]
    # psi_target = [0.0008, 0.0014, 0.0018]
    psi_target = [0.00327967, 0.00456482]

        
    # dstress_dp, stress, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    # print(stress, elastic_potential)
    # exit(0)
    
    
    def objAndGradientEnergy(x):
        model_input = tf.convert_to_tensor(np.hstack((np.tile(x, (len(strain_opt), 1)), strain_opt)))
        # model_input = tf.convert_to_tensor([np.hstack((x, strain_target))])
        _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
        psi = psi.numpy()
        de_dp = de_dp.numpy()
        obj = 0.0
        grad = np.zeros(n_tiling_params)
        for i in range(len(psi_target)):
            obj += 0.5 * np.power((psi[i] - psi_target[i])/psi_target[i], 2)
            grad += (psi[i] - psi_target[i])/psi_target[i] * de_dp[i, :]
        
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    result = minimize(objAndGradientEnergy, x0, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
    # result = minimize(objAndGradientEnergy, x0 ,method='BFGS', jac=True)
    model_input = tf.convert_to_tensor([np.hstack((result.x, strain_target))])
    # dstress_dp, stress, de_dp, elastic_potential = testStep(n_tiling_params, model_input, model)
    print(result.x)
    strain_new, stress_new, psi_new = computeCurrentState(result.x)
    print(psi)
    print(psi_new)
    
    x_axis = []
    for i in range(len(strain)):
        x_axis.append(strain[i][0])
    strain_sp = [strain[3][0], strain[5][0]]
    
    plt.plot(x_axis, psi, label = "init")
    plt.plot(x_axis, psi_new, label = "optimized")
    plt.xlabel("strain_xx")
    plt.ylabel("energy density")
    plt.legend(loc="upper left")
    plt.scatter(strain_sp, psi_target, s=4.0)
    plt.savefig("./results/opt_energy.jpg", dpi=300)
    plt.close()


    # print(stress, elastic_potential)



if __name__ == "__main__":
    # optimizeStrainProfile()
    optimizeDensityProfile()