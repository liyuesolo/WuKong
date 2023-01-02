import os
from functools import cmp_to_key
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
from Optimization import *

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
    stiffness = computeDirectionalStiffness(n_tiling_params, tf.convert_to_tensor(all_data), 
        tf.convert_to_tensor(thetas), model)
    stiffness = stiffness.numpy()
    
    
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
    bounds = []
    IH = 50
    n_sp_theta = 50
    dtheta = np.pi/float(n_sp_theta)
    thetas = np.arange(0.0, np.pi, dtheta)
    # print(len(thetas))
    strain = 0.01
    current_dir = os.path.dirname(os.path.realpath(__file__))
    idx = np.arange(0, len(thetas), 5)

    if IH == 21:
        strain = 0.02
        green = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti = np.array([0.18, 0.7])
        ti_target = np.array([0.1045, 0.65])
        sample_idx = [2, 7, -1]
        theta = 0.0
    elif IH == 50:
        n_sp_theta = 100
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti_default = np.array([0.2308, 0.5])
        # ti = np.array([0.2903, 0.6714])
        ti = ti_default
    elif IH == 67:
        # n_sp_theta = 100
        strain = 0.05
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1])
        # ti = np.array([0.18, 0.68])
        ti = np.array([0.25, 0.85])
        idx = np.arange(0, len(thetas), 5)
        
    elif IH == 28:
        strain = 0.05
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        ti = np.array([0.6, 0.6])
        ti_target = np.array([0.4, 0.8])
        # ti_target = np.array([0.6, 0.6])
        idx = np.arange(0, len(thetas), 5)
    elif IH == 1:
        strain = -0.1
        if strain < 0:
            strain = strain - 0.5 * strain * strain
        else:
            strain = strain + 0.5 * strain * strain
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        # ti = np.array([0.1224, 0.5, 0.1434, 0.625])
        ti = np.array([0.1224,  0.5254, 0.1433, 0.49])
    elif IH == 3:
        strain = 0.1
        if strain < 0:
            strain = strain - 0.5 * strain * strain
        else:
            strain = strain + 0.5 * strain * strain
        n_tiling_params = 4
        bounds.append([0.05, 0.5])
        bounds.append([0.2, 0.8])
        bounds.append([0.08, 0.5])
        bounds.append([0.4, 0.8])
        # ti = np.array([0.1224,  0.5, 0.2253, 0.625])
        ti = np.array([0.12,  0.5, 0.2, 0.625])

    # strain = -0.12
    # strain = strain - 0.5 * strain * strain

    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)

    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    
    # stiffness = generateStiffnessDataThetas(thetas, n_tiling_params, strain, ti, model)

    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    
    # uniaxial_strain = np.reshape(uniaxial_strain, (batch_dim, 3))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    
    stiffness  = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    stiffness = stiffness.numpy()

    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] - np.pi)
        stiffness = np.append(stiffness, stiffness[i])

    plt.polar(thetas, stiffness, label = "tensor", linewidth=3.0)
    plt.savefig("images/stiffness.png", dpi=300)
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
                stiffness = computeDirectionalStiffness(n_tiling_params, tf.convert_to_tensor(nn_inputs), 
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

def generateStiffnessDataThetas(thetas, n_tiling_params, strain, ti, model):
    uniaxial_strain = []
    for theta in thetas:
        uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, ti, False)
        uniaxial_strain.append(uni_strain)
    # print(uniaxial_strain)
    # exit(0)
    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    uniaxial_strain = np.reshape(uniaxial_strain, (batch_dim, 3))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    stiffness = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    stiffness = stiffness.numpy()
    return stiffness

def stiffnessOptimizationSA():
    plot_GT = False
    bounds = []
    IH = 1
    n_sp_theta = 50
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    strain = 0.01
    current_dir = os.path.dirname(os.path.realpath(__file__))
    idx = np.arange(0, len(thetas), 5)

    if IH == 21:
        strain = 0.02
        green = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti = np.array([0.18, 0.7])
        ti_target = np.array([0.1045, 0.65])
        sample_idx = [2, 7, -1]
        theta = 0.0
    elif IH == 50:
        n_sp_theta = 100
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti = np.array([0.2308, 0.5])
        ti_target = np.array([0.2903, 0.6714])
        sample_idx = [2, 7, -1]
    elif IH == 67:
        # n_sp_theta = 100
        strain = 0.05
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1])
        ti = np.array([0.18, 0.68])
        ti_target = np.array([0.25, 0.85])
        idx = np.arange(0, len(thetas), 5)
        
    elif IH == 28:
        strain = 0.05
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        ti = np.array([0.6, 0.6])
        ti_target = np.array([0.4, 0.8])
        # ti_target = np.array([0.6, 0.6])
        idx = np.arange(0, len(thetas), 5)
    elif IH == 1:
        strain = 0.05
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        # test 1
        # ti = np.array([0.1224, 0.5, 0.1434, 0.625])
        # ti_target = np.array([0.1224, 0.6, 0.13, 0.625])
        # test 2
        # ti = np.array([0.1224, 0.6, 0.13, 0.625])
        # ti_target = np.array([0.13, 0.4998, 0.0721, 0.6114])
        # test 3
        # ti = np.array([0.13, 0.4998, 0.0721, 0.6114])
        # ti_target = np.array([0.1224, 0.5, 0.1087, 0.5541])
        # test 4
        # ti = np.array([0.1224, 0.5, 0.1087, 0.5541])
        # ti_target = np.array([0.1224, 0.5, 0.1434, 0.625])
        # test 5
        # ti = np.array([0.1692, 0.4223, 0.0635, 0.6888])
        # ti_target = np.array([0.1224, 0.4724, 0.1406, 0.625])
        # test 6
        # ti = np.array([0.1224, 0.4724, 0.1406, 0.625])
        # ti_target = np.array([0.17, 0.5, 0.12, 0.5])
        # test 7
        # ti = np.array([0.17, 0.5, 0.12, 0.5])
        # ti_target = np.array([0.22, 0.6, 0.08, 0.52])
        # test 8
        ti = np.array([0.22, 0.6, 0.08, 0.52])
        ti_target = np.array([0.1692, 0.4223, 0.0635, 0.6888])
        
        idx = np.arange(0, len(thetas), 5)

    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)

    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    # uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    # print(uniaxial_strain)
    stiffness = generateStiffnessDataThetas(thetas, n_tiling_params, strain, ti, model)
    # idx = [0, 23, 49]
    # idx = [0, 5, 9]

    
    
    # if plot_GT:
    #     data = ""
    #     for k in stiffness:
    #         data += str(k) + ", "
    #     print(data)
        

    # if IH == 50:
        # stiffness_targets = np.array([0.005357881231762475, 0.0053454764198269345, 0.005323076289713101, 0.005291569037720618, 0.0052522045380705784, 0.005206538985992594, 0.005156317853195587, 0.005103432053036575, 0.005049849140188627, 0.004997588281309535, 0.004948602172962785, 0.004904843012360457, 0.004868177620772044, 0.004840440651345217, 0.004823469064749871, 0.004818991757038235, 0.004828978362628706, 0.004855306346270269, 0.0049001325792824854, 0.004965911886655767, 0.005055390854282705, 0.005171798842058322, 0.005319091992881996, 0.005502027464332436, 0.005726526451276003, 0.005999929516542937, 0.006331583277052967, 0.006733359841493947, 0.00722057834175154, 0.007813154325618703, 0.008537263976337377, 0.009427641766171856, 0.010530831464126878, 0.011909725278378426, 0.013649730833703747, 0.01586677564177888, 0.01871692802717734, 0.022404009565604288, 0.027175282123162885, 0.03327769337046987, 0.04081423136805599, 0.04941796486384382, 0.057890418607245774, 0.06158122654443698, 0.07910000717017367, 0.13009581614825785, 0.25367472705223826, 0.47512731203807107, 1.0204514711708237, 1.545156515315592, 1.0226177560044938, 0.4913970812878216, 0.2565621974977346, 0.12514883705159502, 0.07314831557558382, 0.06008642835986133, 0.058832596963625514, 0.051370760458955285, 0.042733790125287746, 0.034839734366138145, 0.028359992691892176, 0.023279511280816297, 0.019359764740570815, 0.016339754532924845, 0.013999665661286846, 0.012170135164418353, 0.010725453771437422, 0.009573281821056216, 0.008646021656700155, 0.007893658420805588, 0.007279120731033306, 0.006774575043915245, 0.006358884296229148, 0.006015915518370427, 0.005733149845429153, 0.005500786642791992, 0.00531114293624395, 0.005158074949472558, 0.00503659274805049, 0.004942658655271803, 0.004872921834920428, 0.0048245386709542, 0.004795009654030171, 0.004782296786893437, 0.0047844060613056825, 0.00479953607388955, 0.0048257329785549715, 0.004861502134963473, 0.004904820697292641, 0.004954039892769007, 0.005007261039364209, 0.005062564561422771, 0.0051179933806987125, 0.005171611640220892, 0.00522150913946486, 0.005265892870544442, 0.005303134799640369, 0.005331845089384196, 0.005350952647260489, 0.005359393895734639])
    stiffness_targets = generateStiffnessDataThetas(thetas, n_tiling_params, strain, ti_target, model)
    if IH == 21:
        mean = np.mean(stiffness_targets)
        stiffness_targets = np.full((len(stiffness_targets), ), mean)
    # exit(0)
    sample_points_theta = thetas[idx]
    batch_dim = len(thetas)
    stiffness_targets_sub = stiffness_targets[idx]

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

        obj = (np.dot(stiffness_current - stiffness_targets_sub, np.transpose(stiffness_current - stiffness_targets_sub)) * 0.5)
        
        grad = np.zeros((n_tiling_params))
        
        for i in range(len(idx)):
            grad += (stiffness_current[i] - stiffness_targets_sub[i]) * stiffness_grad[i].flatten() + \
                (stiffness_current[i] - stiffness_targets_sub[i]) * np.dot(dOdE[i][i], dqdp[i][:3, :]).flatten()
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    if not plot_GT:
        # result = minimize(objAndGradient, ti, method='trust-constr', jac=True, options={'disp' : True}, bounds=bounds)
        result = minimize(objAndGradient, ti, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
        uniaxial_strain_opt = []
        for theta in thetas:
            uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, result.x, False)
            uniaxial_strain_opt.append(uni_strain)

        uniaxial_strain_opt = np.reshape(uniaxial_strain_opt, (batch_dim, 3))
        nn_inputs = tf.convert_to_tensor(np.hstack((np.tile(result.x, (batch_dim, 1)), uniaxial_strain_opt)))
        stiffness_opt = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                        tf.convert_to_tensor(thetas), model)
        stiffness_opt = stiffness_opt.numpy()
        print(result.x)
    
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

    # fdGradient(ti)
    # exit(0)

    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] - np.pi)
        stiffness = np.append(stiffness, stiffness[i])
        stiffness_targets = np.append(stiffness_targets, stiffness_targets[i])
        if not plot_GT:
            stiffness_opt = np.append(stiffness_opt, stiffness_opt[i])
    thetas = np.append(thetas, thetas[0])
    stiffness = np.append(stiffness, stiffness[0])
    stiffness_targets = np.append(stiffness_targets, stiffness_targets[0])
    if not plot_GT:
        stiffness_opt = np.append(stiffness_opt, stiffness_opt[0])
    
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
    if IH == 21:
        ax1.set_ylim(0.05, 0.35)
    elif IH == 1:
        ax1.set_ylim(0, 5.5)
    ax1.plot(thetas,stiffness,lw=2.5, label = "stiffness initial")
    ax1.plot(thetas,stiffness_targets,lw=2.5, label = "stiffness target", linestyle = "dashed")
    # plt.polar(thetas, stiffness, label = "stiffness initial", linewidth=3.0, zorder=0)
    # plt.polar(thetas, stiffness_targets, linestyle = "dashed", label = "stiffness target", linewidth=3.0, zorder=0)
    plt.legend(loc='upper left')
    plt.savefig("stiffness_optimization_IH"+str(IH)+"_initial.png", dpi=300)
    plt.close()
    os.system("convert stiffness_optimization_IH"+str(IH)+"_initial.png -trim stiffness_optimization_IH"+str(IH)+"_initial.png")
    if not plot_GT:
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
        if IH == 21:
            ax1.set_ylim(0.05, 0.35)
        elif IH == 1:
            ax1.set_ylim(0, 5.5)
        # plt.polar(thetas, stiffness_opt, label = "stiffness optimized", linewidth=3.0, zorder=0)
        # plt.polar(thetas, stiffness_targets, linestyle = "dashed", label = "stiffness target", linewidth=3.0, zorder=0)
        ax1.plot(thetas,stiffness_opt,lw=2.5, label = "stiffness optimized")
        ax1.plot(thetas,stiffness_targets,lw=2.5, label = "stiffness target", linestyle = "dashed")
        plt.legend(loc='upper left')
        plt.savefig("stiffness_optimization_IH"+str(IH)+"_optimized.png", dpi=300)
        plt.close()
        os.system("convert stiffness_optimization_IH"+str(IH)+"_optimized.png -trim stiffness_optimization_IH"+str(IH)+"_optimized.png")

if __name__ == "__main__":
    computeStiffness()
    # stiffnessOptimizationSA()