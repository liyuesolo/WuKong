
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
    return psi, tf.cast(stress, tf.float32), tf.cast(C, tf.float32)

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
        
    stiffness = tf.squeeze(tf.math.divide(tf.ones((batch_dim)), tf.expand_dims(dTSd, axis=0)))
    stiffness2 = tf.constant(2.0) * tf.math.divide(tf.squeeze(psi), tf.constant(0.1) * tf.ones((batch_dim)))
    
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

def optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, 
    theta, strain, tiling_params):
    
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
    # strain_opt = result.x
    # strain_tensor = np.reshape([strain_opt[0], 0.5 * strain_opt[-1], 0.5 * strain_opt[-1], strain_opt[1]], (2, 2))
    # model_input = tf.convert_to_tensor([np.hstack((tiling_params, strain_opt))])
    # destress_dp, stress, de_dp, psi = testStep(n_tiling_params, model_input, model)
    # stress = stress.numpy().flatten()
    # stress_tensor = np.reshape([stress[0], stress[-1], stress[-1], stress[1]], (2, 2))
    
    return result.x.astype(np.float32), result.optimality
    



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
    plt.polar(thetas, stiffness2, label = "2Psi/strain^2", linewidth=3.0)
    # plt.polar(thetas, stiffness_gt, label = "stiffness_gt", linewidth=3.0)
    plt.legend(loc="upper left")
    plt.savefig(save_path + "hessian_check.png", dpi=300)
    plt.close()




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


def computeStiffness():
    n_tiling_params = 2
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(327) + "/")
    img_folder = save_path + "/stiffness/"
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH21" + '.tf')

    n_sp_theta = 50
    ti = [0.15, 0.65]
    n_sp_strain = 50
    n_sp_tiling = 10
    strain_range = [0.001, 0.2]
    dstrain = (strain_range[1] - strain_range[0]) / float(n_sp_strain)
    t1_range = [0.105, 0.295]
    t2_range = [0.505, 0.795]
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

                strain_green = strain + 0.5 * np.power(strain, 2.0)
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

if __name__ == "__main__":
    
    computeStiffness()