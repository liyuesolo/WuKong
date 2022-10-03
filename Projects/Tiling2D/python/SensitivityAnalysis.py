
from cProfile import label
from doctest import master
from email.policy import default
from linecache import getlines
import os
from functools import cmp_to_key
from pickletools import optimize
from statistics import mode
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
def testStep(n_tiling_params, lambdas, model):
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(lambdas)
        with tf.GradientTape() as tape:
            tape.watch(lambdas)
            
            elastic_potential = model(lambdas, training=False)
            dedlambda = tape.gradient(elastic_potential, lambdas)
            batch_dim = elastic_potential.shape[0]
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 4])
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
            elastic_potential = model(lambdas)
            dedlambda = tape.gradient(elastic_potential, lambdas)
            batch_dim = elastic_potential.shape[0]
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    d2edp2 = tape_outer.batch_jacobian(de_dp, lambdas)[:, :, 0:n_tiling_params]
    del tape
    del tape_outer
    return d2edp2, de_dp, elastic_potential

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
    strain_target = np.array([0.1, 0.04, 0.002, 0.002])
    x0 = np.array([0.15, 0.55])
    # x0 = np.array([0.125, 0.71])
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
    # print(psi[3], psi[5])
    # exit(0)
    # strain_opt = strain[3:6]
    strain_opt = [strain[3], strain[5]]
    # psi_target = [0.0008, 0.0014, 0.0018]
    psi_target = [0.004, 0.00456482]
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
    # result = minimize(objAndGradientEnergy, x0 ,method='trust-constr', jac=True, hess=hessian, options={'disp' : True}, bounds=bounds)
    result = minimize(objAndGradientEnergy, x0 ,method='Newton-CG', jac=True, hess=hessian, options={'disp' : True, "xtol" : 1e-8})
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
    
    vx = []
    vy = []

    # for polygon in polygons:
    #     for line in polygon:
    #         vx.append(line[0])
    #         vy.append(line[1])

    # max_x, min_x = np.max(vx), np.min(vx)
    # max_y, min_y = np.max(vy), np.min(vy)
    
    # for j in range(len(polygons)):
    #     for i in range(len(polygons[j])):
    #         # polygons[j][i][0] = 100 + ((polygons[j][i][0]) - min_x) * 0.5
    #         # polygons[j][i][1] = 100 + ((polygons[j][i][1]) - min_y) * 0.5
    #         polygons[j][i][0] = 100 + ((polygons[j][i][0])) * 0.5
    #         polygons[j][i][1] = 100 + ((polygons[j][i][1])) * 0.5
    
    return polygons

def explorer():
    update = False
    IH = 21
    current_dir = os.path.dirname(os.path.realpath(__file__))

    if IH == 50:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(69) + "/")
        model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
        model.load_weights(save_path + "IH5040k" + '.tf')
        range_t0 = [0.1, 0.3]
        range_t1 = [0.25, 0.75]
        x0 = np.array([0.2, 0.5])
        params = [0.2, 0.5]
    elif IH == 21:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(52) + "/")
        model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
        model.load_weights(save_path + "full40k" + '.tf')
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
        query_points.append(np.hstack((x0, np.array([strain, 0.06, 0.0004, 0.0004]))))
        
    
    model_input = tf.convert_to_tensor(query_points)

    _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
    
    
    


    def t1Callback(sender):
        update = True
        for pt in query_points:
            pt[0] = dpg.get_value(sender)
        t1 = float(dpg.get_value(sender))
        polygons = getPolygons(IH, [t1, params[1]])
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
        polygons = getPolygons(IH, [params[0], t2])
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
            pt[5] = dpg.get_value(sender)
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
            dpg.add_slider_float(default_value=0.001, min_value = -0.01, max_value=0.01,width=200.0, height=500.0, callback= strainxyCallback)
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
    explorer()
    # optimizeStessProfile()
    # optimizeDensityProfile()