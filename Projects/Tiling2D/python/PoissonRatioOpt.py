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
import time
from Derivatives import *
from Optimization import *
from Samples import*
from PropertyModifier import *


@tf.function
def computeDirectionalPoissonRatio(n_tiling_params, inputs, thetas, model):
    
    thetas = tf.expand_dims(thetas, axis=1)
    
    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)
    n_voigt = tf.concat((tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.cos(thetas) * tf.math.cos(thetas), 
                        -tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)

    psi, stress, C = valueGradHessian(n_tiling_params, inputs, model)
    
    C_inv = tf.linalg.inv(C[0, :, :])
    Sd = tf.linalg.matvec(C_inv, d_voigt[0, :])
    Sn = tf.linalg.matvec(C_inv, n_voigt[0, :])
    dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
    dTSn = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sn, 1), axis=0)
    nu = -tf.divide(dTSn, dTSd)
    
    for i in range(1, C.shape[0]):
        C_inv = tf.linalg.inv(C[i, :, :])
        Sd = tf.linalg.matvec(C_inv, d_voigt[i, :])
        Sn = tf.linalg.matvec(C_inv, n_voigt[i, :])
        
        nu = tf.concat((nu, tf.expand_dims(-tf.divide(tf.tensordot(d_voigt[i, :], Sn, 1), tf.tensordot(d_voigt[i, :], Sd, 1)), axis=0)), 0)
    return tf.squeeze(nu)




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

def computePoissonRatio():
    bounds = []
    IH = 21
    n_sp_theta = 50
    dtheta = np.pi/float(n_sp_theta)
    thetas = np.arange(0.0, np.pi, dtheta)
    strain = 0.02
    if strain < 0:
        strain = strain - 0.5 * strain * strain
    else:
        strain = strain + 0.5 * strain * strain

    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    
    # ti = np.array([0.165, 0.72])
    # ti = np.array([0.25, 0.8])
    # ti = np.array([0.0153, 0.7551])
    # ti = ti_default
    # ti = np.array([0.17, 0.51])
    # ti = np.array([0.17, 0.51])
    # ti = np.array([0.18, 0.7])
    ti = np.array([0.175, 0.52])
    uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    

    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    nu = computeDirectionalPoissonRatio(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    nu = nu.numpy()
    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] + np.pi)
        nu = np.append(nu, nu[i])
    thetas = np.append(thetas, thetas[0])
    nu = np.append(nu, nu[0])
    plt.polar(thetas, nu, label = "tensor", linewidth=3.0)
    plt.savefig("images/poisson_ratio.png", dpi=300)
    plt.close()

def getDirectionPoissonRatio(ti, n_tiling_params, model, strain_cauchy, n_sp_theta = 20, sym=True):
    if strain_cauchy <  0:
        strain = strain_cauchy - 0.5 * strain_cauchy  * strain_cauchy
    else:
        strain = strain_cauchy + 0.5 * strain_cauchy  * strain_cauchy
    
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    
    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    poisson_ratio = computeDirectionalPoissonRatio(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    poisson_ratio = poisson_ratio.numpy()
    if sym:
        for i in range(n_sp_theta):
            thetas= np.append(thetas, thetas[i] + np.pi)
            poisson_ratio = np.append(poisson_ratio, poisson_ratio[i])
        thetas= np.append(thetas, 2*np.pi)
        poisson_ratio = np.append(poisson_ratio, poisson_ratio[0])
    return thetas, poisson_ratio


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
def objGradPoissonRatio(ti, uniaxial_strain, thetas, model):
    batch_dim = uniaxial_strain.shape[0]
    
    thetas = tf.expand_dims(thetas, axis=1)
    
    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)
    n_voigt = tf.concat((tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.cos(thetas) * tf.math.cos(thetas), 
                        -tf.math.sin(thetas) * tf.math.cos(thetas)), 
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
        
        
        C_inv = tf.linalg.inv(C[0, :, :])
        Sd = tf.linalg.matvec(C_inv, d_voigt[0, :])
        Sn = tf.linalg.matvec(C_inv, n_voigt[0, :])
        dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0)
        dTSn = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sn, 1), axis=0)
        nu = -tf.divide(dTSn, dTSd)
        for i in range(1, C.shape[0]):
            C_inv = tf.linalg.inv(C[i, :, :])
            Sd = tf.linalg.matvec(C_inv, d_voigt[i, :])
            Sn = tf.linalg.matvec(C_inv, n_voigt[i, :])
        
            nu = tf.concat((nu, tf.expand_dims(-tf.divide(tf.tensordot(d_voigt[i, :], Sn, 1), tf.tensordot(d_voigt[i, :], Sd, 1)), axis=0)), 0)
    
    
    grad = tape_outer_outer.jacobian(nu, ti)
    dOdE = tape_outer_outer.jacobian(nu, uniaxial_strain)
    del tape
    del tape_outer
    del tape_outer_outer
    return tf.squeeze(nu), tf.squeeze(grad), tf.squeeze(dOdE)

def generatePoissonRatioDataThetas(thetas, n_tiling_params, strain, ti, model):
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
    poisson_ratio = computeDirectionalPoissonRatio(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    poisson_ratio = poisson_ratio.numpy()
    return poisson_ratio

def poissonRatioSA():
    plot_GT = False
    bounds = []
    IH = 50
    n_sp_theta = 50
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    strain = 0.01
    current_dir = os.path.dirname(os.path.realpath(__file__))
    idx = np.arange(0, len(thetas), 5)

    if IH == 21:
        strain = 0.02
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti = np.array([0.165, 0.72])
        ti_target = np.array([0.175, 0.52])

    elif IH == 50:
        n_sp_theta = 50
        n_tiling_params = 2
        strain = 0.02
        strain = strain + 0.5 * strain * strain
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti = np.array([0.25, 0.52])
        ti_target = np.array([0.25, 0.64])
        
    elif IH == 67:
        # n_sp_theta = 100
        strain = 0.01
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1])
        ti = np.array([0.18, 0.7])
        ti_target = np.array([0.25, 0.8])
        
    elif IH == 28:
        strain = 0.02
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        # ti = np.array([0.6, 0.6])
        ti = np.array([0.55, 0.7])
        ti_target = np.array([0.4, 0.8])
    elif IH == 1:
        strain = 0.02
        strain = strain + 0.5 * strain * strain
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        # test 1
        # ti = np.array([0.13, 0.55, 0.13, 0.625])
        # ti_target = np.array([0.1224, 0.6, 0.09, 0.6])
        # test 2
        # ti = np.array([0.1224, 0.6, 0.09, 0.6])
        # ti_target = np.array([0.13, 0.5, 0.08, 0.62])
        # test 3
        # ti = np.array([0.13, 0.5, 0.08, 0.62])
        # ti_target = np.array([0.2, 0.5, 0.1087, 0.55])
        # test 4
        # ti = np.array([0.2, 0.5, 0.1087, 0.55])
        # ti_target = np.array([0.13, 0.55, 0.13, 0.625])
        # test 5
        # ti = np.array([0.15, 0.6, 0.13, 0.6])
        # ti_target = np.array([0.12, 0.45, 0.1, 0.7])
        # test 6
        # ti = np.array([0.10279905, 0.45325127, 0.09960801, 0.70605258])
        # ti_target = np.array([0.18, 0.52, 0.08, 0.55])
        # test 7
        # ti = np.array([0.18605106, 0.51947694, 0.08046355, 0.55166683])
        # ti_target = np.array([0.16, 0.58, 0.08, 0.6])
        # test 8
        ti = np.array([0.16, 0.58, 0.08, 0.6])
        ti_target = np.array([0.15, 0.6, 0.13, 0.6])
        
        idx = np.arange(0, len(thetas), 5)

    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)

    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    poisson_ratio = generatePoissonRatioDataThetas(thetas, n_tiling_params, strain, ti, model)
    # if IH == 50:
        # poisson_ratio_targets = np.array([0.005357881231762475, 0.0053454764198269345, 0.005323076289713101, 0.005291569037720618, 0.0052522045380705784, 0.005206538985992594, 0.005156317853195587, 0.005103432053036575, 0.005049849140188627, 0.004997588281309535, 0.004948602172962785, 0.004904843012360457, 0.004868177620772044, 0.004840440651345217, 0.004823469064749871, 0.004818991757038235, 0.004828978362628706, 0.004855306346270269, 0.0049001325792824854, 0.004965911886655767, 0.005055390854282705, 0.005171798842058322, 0.005319091992881996, 0.005502027464332436, 0.005726526451276003, 0.005999929516542937, 0.006331583277052967, 0.006733359841493947, 0.00722057834175154, 0.007813154325618703, 0.008537263976337377, 0.009427641766171856, 0.010530831464126878, 0.011909725278378426, 0.013649730833703747, 0.01586677564177888, 0.01871692802717734, 0.022404009565604288, 0.027175282123162885, 0.03327769337046987, 0.04081423136805599, 0.04941796486384382, 0.057890418607245774, 0.06158122654443698, 0.07910000717017367, 0.13009581614825785, 0.25367472705223826, 0.47512731203807107, 1.0204514711708237, 1.545156515315592, 1.0226177560044938, 0.4913970812878216, 0.2565621974977346, 0.12514883705159502, 0.07314831557558382, 0.06008642835986133, 0.058832596963625514, 0.051370760458955285, 0.042733790125287746, 0.034839734366138145, 0.028359992691892176, 0.023279511280816297, 0.019359764740570815, 0.016339754532924845, 0.013999665661286846, 0.012170135164418353, 0.010725453771437422, 0.009573281821056216, 0.008646021656700155, 0.007893658420805588, 0.007279120731033306, 0.006774575043915245, 0.006358884296229148, 0.006015915518370427, 0.005733149845429153, 0.005500786642791992, 0.00531114293624395, 0.005158074949472558, 0.00503659274805049, 0.004942658655271803, 0.004872921834920428, 0.0048245386709542, 0.004795009654030171, 0.004782296786893437, 0.0047844060613056825, 0.00479953607388955, 0.0048257329785549715, 0.004861502134963473, 0.004904820697292641, 0.004954039892769007, 0.005007261039364209, 0.005062564561422771, 0.0051179933806987125, 0.005171611640220892, 0.00522150913946486, 0.005265892870544442, 0.005303134799640369, 0.005331845089384196, 0.005350952647260489, 0.005359393895734639])
    poisson_ratio_targets = generatePoissonRatioDataThetas(thetas, n_tiling_params, strain, ti_target, model)
    if IH == 28:
        # poisson_ratio_targets = np.array([1.0305514815075005, 1.012669063448301, 0.9983653930424908, 0.9760854620854097, 0.9469135772911499, 0.9241089858867978, 0.9075302386032793, 0.8972305565527231, 0.8946428027428622, 0.9032537427846475, 0.9267134745911274, 0.9609406400002467, 0.995953056791863, 1.0254782490131515, 1.0470921348157982, 1.0609124576398712, 1.0692259484473527, 1.0750174828160362, 1.0793007918371773, 1.0814428541143122, 1.0799421896850456, 1.0744988899662153, 1.066936858268616, 1.0579452995439098, 1.046264882997938, 1.0309136269602002, 1.0159667567873247, 1.0020042802152218, 0.9855796493607557, 0.9612629744106376, 0.9352652334312358, 0.9139078823741412, 0.8992576412621971, 0.8937920240121994, 0.9013168597102745, 0.9233982121181953, 0.9550951896943425, 0.9872101422046681, 1.0145031876079933, 1.0373101721007572, 1.05660738455303, 1.0713664168056236, 1.0804882453694686, 1.0851594166118306, 1.0869486772361947, 1.0853637130052902, 1.0793267573511756, 1.0708610354670605, 1.0608193315501948, 1.0474468729103958])
        poisson_ratio_targets = np.array([1.0561218321318777, 1.0568443148288544, 1.0574062514622313, 1.055391055536777, 1.049284559766551, 1.0392282181694394, 1.023629980537073, 1.0008446760815706, 0.9728138043603514, 0.9401623204825427, 0.8753084542414213, 0.835619060017621, 0.8089700079630953, 0.791978438726686, 0.7839494395204849, 0.7858520337612478, 0.8006975275390967, 0.832084446388504, 0.8873764831223246, 0.9728263163634734, 1.00703511686131, 1.0194615352674514, 1.0303504405788015, 1.0414715408930273, 1.0514742448035446, 1.0574788991428457, 1.0563811499189961, 1.0507596184540926, 1.0465168154140558, 1.0432899332334102, 1.033105802074758, 1.0170972871271278, 0.9840532240784822, 0.9081628278596467, 0.8667907403499018, 0.8367096178588956, 0.8141035289237027, 0.7977289510478281, 0.78703782722618, 0.7819611161199933, 0.7830033301730911, 0.7914192495436153, 0.808596680642313, 0.8368403931662699, 0.8800483971013053, 0.9378137687220685, 0.9822083655465698, 1.0092538712670118, 1.0288486377603403, 1.0445670596501868])

    if IH == 21:
        mean = np.mean(poisson_ratio_targets)
        poisson_ratio_targets = np.full((len(poisson_ratio_targets), ), mean)
    # exit(0)
    sample_points_theta = thetas[idx]
    batch_dim = len(thetas)
    poisson_ratio_targets_sub = poisson_ratio_targets[idx]

    def objAndGradient(x):
        _uniaxial_strain = []
        dqdp = []
        for theta in thetas:
            uni_strain, dqidpi = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, x, False)
            _uniaxial_strain.append(uni_strain)
            dqdp.append(dqidpi)
        

        ti_TF = tf.convert_to_tensor(x)

        uniaxial_strain_TF = tf.convert_to_tensor(_uniaxial_strain)
        poisson_ratio_current, poisson_ratio_grad, dOdE = objGradPoissonRatio( 
                                            ti_TF, uniaxial_strain_TF, 
                                            tf.convert_to_tensor(thetas), 
                                            model)
        
        poisson_ratio_current = poisson_ratio_current.numpy()[idx]
        poisson_ratio_grad = poisson_ratio_grad.numpy()[idx]
        dOdE = dOdE.numpy()[idx]

        obj = (np.dot(poisson_ratio_current - poisson_ratio_targets_sub, np.transpose(poisson_ratio_current - poisson_ratio_targets_sub)) * 0.5)
        
        grad = np.zeros((n_tiling_params))
        
        for i in range(len(idx)):
            grad += (poisson_ratio_current[i] - poisson_ratio_targets_sub[i]) * poisson_ratio_grad[i].flatten() + \
                (poisson_ratio_current[i] - poisson_ratio_targets_sub[i]) * np.dot(dOdE[i][i], dqdp[i][:3, :]).flatten()
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    if not plot_GT:
        # result = minimize(objAndGradient, ti, method='trust-constr', jac=True, options={'disp' : True}, bounds=bounds)
        tic = time.perf_counter()
        result = minimize(objAndGradient, ti, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
        toc = time.perf_counter()
        print(f"Optimization takes {toc - tic:0.6f} seconds")
        uniaxial_strain_opt = []
        for theta in thetas:
            uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, result.x, False)
            uniaxial_strain_opt.append(uni_strain)

        uniaxial_strain_opt = np.reshape(uniaxial_strain_opt, (batch_dim, 3))
        nn_inputs = tf.convert_to_tensor(np.hstack((np.tile(result.x, (batch_dim, 1)), uniaxial_strain_opt)))
        poisson_ratio_opt = computeDirectionalPoissonRatio(n_tiling_params, nn_inputs, 
                        tf.convert_to_tensor(thetas), model)
        poisson_ratio_opt = poisson_ratio_opt.numpy()
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

    if IH == 50:
        thetas += np.pi * 0.5

    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] - np.pi)
        poisson_ratio = np.append(poisson_ratio, poisson_ratio[i])
        poisson_ratio_targets = np.append(poisson_ratio_targets, poisson_ratio_targets[i])
        if not plot_GT:
            poisson_ratio_opt = np.append(poisson_ratio_opt, poisson_ratio_opt[i])
    thetas = np.append(thetas, thetas[0])
    poisson_ratio = np.append(poisson_ratio, poisson_ratio[0])
    poisson_ratio_targets = np.append(poisson_ratio_targets, poisson_ratio_targets[0])
    min_target, max_target = np.min(poisson_ratio_targets), np.max(poisson_ratio_targets)
    min_init, max_init = np.min(poisson_ratio), np.max(poisson_ratio)
    if not plot_GT:
        poisson_ratio_opt = np.append(poisson_ratio_opt, poisson_ratio_opt[0])
    
        min_opt, max_opt = np.min(poisson_ratio_opt), np.max(poisson_ratio_opt)
        max_pr = np.max([max_init, max_opt, max_target])
        min_pr = np.min([min_init, min_opt, min_target])

    else:
        max_pr = np.max([max_init, max_target])
        min_pr = np.min([min_init, min_target])

    dpr = max_pr - min_pr

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
    
    ax1.set_ylim(min_pr - 0.1 * dpr, max_pr + 0.1 * max_pr)
    
    # ax1.set_ylim(-1.0, 3.5)
    ax1.plot(thetas,poisson_ratio,lw=2.5, label = "poisson ratio initial")
    ax1.plot(thetas,poisson_ratio_targets,lw=2.5, label = "poisson ratio target", linestyle = "dashed")
    # plt.polar(thetas, poisson_ratio, label = "poisson_ratio initial", linewidth=3.0, zorder=0)
    # plt.polar(thetas, poisson_ratio_targets, linestyle = "dashed", label = "poisson_ratio target", linewidth=3.0, zorder=0)
    plt.legend(loc='upper left')
    plt.savefig("poisson_ratio_optimization_IH"+str(IH)+"_initial.png", dpi=300)
    plt.close()
    os.system("convert poisson_ratio_optimization_IH"+str(IH)+"_initial.png -trim poisson_ratio_optimization_IH"+str(IH)+"_initial.png")
    if not plot_GT:
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
        ax1.set_ylim(min_pr - 0.1 * dpr, max_pr + 0.1 * max_pr)
        # ax1.set_ylim(-0.5, 3.0)
        # plt.polar(thetas, poisson_ratio_opt, label = "poisson_ratio optimized", linewidth=3.0, zorder=0)
        # plt.polar(thetas, poisson_ratio_targets, linestyle = "dashed", label = "poisson_ratio target", linewidth=3.0, zorder=0)
        ax1.plot(thetas,poisson_ratio_opt,lw=2.5, label = "poisson ratio optimized")
        ax1.plot(thetas,poisson_ratio_targets,lw=2.5, label = "poisson ratio target", linestyle = "dashed")
        plt.legend(loc='upper left')
        plt.savefig("poisson_ratio_optimization_IH"+str(IH)+"_optimized.png", dpi=300)
        plt.close()
        os.system("convert poisson_ratio_optimization_IH"+str(IH)+"_optimized.png -trim poisson_ratio_optimization_IH" + str(IH) + "_optimized.png")


def findNegativePoissonRatioStructure(save_data = False):
    IH = 21
    n_sp_theta = 50
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))

    strain = 0.1
    strain = strain + 0.5 * strain * strain
    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    candidates = []
    cnt = 0
    indices = []
    if IH == 1:
        sample_data = IH01_samples
    elif IH == 21:
        sample_data = IH21_samples
        folder_name = "IH21"
    elif IH == 28:
        sample_data = IH28_samples
        folder_name = "IH28"
    for ti in sample_data:
        # print(ti)
        uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)

        batch_dim = len(thetas)
        ti_batch = np.tile(ti, (batch_dim, 1))
        nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
        poisson_ratio = computeDirectionalPoissonRatio(n_tiling_params, nn_inputs, 
                        tf.convert_to_tensor(thetas), model)
        poisson_ratio = poisson_ratio.numpy()
        cnt += 1

        for nu in poisson_ratio:
            if nu < 0.0:
                indices.append(cnt)
                candidates.append(ti)
                if save_data:
                    thetas_plot = thetas.copy()
                    poisson_ratio_plot = poisson_ratio.copy()
                    for i in range(n_sp_theta):
                        thetas_plot= np.append(thetas_plot, thetas_plot[i] - np.pi)
                        poisson_ratio_plot = np.append(poisson_ratio_plot, poisson_ratio_plot[i])
                    thetas_plot = np.append(thetas_plot, thetas_plot[0])
                    poisson_ratio_plot = np.append(poisson_ratio_plot, poisson_ratio_plot[0])
                    plt.polar(thetas_plot, poisson_ratio_plot, label = "tensor", linewidth=3.0)
                    plt.savefig("images/poisson_ratio/"+folder_name+"/poisson_ratio_"+str(cnt)+"_.png", dpi=300)
                    # plt.savefig("images/poisson_ratio/"+folder_name+"/poisson_ratio_"+str(cnt)+"_"+str(ti[0])+","+str(ti[1])+","+str(ti[2])+","+str(ti[3])+"_.png", dpi=300)
                    plt.close()
                break
    if save_data:
        f = open("images/poisson_ratio/"+folder_name+"/negative.txt", "w+")
        f.write("np.array([")
        for j in range(len(candidates) - 1):
            candidate = candidates[j]
            f.write("[")
            for i in range(n_tiling_params - 1):
                f.write(str(candidate[i])+", ")
            f.write(str(candidate[n_tiling_params - 1])+"], ")
        f.write("[")
        if len(candidates):
            candidate = candidates[len(candidates) - 1]
        for i in range(n_tiling_params - 1):
            f.write(str(candidate[i])+", ")
        f.write(str(candidate[n_tiling_params - 1])+"]])\n")
        for candidate in candidates:
            for i in range(n_tiling_params - 1):
                f.write(str(candidate[i])+", ")
            f.write(str(candidate[n_tiling_params - 1])+"\n")
        f.close()
    print("{}/{}".format(len(candidates), len(sample_data)))


def fillPolarData(thetas, poisson_ratio):
    n_sp_theta = len(thetas)
    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] + np.pi)
        poisson_ratio = np.append(poisson_ratio, poisson_ratio[i])
    return thetas, poisson_ratio


def poissonRatioModifyUI():
    IH = 28
    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    # ti = np.array([0.1224, 0.6, 0.13, 0.625])
    ti = np.array([0.55, 0.7])
    # ti = np.array([0.6, 0.6])
    # ti = ti_default
    thetas_nn = np.arange(0.0, np.pi, np.pi/float(50))
    thetas = np.arange(0.0, np.pi, np.pi/float(50))
    
    thetas, poisson_ratio = getDirectionPoissonRatio(ti, n_tiling_params, model, 0.02, 20, False)

    thetas_full, poisson_ratio_full = fillPolarData(thetas, poisson_ratio)

    # x, y = pol2cart(poisson_ratio_full, thetas_full)
    x, y = thetas_full, poisson_ratio_full

    min_x, min_y, max_x, max_y = np.min(x), np.min(y), np.max(x), np.max(y)
    dx, dy = max_x - min_x, max_y - min_y

    poly = Polygon(np.column_stack([x, y]), animated=True, visible = False)


    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    fig.set_size_inches(20, 20)
    ax.add_patch(poly)
    p = MacroPropertyModifier(ax, poly, thetas_full, thetas_nn)
    # ax.set_title('Move control points in Cartesian space')
    # ax.set_xlim((min_x - 0.2 * dx, max_x + 0.2 * dx))
    # ax.set_ylim((min_y - 0.2 * dy, max_y + 0.2 * dy))    
    ax.set_ylim((min_y - 0.1 * dy, max_y + 0.1 * dy))    
    # ax.set_ylim(0, 5.5)
    ax.grid(linewidth=3)

    # plt.axis('off')
    # plt.polar([], [])
    plt.show()

if __name__ == "__main__":
    # computePoissonRatio()
    poissonRatioSA()
    # poissonRatioModifyUI()
    # findNegativePoissonRatioStructure(True)