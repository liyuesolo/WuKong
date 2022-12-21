import os
from functools import cmp_to_key
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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from Summary import *

tf.keras.backend.set_floatx('float32')

def relativeL2(y_true, y_pred):
    if (y_true.shape[1] > 1):
        stress_norm = tf.norm(y_true, ord='euclidean', axis=1)
        norm = tf.tile(tf.keras.backend.expand_dims(stress_norm, 1), tf.constant([1, 3]))
        y_true_normalized = tf.divide(y_true, norm)
        y_pred_normalized = tf.divide(y_pred, norm)
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
    else:
        y_true_normalized = tf.ones(y_true.shape, tf.float32)
        y_pred_normalized = tf.divide(y_pred + K.epsilon(), y_true + K.epsilon())
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
        

def absL2(y_true, y_pred):
    if (y_true.shape[1] > 1):
        sigma_xx = K.mean(K.square((y_true[:, 0] - y_pred[:, 0])))
        sigma_yy = K.mean(K.square((y_true[:, 1] - y_pred[:, 1])))
        sigma_xy = K.mean(K.square((y_true[:, 2] - y_pred[:, 2])))
        sigma_yx = K.mean(K.square((y_true[:, 3] - y_pred[:, 3])))
        return sigma_xx + sigma_yy + sigma_xy + sigma_yx
    else:
        return K.mean(K.square((y_true - y_pred)))

def loadDataSplitTest(n_tiling_params, filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    
    for line in open(filename).readlines():
        if len(line.strip().split(" ")) != 11 + n_tiling_params:
            continue
        if "\x00" in line:
            continue
        # exit(0)
        item = [float(i) for i in line.strip().split(" ")]
        if (ignore_unconverging_result):
            if (item[-1] > 1e-6 or math.isnan(item[-1])):
                continue
            if (item[-5] < 1e-6 or item[-5] > 10):
                continue
            # if (np.abs(item[-3] - 1.001) < 1e-6 or np.abs(item[-3] - 0.999) < 1e-6):
            #     continue
        data = item[0:n_tiling_params]
        for i in range(2):
            data.append(item[n_tiling_params+i])
        data.append(2.0*item[n_tiling_params+2])
        # data.append(item[n_tiling_params+2])
        label = item[n_tiling_params+3:n_tiling_params+5]
        label.append(1.0 * item[n_tiling_params+5])
        label.append(item[n_tiling_params+6])
        
        all_data.append(data)
        all_label.append(label)
        
    print("#valid data:{}".format(len(all_data)))
    
    all_data = np.array(all_data[:]).astype(np.float32)
    all_label = np.array(all_label[:]).astype(np.float32) 
    # all_data = np.array(all_data).astype(np.float32)
    # all_label = np.array(all_label).astype(np.float32)
    
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    
    all_data = all_data[indices]
    all_label = all_label[indices]
    
    return all_data, all_label


loss_l2 = tf.keras.losses.MeanSquaredError()
loss_logl2 = tf.keras.losses.MeanSquaredLogarithmicError()
# loss_function = relativeL2

w_grad = tf.constant(1.0, dtype=tf.float32)
w_e = tf.constant(1.0, dtype=tf.float32)

def generator(train_data, train_label):    
    indices = np.arange(train_data.shape[0])
    while True:
        np.random.shuffle(indices)
        yield train_data[indices], train_label[indices]

@tf.function
def trainStep(n_tiling_params, opt, lambdas, sigmas, model, train_vars):
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(train_vars)
        tape.watch(lambdas)
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 3])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, psi)

        loss = grad_loss + e_loss
        
    dLdw = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(dLdw, train_vars))
    gradNorm = tf.math.sqrt(tf.reduce_sum([tf.reduce_sum(gi*gi) for gi in dLdw]))
    # gradNorm = -1.0
    
    del tape
    return grad_loss, e_loss, gradNorm

@tf.function
def testStep(n_tiling_params, lambdas, sigmas, model):
    
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 3])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        
        grad_loss = w_grad * relativeL2(stress_gt, stress_pred)
        e_loss = w_e * relativeL2(potential_gt, psi)
    del tape
    return grad_loss, e_loss, stress_pred, psi

def plot(prefix, prediction, label, gt_only = False):
    def cmp_sigma_xx(i, j):
        return label[i][0] - label[j][0]
    def cmp_sigma_xy(i, j):
        return label[i][2] - label[j][2]
    def cmp_sigma_yx(i, j):
        return label[i][3] - label[j][3]
    def cmp_sigma_yy(i, j):
        return label[i][1] - label[j][1]
        
    indices = [i for i in range(len(label))]
    data_point = [i for i in range(len(label))]

    indices = sorted(indices, key=cmp_to_key(cmp_sigma_xx))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_xx_gt = [sigma_gt_sorted[i][0] for i in range(len(label))]
    sigma_xx = [sigma_sorted[i][0] for i in range(len(label))]
    if not gt_only:
        plt.plot(data_point, sigma_xx, linewidth=1.0, label = "Sigma_xx")
    plt.plot(data_point, sigma_xx_gt, linewidth=1.0, label = "GT Sigma_xx")
    plt.legend(loc="upper left")
    plt.savefig(prefix+"_learned_sigma_xx.png", dpi = 300)
    plt.close()
    
    indices = sorted(indices, key=cmp_to_key(cmp_sigma_yy))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_yy_gt = [sigma_gt_sorted[i][1] for i in range(len(label))]
    sigma_yy = [sigma_sorted[i][1] for i in range(len(label))]
    if not gt_only:
        plt.plot(data_point, sigma_yy, linewidth=1.0, label = "Sigma_yy")
    plt.plot(data_point, sigma_yy_gt, linewidth=1.0, label = "GT Sigma_yy")
    plt.legend(loc="upper left")
    plt.savefig(prefix + "_learned_sigma_yy.png", dpi = 300)
    plt.close()

    indices = sorted(indices, key=cmp_to_key(cmp_sigma_xy))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_xy_gt = [sigma_gt_sorted[i][2] for i in range(len(label))]
    sigma_xy = [sigma_sorted[i][2] for i in range(len(label))]
    if not gt_only:
        plt.plot(data_point, sigma_xy, linewidth=1.0, label = "Sigma_xy")
    plt.plot(data_point, sigma_xy_gt, linewidth=1.0, label = "GT Sigma_xy")
    plt.legend(loc="upper left")
    plt.savefig(prefix + "_learned_sigma_xy.png", dpi = 300)
    plt.close()

    # indices = sorted(indices, key=cmp_to_key(cmp_sigma_yx))
    # sigma_gt_sorted = label[indices]
    # sigma_sorted = prediction[indices]
    # sigma_xy_gt = [sigma_gt_sorted[i][3] for i in range(len(label))]
    # sigma_xy = [sigma_sorted[i][3] for i in range(len(label))]
    # if not gt_only:
    #     plt.plot(data_point, sigma_xy, linewidth=1.0, label = "Sigma_yx")
    # plt.plot(data_point, sigma_xy_gt, linewidth=1.0, label = "GT Sigma_yx")
    # plt.legend(loc="upper left")
    # plt.savefig(prefix + "_learned_sigma_yx.png", dpi = 300)
    # plt.close()

def plotPotentialClean(result_folder, n_tiling_params, tiling_params_and_strain, stress_and_potential, model, prefix = "strain_energy"):
    save_path = result_folder
    
    grad_loss, e_loss, sigma, energy = testStep(n_tiling_params, tf.convert_to_tensor(tiling_params_and_strain), stress_and_potential, model)
    # sigma and energy are the stress and potential from the network

    elastic_potential = model(tf.convert_to_tensor(tiling_params_and_strain), training = False)

    potential_gt = stress_and_potential[:, -1] # last entry is the potential
    # potential_pred = energy.numpy() # prediction 
    potential_pred = elastic_potential.numpy() #identical to above
    indices = [i for i in range(len(potential_gt))]
    
    def compare_energy(i, j):
        return potential_gt[i] - potential_gt[j]
    indices_sorted = sorted(indices, key=cmp_to_key(compare_energy))
    
    # print(potential_pred[indices_sorted][17500:17550])
    # print(potential_gt[indices_sorted][17500:17550])
    gt_sorted = potential_gt[indices_sorted]
    pred_sorted = potential_pred[indices_sorted]
    
    plt.plot(indices, pred_sorted, linewidth=0.8, label = "prediction")
    plt.plot(indices, gt_sorted, linewidth=0.8, label = "GT")
    plt.legend(loc="upper right")
    plt.savefig(save_path + prefix + ".png", dpi = 300)
    plt.close()

    error = []
    mse = 0.0
    for i in range(len(potential_gt)):
        error.append(np.abs(gt_sorted[i] - pred_sorted[i]) / gt_sorted[i])
        mse += np.power((gt_sorted[i] - pred_sorted[i]) / gt_sorted[i], 2.0)
    mse /= float(len(potential_gt))
    print(mse)
    plt.plot(indices, error, linewidth=0.8, label = "error")
    
    plt.savefig(save_path + prefix + "_error.png", dpi = 300)
    plt.close()



def plotPotentialPolar(n_tiling_params, result_folder, model_name):
    full_data = True
    if full_data:
        full_data = "/home/yueli/Documents/ETH/SandwichStructure/Server/0/data.txt" 
        all_data, all_label = loadDataSplitTest(n_tiling_params, full_data, False, False)
    else:
        uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data3.txt"
        # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/training_data_with_strain.txt"
        # uniaxial_data = "/home/yueli/Documents/ETH/SandwichStructure/TrainingData/SingleFamilyUniaxialStrain_backup/training_data.txt"
        all_data, all_label = loadDataSplitTest(n_tiling_params, uniaxial_data, False, False)
    
    
    n_sp_per_strain = 50
    save_path = result_folder
    B = np.fromfile(os.path.join(save_path, model_name + "B.dat"), dtype=float)
    model = loadSingleFamilyModel(n_tiling_params)
    model.load_weights(model_name + '.tf')
    
    grad_loss, e_loss, sigma, energy = testStep(n_tiling_params, all_data, all_label, model)
    print("test loss ", grad_loss + e_loss)
    
    for j in range(len(all_label)//n_sp_per_strain):
        potential = energy.numpy()[j*n_sp_per_strain:(j+1)*n_sp_per_strain]
        potential_gt = all_label[j*n_sp_per_strain:(j+1)*n_sp_per_strain, -1]

        
        theta = np.arange(0.0, np.pi, np.pi/float(n_sp_per_strain))
        for i in range(n_sp_per_strain):
            theta = np.append(theta, theta[i] - np.pi)
            potential = np.append(potential, potential[i])
            potential_gt = np.append(potential_gt, potential_gt[i])

        theta = np.append(theta, theta[0])
        potential = np.append(potential, potential[0])
        potential_gt = np.append(potential_gt, potential_gt[0])

        plt.polar(theta, potential,label="prediction")
        plt.polar(theta, potential_gt,label="gt")
        plt.legend()
        plt.savefig(result_folder + model_name + "_"+str(j)+".png", dpi = 300)
        plt.close()

    # stress_norm = []
    # stress_norm_gt = []
    # for j in range(n_sp_per_strain):
    #     stress_norm[j] = np.linalg.norm(sigma[i])
    #     stress_norm_gt[j] = np.linalg.norm(all_label[i*n_sp_per_strain+j, 0:4])

def testUniAxial(n_tiling_params, count, model_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
    model.load_weights(save_path + model_name + '.tf')
    test_data, test_label = loadDataSplitTest(n_tiling_params, "/home/yueli/Documents/ETH/SandwichStructure/Server/strain_stress.txt", True, True)
    grad_loss, e_loss, sigma, energy = testStep(n_tiling_params,test_data, test_label, model)
    
    plotPotentialClean(save_path, n_tiling_params, test_data, test_label, model)

def testMonotonic(n_tiling_params, count, model_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
    model.load_weights(save_path + model_name + '.tf')
    test_data, test_label = loadDataSplitTest(n_tiling_params, "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/monotonic_test.txt", True, False)
    # test_data, test_label = loadDataSplitTest(n_tiling_params, "/home/yueli/Documents/ETH/SandwichStructure/Server/strain_stress.txt", True, True)
    grad_loss, e_loss, sigma, energy = testStep(n_tiling_params, test_data, test_label, model)
    prefix = "monotonic"
    plotPotentialClean(save_path, n_tiling_params, test_data, test_label, model, prefix)

def validate(n_tiling_params, count, model_name, validation_data, validation_label):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    # model = loadSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params)
    model.load_weights(save_path + model_name + '.tf')
    
    # model.save(save_path + model_name + '.h5')
    grad_loss, e_loss, sigma, energy = testStep(n_tiling_params,validation_data, validation_label, model)
    
    plotPotentialClean(save_path, n_tiling_params, validation_data, validation_label, model)
    plot(save_path + model_name + "_validation", sigma.numpy(), validation_label, False)

    print("validation loss grad: {} energy: {}".format(grad_loss, e_loss))

def train(n_tiling_params, model_name, train_data, train_label, validation_data, validation_label):
    batch_size = 80000 #float32
    # batch_size = 40000 
    
    # model = buildSingleFamilyModel(n_tiling_params)
    model = buildSingleFamilyModelSeparateTilingParamsSwish(n_tiling_params, tf.float32)
    model.load_weights("/home/yueli/Documents/NeuralThickShell/Models/48/" + model_name + '.tf')
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-4)
    max_iter = 40000

    val_lambdasTF = tf.convert_to_tensor(validation_data)
    val_sigmasTF = tf.convert_to_tensor(validation_label)

    losses = [[], []]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # model.load_weights("/home/yueli/Documents/ETH/WuKong/Projects/Tiling2D/python/Models/67/" + model_name + '.tf')
    count = 0
    with open('counter.txt', 'r') as f:
        count = int(f.read().splitlines()[-1])
    f = open("counter.txt", "w+")
    f.write(str(count+1))
    f.close()
    summary = Summary("./Logs/" + str(count) + "/")
    
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    g_norm0 = 0
    for iteration in range(max_iter):
        lambdas, sigmas = next(generator(train_data, train_label))
        if batch_size == -1:
            batch = 1
        else:
            batch = int(np.floor(len(lambdas) / batch_size))
        
        train_loss_grad = 0.0
        train_loss_e = 0.0
        g_norm_sum = 0.0
        for i in range(batch):
            mini_bacth_lambdas = lambdas[i * batch_size:(i+1) * batch_size]
            mini_bacth_sigmas = sigmas[i * batch_size:(i+1) * batch_size]

            lambdasTF = tf.convert_to_tensor(mini_bacth_lambdas)
            sigmasTF = tf.convert_to_tensor(mini_bacth_sigmas)
            
            grad, e, g_norm = trainStep(n_tiling_params, opt, lambdasTF, sigmasTF, model, train_vars)
            
            train_loss_grad += grad
            train_loss_e += e
            g_norm_sum += g_norm
        if (iteration == 0):
            g_norm0 = g_norm_sum
        validation_loss_grad, validation_loss_e, _, _ = testStep(n_tiling_params, val_lambdasTF, val_sigmasTF, model)
        losses[0].append(train_loss_grad + train_loss_e)
        losses[1].append(validation_loss_grad + validation_loss_e)
        print("epoch: {}/{} train_loss_grad: {} train_loss e: {}, validation_loss_grad:{} loss_e:{} |g|: {}, |g_init|: {} ".format(iteration, max_iter, train_loss_grad, train_loss_e, \
                         validation_loss_grad, validation_loss_e, \
                        g_norm_sum, g_norm0))
        summary.saveToTensorboard(train_loss_grad, train_loss_e, validation_loss_grad, validation_loss_e, iteration)
        if iteration % 5000 == 0:
            model.save_weights(save_path + model_name + '.tf')
    
    
    model.save_weights(save_path + model_name + '.tf')
    # fourier_B = model.get_config()['layers'][-1]['config']['B']
    # np.reshape(fourier_B,-1).astype(float).tofile(os.path.join(save_path, model_name + "B.dat"))
    idx = [i for i in range(len(losses[0]))]
    plt.plot(idx, losses[0], label = "train_loss")
    plt.plot(idx, losses[1], label = "validation_loss")
    plt.legend(loc="upper left")
    plt.savefig(save_path + model_name + "_log.png", dpi = 300)
    plt.close()

def test(n_tiling_params, model_name, test_data, test_label):
    
    test_dataTF = tf.convert_to_tensor(test_data)
    test_labelTF = tf.convert_to_tensor(test_label)
    save_path = "./"
    
    model = loadSingleFamilyModel(n_tiling_params)
    model.load_weights(model_name+'.tf')
    
    test_loss, sigma, energy = testStep(test_dataTF, test_labelTF, model)
    plot(model_name + "_test", sigma.numpy(), test_label)
    print("test_lost", test_loss)
    return sigma
    
    
if __name__ == "__main__":
    
    n_tiling_params = 4
    # full_data = "all_data_IH21_shuffled.txt"
    # full_data = "all_data_IH50_shuffled.txt"
    full_data = "all_data_IH01_shuffled.txt"
    # full_data = "all_data_IH67_shuffled.txt"
    # full_data = "all_data_IH28_shuffled.txt"
    
    data_all, label_all = loadDataSplitTest(n_tiling_params, full_data, False, True)    
    

    five_percent = int(len(data_all) * 0.05)
    # five_percent = int(len(data_all) * 0.2)
    
    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]
    # train_data = data_all
    # validation_data = data_all
    # train_label = label_all
    # validation_label = label_all

    test_data = []
    test_label = [] 
    model_name = "IH01"
    
    
    train(n_tiling_params, model_name, 
        train_data, train_label, validation_data, validation_label)
    # validate(n_tiling_params, 32, 
    #     model_name, validation_data, validation_label)
    # testUniAxial(n_tiling_params, 20, "full40k")
    # testMonotonic(n_tiling_params, 81, "IH50")
    # plotPotentialPolar(n_tiling_params, result_folder, model_name)
    