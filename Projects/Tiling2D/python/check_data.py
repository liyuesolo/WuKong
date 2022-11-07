import os
import math
import numpy as np
import matplotlib.pyplot as plt

def loadDataSplitTest(n_tiling_params, filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        if (ignore_unconverging_result):
            if (item[-1] > 1e-6 or math.isnan(item[-1])):
                continue
            if (item[-5] < 1e-5 or item[-5] > 10):
                continue
            if (np.abs(item[-3] - 1.001) < 1e-6 or np.abs(item[-3] - 0.999) < 1e-6):
                continue
            if (np.abs(item[-2] - 1.001) < 1e-6 or np.abs(item[-2] - 0.999) < 1e-6):
                continue
        data = item[0:n_tiling_params]
        for i in range(3):
            data.append(item[n_tiling_params+i])
            
        data.append(item[n_tiling_params+2])
        label = item[n_tiling_params+3:n_tiling_params+6]
        label.append(item[n_tiling_params+5])
        label.append(item[n_tiling_params+6])
        
        all_data.append(data)
        all_label.append(label)
        
    print("#valid data:{}".format(len(all_data)))
    # exit(0)
    start = 0
    end = -1
    all_data = np.array(all_data[start:end]).astype(np.float32)
    all_label = np.array(all_label[start:end]).astype(np.float32) 
    
    # all_data = np.array(all_data).astype(np.float32)
    # all_label = np.array(all_label).astype(np.float32)
    
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    
    all_data = all_data[indices]
    all_label = all_label[indices]
    
    return all_data, all_label

n_tiling_params = 2
# full_data = "/home/yueli/Documents/ETH/SandwichStructure/ServerIH01/all_data_IH01_shuffled.txt"
full_data = "/home/yueli/Documents/ETH/SandwichStructure/ServerIH50/all_data_IH50_shuffled.txt"
# full_data = "/home/yueli/Documents/ETH/SandwichStructure/Server/all_data_IH21_shuffled.txt"  
data_all, label_all = loadDataSplitTest(n_tiling_params, full_data, False, True)

x = label_all[:, 0]
plt.hist(x, bins = 10)
plt.show()
