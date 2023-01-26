import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
def process(i, data):
    IH = -1
    params = data[i]
    # result_folder = "/home/DockerMountFolder/SandwichStructure/ServerToy/" + str(i) + "/"
    # exe_file = "/home/DockerMountFolder/WuKong/build/Projects/Tiling2D/Tiling2D"
    exe_file = "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D"
    result_folder = "/home/yueli/Documents/ETH/SandwichStructure/ServerToy/" + str(i) + "/"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    # os.environ['OMP_THREAD_LIMIT'] = '1'
    # if os.path.exists(result_folder + "structure.vtk"):
    os.system(exe_file+" "+str(IH)+" " + result_folder + " " + str(params[0]))
    
param_list = []
params_range = [[-1.0, 1.0]]
n_sp_params = 400

for i in range(n_sp_params + 1):
    pi = params_range[0][0] + (float(i)/float(n_sp_params))*(params_range[0][1] - params_range[0][0])
    param_list.append([pi])
# print(param_list)
Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(len(param_list)))
# Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(8))
# process(200, param_list)


