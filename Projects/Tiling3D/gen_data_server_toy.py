import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
import sys
def process(i, data):
    
    params = data[i]
    result_folder = "/home/DockerMountFolder/SandwichStructure/ServerToy3D/" + str(i) + "/"
    exe_file = "/home/DockerMountFolder/WuKong/build/Projects/Tiling3D/Tiling3D"
    # exe_file = "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling3D/Tiling3D"
    # result_folder = "/home/yueli/Documents/ETH/SandwichStructure/ServerToy3D/" + str(i) + "/"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    # os.environ['OMP_THREAD_LIMIT'] = '1'
    # if os.path.exists(result_folder + "structure.vtk"):
    # print(exe_file+" "+result_folder + " " + str(params[0]) + " " + str(params[1]) + " " + str(params[2]))
    os.system(exe_file+" "+result_folder + " " + str(params[0]) + " " + str(params[1]))
    
param_list = []
params_range = [[-1.0, 1.0]]
# loading_type = [0, 1, 2]
loading_type = [0]
n_sp_params = 2

for i in range(n_sp_params + 1):
    pi = params_range[0][0] + (float(i)/float(n_sp_params))*(params_range[0][1] - params_range[0][0])
    for loading in loading_type:
        param_list.append([pi, loading])
# print(param_list)
# Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(len(param_list)))
for i in range(int(sys.argv[1]), int(sys.argv[2])):
    process(i, param_list)


