import os
import sys
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
def process(i, data):
    IH = 0
    params = data[i]
    result_folder = "/home/DockerMountFolder/SandwichStructure/ServerIH01/" + str(i) + "/"
    exe_file = "/home/DockerMountFolder/WuKong/build/Projects/Tiling2D/Tiling2D"
    # exe_file = "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D"
    # result_folder = "/home/yueli/Documents/ETH/SandwichStructure/ServerIH01/" + str(i) + "/"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    os.environ['OMP_THREAD_LIMIT'] = '1'
    # if os.path.exists(result_folder + "structure.vtk"):
    os.system(exe_file+" "+str(IH)+" " + result_folder + " 4 " + str(params[0]) + " " + str(params[1]) + " " + str(params[2]) + " " + str(params[3]) + " 0")
    


param_list = []
params_range = [[0.05,0.3], [0.25, 0.75], [0.05, 0.15], [0.4, 0.8]]
default = [0.1224, 0.5, 0.1434, 0.625]
n_sp_params = 5
n_sp_params_single = 20


for i in range(n_sp_params+1):
    pi = params_range[0][0] + (float(i)/float(n_sp_params))*(params_range[0][1] - params_range[0][0])
    for j in range(n_sp_params + 1):
        pj = params_range[1][0] + (float(j)/float(n_sp_params))*(params_range[1][1] - params_range[1][0])
        for k in range(n_sp_params + 1):
            pk = params_range[2][0] + (float(k)/float(n_sp_params))*(params_range[2][1] - params_range[2][0])
            for l in range(n_sp_params + 1):
                pl = params_range[3][0] + (float(l)/float(n_sp_params))*(params_range[3][1] - params_range[3][0])
                param_list.append([pi, pj, pk, pl])

for i in range(len(params_range)):
    for j in range(n_sp_params_single + 1):
        params = default.copy()
        pj = params_range[i][0] + (float(j)/float(n_sp_params_single))*(params_range[i][1] - params_range[i][0])
        params[i] = pj
        param_list.append(params)

# print(len(param_list))
# Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(len(param_list)))
for i in range(int(sys.argv[1]), int(sys.argv[2])):
    process(i, param_list)
# Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(8))
# process(0, param_list)


