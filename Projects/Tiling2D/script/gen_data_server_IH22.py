from email.policy import default
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from joblib import Parallel, delayed, parallel_backend
def process(i, data):
    IH = 46
    params = data[i]
    # result_folder = "/home/DockerMountFolder/SandwichStructure/TrainingData/" + str(i) + "/"
    # exe_file = "/home/DockerMountFolder/WuKong/build/Projects/Tiling2D/Tiling2D"
    exe_file = "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D"
    result_folder = "/home/yueli/Documents/ETH/SandwichStructure/ServerIH50/" + str(i) + "/"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    os.environ['OMP_THREAD_LIMIT'] = '1'
    # if os.path.exists(result_folder + "structure.vtk"):
    os.system(exe_file+" "+str(IH)+" " + result_folder + " 2 " + str(params[0]) + " " + str(params[1]) + " 0")
    


param_list = []
params_range = [[0.1,0.3], [0.25, 0.75], [0.05, 0.4]]
n_sp_params = 20
default_params = [0.2308, 0.5, 0.2253]

for i in range(n_sp_params):
    pi = params_range[0][0] + (float(i)/float(n_sp_params))*(params_range[0][1] - params_range[0][0])
    for j in range(n_sp_params):
        pj = params_range[1][0] + (float(j)/float(n_sp_params))*(params_range[1][1] - params_range[1][0])
        for k in range(n_sp_params):
            pk = params_range[2][0] + (float(k)/float(n_sp_params))*(params_range[2][1] - params_range[2][0])
            param_list.append([pi, pj, pk])

pca = PCA(n_components=3)
pca.fit(param_list)
print(pca.components_)
                

# Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(len(param_list)))
# Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(8))
# process(0, param_list)


