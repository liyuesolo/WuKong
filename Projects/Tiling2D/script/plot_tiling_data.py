from cProfile import label
import matplotlib.pyplot as plt
import os
def plotStrainEnergyCurve(folder):
    data_file = folder + "sample_tiling_along_strain.txt"
    img_file = folder + "energy.png"
    strain_xx = []
    psi = []
    for line in open(data_file).readlines():
        item = [float(i) for i in line.strip().split(" ")]
        strain_xx.append(item[0])
        psi.append(item[-2])
    n_sp = 21
    n_params = len(strain_xx) // n_sp
    for i in range(n_params - 1):
        plt.plot(strain_xx[i*n_sp:(i+1)*n_sp], psi[i*n_sp:(i+1)*n_sp])
    plt.xlabel("strain_xx")
    plt.ylabel("energy density")
    plt.savefig(img_file, dpi=300)
    plt.close()

def plotTilingEnergyCurve(folder):
    data_file = folder + "sample_tiling_along_strain.txt"
    img_file = folder + "energy.png"
    strain_xx = []
    psi = []
    ti_range = [0.1, 0.2]
    dt = 0.1 / 20.0
    for line in open(data_file).readlines():
        item = [float(i) for i in line.strip().split(" ")]
        strain_xx.append(item[0])
        psi.append(item[-2])
    n_sp = 21
    n_params = len(strain_xx) // n_sp
    
    pid = [ti_range[0] + float(i) * dt for i in range(n_params)]
    for i in range(n_sp-1):
        plt.plot(pid, psi[i:-1:n_sp])
    plt.xlabel("tiling param 2")
    plt.ylabel("energy density")
    plt.savefig(img_file, dpi=300)
    plt.close()

def plotVecStrainEnergyCurve(folder):
    plt.title("[-0.2, 0.05] 0.2, 0.001, 0.001")
    data_file = folder + "sample_tiling_vec_along_strain_compression.txt"
    img_file = folder + "vec_energy.png"
    strain_xx = []
    psi = []
    tiling_data = []
    for line in open(data_file).readlines():
        item = [float(i) for i in line.strip().split(" ")]
        tiling_data.append([item[0], item[1]])
        strain_xx.append(item[2])
        if (item[-1] < 1e-6):
            psi.append(item[-2])
        else:
            psi.append(0.002)
    n_sp = 21
    n_params = len(strain_xx) // n_sp
    for i in range(n_params):
        plt.plot(strain_xx[i*n_sp:(i+1)*n_sp], psi[i*n_sp:(i+1)*n_sp], label=str(tiling_data[i * n_sp]))
    plt.xlabel("strain_yy")
    plt.ylabel("energy density")
    plt.legend(loc="upper left")
    plt.savefig(img_file, dpi=300)
    plt.close()

def plotTilingEnergyCurve(folder):
    data_file = folder + "sample_tiling_vec_along_strain.txt"
    img_file = folder + "vec_tiling_energy.png"
    strain_xx = []
    psi = []
    ti_range = [0.1, 0.2]
    dt = 0.1 / 20.0
    for line in open(data_file).readlines():
        item = [float(i) for i in line.strip().split(" ")]
        strain_xx.append(item[0])
        psi.append(item[-2])
    n_sp = 21
    n_params = len(strain_xx) // n_sp
    
    pid = [ti_range[0] + float(i) * dt for i in range(n_params)]
    for i in range(n_sp-1):
        plt.plot(pid, psi[i:-1:n_sp])
    plt.xlabel("tiling param along dir")
    plt.ylabel("energy density")
    plt.savefig(img_file, dpi=300)
    plt.close()


# def plotAgain(folder):
    
#     data_file = folder + "constant_energy.txt"
#     img_file = folder + "constant_energy.png"
#     strain_xx = []
#     psi = []
#     tiling_data = []
#     for line in open(data_file).readlines():
#         item = [float(i) for i in line.strip().split(" ")]
#         tiling_data.append([item[0], item[1]])
#         strain_xx.append(item[2])
#         psi.append(item[-2])
#         # if (item[-1] < 1e-6):
#             # psi.append(item[-2])
#         # else:
#             # psi.append(0.002)
#     n_sp = 21
#     n_params = len(strain_xx) // n_sp
#     for i in range(n_params):
#         plt.plot(strain_xx[i*n_sp:(i+1)*n_sp], psi[i*n_sp:(i+1)*n_sp], label=str(tiling_data[i * n_sp]))
#     plt.xlabel("strain_xx")
#     plt.ylabel("energy density")
#     plt.legend(loc="upper left")
#     plt.savefig(img_file, dpi=300)
#     plt.close()

def plotAgain(folder):
    
    data_file = folder + "constant_energy.txt"
    img_file = folder + "constant_energy"
    strain_xx = []
    psi = []
    tiling_data = []
    for line in open(data_file).readlines():
        item = [float(i) for i in line.strip().split(" ")]
        tiling_data.append([item[0], item[1]])
        strain_xx.append(item[2])
        psi.append(item[-2])
    pred = []
    for line in open(folder + "network.txt").readlines():
        pred = [float(i) for i in line.strip().split(' ')]
    
    for i in range(len(strain_xx)):
        
        plt.plot(strain_xx, psi, linewidth=2, label = "sim")
        plt.plot(strain_xx, pred, linewidth=2, label = "NN")
        
        plt.plot(strain_xx[i], psi[i], "bo")
        plt.plot(strain_xx[i], pred[i], "go")
        plt.xlabel("strain_xx")
        plt.legend(loc="upper left")
        plt.ylabel("energy density")
        plt.savefig(img_file + "_" + str(i) + ".png", dpi = 300)
        plt.close()
        os.system("convert " + folder + str(i) + ".png " + img_file + "_" + str(i) + ".png +append " + folder + "sim_" + str(i) + ".png")
    

plotAgain("/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/")