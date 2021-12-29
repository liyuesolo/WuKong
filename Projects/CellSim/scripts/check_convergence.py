
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing._private.utils import nulp_diff

def loadLogCheckConvergence(base_dir):
    log_file = base_dir + "log.txt"
    iterations = []
    residual_norms = []
    num_indefinite = []
    num_bad_dir = []
    num_ls = []
    dot_prod = []
    du = []
    for line in open(log_file).readlines():
        if "iter" in line and "newton iter" not in line:
            iteration = line.split("/")[0].split("iter ")[-1]
            residual_norm = line.split("residual_norm ")[-1].split(" tol")[0]
            iterations.append(int(iteration))
            residual_norms.append(float(residual_norm))
        elif "# regularization step" in line:
            items = line.split(" ")
            num_indefinite.append(int(items[5]))
            num_bad_dir.append(int(items[9]))
        elif "# ls" in line:
            items = line.split(" ")
            num_ls.append(int(items[2]))
            du.append(float(items[-1]))
        elif "dot(search, -gradient)" in line:
            dot_prod.append(float(line.split(" ")[-1]))
    iterations = np.array(iterations)
    residual_norms = np.array(residual_norms)
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Static solve without IPC')
    axs[0, 0].plot(iterations, residual_norms)
    axs[0, 0].set_title('residual norm')
    axs[0, 1].plot(iterations[:-1], num_indefinite)
    axs[0, 1].set_title('indefinite hessian reg cnt')
    axs[0, 2].plot(iterations[:-1], num_bad_dir)
    axs[0, 2].set_title('dot(search, -gradient) > 1e-3 cnt')
    axs[1, 0].plot(iterations[:-1], num_ls)
    axs[1, 0].set_title('ls cnt')
    axs[1, 1].plot(iterations[:-1], du)
    axs[1, 1].set_title('newton stepsize')
    axs[1, 2].plot(iterations[:-1], dot_prod)
    axs[1, 2].set_title('dot(search, -gradient)')
    fig.set_size_inches(14, 10)
    plt.savefig(base_dir + "log.png", dpi=300)
    
if __name__ == "__main__":
    base_dir = "/home/yueli/Documents/ETH/WuKong/output/cells/cell_fix_3_points_without_IPC_1144/"
    loadLogCheckConvergence(base_dir)
