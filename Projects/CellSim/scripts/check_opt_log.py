import matplotlib.pyplot as plt
import numpy as np

def loadLogCheckConvergence(base_dir):
    log_file = base_dir + "opt_log.txt"
    iterations = []
    residual_norms = []
    energy = []
    for line in open(log_file).readlines():
        if "[GD]" in line.strip() and "\t" not in line:
            iteration = line.split("|g|")[0].split("iter ")[-1]
            residual_norm = line.split("|g| ")[-1].split(" max")[0]
            obj = line.split("obj: ")[-1]
            iterations.append(int(iteration))
            residual_norms.append(float(residual_norm))
            energy.append(float(obj))
        
    iterations = np.array(iterations)
    residual_norms = np.array(residual_norms)
    energy = np.array(energy)
    
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('min uTu Gradient Descent')
    axs[0].plot(iterations, residual_norms)
    axs[0].set_title('residual norm')
    axs[1].plot(iterations, energy)
    axs[1].set_title('energy')
    fig.set_size_inches(14, 10)
    plt.savefig(base_dir + "log.png", dpi=300)
    
if __name__ == "__main__":
    base_dir = "/home/yueli/Documents/ETH/WuKong/output/cells/opt/"
    loadLogCheckConvergence(base_dir)