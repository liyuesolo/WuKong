import matplotlib.pyplot as plt
import numpy as np

def loadLogCheckConvergence(base_dir, solver):
    log_file = base_dir + "log.txt"
    iterations = []
    residual_norms = []
    energy = []
    smallest_evs = []
    ls_cnt = []
    for line in open(log_file).readlines():
        if "[" + solver +"]" in line.strip() and "\t" not in line:
            iteration = line.split("|g|")[0].split("iter ")[-1]
            residual_norm = line.split("|g| ")[-1].split(" |g_init|")[0]
            obj = line.split("obj: ")[-1]
            iterations.append(int(iteration))
            residual_norms.append(float(residual_norm))
            energy.append(float(obj))
        if "forward simulation hessian eigen values:" in line.strip():
            smallest_evs.append(float(line.strip().split(' ')[-1]))
        if "# ls" in line.strip():
            ls_cnt.append(float(line.strip().split(" ")[-1]))
        
    iterations = np.array(iterations)
    residual_norms = np.array(residual_norms)
    energy = np.array(energy)
    smallest_evs = np.array(smallest_evs)
    ls_cnt = np.array(ls_cnt)

    n = np.min([len(iterations), len(residual_norms), len(energy), len(smallest_evs), len(ls_cnt)])
    
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(solver + 'log - power 2, without sim term, 0.25 * cell_length')
    axs[0, 0].plot(iterations[:n], residual_norms[:n])
    axs[0, 0].set_title('residual norm')
    axs[0, 1].plot(iterations[:n], energy[:n])
    axs[0, 1].set_title('energy')
    axs[1, 0].plot(iterations[:n], smallest_evs[:n])
    axs[1, 0].set_title('smallest ev')
    axs[1, 1].plot(iterations[:n], ls_cnt[:n])
    axs[1, 1].set_title('ls cnt')
    fig.set_size_inches(14, 10)
    plt.savefig(base_dir + solver + "_log.png", dpi=300)


if __name__ == "__main__":
    base_dir = "/home/yueli/Documents/ETH/WuKong/output/cells/458/"
    # loadLogCheckConvergence(base_dir)
    # loadSQPLogCheckConvergence(base_dir)
    loadLogCheckConvergence(base_dir, "SQP")