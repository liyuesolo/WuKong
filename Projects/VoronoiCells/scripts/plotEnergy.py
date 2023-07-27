import numpy as np
import matplotlib.pyplot as plt

def plotEnergyAlongGradient(filename):
    lines = open(filename).readlines()
    energies = [float(i) for i in lines[0].split(" ")[:-1]]
    step_sizes = [float(i) for i in lines[1].split(" ")[:-1]]
    plt.figure(figsize=(16,9))
    plt.plot(step_sizes, energies, linewidth = 2)
    plt.xticks(np.arange(min(step_sizes), max(step_sizes)+1, 10.0))
    plt.grid()
    plt.savefig("energy_small.png", dpi=300)
    plt.close()
    # plt.show()
    


plotEnergyAlongGradient("energy.txt")