import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import Parallel, delayed

base_folder = "/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/"
def plotForceDisplacementCurve(idx):
    filename = base_folder + str(idx) + "/log.txt"
    image = base_folder + str(idx) + "/" + "force_displacement_curve"
    line_cnt = 0
    displacement = []
    force = []
    for line in open(filename).readlines():
        line_cnt += 1
        if line_cnt % 2 == 1:
            continue
        elif line_cnt == 2:
            displacement = [float(i) for i in line.strip().split(' ')]
        elif line_cnt == 4:
            force = [float(i) for i in line.strip().split(' ')]
    
    for i in range(len(displacement)):
        
        plt.plot(displacement, force, linewidth=2)
        
        plt.plot(displacement[i], force[i], "bo")
        plt.xlabel("displacement in cm")
        plt.ylabel("force in N")
        plt.savefig(image + "_" + str(i) + ".png", dpi = 300)
        plt.close()

def plotForceDisplacementCurveDifferentResolution(idx):
    base_folder = "/home/yueli/Documents/ETH/SandwichStructure/"
    filename = base_folder + "/convergence_test/res" + str(idx) + "/log.txt"
    image = base_folder + "/convergence_test/plots/res" + str(idx) + "_force_displacement_curve.png"
    line_cnt = 0
    displacement = []
    force = []
    for line in open(filename).readlines():
        line_cnt += 1
        if line_cnt % 2 == 1:
            continue
        elif line_cnt == 2:
            displacement = [float(i) for i in line.strip().split(' ')]
        elif line_cnt == 4:
            force = [float(i) for i in line.strip().split(' ')]
    
    plt.plot(displacement, force, linewidth=2)
    plt.xlabel("displacement in cm")
    plt.ylabel("force in N")
    plt.savefig(image, dpi = 300)
    plt.close()

Parallel(n_jobs=8)(delayed(plotForceDisplacementCurve)(idx) for idx in range(2, 3))