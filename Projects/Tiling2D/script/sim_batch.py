from genericpath import isdir
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
def process(i, parallel):
    if not os.path.isdir("/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(i)):
        os.mkdir("/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(i))
    if parallel:
        os.system("taskset --cpu-list " + str(i%8) + " /home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D " + str(i)
            + " /home/yueli/Documents/ETH/ 0 >> /home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(i) + "/out.txt")
    else:
        os.system("/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D " + str(i)
            + " /home/yueli/Documents/ETH/ 0 >> /home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(i) + "/out.txt")

def renderOBJ(i):
    os.system("taskset --cpu-list " + str(i%8) + " /home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D " + str(i)
     + " /home/yueli/Documents/ETH/ 1")

def renderStress(i):
    os.system("taskset --cpu-list " + str(i%8) + " /home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D " + str(i)
     + " /home/yueli/Documents/ETH/ 5")

def resumeSim(i, parallel):
    if (parallel):
        os.system("taskset --cpu-list " + str(i%8) + " /home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D " + str(i)
        + " /home/yueli/Documents/ETH/ 2 >> /home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(i) + "/out.txt")
    else:
        os.system("/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D " + str(i)
        + " /home/yueli/Documents/ETH/ 2 >> /home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(i) + "/out.txt")

def generateCurveData(i):
    os.system("taskset --cpu-list " + str(i%8) + " /home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D " + str(i)
     + " /home/yueli/Documents/ETH/ 3")

def test(i):
    os.system("taskset --cpu-list " + str(i%8) + " /home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D " + str(i)
     + " /home/yueli/Documents/ETH/ 4")

def imgToVideo(tiling_idx):
    os.chdir("/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(tiling_idx))
    for i in range(41):
        percent = i * 2
        name = ""
        if (percent < 10):
            name = "stress_0.0" + str(percent) + "0000.png"
        else:
            name = "stress_0." + str(percent) + "0000.png"
        os.system("cp " + name + " stress_" + str(i) + ".png")
    os.system("ffmpeg -y -r 10 -start_number 0 -i stress_%d.png -c:v mpeg4 tiling_" + str(tiling_idx) +".mp4" )

def concatImage(tiling_idx):
    os.chdir("/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(tiling_idx))
    if not os.path.exists("0.000000.png"):
        return
    for i in range(41):
        percent = i * 2
        name = ""
        if (percent < 10):
            name = "0.0" + str(percent) + "0000.png"
        else:
            name = "0." + str(percent) + "0000.png"
        os.system("cp " + name + " " + str(i) + ".png")
        os.system("convert " + str(i) + ".png" + " force_displacement_curve_" + str(i) + ".png +append tiling_" + str(tiling_idx) + "_" + str(i) + ".png")
    os.system("ffmpeg -y -r 10 -start_number 0 -i tiling_"+str(tiling_idx)+"_%d.png -c:v mpeg4 tiling_" + str(tiling_idx) +".mp4" )

def plotForceDisplacementCurve(idx):
    base_folder = "/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/"

    filename = base_folder + str(idx) + "/log.txt"
    if not os.path.exists(filename):
        return
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

def loadForcedDisplacement(filename):
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
    return force, displacement

def plotSeveralCurvesTogether():
    for IH in range(10):
        tiling_indices = [IH * 16 + i for i in range(16)]
        base_folder = "/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/"
        image = base_folder + "tiling_IH"+str(IH)+".png"
        forces = []
        displacements = []
        for idx in tiling_indices:
            if not os.path.exists(base_folder + str(idx) + "/log.txt"):
                continue
            force, displacement = loadForcedDisplacement(base_folder + str(idx) + "/log.txt")
            displacements.append(displacement)
            forces.append(force)
        for i in range(len(displacements)):
            plt.plot(displacements[i][:-5], forces[i][:-5], linewidth=1.5, label="param_"+str(i))
        # plt.legend(loc="upper left")
        plt.title("IH" + str(IH))
        plt.xlabel("displacement in cm")
        plt.ylabel("force in N")
        plt.savefig(image, dpi = 300)
        plt.close()

def gatherAllVideos(tiling_idx):
    os.chdir("/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(tiling_idx))
    os.system("cp tiling_" + str(tiling_idx) + ".mp4 ../../videos/tiling_" + str(tiling_idx) + ".mp4")

def pipeLine():
    idx_range = [i for i in range(133)]
    # Parallel(n_jobs=8)(delayed(resumeSim)(i, False) for i in idx_range)
    # Parallel(n_jobs=8)(delayed(renderOBJ)(i) for i in idx_range)
    Parallel(n_jobs=8)(delayed(plotForceDisplacementCurve)(i) for i in idx_range)
    Parallel(n_jobs=8)(delayed(concatImage)(i) for i in idx_range)
    
# pipeLine()
# Parallel(n_jobs=8)(delayed(process)(i, True) for i in range(200))
# Parallel(n_jobs=8)(delayed(concatImage)(i) for i in range(2, 3))
# Parallel(n_jobs=8)(delayed(gatherAllVideos)(i) for i in range(133))
plotSeveralCurvesTogether()
