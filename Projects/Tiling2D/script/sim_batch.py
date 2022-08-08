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

def pipeLine():
    idx_range = [i for i in range(100)]
    Parallel(n_jobs=8)(delayed(resumeSim)(i, False) for i in idx_range)
    Parallel(n_jobs=8)(delayed(renderOBJ)(i) for i in idx_range)
    Parallel(n_jobs=8)(delayed(plotForceDisplacementCurve)(i) for i in idx_range)
    Parallel(n_jobs=8)(delayed(concatImage)(i) for i in idx_range)
    
pipeLine()
# Parallel(n_jobs=8)(delayed(process)(i, False) for i in [1111])
# Parallel(n_jobs=8)(delayed(concatImage)(i) for i in range(2, 3))
# Parallel(n_jobs=8)(delayed(imgToVideo)(i) for i in [1111])
