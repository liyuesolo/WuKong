import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# embryo = scipy.io.loadmat('/home/yueli/Downloads/drosophila_fused.mat')['embryo']
embryo = scipy.io.loadmat('/home/yueli/Downloads/drosophila_raw.mat')['embryo']
frames = len(embryo)
# frames = 1

def generateRenderedImages():
    for frame in range(frames):
    # for frame in range(60, 61):
        # fig = plt.figure()
        # fig.set_size_inches(18.5, 10.5)
        # ax = fig.gca(projection='3d')
        xdata = embryo[frame][0][:, 0]
        ydata = embryo[frame][0][:, 1]
        zdata = embryo[frame][0][:, 2]
        csize = embryo[frame][0][:, 3] * 1.0
        f = open("/home/yueli/Documents/ETH/WuKong/output/cells/gt_imgs/mesh"+str(frame)+".obj", "w+")
        for i in range(len(xdata)):
            f.write("v " + str(xdata[i]) + " " + str(ydata[i]) + " " + str(zdata[i]) + "\n")
        f.close()
        # ax.scatter3D(xdata, ydata, zdata, s=csize, color="b")
        
        # plt.axis('off')
        # plt.savefig("/home/yueli/Documents/ETH/WuKong/output/cells/gt_imgs/" + str(frame) +".jpg")
        
        # plt.close()

def generateMesh():
    for frame in range(frames):
        xdata = embryo[frame][0][:, 0]
        ydata = embryo[frame][0][:, 1]
        zdata = embryo[frame][0][:, 2]
        f = open("/home/yueli/Documents/ETH/WuKong/output/cells/gt_imgs/raw_mesh"+str(frame)+".obj", "w+")
        for i in range(len(xdata)):
            f.write("v " + str(xdata[i]) + " " + str(ydata[i]) + " " + str(zdata[i]) + "\n")
        f.close()

def reconstructMeshLabPoisson():
    base_dir = "/home/yueli/Documents/ETH/WuKong/output/cells/gt_imgs/"
    for frame in range(frames):
        cmd = "meshlabserver -i " + base_dir + "raw_mesh" + str(frame) + ".obj -o " + base_dir + "/surface_raw_" + str(frame) + ".obj -s poisson.mlx"
        os.system(cmd)

reconstructMeshLabPoisson()
# generateMesh()