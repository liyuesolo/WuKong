import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from multiprocessing import Pool

def loadTxtFile(filename):
    coord_xyz = []
    time_stamp = []
    cell_ids = []
    parent_ids = []
    track_ids = []
    node_score = []
    edge_score = []

    # entry = []

    line_cnt = 0
    for line in open(filename).readlines():
        line_cnt += 1
        if line_cnt == 1:
            continue
        data = line.strip().split(', ')
        coord_xyz.append([int(data[3]), int(data[2]), int(data[1])])
        time_stamp.append(int(data[0]))
        cell_ids.append(int(data[4]))
        parent_ids.append(int(data[5]))
        track_ids.append(int(data[6]))
        node_score.append(float(data[7]))
        edge_score.append(float(data[8]))

        # entry.append([int(data[0]),
        # [int(data[3]), int(data[2]), int(data[1])],
        # int(data[4]), int(data[5]), int(data[6]), float(data[7]), float(data[8])])
    
    
    data = [time_stamp, coord_xyz, cell_ids, parent_ids, track_ids, node_score, edge_score]
    
    return data



def savePointCloud(coord_xyz):
    # frames = trackingNeuroblastCurated.shape[0]
    f = open("/home/yueli/Documents/ETH/WuKong/output/cells/2014data/" + str(Frame) + ".obj", "w+")
    for i in range(1000):
        f.write("v " + str(coord_xyz[i, 0]) + " " + str(coord_xyz[i, 1]) + " " + str(coord_xyz[i, 2]) + "\n")
    f.close()
    
    # for frame in range(frames):
    #     f = open("/home/yueli/Documents/ETH/WuKong/output/cells/2014/point_cloud"+str(frame)+".obj", "w+")
    #     f.close()

def saveSingleFrame(frame, data):
    
    f = open("/home/yueli/Documents/ETH/WuKong/output/cells/2014data/side2_" + str(frame) + ".obj", "w+")
    coord_xyz = data[1]
    for i in range(len(data[0])):
        if data[0][i] == frame:
            f.write("v " + str(coord_xyz[i][0]) + " " + str(coord_xyz[i][1]) + " " + str(coord_xyz[i][2]) + "\n")
    f.close()            

def reconstructMeshLabPoisson():
    base_dir = "/home/yueli/Documents/ETH/WuKong/output/cells/2014data/"
    for frame in range(450):
        cmd = "meshlabserver -i " + base_dir + "side2_" + str(frame) + ".obj -o " + base_dir + "/side2_surface_" + str(frame) + ".obj -s poisson.mlx"
        os.system(cmd)

def reconstructTrackingCells():
    base_dir = "/home/yueli/Documents/ETH/WuKong/output/cells/tracking_data/"
    for frame in range(100):
        cmd = "meshlabserver -i " + base_dir + "frame" + str(frame) + ".obj -o " + base_dir + "/surface_frame_" + str(frame) + ".obj -s poisson.mlx"
        os.system(cmd)

if __name__ == "__main__":
    drosophila_data = loadTxtFile("/home/yueli/Downloads/drosophila_data/drosophila_side_2_tracks_071621.txt")
    # drosophila_data = loadTxtFile("/home/yueli/Downloads/drosophila_data/test.txt")
    # for i in range(450):
    #     saveSingleFrame(i, drosophila_data)

    # reconstructMeshLabPoisson()
    reconstructTrackingCells()
