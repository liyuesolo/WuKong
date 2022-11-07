import numpy as np


def getStVKData(n_tiling_params, filename):
    
    f = open("stvk_gt.txt", "w+")
    cnt = 0
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")]
        cnt += 1
        if (cnt > 1008):
            return
        green_strain = np.reshape(np.array([item[n_tiling_params + 0], item[n_tiling_params + 2], 
            item[n_tiling_params + 2], item[n_tiling_params + 1]]), (2, 2))
        
        S = np.trace(green_strain) * np.eye(2, dtype=np.float32) + 2.0 * green_strain

        for i in range(n_tiling_params, n_tiling_params + 3):
            f.write(str(item[i]) + " ")
        f.write(str(S[0][0]) + " " + str(S[1][1]) + " " + str(S[0][1]) + " ")
        for i in range(n_tiling_params + 6, len(item)):
            f.write(str(item[i]) + " ")
        f.write("\n")
    f.close()


getStVKData(2, "/home/yueli/Documents/ETH/SandwichStructure/ServerIH21/all_data_IH21.txt")