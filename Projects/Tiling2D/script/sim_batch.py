import os
from joblib import Parallel, delayed
def process(i):
    os.mkdir("/home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(i))
    os.system("taskset --cpu-list " + str(i%8) + " ../../../build/Projects/Tiling2D/Tiling2D " + str(i)
     + " >> /home/yueli/Documents/ETH/SandwichStructure/ForceDisplacementCurve/" + str(i) + "/log.txt")
    # os.system("taskset --cpu-list 0 ../../../build/Projects/Tiling2D/Tiling2D " + str(i))
    
results = Parallel(n_jobs=8)(delayed(process)(i) for i in range(16))
