import os
import numpy as np

count = 0
with open('counter.txt', 'r') as f:
    count = int(f.read().splitlines()[-1])
f = open("counter.txt", "w+")
f.write(str(count+1))
f.close()

exp_id = count+1

data_folder = "./output/cells/"
exp_folder = data_folder + str(exp_id)
os.mkdir(exp_folder)

cmd = "./build/Projects/CellSim/CellSim" + " " + exp_folder + " >> " + exp_folder + "/log.txt"

# additional_comment = "sim tol 1e-9 project dOdx"
additional_comment = "gradient descent without box constraint"

f = open(exp_folder + "/comments.txt", "w+")
f.write(additional_comment)
f.close()

# print(cmd)
os.system(cmd)