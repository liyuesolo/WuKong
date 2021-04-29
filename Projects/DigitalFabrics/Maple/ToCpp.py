import shutil
import os
def mcgToCpp(filename):
    f = open(filename + ".tmp", "w+")
    for line in open(filename).readlines():
        rhs = line.split(" = ")[-1]
        lhs = line.split(" = ")[0]
        newline = line
        location = lhs.find('t')
        if (location != -1 and '=' in line):
            newline = 'T ' + lhs.strip() + ' = ' + rhs
        else:
            newline = line.strip() + "\n"
        f.write(newline)
    f.close()
    os.system("rm " + filename)
    os.system("mv " + filename + ".tmp " + filename)
    
# mcgToCpp("YarnBendPBCV.mcg")
# mcgToCpp("YarnBendPBCF.mcg")
# mcgToCpp("YarnBendPBCJ.mcg")
# mcgToCpp("YarnBendV.mcg")
# mcgToCpp("YarnBendF.mcg")
# mcgToCpp("YarnBendJ.mcg")


mcgToCpp("YarnBendPBCSDV.mcg")
mcgToCpp("YarnBendPBCSDF.mcg")
mcgToCpp("YarnBendPBCSDJ.mcg")