import os
import numpy as np
for theta in np.linspace(0, 2.0 * np.math.pi, 40, True):
    os.system("./build/Projects/DigitalFabrics/DigitalFabrics " + str(theta))