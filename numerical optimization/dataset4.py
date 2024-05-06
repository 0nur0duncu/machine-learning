import numpy as np
import math

ti = [-10.0000, -8.9500, -7.9000, -6.8500, -5.8000, -4.7500, -3.7000, -2.6500, -1.6000, -0.5500, 0.5000, 1.5500, 2.6000, 3.6500, 4.7000, 5.7500, 6.8000, 7.8500, 8.9000, 9.9500]
yi = [-0.0116, 0.2860, 0.3363, 0.0980, -0.0604, -0.1570, -0.1363, 0.3304, 0.9160, 1.1664, 1.1409, 0.9474, 0.2325, 0.1845, -0.1749, 0.0926, 0.1228, 0.3232, 0.0579, 0.2179]

#-----------------------
import matplotlib.pyplot as plt
plt.scatter(ti, yi, color='darkred')
plt.xlabel('ti')
plt.ylabel('yi')
plt.title('Dataset 4', fontstyle='italic')
plt.grid(color='green', linestyle='--', linewidth=0.1)
plt.show()
#------------------------