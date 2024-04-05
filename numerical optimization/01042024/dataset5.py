import numpy as np
import math

ti = np.arange(-4,4,0.4)
yi = [0.6294 * math.exp(0.8116 * t) + np.random.random() * 1.90 for t in ti]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.scatter(ti,yi,color='darkred')
    plt.xlabel('ti')
    plt.ylabel('yi')
    plt.title('Dataset 5', fontstyle='italic')
    plt.grid(color='green', linestyle='--', linewidth=0.1)
    plt.show()