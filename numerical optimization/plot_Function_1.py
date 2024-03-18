import numpy as np
import matplotlib.pyplot as plt
from Function_1 import F, FT, FTT

x = np.linspace(0,4)
plt.plot(x,F(x), label="F(x)", color="black")
plt.plot(x,FT(x),label="F(x)", color="black", linestyle='--')
plt.xlim(0.5,3)
plt.ylim(-2,2)
plt.grid()
plt.xlabel("x")
plt.ylabel("F(x),FT(x)")
plt.legend(loc="upper right")
plt.show()