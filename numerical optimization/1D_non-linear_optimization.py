import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return x**3 + 4*x**2 - 7*x + 6

def F(x):
  return 3*x**2 + 8*x - 7

def FF(x):
  return 6*x + 8

x = np.linspace(-5,5)

x1,x2 = np.roots([3,8,-7])

print("FF(x1):",FF(x1),"FF(x2)",FF(x2))

plt.figure()
plt.plot(x,f(x))
plt.scatter([x1,x2], [0, 0], color='red', label='Kökler')
plt.show()