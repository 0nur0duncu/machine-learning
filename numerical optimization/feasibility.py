import numpy as np
import matplotlib.pyplot as plt

# Eşitsizlik denklemi: x1 + 3*x2 + 2*x3 - 8 <= 0

# Eşitsizlik doğrusunu tanımla
def inequality_plane(x1, x2):
    return (8 - x1 - 3*x2) / 2

# Eşitsizlik bölgesinde noktaları oluştur
x1 = np.arange(0, 10, 5)
x2 = np.arange(0, 3, 1)
X1, X2 = np.meshgrid(x1, x2)
Y = inequality_plane(X1, X2)

# 3D grafik oluştur
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Eşitsizlik bölgesini görselleştir
ax.plot_surface(X1, X2, Y, alpha=0.5)

# Eksen etiketleri
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

plt.show()
