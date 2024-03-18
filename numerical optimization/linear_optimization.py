import numpy as np
"""
A.X = B
"""
A = np.matrix([[1,2,4],[1,3,9],[1,5,25]])

B = np.array([1,6,4]).transpose()

X = np.linalg.inv(A).dot(B)

print(X)