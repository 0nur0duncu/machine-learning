from dataset1 import ti,yi
import numpy as np 

def polinom(x,T):
	yhat=[]
	for ti in T:
		yhat.append(x[0] + x[1]*ti + x[2]*ti**2)
	return yhat


N = len(ti)
# hstack matris formatında yazmamızı sağlar
J = -np.hstack((np.ones((N,1)),np.array(ti).reshape(N,1),np.array(ti).reshape(N,1)**2))
A = np.linalg.inv(J.transpose().dot(J))
B = J.transpose().dot(yi)
x = -A.dot(B)

T = np.arange(-3,3,0.1)
yhat = polinom(x,T)
#----------------------------
import matplotlib.pyplot as plt 
plt.plot(T, yhat, color='green', linestyle='solid', linewidth=1)
plt.scatter(ti,yi, color='darkred', marker='x')
plt.xlabel('ti')
plt.ylabel('yi')
plt.title('Polinom Modeli', fontstyle='italic')
plt.grid(color = 'green', linestyle = '--', linewidth=0.1)
plt.legend(['polinom modeli', 'gercek veri'])
plt.show()