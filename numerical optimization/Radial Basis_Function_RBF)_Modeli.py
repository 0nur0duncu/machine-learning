import numpy as np
import math
from dataset4 import ti,yi

#-----------------------------
def rbfIO(t,x,c,s):
	yhat = []
	for i in range(0,len(t)):
		ti = t[i]
		tmp = 0
		for j in range(0,len(x)):
			tmp += x[j]*rbf(t[i],c[j],s)
		yhat.append(tmp)
	return yhat

#-----------------------------
def rbf(t,c,s):
	h = math.exp(-(t-c)**2/s**2)
	return h
#-----------------------------

def obtainX(t,y,n):  #obtainX optimum x 
        N=len(y)
        L = max(t)-min(t)
        s = L/n
        c = []
        
        for i in range(1,n+1):
            c.append(min(t) + i*L/(n+1)) 
        J = np.zeros((N,n))
        for i in range(0,N):
            for j in range(0,n):
                J[i,j] = -rbf(t[i],c[j],s)
        A = np.linalg.inv(J.transpose().dot(J))
        B = J.transpose().dot(y)
        x = -A.dot(B)
        return x,c,s  

#----------------------------------------------------

def plotresult(dugumsayisi,x,c,s,validationPerf):
        import matplotlib.pyplot as plt 
        T = np.arange(-15,15,0.1)
        yhat=rbfIO(T,x,c,s)
        #plot the points
        plt.scatter(ti,yi,color='darkred',marker="x")
        plt.plot(T, yhat,color='green',linestyle='solid',linewidth=1)
        plt.xlabel('ti')
        plt.ylabel('yi')
        plt.title(str(dugumsayisi) + " dugumlu RBF modeli" + " FV: " + str(validationPerf), fontstyle='italic')
        plt.grid(color = 'green', linestyle = '--', linewidth=0.1)
        plt.legend(["gercek veri","polinom modeli"])
        plt.show()
		
#-----------------------------------------------------------------------------------
    
N = len(yi) # veri sayımız
itra = np.arange(0,N,2) #train tek sayıları tranin için ayırırken homojen ayırmalıyız
ival = np.arange(1,N,2) #validation  çift sayıları valaidation için ayırdık

trainingInput = np.array(ti)[itra]
trainingOutput = np.array(yi)[itra]

validationInput = np.array(ti)[ival]
validationOutput = np.array(yi)[ival]

DER = []
VAL = []

for dugumsayisi in range(1,10):
	x,c,s = obtainX(trainingInput, trainingOutput,dugumsayisi)
	yhat = rbfIO(validationInput, x,c,s)
	validationError = validationOutput-yhat
	validationPerf = sum(validationError**2)
	DER.append(dugumsayisi)
	VAL.append(validationPerf)
	plotresult(dugumsayisi,x,c,s,validationPerf)
       
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt 
plt.bar(DER,VAL,color='darkred')
plt.scatter(DER,VAL, color='darkred')
plt.xlabel('düğüm sayısı')
plt.ylabel('validation performansi')
plt.title('RBF Modeli', fontstyle='italic')
plt.grid(color = 'green', linestyle = '--', linewidth=0.1)
plt.show()
