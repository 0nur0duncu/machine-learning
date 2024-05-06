import numpy as np
import math
from dataset2 import ti,yi

#-----------------------------

def polinomIO(t,x):
	yhat = []
	for i in range(0,len(t)):
		ti = t[i]
		tmp = 0
		for j in range(0,len(x)):
			tmp += x[j]*ti**j
		yhat.append(tmp)
	return yhat

#-----------------------------


def obtainX(polinomderecesi, ti, yi):  #obtainX optimum x 
	numofdata=len(ti)
	J = -np.ones((numofdata,1)) #bir -1 lerden oluşan bir sütun vektör oluşturur
	for i in range(1,polinomderecesi+1):
		J = np.hstack((J,-np.ones((numofdata,1))*np.array(ti).reshape(numofdata,1)**i)) 
	A = np.linalg.inv(J.transpose().dot(J))
	B = J.transpose().dot(yi)
	x = -A.dot(B)
	return x  #x değişkeni, polinom regresyon modelinin katsayılarını içeren bir sütun vektörüdür. 
  #hstack() işlevi, J matrisine sütun ekler

#----------------------------

N = len(yi) # veri sayımız
itra = np.arange(0,N,2) #train tek sayıları tranin için ayırırken homojen ayırmalıyız
ival = np.arange(1,N,2) #validation  çift sayıları valaidation için ayırdık

trainingInput = np.array(ti)[itra]
trainingOutput = np.array(yi)[itra]

validationInput = np.array(ti)[ival]
validationOutput = np.array(yi)[ival]
del ti,yi


DER = []
VAL = []

for polinomderecesi in range(0,9):
	x = obtainX(polinomderecesi, trainingInput, trainingOutput)
	yhat = polinomIO(validationInput, x)
	validationError = validationOutput-yhat
	validationPerf = sum(validationError**2)
	DER.append(polinomderecesi)
	VAL.append(validationPerf)


#----------------------------


import matplotlib.pyplot as plt 
plt.scatter(DER,VAL, color='darkred')
plt.xlabel('polinom derecesi')
plt.ylabel('validation performansi')
plt.title('Polinom Modeli', fontstyle='italic')
plt.grid(color = 'green', linestyle = '--', linewidth=0.1)
plt.show()