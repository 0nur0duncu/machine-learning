import numpy as np
import math
from ornekFonksiyon2 import f,jacobian,error
#---------------------------------------------------------------
MaxIter = 10000
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e+99
muscal =10
mu = 1
#-----------------------------------------------------------------
x1 = [0.0]
x2 = [0.0]
xk = np.array([x1[0],x2[0]])
k = 0; C1 = True; C2 = True; C3 = True; C4 = True; C5 = True;

I = np.identity(2)
sk = 1.0
print('k',k,' x1:',format(xk[0],'f'),' x2:',format(xk[1],'f'),' f',format(f(xk)))
while C1 & C2 & C3 & C4 & C5:
    loop =True
    while loop:
        g= 2*jacobian(xk).transpose().dot(error(xk))
        g= np.array(g.tolist()[0]) #boyut uyumu için
        H = 2*jacobian(xk).transpose().dot(jacobian(xk))
        zk = -np.linalg.inv(H+mu+I).dot(g)
        zk = np.array(zk.tolist()[0]) #boyut uyumu için
        if f(xk+zk)<f(xk):
            pk = zk
            xk =xk + sk*pk #güncelle 
            mu = mu/muscal; #hızlandır
            loop =False; # iç döngüyü bitir
        else:
            mu = mu*muscal #yavaşlat
            C5 = mu < mumax; #yerel minimum testi
            if not C5:
                loop = False
            
    k += 1
    x1.append(xk[0])
    x2.append(xk[1])
    print('k: ',k,' x1:',format(xk[0],'f'),' x2:',format(xk[1],'f'),' sk:',format(sk))
    #-----------------------------------------------------------
    C1=k<MaxIter
    C2=epsilon1<abs(f(xk)-f(xk+sk*pk))
    C3=epsilon2<np.linalg.norm(sk*pk)
    C4=epsilon3<np.linalg.norm(g)
    C5=mu<mumax
    #---------------------------------------  
#-------------------------------------------------------
if not C1:
    print('max.iterasyon aşıldı')
if not C2:
    print('fonksiyon değeri değişmiyor')
if not C3:
    print('ilerleme yönü bulunmuyor')
if not C4:
    print('gradyan sıfıra çok yakın')
#print('xkbest: ', xkbest)
    
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt 
#plot the points
plt.plot(x1,x2,color='darkred',linestyle='solid',linewidth=1, marker ='o',markerfacecolor='black',markersize=2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(' Levenberg-Marquardt (LM)  Algoritması', fontstyle='italic')
plt.grid(color = 'green', linestyle = '--', linewidth=0.1)
plt.show()



                 

            