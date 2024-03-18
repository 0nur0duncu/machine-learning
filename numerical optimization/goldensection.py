import math
from Function_1 import F
    
xalt = 0
xust = 4
dx = 0.0001
alpha = (1+math.sqrt(5))/2
tau = 1-1/alpha
epsilon = dx/(xust-xalt)
N = round(-2.078*math.log(epsilon))

k = 0
x1 = xalt + tau*(xust-xalt); f1 = F(x1)
x2 = xust - tau*(xust-xalt); f2 = F(x2)
print(k, x1, x2, f1, f2)

for k in range(0,N):
    if f1>f2:
        xalt = 1*x1; x1 = 1*x2; f1 = 1*f2
        x2 = xust - tau*(xust-xalt); f2 = F(x2)
    else:
        xust = 1*x2; x2 = 1*x1; f2 = 1*f1
        x1 = xalt + tau*(xust-xalt); f1 = F(x1)
    print(k+1, x1, x2, f1, f2)

x = 0.5*(x1+x2)
print(x)
