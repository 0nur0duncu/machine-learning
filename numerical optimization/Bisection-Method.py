import math
from Function_1 import FT

hata = 1e-6

a = -5
b = 5

if FT(a) * FT(b) >= 0:
    print("No root exists between {:.4f} and {:.4f}.".format(a, b))
else:
    i = 0
    x0 = (a + b) / 2
    print(f"{i:d}. step, interval= [{a:.4f}, {b:.4f}]")
    
    while True:
        i += 1
        print("{:d}. step, approximate root= {:.4f}\t ".format(i, x0), end='')

        x1 = x0
        if FT(x0) * FT(a) < 0:
            b = x0
        else:
            a = x0

        print("new interval= [{:.4f}, {:.4f}]".format(a, b))

        x0 = (a + b) / 2

        if abs(x0 - x1) <= hata:
            break

    print("{:d}. step, exited from the loop. Approximate root in step {:d}= {:.4f}".format(i + 1, i + 1, x0))
    print("{:.3f} error, approximate root = {:.4f}".format(hata, x0))
    print("f({:.4f})= {:.3f}".format(x0, FT(x0)))
