import numpy as np
from Function_1 import F, FT, FTT

xk = 0.5
error = 1e-8

delta_xk = -FT(xk) / FTT(xk)
step = 0

while FT(xk + delta_xk):
  delta_xk = -FT(xk) / FTT(xk)
  print(f"step={step}",f"xk={xk:.4f}",f"F(x{xk:.4f})={F(xk):.4f}",f"FT(x{xk:.4f})={FT(xk):.4f}",f"FTT({xk:.4f})={FTT(xk):.4f}",f"delta_xk={delta_xk:.4f}")
  x = xk
  xk = xk + delta_xk
  step += 1
  if abs(x - xk) <= error:
    break