def F(x):
  return (x-1)**2*(x-2)*(x-3)

def FT(x):
  return 2*(x-1)*(x-2)*(x-3) + (x-1)**2*(2*x - 5)

def FTT(x):
  return 2*(x-2)*(x-3) + 2*(x-1)*(2*x - 5) + 2*(x-1)*(2*x - 5) + 2*(x-1)**2