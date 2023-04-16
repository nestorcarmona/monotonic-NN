import numpy as np

# defining the functions

# D_{f_i} ->  R with D_{f_i} = [0, 1]x[0, 1]x ... x[0, 1] (times the dimensions)

# monotonic increasing in x
def f1(x):
    return (x+1)**2

# monotonic decreasing in x
def f2(x):
    return -(x+1)**2

# monotonic increasing in x and y
def f3(x, y):
    return x+y

# monotonic increasing in x and decreasing in y
def f4(x, y):
    return x-y

# non-monotonic in x
def f5(x):
    return np.cos(x)

# monotonic increasing in x and non-monotonic in y
def f6(x, y):
    return x**3 + 0.5*np.sin(4*np.pi*y)

# non-monotonic in x and y
def f7(x, y):
    return np.sin(4*np.pi*x) + np.cos(4*np.pi*y)

# monotonic increasing in x, y and z
def f8(x, y, z):
    return x+y+z
