import numpy as np

# defining the functions

# D_{f_i} ->  R with D_{f_i} = [0, 1]x[0, 1]x ... x[0, 1] (times the dimensions)
mean = 0
std = 0.01

# monotonic increasing in x
def f1(x):
    noise = np.random.normal(mean, std, x.shape)
    return (x+1)**2 + noise

# monotonic decreasing in x
def f2(x):
    noise = np.random.normal(mean, std, x.shape)
    return -(x+1)**2  + noise

# monotonic increasing in x and y
def f3(x, y):
    noise = np.random.normal(mean, std, x.shape)
    return x+y  + noise

# monotonic increasing in x and decreasing in y
def f4(x, y):
    noise = np.random.normal(mean, std, x.shape)
    return x-y  + noise

# non-monotonic in x
def f5(x):
    noise = np.random.normal(mean, std, x.shape)
    return np.cos(x)  + noise

# monotonic increasing in x and non-monotonic in y
def f6(x, y):
    noise = np.random.normal(mean, std, x.shape)
    # 
    return x**2 + y**2

# monotonic increasing in x, y and z
def f7(x, y):
    noise = np.random.normal(mean, std, x.shape)
    return x**3 + 0.5*np.sin(4*np.pi*y)  + noise

# non-monotonic in x and y
def f8(x):
    noise = np.random.normal(mean, std, x.shape)
    return x**3 + x**2 + x + 1  # + noise


