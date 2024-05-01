import numpy as np

def cumtrapz_python(pts, x=None, dx=1.0, initial=0.0):
    pts = np.array(pts)
    Nsize = len(pts)
    res = np.zeros(Nsize)
    res[0] = initial
    
    if x is None:
        x = np.arange(Nsize) * dx
        default_dx = True
    else:
        x = np.array(x)
        default_dx = (len(x) != Nsize)

    for i in range(1, Nsize):
        dx_used = dx if default_dx else x[i] - x[i - 1]
        res[i] = res[i - 1] + (pts[i] + pts[i - 1]) * dx_used / 2.0
    
    return res

pts = [100, 20, 100, 100]
x = [0, 1, 2, 3]
result = cumtrapz_python(pts, x=x, initial=0)
print("Cumulative Trapezoidal Integration:", result)

# Using default dx
result_default_dx = cumtrapz_python(pts, dx=1.0, initial=0)
print("Cumulative Trapezoidal Integration with default dx:", result_default_dx)
