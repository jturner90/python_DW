import numpy as np

def deriv14_const_dx(y, dx=1.0):
    y = np.array(y)
    N = len(y)
    dy = np.zeros_like(y)
    
    if N < 5:
        raise ValueError("Input vector y must have at least 5 elements.")
    
    # Boundary points using one-sided finite differences
    dy[0] = (-25.0 * y[0] + 48.0 * y[1] - 36.0 * y[2] + 16.0 * y[3] - 3.0 * y[4]) / (12.0 * dx)
    dy[1] = (-3.0 * y[0] - 10.0 * y[1] + 18.0 * y[2] - 6.0 * y[3] + y[4]) / (12.0 * dx)
    
    # Central points using central differences
    for i in range(2, N - 2):
        dy[i] = (y[i - 2] - 8.0 * y[i - 1] + 8.0 * y[i + 1] - y[i + 2]) / (12.0 * dx)
    
    # Boundary points using one-sided finite differences
    dy[-2] = (3.0 * y[-1] + 10.0 * y[-2] - 18.0 * y[-3] + 6.0 * y[-4] - y[-5]) / (12.0 * dx)
    dy[-1] = (25.0 * y[-1] - 48.0 * y[-2] + 36.0 * y[-3] - 16.0 * y[-4] + 3.0 * y[-5]) / (12.0 * dx)
    
    return dy

# Example usage
y = [0,2,3,4,5,-60]
derivatives = deriv14_const_dx(y)
print("Derivatives:", derivatives)
