import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as optimize
from scipy.interpolate import interp1d


def deriv14_const_dx(y, dx=1.0):
    y = np.array(y)  # Ensure y is a NumPy array for vectorized operations
    N = len(y)
    dy = np.zeros_like(y, dtype=float)

    if N >= 5:
        # Compute derivatives for the first and last points using one-sided differences
        dy[0] = (-25.0 * y[0] + 48.0 * y[1] - 36.0 * y[2] + 16.0 * y[3] - 3.0 * y[4]) / (12.0 * dx)
        dy[1] = (-3.0 * y[0] - 10.0 * y[1] + 18.0 * y[2] - 6.0 * y[3] + y[4]) / (12.0 * dx)
        
        # Compute derivatives for the central points using central differences
        for i in range(2, N - 2):
            dy[i] = (y[i - 2] - 8.0 * y[i - 1] + 8.0 * y[i + 1] - y[i + 2]) / (12.0 * dx)
        
        # Compute derivatives for the second last and last points using one-sided differences
        dy[-2] = (3.0 * y[-1] + 10.0 * y[-2] - 18.0 * y[-3] + 6.0 * y[-4] - y[-5]) / (12.0 * dx)
        dy[-1] = (25.0 * y[-1] - 48.0 * y[-2] + 36.0 * y[-3] - 16.0 * y[-4] + 3.0 * y[-5]) / (12.0 * dx)
    
    return dy

# Example usage:
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dx = 1.0
dy = deriv14_const_dx(y, dx)
print("Derivative:", dy)



def _pathDeriv(pts):
    pts = np.array(pts)
    N = len(pts)
    if N >= 5:
        # Assuming deriv14_const_dx has been properly defined to handle numpy arrays
        res = deriv14_const_dx(pts)
    elif N > 2:
        res = np.zeros_like(pts)
        # Using finite differences for the first and last point and central differences for the others
        res[0] = -1.5 * pts[0] + 2.0 * pts[1] - 0.5 * pts[2]
        for i in range(1, N - 1):
            res[i] = (pts[i + 1] - pts[i - 1]) / 2.0
        res[-1] = 1.5 * pts[-1] - 2.0 * pts[-2] + 0.5 * pts[-3]
    else:
        # If there are fewer than three points, calculate a constant derivative
        res = np.full_like(pts, pts[-1] - pts[0])
    
    return res


# Example usage:
pts = np.array([[10, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
#pts = np.array([[200, 0, 0], [200, 0, 0]])

derivatives = _pathDeriv(pts)
print("Derivatives:\n", derivatives)

cumtrapsarg = [np.sqrt(a @ b) for a, b in zip(derivatives, derivatives)]

print('cumtraps', cumtrapsarg)



def find_multimin_arg_gsl_wraper(func, params, x_start, epsabs):
    NDim = len(x_start)
    
    # Define the objective function
    def objective(x):
        return func(x, params)
    
    # Initialize the minimizer
    result = optimize.minimize(objective, x_start, method='Nelder-Mead', tol=epsabs)
    
    if result.success or result.status == 3:  # '3' indicates that the algorithm converged
        return result.x
    else:
        # Handle cases where optimization fails
        raise RuntimeError("Optimization failed: " + result.message)

# Example Usage:
# Define your multimin_func and params appropriately
def multimin_func(x, params):
    # Example multimin_func implementation
    return np.sum((x - params)**2)

params = np.array([1.0, 2.0, 3.0])  # Example parameters
x_start = np.array([0.0, 0.0, 0.0])  # Initial guess
epsabs = 1e-6  # Tolerance for convergence

# Call the wrapper function
result = find_multimin_arg_gsl_wraper(multimin_func, params, x_start, epsabs)
print("Optimal solution:", result)

def cumtrapz(pts, x = [], dx = 1, initial = 0):
    Nsize = len(pts)
    default_dx = (len(x) != Nsize)
    res = np.zeros(Nsize)
    res[0] = initial
    for i in range(1, Nsize):
        dx_used = dx if default_dx else x[i] - x[i-1]
        res[i] = res[i-1] + (pts[i] + pts[i-1]) * dx_used / 2.0
    return res

#cumtrapsarg = [np.sqrt(a @ b) for a, b in zip(derivatives, derivatives)]
print("actual cumtraps", cumtrapz(cumtrapsarg))

# Example Usage:
pts = np.array([1, 2, 3, 4, 5])  # Example data points
x = np.array([0, 1, 2, 3, 4])  # Example x values
dx = 1.0  # Example step size
initial = 0.0  # Example initial value
result = cumtrapz(pts, x, dx, initial)
print("Result:", result)


from scipy.interpolate import CubicSpline

class PathInterpolater:
    def __init__(self, x, y):
        self.spline = CubicSpline(x, y)
    
    def valAt(self, x_val):
        return self.spline(x_val)



#path_interpolator = interp1d(p_dist, pts)

from scipy.integrate import solve_ivp
import numpy as np

# Define the ODE function
def _func_refine_dist(t, y, spline_path):
    dpdx = spline_path.pts_at_dist(t, 1)
    d_dist = np.sqrt(np.dot(dpdx, dpdx))
    return d_dist

# Define the ODE solver function
def rk_solver(func, t_span, y0, spline_path, atol=1e-8, rtol=1e-6):
    sol = solve_ivp(func, t_span, y0, args=(spline_path,), method='RK45', atol=atol, rtol=rtol)
    return sol.y.T

# Assuming p_dist, pts, and _L are defined elsewhere
# Define the time span
t_span = [0.0, 1.0]  # Adjust the time span as needed

# Initial condition
y0 = [0.0]  # Adjust the initial condition as needed

# Call the Runge-Kutta solver
#dist_tmp = rk_solver(_func_refine_dist, t_span, y0, spline_path)

# Update p_dist
#p_dist = dist_tmp[:, 0]
#
## Update _L
#L = p_dist[-1]
#
## Set data for _path_Inter
#path_Inter.SetData(pts, p_dist)


# argument of  VD p_dist = cumtrapz(pow(dpts*dpts,0.5));



#
#
#class SplinePath:
#    def __init__(self, pts, V_, V_spline_samples, extend_to_minima, re_eval_distances):
#        self.pts = np.array(pts)
#        self.V_ = V_
#        self.V_spline_samples = V_spline_samples
#
#        # Calculate derivatives along the path
#        self.dpts = self._pathDeriv(self.pts)
#
#        # Extend the path to minima if necessary
#        if extend_to_minima:
#            self.extend_path_to_minima()
#
#        # Setup the spline interpolation based on the path
#        self.setup_spline_interpolation()
#
#        # Optionally re-evaluate distances for accuracy
#        if re_eval_distances:
#            self.re_evaluate_distances()
#
#    def _pathDeriv(self, pts):
#        N = len(pts)
#        dy = np.zeros_like(pts)
#        if N >= 5:
#            dy[0] = (-25*pts[0] + 48*pts[1] - 36*pts[2] + 16*pts[3] - 3*pts[4]) / 12.0
#            dy[1] = (-3*pts[0] - 10*pts[1] + 18*pts[2] - 6*pts[3] + pts[4]) / 12.0
#            for i in range(2, N-2):
#                dy[i] = (pts[i-2] - 8*pts[i-1] + 8*pts[i+1] - pts[i+2]) / 12.0
#            dy[-2] = (3*pts[-1] + 10*pts[-2] - 18*pts[-3] + 6*pts[-4] - pts[-5]) / 12.0
#            dy[-1] = (25*pts[-1] - 48*pts[-2] + 36*pts[-3] - 16*pts[-4] + 3*pts[-5]) / 12.0
#        return dy
#
#
#
#    def cumtrapz_python(pts, x=None, dx=1.0, initial=0.0):
#        pts = np.array(pts)
#        Nsize = len(pts)
#        res = np.zeros(Nsize)
#        res[0] = initial
#
#        if x is None:
#            x = np.arange(Nsize) * dx
#            default_dx = True
#        else:
#            x = np.array(x)
#            default_dx = (len(x) != Nsize)
#
#        for i in range(1, Nsize):
#            dx_used = dx if default_dx else x[i] - x[i - 1]
#            res[i] = res[i - 1] + (pts[i] + pts[i - 1]) * dx_used / 2.0
#
#        return res
#
#
#    def extend_path_to_minima(self):
#        # Extend the path at both the front and the back
#        self.extend_at_endpoint(self.pts[0], self.dpts[0], 'front')
#        self.extend_at_endpoint(self.pts[-1], self.dpts[-1], 'end')
#
#    def extend_at_endpoint(self, point, deriv, position):
#        # Placeholder for extending the path based on gradient descent or similar
#        pass
#
#    def setup_spline_interpolation(self):
#        distances = np.sqrt(np.sum(np.diff(self.pts, axis=0)**2, axis=1))
#        cumulative_distances = np.cumsum(distances)
#        cumulative_distances = np.insert(cumulative_distances, 0, 0)  # Ensure starting from zero
#
#        # Check that the lengths match
#        if len(cumulative_distances) != len(self.pts):
#            raise ValueError(f"Length mismatch: {len(cumulative_distances)} distances vs {len(self.pts)} points")
#
#        # Setup the spline with proper boundary conditions if necessary
#        self.spline = InterpolatedUnivariateSpline(cumulative_distances, self.pts, k=3, ext=3)
#
#
#    def re_evaluate_distances(self):
#        # Placeholder for improving the accuracy of the path distance calculation
#        pass
#
#    # Example methods for interacting with the spline
#    def get_position_at(self, distance):
#        return self.spline(distance)
#
#    def get_velocity_at(self, distance):
#        return self.spline.derivative()(distance)
#
## Example usage
#pts = [[0, 0], [1, 1], [2, 4], [3, 9], [4, 16]]
#V_ = lambda x: np.sum(x**2)  # Simple quadratic potential
#spline_path = SplinePath(pts, V_, 10, True, False)
