import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as optimize
from scipy.interpolate import interp1d
from scipy import optimize, interpolate, integrate
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from collections import namedtuple

def Nbspld2(t, x, k=3):
    """Same as :func:`Nbspl`, but returns first and second derivatives too."""
    kmax = k
    if kmax > len(t)-2:
        raise Exception("Input error in Nbspl: require that k < len(t)-2")
    t = np.array(t)
    x = np.array(x)[:, np.newaxis]
    N = 1.0*((x > t[:-1]) & (x <= t[1:]))
    dN = np.zeros_like(N)
    d2N = np.zeros_like(N)
    for k in range(1, kmax+1):
        dt = t[k:] - t[:-k]
        _dt = dt.copy()
        _dt[dt != 0] = 1./dt[dt != 0]
        d2N = d2N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - d2N[:,1:]*(x-t[k+1:])*_dt[1:] \
            + 2*dN[:,:-1]*_dt[:-1] - 2*dN[:,1:]*_dt[1:]
        dN = dN[:,:-1]*(x-t[:-k-1])*_dt[:-1] - dN[:,1:]*(x-t[k+1:])*_dt[1:] \
            + N[:,:-1]*_dt[:-1] - N[:,1:]*_dt[1:]
        N = N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - N[:,1:]*(x-t[k+1:])*_dt[1:]
    return N, dN, d2N
    

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
#
## Example usage:
#y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#dx = 1.0
#dy = deriv14_const_dx(y, dx)
##print("Derivative:", dy)
#


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


## Example usage:
#pts = np.array([[10, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
##pts = np.array([[200, 0, 0], [200, 0, 0]])
#
#derivatives = _pathDeriv(pts)
##print("Derivatives:\n", derivatives)
#
#cumtrapsarg = [np.sqrt(a @ b) for a, b in zip(derivatives, derivatives)]
#
##print('cumtraps', cumtrapsarg)



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
#print("Optimal solution:", result)

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
#print("actual cumtraps", cumtrapz(cumtrapsarg))
#print("scipy cumtraps", integrate.cumtrapz(cumtrapsarg))

# Example Usage:
pts = np.array([1, 2, 3, 4, 5])  # Example data points
x = np.array([0, 1, 2, 3, 4])  # Example x values
dx = 1.0  # Example step size
initial = 0.0  # Example initial value
result = cumtrapz(pts, x, dx, initial)
#print("Result:", result)



class SplinePath:
    def __init__(self, pts, V, dV=None, V_spline_samples=100, extend_to_minima=False, reeval_distances=True):
        assert len(pts) > 1
        # Find derivatives
        dpts = _pathDeriv(pts)
#        print(dpts)
        # Extend the path if necessary
        if extend_to_minima:
            def V_lin(x, p0, dp0, V): return V(p0 + x * dp0)
            xmin = optimize.fmin(V_lin, 0.0, args=(pts[0], dpts[0], V), xtol=1e-6, disp=0)[0]
            if xmin > 0.0: xmin = 0.0
            nx = int(np.ceil(abs(xmin) - 0.5)) + 1
            x = np.linspace(xmin, 0, nx)[:, np.newaxis]
            pt_ext = pts[0] + x * dpts[0]
            pts = np.append(pt_ext, pts[1:], axis=0)
            xmin = optimize.fmin(V_lin, 0.0, args=(pts[-1], dpts[-1], V), xtol=1e-6, disp=0)[0]
            if xmin < 0.0: xmin = 0.0
            nx = int(np.ceil(abs(xmin) - 0.5)) + 1
            x = np.linspace(xmin, 0, nx)[::-1, np.newaxis]
            pt_ext = pts[-1] + x * dpts[-1]
            pts = np.append(pts[:-1], pt_ext, axis=0)
             # extend at the end of the path
            xmin = optimize.fmin(V_lin, 0.0, args=(pts[-1], dpts[-1], V),
                                 xtol=1e-6, disp=0)[0]
            if xmin < 0.0: xmin = 0.0
            nx = int(np.ceil(abs(xmin)-.5)) + 1
            x = np.linspace(xmin, 0, nx)[::-1, np.newaxis]
            pt_ext = pts[-1] + x*dpts[-1]
            pts = np.append(pts[:-1], pt_ext, axis=0)
            # Recalculate the derivative
            dpts = _pathDeriv(pts)
#            print(dpts)
            # 3. Find knot positions and fit the spline.
        pdist = integrate.cumtrapz(np.sqrt(np.sum(dpts*dpts, axis=1)),
                                   initial=0.0)
#        print(pdist)
        self.L = pdist[-1]
#        print('L crude', self.L)
        k = min(len(pts)-1, 3)  # degree of the spline
        self._path_tck = interpolate.splprep(pts.T, u=pdist, s=0, k=k)[0]
        # Assuming self._path_tck and pdist are already defined
        # Generate a range of parameter values (u) to evaluate the spline
        u_fine = np.linspace(pdist[0], pdist[-1], 300)  # More points for a smoother curve

        # Evaluate the spline at these parameter values
        spline_points = splev(u_fine, self._path_tck)
        plt.figure(figsize=(8, 6))
        plt.plot(spline_points[0], spline_points[1], 'b', label='Fitted Spline')
        plt.scatter(pts[:, 0], pts[:, 1], color='red', zorder=5, label='Original Points')  # Assuming pts is available
        # 4. Re-evaluate the distance to each point.
        if reeval_distances:
            def dpdx(_, x):
                dp = np.array(interpolate.splev(x, self._path_tck, der=1))
                return np.sqrt(np.sum(dp*dp))
            pdist = integrate.odeint(dpdx, 0., pdist,
                                     rtol=0, atol=pdist[-1]*1e-8)[:,0]
#            print(pdist)
            self.L = pdist[-1]
#            print('L refined', self.L)
            self._path_tck = interpolate.splprep(pts.T, u=pdist, s=0, k=k)[0]
            spline_points2 = splev(u_fine, self._path_tck)
            plt.plot(spline_points2[0], spline_points2[1], 'r', label='Fitted Spline Improved')
            # Finalizing the plot
            plt.title('Comparison of Initial and Refined Spline Fits')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
#            plt.show()
         # Now make the potential spline.
        self._V = V
        self._dV = dV
        self._V_tck = None
        
#        plt.figure(figsize=(8, 6))
#        if V_spline_samples is not None:
#            x = np.linspace(0, self.L, V_spline_samples)
#            y = self._V(x)
#            plt.plot(x, y, label='Initial V(x)', marker='o', linestyle='-', color='blue')
#            print("x array", x )
#            x_ext = np.arange(x[1], self.L * 0.2, x[1])
#            x = np.append(-x_ext[::-1], x)
#            x = np.append(x, self.L + x_ext)
#            y = self._V(x)  # Correctly accessing the potential function
#            plt.plot(x, y, label='Extended V(x)', marker='x', linestyle='--', color='red')
#            self._V_tck = interpolate.splrep(x, y, s=0)
#            plt.xlabel('X')
#            plt.ylabel('V(X)')
#            plt.title('Plot of V(X) Against X: Initial vs Extended')
#            plt.show()
#            self._V_tck = splrep(x, y, s=0)
    def V(self, x):
        """The potential as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=4)
        else:
            pts = interpolate.splev(x, self._path_tck)
            return self._V(np.array(pts).T)

    def dV(self, x):
        """`dV/dx` as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=1)
        else:
            pts = interpolate.splev(x, self._path_tck)
            dpdx = interpolate.splev(x, self._path_tck, der=1)
            dV = self._dV(np.array(pts).T)
            return np.sum(dV.T*dpdx, axis=0)

    def d2V(self, x):
        """`d^2V/dx^2` as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=2)
        else:
            raise RuntimeError("No spline specified. Cannot calculate d2V.")

    def pts(self, x):
        """
        Returns the path points as a function of the distance `x` along the
        path. Return value is an array with shape ``(len(x), N_dim)``.
        """
        pts = interpolate.splev(x, self._path_tck)
        return np.array(pts).T


# This function is naturally vectorized because ** and - are ufuncs and follow broadcasting.
# Path points, define a simple linear path for simplicity
#pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 9]])  # Example points
#pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 9]])  # Example points
#
#
#
## Create instance
#path = SplinePath(pts, V)
#parameter_values = np.linspace(0, 11, 20)  # Assuming the path parameters are normalized from 0 to 1
#points_interp = path.pts(parameter_values)
#
#
#plt.figure(figsize=(8, 6))
#x = pts[:, 0]  # X coordinates
#y = pts[:, 1]  # Y coordinates
#
#x2 = points_interp[:, 0]  # X coordinates of interpolated points
#y2 = points_interp[:, 1]  # Y coordinates of interpolated points
#
#
#
#plt.figure(figsize=(8, 6))  # Create a figure with a specific size
#plt.scatter(x, y, c='blue', marker='o', label='Points')  # Scatter plot of the points
#plt.scatter(x2, y2, c='red', marker='o', label='Spleen Points')  # Scatter plot of the points
#
#plt.title('Scatter Plot of Points')  # Title of the plot
#plt.xlabel('X Coordinate')  # Label for the x-axis
#plt.ylabel('Y Coordinate')  # Label for the y-axis
#plt.grid(True)  # Enable grid for better readability
#plt.legend()  # Show legend
##plt.show()  # Display the plot

from collections import namedtuple
import numpy as np

_step_rval = namedtuple("step_rval", "stepsize fRatio")
_forces_rval = namedtuple("forces_rval", "F_norm dV")

class Deformation_Spline:
    def __init__(self, phi, dphidr, dV, nb=10, kb=3, v2min=0.0,
                 fix_start=False, fix_end=False, save_all_steps=False):
        # First step: convert phi to a set of path lengths.
        phi = np.asanyarray(phi)
        dphi = phi[1:] - phi[:-1]
        dL = np.sqrt(np.sum(dphi * dphi, axis=-1))
        y = np.cumsum(dL)
        self._L = y[-1]
        self._t = np.append(0, y) / self._L
        self._t[0] = 1e-100  # Ensures first data point is included in any bin

        # Create the starting spline
        t0 = np.append(np.append([0.] * (kb - 1), np.linspace(0, 1, nb + 3 - kb)), [1.] * (kb - 1))
        self._X, self._dX, self._d2X = Nbspld2(t0, self._t, kb)
        self._t = self._t[:, np.newaxis]  # Reshape for use in calculations

        # Subtract off the linear component from phi.
        phi0, phi1 = phi[:1], phi[-1:]  # Start and end points
        phi_lin = phi0 + (phi1 - phi0) * self._t
        self._beta, residues, rank, s = np.linalg.lstsq(self._X, phi - phi_lin, rcond=-1)
        
        # Save the points for future use
        self.phi = phi  # Save original phi
        self.v2 = np.asanyarray(dphidr)[:, np.newaxis] ** 2  # Save squared derivatives
        self.dV = dV
        self.F_list = []
        self.phi_list = []
        self._phi_prev = self._F_prev = None
        self.save_all_steps = save_all_steps
        self.fix_start = fix_start
        self.fix_end = fix_end
        self.num_steps = 0

        # Ensure v2 isn't too small
        v2 = dphidr ** 2
        v2min *= np.max(np.sum(dV(self.phi) ** 2, -1) ** .5 * self._L / nb)
        v2[v2 < v2min] = v2min
        self.v2 = v2[:, np.newaxis]

    def forces(self):
        """ Calculate the normal force and potential gradient on the path. """
        X, dX, d2X = self._X, self._dX, self._d2X
        beta = self._beta
        phi = self.phi
        dphi = np.sum(beta[np.newaxis, :, :] * dX[:, :, np.newaxis], axis=1) + (phi[-1] - phi[1])[np.newaxis, :]
        d2phi = np.sum(beta[np.newaxis, :, :] * d2X[:, :, np.newaxis], axis=1)
        dphi_sq = np.sum(dphi * dphi, axis=-1)[:, np.newaxis]
        dphids = dphi / np.sqrt(dphi_sq)
        d2phids2 = (d2phi - dphi * np.sum(dphi * d2phi, axis=-1)[:, np.newaxis] / dphi_sq) / dphi_sq
        dV = self.dV(phi)
        dV_perp = dV - np.sum(dV * dphids, axis=-1)[:, np.newaxis] * dphids
        F_norm = d2phids2 * self.v2 - dV_perp
        if self.fix_start:
            F_norm[0] = 0.0
        if self.fix_end:
            F_norm[-1] = 0.0
        return _forces_rval(F_norm, dV)

    def step(self, stepsize, minstep, diff_check=0.1, step_decrease=2.):
        """
        Take adaptive steps in the direction of the normal force to optimize the path.
        """
        F1, dV = self.forces()
        F_max = np.max(np.sqrt(np.sum(F1 * F1, -1)))
        dV_max = np.max(np.sqrt(np.sum(dV * dV, -1)))
        fRatio = F_max / dV_max

        if self.save_all_steps:
            self.phi_list.append(self.phi)
            self.F_list.append(F1)

        while True:
            phi2 = self.phi + F1 * (stepsize * 0.5)
            F2, _ = self.forces()  # Update to only retrieve forces as needed
            if stepsize <= minstep:
                stepsize = minstep
                break
            DF_max = np.max(np.abs(F2 - F1), axis=0)
            F_max = np.max(np.abs(F1), axis=0)
            if (DF_max < diff_check * F_max).all():
                break
            stepsize /= step_decrease
        self.phi = phi2 + F2 * (stepsize * 0.5)
        return _step_rval(stepsize, fRatio)

    
def V(x):
    return x**3
    
def dV(x):
    return 3*x**2

# Create instance with a sample path
phi = np.array([[0,1], [1,2], [2,3], [3, 6]])
def dphidr(x):
    # Example derivative function; adjust based on your data's nature
    return np.gradient(x, axis=0)

# Instantiate and call forces
path_instance = Deformation_Spline(phi, dphidr=dphidr(phi), dV=dV)
forces_output = path_instance.forces()



#print("Shape of phi:", phi.shape)
#print("Shape of F_norm:", forces_output.F_norm.shape)
print("phi:", phi)
print("F_norm:", forces_output.F_norm)
#F_norm_corrected = np.sum(forces_output.F_norm, axis=1)
#print("Corrected shape of F_norm:", F_norm_corrected.shape)

# Define the step parameters
stepsize = 0.1
minstep = 0.01

path_instance = Deformation_Spline(phi, dphidr=dphidr(phi), dV=dV)

# Perform the path optimization step
step_output = path_instance.step(stepsize, minstep)

# Access and print the updated path
phi_updated = path_instance.phi

print("x axis phi", phi[:, 0])
print("x axis updated phi", phi_updated[:, 0])
print("y axis phi", phi[:, 1])
print("y axis updated phi", phi_updated[:, 1])

exit()








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
