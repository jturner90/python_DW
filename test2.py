import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

from math import ceil

class SplinePath:
    def __init__(self, pts, V_, V_spline_samples, extend_to_minima, re_eval_distances):
        self.pts = np.array(pts)
        self.V_ = V_
        
        # Calculate derivatives along the path
        self.dpts = self.path_deriv(self.pts)
#        print(self.dpts)
        # Extend the path to minima if requested
        if extend_to_minima:
            self.extend_to_minima()

    def path_deriv(self, pts):
        # Placeholder: Compute derivatives of the path
        # This should return an array where each row is the derivative at the corresponding point
#        print(np.gradient(pts, axis=0))
        return np.gradient(pts, axis=0)

    def extend_to_minima(self):
        # Extend the front of the path
        front = {'point': self.pts[0], 'deriv': self.dpts[0], 'V': self.V_}
        xmin, minima = self.find_minima(front, extend_direction='front')
        self.extend_path(self.pts[0], self.dpts[0], xmin, minima, prepend=True)

        # Extend the end of the path
        end = {'point': self.pts[-1], 'deriv': self.dpts[-1], 'V': self.V_}
        xmin, minima = self.find_minima(end, extend_direction='end')
        self.extend_path(self.pts[-1], self.dpts[-1], xmin, minima, prepend=False)

    def find_minima(self, param, extend_direction):
        # Using scipy's minimize to find the local minima
        res = minimize(lambda x: self.V_(param['point'] + x * param['deriv']), [0.0], tol=1e-6)
        xmin = res.x[0]
        xmin = max(xmin, 0.0) if extend_direction == 'end' else min(xmin, 0.0)
        minima = param['point'] + xmin * param['deriv']
        print(xmin, minima)
        return xmin, minima

    def extend_path(self, point, deriv, xmin, minima, prepend):
        # Calculate the number of points to extend and the distance between them
        nx = ceil(abs(xmin) - 0.5)
        if nx > 0:
            dx = xmin / nx
            new_points = [point + i * dx * deriv for i in range(nx)]
            if prepend:
                self.pts = np.vstack([minima] + new_points + [self.pts])
            else:
                self.pts = np.vstack([self.pts] + new_points + [minima])
        self.dpts = self.path_deriv(self.pts)  # Re-evaluate derivatives if path is extended
        
    def plot_path(self):
        # Plot the original points
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Path')
        original_pts = np.array(self.pts)  # assuming self.pts contains all path points
        plt.scatter(original_pts[:, 0], original_pts[:, 1], c='red', label='Original Points')
        plt.plot(original_pts[:, 0], original_pts[:, 1], 'r--', label='Path')

        # Plot the path with extensions
        plt.subplot(1, 2, 2)
        plt.title('Extended Path')
        extended_pts = np.array(self.pts)  # assuming self.pts now also contains the extended points
        plt.scatter(extended_pts[:, 0], extended_pts[:, 1], c='blue', label='Extended Points')
        plt.plot(extended_pts[:, 0], extended_pts[:, 1], 'b-', label='Path')

        # Annotate extended points with their coordinates
        for x, y in extended_pts:
            plt.text(x, y, f'({x:.2f}, {y:.2f})', color='blue', fontsize=8)

        # Optionally, show derivatives as arrows
        for point, deriv in zip(extended_pts, self.dpts):
            plt.gca().add_patch(Arrow(point[0], point[1], deriv[0]*0.1, deriv[1]*0.1, width=0.05, color='green'))

        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.show()

# Example usage of the SplinePath with a potential function
pts = np.array([[0, 0], [1, 1], [2, 2]])
V_ = lambda x: np.sum(x**2)  # Simple quadratic potential
spline = SplinePath(pts, V_, 10, True, False)
spline.plot_path()
