import numpy as np
from scipy.optimize import minimize

def minimize_function(func, initial_guess, tol=1e-6):
    """
    Minimizes a multidimensional function using the Nelder-Mead algorithm.

    Parameters:
    - func: The function to minimize. Must take a single argument, a numpy array of parameters.
    - initial_guess: Initial guess for the parameters as a numpy array.
    - tol: Tolerance for termination.

    Returns:
    - A dictionary containing the result of the minimization process.
    """

    # Define the options for the minimization process
    options = {'xatol': tol, 'fatol': tol, 'disp': True}

    # Perform the minimization
    result = minimize(func, initial_guess, method='Nelder-Mead', options=options)

    # Check if the optimization was successful and print results
    if result.success:
        print("Optimization successful.")
        print("Minimum found at:", result.x)
        print("Function value at minimum:", result.fun)
    else:
        print("Optimization failed.")
        print("Reason:", result.message)

    return result

# Example usage:
# Define a function to minimize, e.g., a simple quadratic function
def example_function(x):
    return (x[0] - 10)**2 + (x[1] - 2)**2 + (x[2] - 3)**2

# Initial guess for the parameters
initial_guess = np.array([-10, 0, 0])

# Call the minimize function
result = minimize_function(example_function, initial_guess)

