import math
import numpy as np

class A4DW:
    def __init__(self, mu2, g1, g2):
        #    define input parameters for potential, including dimension of scalar field (3 in this case)
        self._N_VEVs = 3
        
        self._mu2 = mu2
        self._g1 = g1
        self._g2 = g2
        

        # Prepare the list of potential local minima candidates, these are three vectors in (phi1, phi2, phi3)
        self._candidates = self.prepare_local_minima_candidates()

        # For input parameters musq, g1, g2 this return the vacua
        self.candidate_potentials = [self.Vtotal(candidate) for candidate in self._candidates]

        # Check the Hessian matrix for each candidate to see if it is a local minimum, if it is return True else False
        self.candidate_stability = [self.check_hessian(candidate) for candidate in self._candidates]


        print(self.candidate_stability)

        self._v1 = 0
        self._v2 = 0
        self._v3 = 0
        
        #  this loops over the candidate vacua and find the single stable vacua
        for cand, is_stable in zip(self._candidates, self.candidate_stability):
            print(is_stable, cand)
            if is_stable and ((abs(cand[0]) > self._v1) and (abs(cand[1]) > self._v2) and (abs(cand[2]) > self._v3) ):
                self._v1 = cand[0]
                self._v2 = cand[1]
                self._v3 = cand[2]
           
        print(self._v1,self._v2,self._v3)
       
        # initialisation is complete, after this next action full kink function will be called
        vt1   = np.sqrt(self.mu2 / (3.0 * self.g1 + 2.0 * self.g2))
        left  = [-vt1, vt1, vt1]
        right = [vt1, vt1, vt1]
        pts_init = [left, right]

        # Define the function that takes field and scale and applies the Vtotal method
        vtol = lambda field, scale: self.Vtotal(field, scale)

        # Define the function that takes field and scale and applies the dVtotal method
        dvtol = lambda field, scale: self.dVtotal(field, scale)
        
        
        
        
    #       self.set_potential_parameters()
       
    #    construct scalar potential

    def Vtotal(self, field_values, scale=1.0):
        phi1sq = field_values[0]**2
        phi2sq = field_values[1]**2
        phi3sq = field_values[2]**2
       
        tmp = 0
        tmp += -self._mu2 *(phi1sq + phi2sq + phi3sq) / 2.0
        tmp +=  self._g1 * (phi1sq + phi2sq + phi3sq) ** 2 / 4.0
        tmp +=  self._g2 * (phi1sq * phi2sq + phi2sq * phi3sq + phi3sq * phi1sq) / 2.0

        return tmp * scale

    #    construct first derivative of scalar potential, wrt phi1, phi2, phi3 NB ask about scale^3

    def dVtotal(self, field_values, scale=1.0):
        phi1 = field_values[0]
        phi2 = field_values[1]
        phi3 = field_values[2]
        
        phi1sq = phi1**2
        phi2sq = phi2**2
        phi3sq = phi3**2

        # Initializing the result vector
        res = np.zeros(self._N_VEVs)
        
        # Calculating the derivatives
        res[0] = phi1 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi2sq + phi3sq))
        res[1] = phi2 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi1sq + phi3sq))
        res[2] = phi3 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi1sq + phi2sq))

        # Scaling the result
        return res / np.power(scale, 3)

#    construct second derivative of scalar potential  wrt phi1, phi2, phi3 to form Hessian Matrix
#    res is a 3x3 Hessian Matrix

    def d2Vtotal(self, field_values, scale=1.0):
        phi1 = field_values[0]
        phi2 = field_values[1]
        phi3 = field_values[2]
      
        phi1sq = phi1**2
        phi2sq = phi2**2
        phi3sq = phi3**2

        # Initialize the Hessian matrix
        res = np.zeros((self._N_VEVs, self._N_VEVs))

        # Fill the matrix with second derivatives
        res[0][0] = -self._mu2 + self._g1 * (3.0 * phi1sq + phi2sq + phi3sq ) + self._g2 * (phi2sq + phi3sq )
        res[0][1] = 2.0 * (self._g1 + self._g2) * phi1 * phi2
        res[0][2] = 2.0 * (self._g1 + self._g2) * phi1 * phi3
        res[1][0] = res[0][1]  # Symmetric element
        res[1][1] = -self._mu2 + self._g1 * (phi1sq + 3.0 * phi2sq + phi3sq ) + self._g2 * (phi1sq + phi3sq )
        res[1][2] = 2.0 * (self._g1 + self._g2) * phi2 * phi3
        res[2][0] = res[0][2]  # Symmetric element
        res[2][1] = res[1][2]  # Symmetric element
        res[2][2] = -self._mu2 + self._g1 * (phi1sq + phi2sq + 3.0 * phi3sq ) + self._g2 * (phi1sq + phi2sq)

        # Scale the Hessian matrix
        return res / np.power(scale, 2)

# this function allows user to input possible minima, which still need to be tested

    def prepare_local_minima_candidates(self):
        local_extreme = []

        # Initial zero point
        local_extreme.append([0, 0, 0])

        # First condition based points
        if self._mu2 / self._g1 >= 0:
            root = math.sqrt(self._mu2 / self._g1)
            local_extreme.extend([
                [0, 0, root],
                [0, 0, -root],
                [0, root, 0],
                [0, -root, 0],
                [root, 0, 0],
                [-root, 0, 0]
            ])

        # Second condition based points
        if self._mu2 / (3.0 * self._g1 + 2.0 * self._g2) >= 0:
            root = math.sqrt(self._mu2 / (3.0 * self._g1 + 2.0 * self._g2))
            local_extreme.extend([
                [root, root, root],
                [-root, root, root],
                [root, -root, root],
                [root, root, -root],
                [-root, -root, root],
                [root, -root, -root],
                [-root, root, -root],
                [-root, -root, -root]
            ])

        return local_extreme

# This function takes the matrix output of d2Vtotal and asks if all eigenvalues are posiive, if so then return True else False

    def check_hessian(self, field_values):
        # Get the Hessian matrix
        HM = self.d2Vtotal(field_values)

        # Compute eigenvalues of the Hessian matrix
        eigenvalues = np.linalg.eigvalsh(HM)

        # Check if all eigenvalues are greater than zero
        if np.all(eigenvalues > 0):
            return True
        else:
            return False
