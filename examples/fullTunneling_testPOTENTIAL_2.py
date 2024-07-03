import numpy as np
import matplotlib.pyplot as plt

from cosmoTransitions import pathDeformation as pd


class Potential:
    """
    A sample potential. The depth of the absolute minimum is controlled with
    the parameters `fx` and `fy`.

    This potential has no physical significance whatsoever.
    """
    def __init__(self, mu2, g1, g2):
        #    define input parameters for potential, including dimension of scalar field (3 in this case)
        self._N_VEVs = 3
        
        self._mu2 = mu2
        self._g1 = g1
        self._g2 = g2
        

        
    #       self.set_potential_parameters()
       
    #    construct scalar potential

    def V(self, field_values, scale=1.0):
        phi1, phi2, phi3  = field_values[...,0], field_values[...,1], field_values[...,2]
        
        phi1sq = phi1**2
        phi2sq = phi2**2
        phi3sq = phi3**2
       
        tmp = 0
        tmp += -self._mu2 *(phi1sq + phi2sq + phi3sq) / 2.0
        tmp +=  self._g1 * (phi1sq + phi2sq + phi3sq) ** 2 / 4.0
        tmp +=  self._g2 * (phi1sq * phi2sq + phi2sq * phi3sq + phi3sq * phi1sq) / 2.0

        return tmp * scale

    #    construct first derivative of scalar potential, wrt phi1, phi2, phi3 NB ask about scale^3
    
    def Vtotal(self, field_values, scale=1.0):
        phi1, phi2, phi3  = field_values[...,0], field_values[...,1], field_values[...,2]
        
        phi1sq = phi1**2
        phi2sq = phi2**2
        phi3sq = phi3**2
       
        tmp = 0
        tmp += -self._mu2 *(phi1sq + phi2sq + phi3sq) / 2.0
        tmp +=  self._g1 * (phi1sq + phi2sq + phi3sq) ** 2 / 4.0
        tmp +=  self._g2 * (phi1sq * phi2sq + phi2sq * phi3sq + phi3sq * phi1sq) / 2.0

        return tmp * scale
        
    def dV(self, field_values, scale=1.0):
        phi1, phi2, phi3  = field_values[...,0], field_values[...,1], field_values[...,2]
        
        phi1sq = phi1**2
        phi2sq = phi2**2
        phi3sq = phi3**2

        # Initializing the result vector
        res = np.zeros(self._N_VEVs)
        
        # Calculating the derivatives
        dVdx = phi1 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi2sq + phi3sq))
        dVdy = phi2 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi1sq + phi3sq))
        dVdz = phi3 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi1sq + phi2sq))
        
        rval = np.empty_like(field_values)
        rval[...,0] = dVdx
        rval[...,1] = dVdy
        rval[...,2] = dVdz
        
        print("rval shape", rval.shape)
        # Scaling the result
        return rval

def makePlots():
    # Thin-walled instanton
#    plt.figure()
#    ax = plt.subplot(221)
    m = Potential(mu2=1 , g1=.5, g2=-.5)
    m.dV(np.array([.1, .1, .1]))
#    m.plotContour()
    Y = pd.fullTunneling([[-1,1.,1],[1,1,1]], m.V, m.dV)
    #print(Y)
    #    print("GETTTTT TOTAL",  get_total_energy(Y.profile1D[0], Y.profile1D[1], m))

#    r = Y[:,0]
#    Phi = Y[:,1]
#    print(get_total_energy(r, Phi))
#    ax.plot(Y.Phi[:,0], Y.Phi[:,1], 'k', lw=1.5)
#    ax.set_xlabel(r"$\phi_x$")
#    ax.set_ylabel(r"$\phi_y$")
#    ax.set_title("Thin-walled")
#    ax = plt.subplot(223)
#    ax.plot(Y.profile1D.R, Y.profile1D.Phi, 'r')
#    ax.set_xlabel("$r$")
#    ax.set_ylabel(r"$|\phi(r) - \phi_{\rm absMin}|$")
#
#    # Thick-walled instanton
#    ax = plt.subplot(222)
#    m = Potential(c=5, fx=0., fy=80.)
#    m.plotContour()
#    Y = pd.fullTunneling([[1,1.],[0,0]], m.V, m.dV)
#    ax.plot(Y.Phi[:,0], Y.Phi[:,1], 'k', lw=1.5)
#    ax.set_xlabel(r"$\phi_x$")
#    ax.set_title("Thick-walled")
#    ax = plt.subplot(224)
#    ax.plot(Y.profile1D.R, Y.profile1D.Phi, 'r')
#    ax.set_xlabel("$r$")
#
#    plt.show()


if __name__ == "__main__":
    makePlots()
