import numpy as np
from scipy.linalg import eigvalsh_tridiagonal as eig

# Hu, Xu, Yan, J. Chem. Phys. 133, 101106 (2010)

def __eta(order, z):
    """Compute expansion coefficients eta according to the paragraph around
    equations (12) and (13).
    """
    Am = 1/4 # A1
    A  = 5/4 # A2

    Bm = 3      # B1
    B  = 15+z/4 # B2

    dBm = 0   # B1'=dB1/dz
    dB  = 1/4 # B2'=dB2/dz

    # Eq. (12)
    for M in range(3, order+1):
        Amm = Am
        Bmm = Bm
        dBmm = dBm

        Am = A
        Bm = B
        dBm = dB

        dB = (2*M+1)*dBm + Bmm/4 + z*dBmm/4 # derivative of Eq. (12)
        A = (2*M+1)*Am + z*Amm/4            # Eq. (12)
        B = (2*M+1)*Bm + z*Bmm/4            # Eq. (12)

        # rescale to prevent overflows
        Am /= A
        Bm /= A
        B  /= A
        dBm /= A
        dB  /= A
        A = 1

    return A/(2*dB)

def psd(N):
    """Compute poles xi and expansion coefficients eta for Pade spectrum
    decomposition (PSD) of order N
    
    This function computes the poles xi_j (at imaginary frequency) and the
    expansion coefficients eta_j for the Pade spectrum decomposition of order
    N, see reference [1]. The poles are returned as array xi, the coefficients
    are returned as array eta.
    
    References:
    [1] Hu, Xu, Yan, J. Chem. Phys. 133, 101106 (2010)
    
    returns: xi, eta
    """
    ndim = 2*N # dimension of matrix

    # diagonal elements
    d = np.zeros(ndim)

    # off-diagonal elements
    j = np.arange(1,ndim)
    e = 1/np.sqrt((2*j+1)*(2*j+3)) # Eq. (14)

    # compute eigenvalues of a real, symmetric, tridiagonal matrix
    v = eig(d,e, select="i", select_range=(0,N-1))
    xi = -2/v

    eta = __eta(ndim, -xi**2)

    return xi, eta

if __name__ == "__main__":
    N = 100
    print(psd(N))
