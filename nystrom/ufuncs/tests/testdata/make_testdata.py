import numpy as np
from mpmath import *

mp.dps = 40
maxterms = 1e6

def mp_Inu_e(nu, z):
    """
    exponentially scaled modified Bessel function of the first kind
    """
    return besseli(nu, z, maxterms=maxterms)*exp(-sqrt(nu**2 + z**2) + nu*asinh(nu/z))

def mp_Knu_e(nu, z):
    """
    exponentially scaled modified Bessel function of the second kind
    """
    return besselk(nu, z, maxterms=maxterms)*exp(sqrt(nu**2 + z**2) - nu*asinh(nu/z))

if __name__ == '__main__':
    params = [(1.002500000000000000e+03, 1.002000000000000113e-01),
              (1.002500000000000000e+03, 1.002000000000000000e+03),
              (1.002500000000000000e+03, 1.002000000000000000e+07),
              (1.001150000000000000e+04, 1.001100000000000101e+00),
              (1.001150000000000000e+04, 1.001100000000000000e+04),
              (1.001150000000000000e+04, 1.001100000000000000e+08),
              (9.994150000000000000e+04, 9.994100000000001316e+00),
              (9.994150000000000000e+04, 9.994100000000000000e+04),
              (9.994150000000000000e+04, 9.994100000000000000e+08)]

    data = np.empty((len(params), 4))
    for i, (nu, z) in enumerate(params):
        print('nu:', nu, 'z:', z)
        data[i ,0] = nu
        data[i, 1] = z
        data[i, 2] = float(mp_Inu_e(mpf(nu), mpf(z)))
        data[i, 3] = float(mp_Knu_e(mpf(nu), mpf(z)))

    np.savetxt('InuKnue.dat', data, header='nu, z, Inue(nu,z), Knue(nu,z)')