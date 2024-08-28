import numpy as np
from multiprocessing import cpu_count
from math import sqrt
from scipy.constants import k as kB
from scipy.constants import c, hbar
from scipy.integrate import quad
from ..plane.reflection_coefficients import def_reflection_coeff
from .plsp_interaction import contribution_finite, contribution_zero
from ..ufuncs.integration import fc_quadrature
from ..ufuncs.summation import msd_sum, psd_sum

class plane_sphere_system:
    def __init__(self, T, L, R, mat_sphere, mat_plane, mat_medium):
        self.T = T
        self.L = L
        self.R = R
        self.mat_sphere = mat_sphere
        self.mat_plane = mat_plane
        self.mat_medium = mat_medium

        self.calculate_n0_TE = mat_plane.materialclass != "dielectric" and mat_plane.materialclass != "drude" and mat_sphere.materialclass != "dielectric" and mat_sphere.materialclass != "drude"

        # Nystrom discretization scheme
        self.k_quad = fc_quadrature

        if T == 0.:
            self.automatic_integration = True
            # used if above is False
            self.npoints = 40
            self.K_quad = fc_quadrature


    def add_plane_layers(self, mats_layers, thicknesses):
        self.mats_plane = mats_layers.insert(0, self.mat_plane)
        self.thicknesses = thicknesses


    def calculate(self, observable, ht_limit=False, etaM=5.1, M=None, etaN=5.6, N=None, etalmax=12., lmax=None, fs='psd', epsrel=1.e-8, O=None, cores=cpu_count(), debug=False):
        if observable == "energy":
            j = 0
        if observable == "force":
            j = 1
        if observable == "forcegradient":
            j = 2

        if hasattr(self, 'thicknesses'):
            plane_refl_coeff = def_reflection_coeff(self.mat_medium, self.mats_plane, self.thicknesses)
        else:
            plane_refl_coeff = def_reflection_coeff(self.mat_medium, [self.mat_plane], [None])

        rho = max(self.R / self.L, 50.)
        if M == None:
            self.M = int(etaM * sqrt(rho))
        else:
            self.M = M
        if N == None:
            self.N = int(etaN * sqrt(rho))
        else:
            self.N = N
        if lmax == None:
            self.lmax = int(etalmax * rho)
        else:
            self.lmax = lmax

        nds, wts = self.k_quad(self.N)
        k = nds/self.L
        w = wts/self.L

        nfunc_medium = lambda xi: sqrt(self.mat_medium.epsilon(xi))
        nfunc_sphere = lambda xi: sqrt(self.mat_sphere.epsilon(xi))

        if self.T == 0.:
            self.f = lambda x: \
                contribution_finite(self.R, self.L, x/self.L, nfunc_medium(c * x / self.L) * x/self.L,
                                    nfunc_sphere(c * x / self.L) / nfunc_medium(c * x / self.L), plane_refl_coeff, self.N,
                                    self.M,
                                    k, w, self.lmax, cores, observable, debug)[j]

            if self.automatic_integration:
                result = quad(self.f, 0., np.inf)[0]
                return hbar*c/2/np.pi/self.L*result
            else:
                X, WX = self.K_quad(self.npoints)
                result = 0.
                for x, wx in zip(X, WX):
                    result += wx*self.f(x)
                return hbar * c / 2 / np.pi / self.L * result
        else:
            # ZERO FREQUENCY
            materialclass_sphere = self.mat_sphere.materialclass
            if materialclass_sphere == "dielectric":
                alpha_sphere = nfunc_sphere(0.) ** 2 / nfunc_medium(0.) ** 2
            elif materialclass_sphere == "plasma":
                alpha_sphere = self.mat_sphere.K_plasma * self.R
            else:  # will not be used
                alpha_sphere = 0.

            if debug:
                print('# K*L, logdet, t_matrix, t_fft, t_logdet')

            f_n0_TM, f_n0_TE = contribution_zero(self.R, self.L, alpha_sphere,
                                                 materialclass_sphere, plane_refl_coeff, self.N, self.M,
                                                 k,
                                                 w, self.lmax, cores, observable, self.calculate_n0_TE, debug)
            self.f_n0_TM = 0.5 * kB * self.T * f_n0_TM
            self.f_n0_TE = 0.5 * kB * self.T * f_n0_TE

            if ht_limit == True:
                self.result = self.f_n0_TM + self.f_n0_TE
                return self.result[j]

            # FINITE FREQUENCY
            if fs == 'psd':
                fsum = psd_sum
            elif fs == 'msd':
                fsum = msd_sum
            else:
                raise ValueError('Supported values for fs are either \'psd\' or \'msd\'!')

            self.f = lambda k0: \
                contribution_finite(self.R, self.L, k0, nfunc_medium(c * k0) * k0,
                                    nfunc_sphere(c * k0) / nfunc_medium(c * k0), plane_refl_coeff, self.N, self.M,
                                    k, w, self.lmax, cores, observable, debug)

            self.f_n1 = fsum(self.T, self.L, self.f, epsrel=epsrel, order=O)
            self.result = self.f_n0_TM + self.f_n0_TE + self.f_n1
            return self.result[j]