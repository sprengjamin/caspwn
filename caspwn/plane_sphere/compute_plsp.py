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
    r"""Casimir interaction between a plane and a sphere.

    Parameters
    ----------
    T : float
        temperature in K
    L : float
        surface-to-surface distance in m
    R : float
        sphere radius in m
    mat_sphere, mat_plane, mat_medium : material
        materials of the sphere, the plane and the medium in between

    """
    # aspect ratio R/L above which sparse round-trip matrices are used by default
    sparse_threshold = 100.

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
        r"""Turns the plane into a layered system.

        The material ``mat_plane`` becomes the top layer facing the sphere, followed by the materials in ``mats_layers``.
        The last material is the substrate. At most four materials in total are supported.

        Parameters
        ----------
        mats_layers : list
            materials below the top layer, the last one being the substrate
        thicknesses : list
            thicknesses of all layers except the substrate in meters, starting with the top layer

        """
        self.mats_plane = [self.mat_plane] + list(mats_layers)
        self.thicknesses = thicknesses


    def calculate(self, observable, ht_limit=False, etaM=5.1, M=None, etaN=5.6, N=None, etalmax=12., lmax=None, fs='psd', epsrel=1.e-8, O=None, cores=cpu_count(), debug=False, sparse=None):
        r"""Computes the Casimir free energy, force or force gradient.

        Parameters
        ----------
        observable : string
            "energy", "force" or "forcegradient"
        ht_limit : boolean
            if True, only the zero-frequency contribution (high-temperature limit) is computed
        etaM, etaN, etalmax : float
            discretization parameters, M = etaM*sqrt(R/L), N = etaN*sqrt(R/L) and lmax = etalmax*R/L with R/L at
            least 50
        M, N, lmax : int
            angular and radial discretization order and multipole cut-off, overwrite the parameters above
        fs : string
            Matsubara frequency summation, "psd" (Pade spectrum decomposition) or "msd" (plain Matsubara sum)
        epsrel : float
            relative accuracy of the frequency summation
        O : int
            order of the Pade spectrum decomposition, determined from epsrel if None
        cores : int
            number of threads used for the computation of the matrix elements
        debug : boolean
            print intermediate results and timings for each frequency
        sparse : boolean or None
            use sparse round-trip matrices; if None, they are used if R/L is at least ``sparse_threshold``

        Returns
        -------
        float
            free energy in J, force in N or force gradient in N/m (a negative force is attractive). For T > 0 the
            attribute ``result`` holds the array [energy, force, forcegradient] where only the requested observable
            and the ones before it are computed.

        """
        if observable == "energy":
            j = 0
        elif observable == "force":
            j = 1
        elif observable == "forcegradient":
            j = 2
        else:
            raise ValueError("Supported observables are 'energy', 'force' and 'forcegradient'!")

        if hasattr(self, 'thicknesses'):
            plane_refl_coeff = def_reflection_coeff(self.mat_medium, self.mats_plane, self.thicknesses)
        else:
            plane_refl_coeff = def_reflection_coeff(self.mat_medium, [self.mat_plane], [None])

        if sparse == None:
            self.sparse = self.R / self.L >= self.sparse_threshold
        else:
            self.sparse = sparse

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
                                    k, w, self.lmax, cores, observable, debug, self.sparse)[j]

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
                                                 w, self.lmax, cores, observable, self.calculate_n0_TE, debug, self.sparse)
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
                                    k, w, self.lmax, cores, observable, debug, self.sparse)

            self.f_n1 = fsum(self.T, self.L, self.f, epsrel=epsrel, order=O)
            self.result = self.f_n0_TM + self.f_n0_TE + self.f_n1
            return self.result[j]