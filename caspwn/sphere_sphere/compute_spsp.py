import numpy as np
from multiprocessing import cpu_count
from math import sqrt
from scipy.constants import k as kB
from scipy.constants import c, hbar
from scipy.integrate import quad
from .spsp_interaction import contribution_finite, contribution_zero
from ..ufuncs.integration import fc_quadrature
from ..ufuncs.summation import msd_sum, psd_sum

class sphere_sphere_system:
    r"""Casimir interaction between two spheres.

    Parameters
    ----------
    T : float
        temperature in K
    L : float
        surface-to-surface distance in m
    R1, R2 : float
        sphere radii in m
    mat_sphere1, mat_sphere2, mat_medium : material
        materials of the spheres and the medium in between

    """
    # (R1+R2)/L above which sparse round-trip matrices are used by default. For (R1+R2)/L up to a few thousand the
    # sparse path is not faster than the dense one, it is therefore switched off by default.
    sparse_threshold = np.inf

    def __init__(self, T, L, R1, R2, mat_sphere1, mat_sphere2, mat_medium):
        self.T = T
        self.L = L
        self.R1 = R1
        self.R2 = R2
        self.mat_sphere1 = mat_sphere1
        self.mat_sphere2 = mat_sphere2
        self.mat_medium = mat_medium

        self.calculate_n0_TE = mat_sphere1.materialclass != "dielectric" and mat_sphere1.materialclass != "drude" and mat_sphere2.materialclass != "dielectric" and mat_sphere2.materialclass != "drude"

        # Nystrom discretization scheme
        self.k_quad = fc_quadrature

        if T == 0.:
            self.automatic_integration = True
            # used if above is False
            self.npoints = 40
            self.K_quad = fc_quadrature


    def calculate(self, observable, ht_limit=False, etaM=6.2, M=None, etaNin=8.4, Nin=None, etaNout=5.3, Nout=None, etalmax1=12., lmax1=None, etalmax2=12., lmax2=None, fs='psd', epsrel=1.e-8, O=None, cores=cpu_count(), debug=False, sparse=None):
        r"""Computes the Casimir free energy, force or force gradient.

        Parameters
        ----------
        observable : string
            "energy", "force" or "forcegradient"
        ht_limit : boolean
            if True, only the zero-frequency contribution (high-temperature limit) is computed
        etaM, etaNin, etaNout, etalmax1, etalmax2 : float
            discretization parameters, M = etaM*sqrt((R1+R2)/L), Nin = etaNin*sqrt((R1+R2)/L),
            Nout = etaNout*sqrt(Reff/L) with Reff = R1*R2/(R1+R2) and lmax1/2 = etalmax1/2*R1/2/L, where the ratios are
            at least 50
        M, Nin, Nout, lmax1, lmax2 : int
            discretization orders and multipole cut-offs, overwrite the parameters above
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
            use sparse round-trip matrices; if None, they are used if (R1+R2)/L is at least ``sparse_threshold``

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

        if sparse == None:
            self.sparse = (self.R1 + self.R2) / self.L >= self.sparse_threshold
        else:
            self.sparse = sparse

        rhosum = max((self.R1 + self.R2) / self.L, 50)
        if Nin == None:
            self.Nin = int(etaNin * sqrt(rhosum))
        else:
            self.Nin = Nin
        if Nout == None:
            Reff = 1 / (1 / self.R1 + 1 / self.R2)
            rhoeff = max(Reff / self.L, 50)
            self.Nout = int(etaNout * sqrt(rhoeff))
        else:
            self.Nout = Nout
        if M == None:
            self.M = int(etaM * sqrt(rhosum))
        else:
            self.M = M
        rho1 = max(self.R1 / self.L, 50)
        if lmax1 == None:
            self.lmax1 = int(etalmax1 * rho1)
        else:
            self.lmax1 = lmax1
        rho2 = max(self.R2 / self.L, 50)
        if lmax2 == None:
            self.lmax2 = int(etalmax2 * rho2)
        else:
            self.lmax2 = lmax2

        nds_in, wts_in = self.k_quad(self.Nin)
        k_inner = nds_in/self.L
        w_inner = wts_in/self.L
        nds_out, wts_out = self.k_quad(self.Nout)
        k_outer = nds_out / self.L
        w_outer = wts_out / self.L

        nfunc_medium = lambda xi: sqrt(self.mat_medium.epsilon(xi))
        nfunc_sphere1 = lambda xi: sqrt(self.mat_sphere1.epsilon(xi))
        nfunc_sphere2 = lambda xi: sqrt(self.mat_sphere2.epsilon(xi))

        if self.T == 0.:
            self.f = lambda x: \
                contribution_finite(self.R1, self.R2, self.L, nfunc_medium(c * x / self.L) * x / self.L,
                                    nfunc_sphere1(c * x / self.L) / nfunc_medium(c * x / self.L),
                                    nfunc_sphere2(c * x / self.L) / nfunc_medium(c * x / self.L), self.Nout, self.Nin, self.M,
                                    k_outer, w_outer, k_inner,
                                    w_inner, self.lmax1, self.lmax2, cores, observable, debug, self.sparse)[j]

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
            materialclass_sphere1 = self.mat_sphere1.materialclass
            if materialclass_sphere1 == "dielectric":
                alpha_sphere1 = nfunc_sphere1(0.) ** 2 / nfunc_medium(0.) ** 2
            elif materialclass_sphere1 == "plasma":
                alpha_sphere1 = self.mat_sphere1.K_plasma * self.R1
            else:  # will not be used
                alpha_sphere1 = 0.
            materialclass_sphere2 = self.mat_sphere2.materialclass
            if materialclass_sphere2 == "dielectric":
                alpha_sphere2 = nfunc_sphere2(0.) ** 2 / nfunc_medium(0.) ** 2
            elif materialclass_sphere2 == "plasma":
                alpha_sphere2 = self.mat_sphere2.K_plasma * self.R2
            else:  # will not be used
                alpha_sphere2 = 0.

            if debug:
                print('# K*L, logdet, t_matrix, t_fft, t_logdet')

            f_n0_TM, f_n0_TE = contribution_zero(self.R1, self.R2, self.L, alpha_sphere1, alpha_sphere2,
                                                 materialclass_sphere1, materialclass_sphere2, self.Nout, self.Nin, self.M, k_outer,
                                                 w_outer, k_inner, w_inner, self.lmax1, self.lmax2, cores, observable, self.calculate_n0_TE, debug, self.sparse)

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
                contribution_finite(self.R1, self.R2, self.L, nfunc_medium(c * k0) * k0,
                                    nfunc_sphere1(c * k0) / nfunc_medium(c * k0),
                                    nfunc_sphere2(c * k0) / nfunc_medium(c * k0), self.Nout, self.Nin, self.M,
                                    k_outer, w_outer, k_inner,
                                    w_inner, self.lmax1, self.lmax2, cores, observable, debug, self.sparse)

            self.f_n1 = fsum(self.T, self.L, self.f, epsrel=epsrel, order=O)
            self.result = self.f_n0_TM + self.f_n0_TE + self.f_n1
            return self.result[j]