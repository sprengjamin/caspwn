"""
Compare the sparse matrix operations of the sphere-sphere geometry with the dense ones.
"""
import numpy as np
import pytest
from math import sqrt
from caspwn.ufuncs.integration import fc_quadrature
from caspwn.sphere_sphere.spsp_interaction import contribution_finite, contribution_zero


def discretization(R1, R2, L):
    rhosum = max((R1 + R2) / L, 50)
    Nin = int(8.4 * sqrt(rhosum))
    Nout = int(5.3 * sqrt(max(1 / (1 / R1 + 1 / R2) / L, 50)))
    M = int(6.2 * sqrt(rhosum))
    lmax1 = int(12 * max(R1 / L, 50))
    lmax2 = int(12 * max(R2 / L, 50))
    k_inner, w_inner = fc_quadrature(Nin)
    k_outer, w_outer = fc_quadrature(Nout)
    return Nout, Nin, M, k_outer / L, w_outer / L, k_inner / L, w_inner / L, lmax1, lmax2


@pytest.mark.parametrize("R1, R2", [(150., 50.), (100., 100.)])
@pytest.mark.parametrize("K", [0.1, 1.])
def test_finite(R1, R2, K):
    L = 1.
    Nout, Nin, M, ko, wo, ki, wi, lmax1, lmax2 = discretization(R1, R2, L)
    dense = contribution_finite(R1, R2, L, K, np.inf, 2.5, Nout, Nin, M, ko, wo, ki, wi, lmax1, lmax2, 2, "forcegradient", False, False)
    sparse = contribution_finite(R1, R2, L, K, np.inf, 2.5, Nout, Nin, M, ko, wo, ki, wi, lmax1, lmax2, 2, "forcegradient", False, True)
    np.testing.assert_allclose(sparse, dense, rtol=1.e-12)


@pytest.mark.parametrize("materialclass, alpha", [("PEC", 0.), ("dielectric", 3.)])
def test_zero(materialclass, alpha):
    R1, R2, L = 150., 50., 1.
    Nout, Nin, M, ko, wo, ki, wi, lmax1, lmax2 = discretization(R1, R2, L)
    args = (R1, R2, L, alpha, alpha, materialclass, materialclass, Nout, Nin, M, ko, wo, ki, wi, lmax1, lmax2, 2, "forcegradient", True, False)
    dense_TM, dense_TE = contribution_zero(*args, False)
    sparse_TM, sparse_TE = contribution_zero(*args, True)
    np.testing.assert_allclose(sparse_TM, dense_TM, rtol=1.e-12)
    np.testing.assert_allclose(sparse_TE, dense_TE, rtol=1.e-12)
