"""
Compare the sparse (envelope) matrix operations of the plane-sphere geometry with the dense ones.
"""
import numpy as np
import pytest
from caspwn.ufuncs.integration import fc_quadrature
from caspwn.plane_sphere.plsp_interaction import construct_roundtrip_finite, construct_roundtrip_zero, \
    compute_matrix_operations
from caspwn.plane_sphere.plsp_sparse import envelope_structure, construct_roundtrip_finite_sparse, \
    construct_roundtrip_zero_sparse, compute_matrix_operations_sparse, observable_order


def banded_data(N, M, w, bandwidth, amplitude, nblocks, seed=0):
    # random matrix elements for index pairs i >= j with i-j <= bandwidth
    rng = np.random.default_rng(seed)
    pairs = np.array([(i, j) for i in range(N) for j in range(max(0, i-bandwidth), i+1)], dtype=np.int64)
    row = np.ascontiguousarray(pairs[:, 0])
    col = np.ascontiguousarray(pairs[:, 1])
    # compensate quadrature weights such that 1-M is diagonally dominant
    scale = amplitude*M*2*np.pi/np.sqrt(w[row]*w[col])
    data = [scale*rng.uniform(-1., 1., len(row)) for _ in range(nblocks)]
    return row, col, data


def exact_reference(mat, dL_mat, d2L_mat):
    A = np.eye(len(mat)) - mat
    X = np.linalg.inv(A)
    Z = X @ dL_mat
    return np.linalg.slogdet(A)[1], np.trace(Z), np.trace(X @ d2L_mat) + np.trace(Z @ Z)


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("observable", ["energy", "force", "forcegradient"])
def test_finite(symmetric, observable):
    N, M = 40, 10
    k, w = fc_quadrature(N)
    K = 0.3
    kappa = np.sqrt(K**2 + k**2)
    rTM = 0.9*np.ones(N)
    rTE = -0.8*np.ones(N)
    if not symmetric:
        rTM[::3] *= -0.5
    row, col, (dataTMTM, dataTETE, dataTMTE, dataTETM) = banded_data(N, M, w, 4, 0.05, 4)

    mat, dL_mat, d2L_mat = construct_roundtrip_finite(row, col, dataTMTM, dataTETE, dataTMTE, dataTETM, N, M, kappa, w, rTM, rTE)
    dense = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)

    first, ptr = envelope_structure(row, col, N, 2)
    env = np.empty((3, 2*ptr[-1]))
    sym = construct_roundtrip_finite_sparse(env, first, ptr, row, col, dataTMTM, dataTETE, dataTMTE, dataTETM, N, M, w, rTM, rTE)
    assert sym == symmetric
    sparse = compute_matrix_operations_sparse(env, first, ptr, np.repeat(kappa, 2), observable_order(observable), sym)

    ref = exact_reference(mat, dL_mat, d2L_mat)
    order = observable_order(observable)
    np.testing.assert_allclose(sparse[:order+1], ref[:order+1], rtol=1.e-12)
    np.testing.assert_allclose(sparse, dense, rtol=1.e-12)


@pytest.mark.parametrize("symmetric", [True, False])
def test_zero(symmetric):
    N, M = 40, 10
    k, w = fc_quadrature(N)
    rp = 0.9*np.ones(N)
    if not symmetric:
        rp[::3] *= -0.5
    row, col, (data,) = banded_data(N, M, w, 5, 0.05, 1, seed=1)

    mat, dL_mat, d2L_mat = construct_roundtrip_zero(row, col, data, N, M, k, w, rp)
    dense = compute_matrix_operations(mat, dL_mat, d2L_mat, "forcegradient")

    first, ptr = envelope_structure(row, col, N, 1)
    env = np.empty((3, 2*ptr[-1]))
    sym = construct_roundtrip_zero_sparse(env, first, ptr, row, col, data, M, w, rp)
    assert sym == symmetric
    sparse = compute_matrix_operations_sparse(env, first, ptr, k, 2, sym)

    np.testing.assert_allclose(sparse, exact_reference(mat, dL_mat, d2L_mat), rtol=1.e-12)
    np.testing.assert_allclose(sparse, dense, rtol=1.e-12)


def test_trace_approximation():
    N, M = 40, 10
    k, w = fc_quadrature(N)
    rp = 0.9*np.ones(N)
    row, col, (data,) = banded_data(N, M, w, 5, 1.e-14, 1, seed=2)

    mat, dL_mat, d2L_mat = construct_roundtrip_zero(row, col, data, N, M, k, w, rp)
    dense = compute_matrix_operations(mat, dL_mat, d2L_mat, "forcegradient")

    first, ptr = envelope_structure(row, col, N, 1)
    env = np.empty((3, 2*ptr[-1]))
    sym = construct_roundtrip_zero_sparse(env, first, ptr, row, col, data, M, w, rp)
    sparse = compute_matrix_operations_sparse(env, first, ptr, k, 2, sym)
    np.testing.assert_allclose(sparse, dense, rtol=1.e-12)
