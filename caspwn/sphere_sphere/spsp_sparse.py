r""" Sparse round-trip matrices for the sphere-sphere geometry

The round-trip matrix is the product :math:`\mathcal{M}=\mathcal{M}_1\mathcal{M}_2` of the reflection matrices of
both spheres, where :math:`\mathcal{M}_1` maps the inner to the outer discretization and :math:`\mathcal{M}_2` vice
versa. For large aspect ratios both reflection matrices are sparse. They are stored in CSR format with interleaved
polarizations, their product is computed directly into the envelope storage of ``plane_sphere/plsp_sparse.py`` and
:math:`\log\det(1-\mathcal{M})` and its derivatives are obtained from the differentiated envelope LU decomposition.

Each element of a reflection matrix depends on :math:`L` through a factor :math:`\exp[-r(\kappa_i+\kappa_j)L]`, such
that its derivatives are the matrix element multiplied by :math:`f_{ij}=-r(\kappa_i+\kappa_j)` and :math:`f_{ij}^2`.
For the product this implies

.. math::
    \partial^n_L\mathcal{M}_{ik} = \sum_j (f^{(1)}_{ij} + f^{(2)}_{jk})^n \mathcal{M}_{1,ij}\mathcal{M}_{2,jk}\,.

"""
import numpy as np
from math import sqrt
from numba import njit
from ..plane_sphere.plsp_sparse import envelope_lu_logdet_explicit


@njit("Tuple((int64[:], int64[:], int64[:]))(int64[:], int64[:], int64, int64, int64)", cache=True)
def reflection_structure(row, col, N1, N2, npol):
    r"""CSR structure of a reflection matrix with interleaved polarizations.

    The order of assignments (and thus which element is kept when several assignments address the same matrix element)
    follows ``construct_matrix_finite`` (npol=2) and ``construct_roundtrip_zero`` (npol=1) in ``spsp_interaction.py``.

    Parameters
    ----------
    row, col : ndarray
        1D arrays of ints, row and column index of the matrix elements
    N1, N2 : int
        number of radial nodes for the rows and columns, respectively
    npol : int
        number of polarizations, 2 for finite and 1 for zero frequency

    Returns
    -------
    (ndarray, ndarray, ndarray)
        indptr, indices : CSR structure of the (npol*N1)x(npol*N2) matrix
        code : for each stored element ``l*nblocks + b`` with the index ``l`` of the data arrays and the polarization
        block ``b`` (TMTM, TETE, TMTE, TETM for npol=2), ``nblocks=npol**2``

    """
    nblocks = npol**2
    n1 = npol*N1
    n2 = npol*N2
    src = np.full((n1, n2), -1, dtype=np.int64)
    for l in range(len(row)):
        i, j = row[l], col[l]
        if npol == 2:
            src[2*i, 2*j] = l*nblocks
            src[2*i+1, 2*j+1] = l*nblocks + 1
            src[2*i, 2*j+1] = l*nblocks + 2
            src[2*i+1, 2*j] = l*nblocks + 3
            if N1 == N2:
                src[2*j, 2*i] = src[2*i, 2*j]
                src[2*j+1, 2*i+1] = src[2*i+1, 2*j+1]
                src[2*j+1, 2*i] = src[2*i, 2*j+1]
                src[2*j, 2*i+1] = src[2*i+1, 2*j]
        else:
            src[i, j] = l
            if N1 == N2:
                src[j, i] = src[i, j]
    indptr = np.zeros(n1 + 1, dtype=np.int64)
    for i in range(n1):
        indptr[i+1] = indptr[i] + np.sum(src[i] >= 0)
    indices = np.empty(indptr[-1], dtype=np.int64)
    code = np.empty(indptr[-1], dtype=np.int64)
    p = 0
    for i in range(n1):
        for j in range(n2):
            if src[i, j] >= 0:
                indices[p] = j
                code[p] = src[i, j]
                p += 1
    return indptr, indices, code


@njit("float64[:](int64[:], int64[:], int64, float64[:], float64[:], float64)", cache=True)
def reflection_factors(indptr, indices, npol, kappa1, kappa2, r):
    r"""Factors :math:`f_{ij}=-r(\kappa_i+\kappa_j)` relating the matrix elements to their derivatives.

    Parameters
    ----------
    indptr, indices : ndarray
        CSR structure, see ``reflection_structure``
    npol : int
        number of polarizations
    kappa1, kappa2 : ndarray
        imaginary z-component of the wave vectors for rows and columns
    r : float
        relative translation distance contained in the reflection matrix

    Returns
    -------
    ndarray

    """
    factors = np.empty(len(indices))
    for i in range(len(indptr)-1):
        for p in range(indptr[i], indptr[i+1]):
            factors[p] = -r*(kappa1[i//npol] + kappa2[indices[p]//npol])
    return factors


@njit("float64[:](int64[:], int64[:], int64[:], int64, float64[:,:], int64, float64[:], float64[:])", cache=True)
def reflection_values(code, row, col, nblocks, data, M, w1, w2):
    r"""Values of the stored matrix elements of a reflection matrix.

    Parameters
    ----------
    code : ndarray
        see ``reflection_structure``
    row, col : ndarray
        1D arrays of ints, row and column index of the matrix elements
    nblocks : int
        number of polarization blocks
    data : ndarray
        2D array of shape (nblocks, len(row)), matrix elements for each polarization block
    M : int
        angular discretization order
    w1, w2 : ndarray
        weights of the radial quadrature for rows and columns

    Returns
    -------
    ndarray

    """
    values = np.empty(len(code))
    for p in range(len(code)):
        l = code[p] // nblocks
        b = code[p] % nblocks
        values[p] = 1/M/2/np.pi*sqrt(w1[row[l]]*w2[col[l]])*data[b, l]
    return values


@njit("Tuple((int64[:], int64[:]))(int64[:], int64[:], int64[:], int64[:])", cache=True)
def product_envelope_structure(indptr1, indices1, indptr2, indices2):
    r"""Envelope of the product of two sparse matrices given in CSR format.

    Returns
    -------
    (ndarray, ndarray)
        first, ptr : see ``plane_sphere/plsp_sparse.py``

    """
    n = len(indptr1) - 1
    first = np.arange(n)
    for i in range(n):
        for p in range(indptr1[i], indptr1[i+1]):
            j = indices1[p]
            for q in range(indptr2[j], indptr2[j+1]):
                k = indices2[q]
                lo = min(i, k)
                hi = max(i, k)
                if lo < first[hi]:
                    first[hi] = lo
    ptr = np.empty(n + 1, dtype=np.int64)
    ptr[0] = 0
    for i in range(n):
        ptr[i+1] = ptr[i] + i - first[i] + 1
    return first, ptr


@njit("void(float64[:,:], int64[:], int64[:], int64[:], int64[:], float64[:], float64[:], int64[:], int64[:], float64[:], float64[:])", cache=True)
def product_envelope(mat, first, ptr, indptr1, indices1, values1, factors1, indptr2, indices2, values2, factors2):
    r"""Computes the round-trip matrix :math:`\mathcal{M}=\mathcal{M}_1\mathcal{M}_2` and its first two derivatives
    into envelope storage ``mat[0]``, ``mat[1]`` and ``mat[2]``.

    """
    S = ptr[-1]
    n = len(indptr1) - 1
    # each row of the product is accumulated in a dense buffer
    buf0 = np.zeros(n)
    buf1 = np.zeros(n)
    buf2 = np.zeros(n)
    for i in range(n):
        kmin = n
        kmax = -1
        for p in range(indptr1[i], indptr1[i+1]):
            j = indices1[p]
            x = values1[p]
            f = factors1[p]
            for q in range(indptr2[j], indptr2[j+1]):
                k = indices2[q]
                g = f + factors2[q]
                v = x*values2[q]
                buf0[k] += v
                buf1[k] += g*v
                buf2[k] += g*g*v
                kmin = min(kmin, k)
                kmax = max(kmax, k)
        # lower part of row i, the diagonal is stored in the upper part only
        mat[:, ptr[i] + i - first[i]] = 0.
        for k in range(first[i], i):
            pos = ptr[i] + k - first[i]
            mat[0, pos] = buf0[k]
            mat[1, pos] = buf1[k]
            mat[2, pos] = buf2[k]
        # upper part (column-wise storage) including diagonal
        for k in range(i, n):
            if i >= first[k]:
                pos = S + ptr[k] + i - first[k]
                mat[0, pos] = buf0[k]
                mat[1, pos] = buf1[k]
                mat[2, pos] = buf2[k]
        for k in range(kmin, kmax+1):
            buf0[k] = 0.
            buf1[k] = 0.
            buf2[k] = 0.


@njit("UniTuple(float64, 3)(float64[:,:], int64[:], int64[:], int64)", cache=True)
def compute_matrix_operations_sparse(mat, first, ptr, order):
    r"""Sparse counterpart of ``compute_matrix_operations`` in ``spsp_interaction.py``.

        Parameters
        ----------
        mat : ndarray
            2D array of shape (3, 2*S), round-trip matrix and its first two derivatives in envelope storage as computed
            by ``product_envelope``. It is overwritten.
        first, ptr : ndarray
            envelope structure
        order : int
            0, 1 or 2 for the observables "energy", "force" and "forcegradient", respectively

        Returns
        -------
        (float, float, float)

    """
    n = len(first)
    S = ptr[-1]
    norm2_mat = 0.
    for l in range(2*S):
        norm2_mat += mat[0, l]**2
    tr_mat = 0.
    tr_dL_mat = 0.
    tr_d2L_mat = 0.
    for i in range(n):
        d = S + ptr[i+1] - 1
        tr_mat += mat[0, d]
        tr_dL_mat += mat[1, d]
        tr_d2L_mat += mat[2, d]
    norm_mat = sqrt(norm2_mat)
    if tr_mat == 0.:
        assert(norm_mat == 0.)
        return 0., 0., 0.
    if abs(norm_mat**2/(1-norm_mat)/tr_mat) < 1.e-10:
        # make trace approximation
        if order == 0:
            return -tr_mat, 0., 0.
        if order == 1:
            return -tr_mat, tr_dL_mat, 0.
        return -tr_mat, tr_dL_mat, tr_d2L_mat
    return envelope_lu_logdet_explicit(mat, first, ptr, order, False)


class sparse_roundtrip:
    r"""Sparse structure of the round-trip matrices of the sphere-sphere geometry for one frequency/wavenumber.

    Parameters
    ----------
    row1, col1, row2, col2 : ndarray
        row and column indices of the non-vanishing elements of the reflection matrices of sphere 1 and 2
    N_outer, N_inner : int
        outer and inner radial discretization order
    npol : int
        number of polarizations, 2 for finite and 1 for zero frequency
    kappa_outer, kappa_inner : ndarray
        imaginary z-component of the wave vectors for the outer and inner discretization
    w_outer, w_inner : ndarray
        weights of the outer and inner radial quadrature
    M : int
        angular discretization order
    r : float
        relative translation distance contained in the reflection matrices

    """
    def __init__(self, row1, col1, row2, col2, N_outer, N_inner, npol, kappa_outer, kappa_inner, w_outer, w_inner, M, r):
        self.row1, self.col1, self.row2, self.col2 = row1, col1, row2, col2
        self.npol = npol
        self.M = M
        self.w_outer, self.w_inner = w_outer, w_inner
        self.indptr1, self.indices1, self.code1 = reflection_structure(row1, col1, N_outer, N_inner, npol)
        self.indptr2, self.indices2, self.code2 = reflection_structure(row2, col2, N_inner, N_outer, npol)
        self.factors1 = reflection_factors(self.indptr1, self.indices1, npol, kappa_outer, kappa_inner, r)
        self.factors2 = reflection_factors(self.indptr2, self.indices2, npol, kappa_inner, kappa_outer, r)
        self.first, self.ptr = product_envelope_structure(self.indptr1, self.indices1, self.indptr2, self.indices2)
        self.mat = np.empty((3, 2*self.ptr[-1]))

    def matrix_operations(self, data1, data2, order):
        r"""Computes the quantities of ``compute_matrix_operations`` for the given reflection matrix elements.

        Parameters
        ----------
        data1, data2 : ndarray
            2D arrays of shape (npol**2, number of elements), elements of the reflection matrices for each polarization
            block (TMTM, TETE, TMTE, TETM for finite frequencies)
        order : int
            0, 1 or 2 for the observables "energy", "force" and "forcegradient", respectively

        Returns
        -------
        (float, float, float)

        """
        nblocks = self.npol**2
        values1 = reflection_values(self.code1, self.row1, self.col1, nblocks, data1, self.M, self.w_outer, self.w_inner)
        values2 = reflection_values(self.code2, self.row2, self.col2, nblocks, data2, self.M, self.w_inner, self.w_outer)
        product_envelope(self.mat, self.first, self.ptr, self.indptr1, self.indices1, values1, self.factors1,
                         self.indptr2, self.indices2, values2, self.factors2)
        return compute_matrix_operations_sparse(self.mat, self.first, self.ptr, order)
