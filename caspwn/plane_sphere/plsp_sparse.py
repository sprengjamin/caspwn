r""" Sparse round-trip matrices for the plane-sphere geometry

For large aspect ratios :math:`R/L` most matrix elements of the round-trip operator are numerically zero (see
``isFinite`` in ``sphere/reflection_matrix.py``). When the polarization blocks are interleaved, the non-vanishing
elements lie within a narrow envelope around the diagonal, i.e. every row :math:`i` only has non-zero elements in the
columns :math:`\mathtt{first}[i]\leq j\leq i` (and symmetrically for the columns). An LU decomposition without pivoting
does not create any fill-in outside of this envelope.

The round-trip matrix :math:`\mathcal{M}` is thus stored in envelope (profile) format and :math:`\log\det(1-\mathcal{M})`
is computed from an envelope LU decomposition. Instead of computing the inverse of :math:`1-\mathcal{M}`, the
derivatives of the logdet with respect to the separation :math:`L` are obtained by differentiating the LU decomposition
itself (forward-mode differentiation up to second order). All operations then stay within the envelope.

Storage
-------
For an envelope described by ``first`` and ``ptr`` with total size ``S = ptr[-1]`` a matrix is stored in a 1D array of
length ``2*S``. The element :math:`(i, j)` is located at

    * ``ptr[i] + j - first[i]`` if :math:`j < i` (strictly lower part, row-wise),
    * ``S + ptr[j] + i - first[j]`` if :math:`i \leq j` (upper part including diagonal, column-wise).

"""
import numpy as np
from math import sqrt, log
from numba import njit


@njit("Tuple((int64[:], int64[:]))(int64[:], int64[:], int64, int64)", cache=True)
def envelope_structure(row, col, N, nblocks):
    r"""Envelope of the round-trip matrix with interleaved polarization blocks.

    The matrix index ``b*N + i`` (polarization block ``b``, radial index ``i``) is mapped to ``i*nblocks + b``.

    Parameters
    ----------
    row, col : ndarray
        1D arrays of ints, row and column index of the non-vanishing matrix elements
    N : int
        dimension of each polarization block
    nblocks : int
        number of polarization blocks

    Returns
    -------
    (ndarray, ndarray)
        first : first column index within the envelope for each row
        ptr : start of each row in the envelope storage, ``ptr[-1]`` is the total size

    """
    n = N * nblocks
    first = np.arange(n)
    for l in range(len(row)):
        lo = min(row[l], col[l])
        hi = max(row[l], col[l])
        for a in range(nblocks):
            r = hi * nblocks + a
            if lo * nblocks < first[r]:
                first[r] = lo * nblocks
    ptr = np.empty(n + 1, dtype=np.int64)
    ptr[0] = 0
    for i in range(n):
        ptr[i + 1] = ptr[i] + i - first[i] + 1
    return first, ptr


@njit("int64(int64[:], int64[:], int64, int64, int64)", cache=True)
def envelope_position(first, ptr, S, i, j):
    r"""Position of matrix element :math:`(i, j)` in the envelope storage."""
    if j < i:
        return ptr[i] + j - first[i]
    else:
        return S + ptr[j] + i - first[j]


@njit("boolean(float64[:,:], int64[:], int64[:], int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], int64, int64, float64[:], float64[:], float64[:])", cache=True)
def construct_roundtrip_finite_sparse(mat, first, ptr, row, col, dataTMTM, dataTETE, dataTMTE, dataTETM, N, M, w, rTM, rTE):
    r"""Construct the round-trip matrix in envelope storage.

        The matrix elements are identical to the ones of ``construct_roundtrip_finite``. Its derivatives are not stored,
        since they are obtained from the matrix elements by multiplication with :math:`-(\kappa_i+\kappa_j)` and
        :math:`(\kappa_i+\kappa_j)^2`, respectively.

        Parameters
        ----------
        mat : ndarray
            2D array of shape (3, 2*S), the round-trip matrix is written to ``mat[0]``
        first, ptr : ndarray
            envelope structure, see ``envelope_structure``
        row, col : ndarray
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : ndarray
            1D array of floats, matrix elements
        N, M : int
            positive, radial and angular discretization order
        w : ndarray
            weights of radial quadrature
        rTM, rTE : ndarray
            Fresnel reflection coefficients for TM and TE polarization

        Returns
        -------
        boolean
            True if the round-trip matrix is symmetric

    """
    S = ptr[-1]
    mat[0, :] = 0.
    sym = np.all(np.sign(rTM) == np.sign(rTM[0])) and np.all(np.sign(rTE) == np.sign(rTE[0]))
    for l in range(len(row)):
        i, j = row[l], col[l]
        # interleaved indices of TM and TE
        iTM, iTE, jTM, jTE = 2*i, 2*i+1, 2*j, 2*j+1
        weight = 1/M/2/np.pi*sqrt(w[i]*w[j])
        # the order of assignments follows construct_roundtrip_finite
        if sym:
            vTMTM = np.sign(rTM[i])*sqrt(rTM[i]*rTM[j])*weight*dataTMTM[l]
            vTETE = np.sign(rTE[i])*sqrt(rTE[i]*rTE[j])*weight*dataTETE[l]
            vTMTE = sqrt(-rTM[i]*rTE[j])*weight*dataTMTE[l]
            vTETM = sqrt(-rTE[i]*rTM[j])*weight*dataTETM[l]
            mat[0, envelope_position(first, ptr, S, iTM, jTM)] = vTMTM
            mat[0, envelope_position(first, ptr, S, jTM, iTM)] = vTMTM
            mat[0, envelope_position(first, ptr, S, iTE, jTE)] = vTETE
            mat[0, envelope_position(first, ptr, S, jTE, iTE)] = vTETE
            mat[0, envelope_position(first, ptr, S, iTM, jTE)] = vTMTE
            mat[0, envelope_position(first, ptr, S, jTE, iTM)] = vTMTE
            mat[0, envelope_position(first, ptr, S, iTE, jTM)] = vTETM
            mat[0, envelope_position(first, ptr, S, jTM, iTE)] = vTETM
        else:
            mat[0, envelope_position(first, ptr, S, iTM, jTM)] = weight*rTM[i]*dataTMTM[l]
            mat[0, envelope_position(first, ptr, S, jTM, iTM)] = weight*rTM[j]*dataTMTM[l]
            mat[0, envelope_position(first, ptr, S, iTE, jTE)] = weight*rTE[i]*dataTETE[l]
            mat[0, envelope_position(first, ptr, S, jTE, iTE)] = weight*rTE[j]*dataTETE[l]
            mat[0, envelope_position(first, ptr, S, iTM, jTE)] = weight*rTM[i]*dataTMTE[l]
            mat[0, envelope_position(first, ptr, S, jTE, iTM)] = weight*rTE[j]*dataTETM[l]
            mat[0, envelope_position(first, ptr, S, iTE, jTM)] = weight*rTE[i]*dataTETM[l]
            mat[0, envelope_position(first, ptr, S, jTM, iTE)] = weight*rTM[j]*dataTMTE[l]
    return sym


@njit("boolean(float64[:,:], int64[:], int64[:], int64[:], int64[:], float64[:], int64, float64[:], float64[:])", cache=True)
def construct_roundtrip_zero_sparse(mat, first, ptr, row, col, data, M, w, rp):
    r"""Construct the round-trip matrix of one polarization at zero frequency in envelope storage.

        The matrix elements are identical to the ones of ``construct_roundtrip_zero``.

        Parameters
        ----------
        mat : ndarray
            2D array of shape (3, 2*S), the round-trip matrix is written to ``mat[0]``
        first, ptr : ndarray
            envelope structure, see ``envelope_structure``
        row, col : ndarray
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : ndarray
            1D array of floats, matrix elements
        M : int
            angular discretization order
        w : ndarray
            weights of radial quadrature
        rp : ndarray
            Fresnel reflection coefficients for a given polarization

        Returns
        -------
        boolean
            True if the round-trip matrix is symmetric

    """
    S = ptr[-1]
    mat[0, :] = 0.
    sym = np.all(np.sign(rp) == np.sign(rp[0]))
    for l in range(len(data)):
        i, j = row[l], col[l]
        weight = 1/M/2/np.pi*sqrt(w[i]*w[j])
        if sym:
            v = np.sign(rp[i])*sqrt(rp[i]*rp[j])*weight*data[l]
            mat[0, envelope_position(first, ptr, S, i, j)] = v
            mat[0, envelope_position(first, ptr, S, j, i)] = v
        else:
            mat[0, envelope_position(first, ptr, S, i, j)] = weight*data[l]*rp[i]
            mat[0, envelope_position(first, ptr, S, j, i)] = weight*data[l]*rp[j]
    return sym


@njit("UniTuple(float64, 3)(float64[:,:], int64[:], int64[:], int64, boolean)", cache=True)
def envelope_lu(mat, first, ptr, order, sym):
    r"""Envelope LU decomposition of :math:`\mathcal{A}=1-\mathcal{M}` (without pivoting) together with its first two
        derivatives with respect to :math:`L`.

        Parameters
        ----------
        mat : ndarray
            2D array of shape (3, 2*S) containing :math:`\mathcal{A}`, :math:`\partial_L\mathcal{A}` and
            :math:`\partial^2_L\mathcal{A}` in envelope storage. It is overwritten by the decomposition.
        first, ptr : ndarray
            envelope structure, see ``envelope_structure``
        order : int
            0, 1 or 2 for the observables "energy", "force" and "forcegradient", respectively
        sym : boolean
            True if the matrix is symmetric

        Returns
        -------
        (float, float, float)
            same as ``compute_matrix_operations_sparse``

    """
    n = len(first)
    S = ptr[-1]
    a0 = mat[0]
    a1 = mat[1]
    a2 = mat[2]

    logdet = 0.
    dL_logdet = 0.
    d2L_logdet = 0.
    for i in range(n):
        fi = first[i]
        pi = ptr[i]
        for j in range(fi, i):
            fj = first[j]
            pj = ptr[j]
            kmin = max(fi, fj)
            # L[i, j] = (A[i, j] - sum_k L[i, k] U[k, j]) / U[j, j]
            p = pi + j - fi
            s0 = a0[p]
            s1 = a1[p]
            s2 = a2[p]
            for k in range(kmin, j):
                pl = pi + k - fi
                pu = S + pj + k - fj
                s0 -= a0[pl]*a0[pu]
                if order >= 1:
                    s1 -= a1[pl]*a0[pu] + a0[pl]*a1[pu]
                if order == 2:
                    s2 -= a2[pl]*a0[pu] + 2*a1[pl]*a1[pu] + a0[pl]*a2[pu]
            pd = S + pj + j - fj
            u0 = a0[pd]
            u1 = a1[pd]
            u2 = a2[pd]
            x0 = s0/u0
            x1 = (s1 - x0*u1)/u0
            x2 = (s2 - 2*x1*u1 - x0*u2)/u0
            # U[j, i] = A[j, i] - sum_k L[j, k] U[k, i]
            q = S + pi + j - fi
            if sym:
                # U[j, i] = L[i, j] U[j, j]
                a0[q] = s0
                a1[q] = s1
                a2[q] = s2
            else:
                t0 = a0[q]
                t1 = a1[q]
                t2 = a2[q]
                for k in range(kmin, j):
                    pl = pj + k - fj
                    pu = S + pi + k - fi
                    t0 -= a0[pl]*a0[pu]
                    if order >= 1:
                        t1 -= a1[pl]*a0[pu] + a0[pl]*a1[pu]
                    if order == 2:
                        t2 -= a2[pl]*a0[pu] + 2*a1[pl]*a1[pu] + a0[pl]*a2[pu]
                a0[q] = t0
                a1[q] = t1
                a2[q] = t2
            a0[p] = x0
            a1[p] = x1
            a2[p] = x2
        # U[i, i] = A[i, i] - sum_k L[i, k] U[k, i]
        pd = S + pi + i - fi
        s0 = a0[pd]
        s1 = a1[pd]
        s2 = a2[pd]
        for k in range(fi, i):
            pl = pi + k - fi
            pu = S + pi + k - fi
            s0 -= a0[pl]*a0[pu]
            if order >= 1:
                s1 -= a1[pl]*a0[pu] + a0[pl]*a1[pu]
            if order == 2:
                s2 -= a2[pl]*a0[pu] + 2*a1[pl]*a1[pu] + a0[pl]*a2[pu]
        a0[pd] = s0
        a1[pd] = s1
        a2[pd] = s2
        if s0 <= 0.:
            # 1-M is not positive definite, which should not happen for physical round-trip matrices
            return np.nan, np.nan, np.nan
        logdet += log(s0)
        dL_logdet += s1/s0
        d2L_logdet += s2/s0 - (s1/s0)**2

    if order == 0:
        return logdet, 0., 0.
    if order == 1:
        return logdet, -dL_logdet, 0.
    return logdet, -dL_logdet, -d2L_logdet




@njit("UniTuple(float64, 3)(float64[:,:], int64[:], int64[:], float64[:], int64, boolean)", cache=True)
def envelope_lu_logdet(mat, first, ptr, kappa, order, sym):
    r"""Computes the quantities of ``compute_matrix_operations_sparse`` exactly, i.e. without trace approximation.

        Parameters and return values are the same as for ``compute_matrix_operations_sparse``. The derivatives of the
        matrix elements are :math:`\partial_L\mathcal{M}_{ij} = -(\kappa_i+\kappa_j)\mathcal{M}_{ij}` and
        :math:`\partial_L^2\mathcal{M}_{ij} = (\kappa_i+\kappa_j)^2\mathcal{M}_{ij}`.

    """
    n = len(first)
    S = ptr[-1]
    a0 = mat[0]
    a1 = mat[1]
    a2 = mat[2]

    # A = 1 - M, A' = -dL_M and A'' = -d2L_M
    for i in range(n):
        for j in range(first[i], i+1):
            factor = kappa[i] + kappa[j]
            # (i, j) in lower part, (j, i) in upper part, the diagonal is stored in the upper part only
            lo = ptr[i] + j - first[i]
            up = S + ptr[i] + j - first[i]
            for p in (lo, up):
                m = a0[p]
                a0[p] = -m
                a1[p] = factor*m
                a2[p] = -factor**2*m
        a0[S + ptr[i+1] - 1] += 1.

    return envelope_lu(mat, first, ptr, order, sym)


@njit("UniTuple(float64, 3)(float64[:,:], int64[:], int64[:], int64, boolean)", cache=True)
def envelope_lu_logdet_explicit(mat, first, ptr, order, sym):
    r"""Same as ``envelope_lu_logdet``, but ``mat[0]``, ``mat[1]`` and ``mat[2]`` contain :math:`\mathcal{M}`,
        :math:`\partial_L\mathcal{M}` and :math:`\partial^2_L\mathcal{M}` explicitly in envelope storage.

    """
    n = len(first)
    S = ptr[-1]
    for l in range(2*S):
        mat[0, l] = -mat[0, l]
        mat[1, l] = -mat[1, l]
        mat[2, l] = -mat[2, l]
    for i in range(n):
        mat[0, S + ptr[i+1] - 1] += 1.
    return envelope_lu(mat, first, ptr, order, sym)


@njit("UniTuple(float64, 3)(float64[:,:], int64[:], int64[:], float64[:], int64, boolean)", cache=True)
def compute_matrix_operations_sparse(mat, first, ptr, kappa, order, sym):
    r"""Sparse counterpart of ``compute_matrix_operations``.

        Computes the 3-tuple

        .. math::
            \begin{aligned}
            \log\det(1-\mathcal{M})\,,\\
            \mathrm{tr}\left[\frac{\partial_L\mathcal{M}}{1-\mathcal{M}}\right]\,,\\
            \mathrm{tr}\left[\frac{\partial_L^2\mathcal{M}}{1-\mathcal{M}} + \left(\frac{\partial_L\mathcal{M}}{1-\mathcal{M}}\right)^2\right]\,.
            \end{aligned}

        The round-trip matrix is LU decomposed (without pivoting) within its envelope,
        :math:`1-\mathcal{M} = \mathcal{L}\mathcal{U}` with :math:`\mathrm{diag}(\mathcal{L})=1`. The decomposition is
        differentiated twice with respect to :math:`L` alongside, such that

        .. math::
            \begin{aligned}
            \log\det(1-\mathcal{M}) &= \sum_i \log \mathcal{U}_{ii}\,,\\
            \partial_L\log\det(1-\mathcal{M}) &= \sum_i \frac{\mathcal{U}'_{ii}}{\mathcal{U}_{ii}}\,,\\
            \partial_L^2\log\det(1-\mathcal{M}) &= \sum_i \left[\frac{\mathcal{U}''_{ii}}{\mathcal{U}_{ii}} - \left(\frac{\mathcal{U}'_{ii}}{\mathcal{U}_{ii}}\right)^2\right]\,.
            \end{aligned}

        Parameters
        ----------
        mat : ndarray
            2D array of shape (3, 2*S), ``mat[0]`` contains the round-trip matrix in envelope storage, ``mat[1]`` and
            ``mat[2]`` are used as workspace. All rows are overwritten.
        first, ptr : ndarray
            envelope structure, see ``envelope_structure``
        kappa : ndarray
            imaginary z-component of the wave vectors for each (interleaved) matrix index
        order : int
            0, 1 or 2 for the observables "energy", "force" and "forcegradient", respectively
        sym : boolean
            True if the round-trip matrix is symmetric

        Returns
        -------
        (float, float, float)

    """
    n = len(first)
    S = ptr[-1]
    a0 = mat[0]
    a1 = mat[1]
    a2 = mat[2]

    # trace and norm of round-trip matrix
    norm2_mat = 0.
    for l in range(2*S):
        norm2_mat += a0[l]**2
    tr_mat = 0.
    tr_dL_mat = 0.
    tr_d2L_mat = 0.
    for i in range(n):
        m_ii = a0[S + ptr[i+1] - 1]
        tr_mat += m_ii
        tr_dL_mat += -2*kappa[i]*m_ii
        tr_d2L_mat += 4*kappa[i]**2*m_ii
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
    return envelope_lu_logdet(mat, first, ptr, kappa, order, sym)


def observable_order(observable):
    r"""Number of derivatives with respect to :math:`L` needed for the given observable."""
    if observable == "energy":
        return 0
    elif observable == "force":
        return 1
    elif observable == "forcegradient":
        return 2
    else:
        raise ValueError
