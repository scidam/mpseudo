'''
Parallel computation of pseudospecta of a square matrix by its definition.

Author: Dmitry E. Kislov
E-mail: kislov@easydan.com
Date: 25 Nov. 2015
'''

from __future__ import print_function

import multiprocessing
import warnings
import numpy as np
import itertools
from lib2to3.pgen2.token import LBRACE


__all__ = ['gersgorin_bounds', 'pseudo', 'eigen_bounds']


def gersgorin_bounds(A):
    '''Localize eigenvalues of a matrix in a complex plane.

    The function uses well known S.A. Gersgorin (1931) theorem about
    matrix eigenvalues localization: the eigenvalues lie in the closed region
    of the complex plane consisting of all the rings:

    :param A: the input matrix as a ``numpy.array`` or 2D list with ``A.shape==(n, m)``.
              For rectangular matrices bounding box is computed for the largest square submatrix with shape min(n,m) x min(n,m).

    .. math::
       |z-a_{kk}|\leq R_k - |a_{kk}|, R_k=\sum\limits_{i=1}^n|a_{ki}|
    '''

    n, m = np.shape(A)
    if n <= m:
        B = A[:n, :n]
    else:
        B = A[:m, :m]
        n = m
    _A = np.abs(B)
    Rk = np.sum(_A, axis=1)
    radii = [Rk[k] - _A[k, k] for k in range(n)]
    rbounds = [B[k, k].real - radii[k] for k in range(n)]
    rbounds.extend([B[k, k].real + radii[k] for k in range(n)])
    cbounds = [B[k, k].imag - radii[k] for k in range(n)]
    cbounds.extend([B[k, k].imag + radii[k] for k in range(n)])
    return [np.min(rbounds), np.max(rbounds), np.min(cbounds), np.max(cbounds)]


def eigen_bounds(A, percent=0.1):
    '''Build pseudospectra bounds on matrix eigenvalues

    :param A: the input matrix as a ``numpy.array`` or 2D list with ``A.shape==(n, m)``.
              For rectangular matrices bounding box is computed for the largest square
              submatrix with shape min(n,m) x min(n,m).

    :param percent: an indent for bounding box construction (default is 0.1).
                    Bound values are computed as extreme egienvalues +/- percent*residual,
                    where residual is a maximal distance between all possible 
                    pairs of eigenvalues.
    '''

    n, m = np.shape(A)
    if n <= m:
        B = A[:n, :n]
    else:
        B = A[:m, :m]
    eigvals = np.linalg.eigvals(B)
    reals = np.real(eigvals)
    imags = np.imag(eigvals)
    lbr = np.min(reals)
    ubr = np.max(reals)
    lbc = np.min(imags)
    ubc = np.max(imags)
    residual = np.max([abs(x-y) for x, y in itertools.combinations(eigvals, 2)])
    return [lbr-percent*residual,
            ubr+percent*residual,
            lbc-percent*residual,
            ubc+percent*residual]


def _safe_bbox(bbox, A):
    '''converts bbox array to the array of type [float, float, float, float].
    '''
    assert len(bbox) >= 4, "Length of bbox should be equal or greater 4."
    try:
        res = [float(bbox[i]) for i in range(4)]
    except (TypeError, ValueError):
        warnings.warn('Invalid bbox-array. Gershgorin circles will be used.',
                      RuntimeWarning)
        res = gersgorin_bounds(A)
    return res


def _calc_pseudo(A, x, y, n, m):
    def ff(x, y): return np.linalg.svd((x+(1j)*y)*np.eye(n, m) - A,
                                       compute_uv=False)[-1]
    return [ff(_x, _y) for _x, _y in zip(x, y)]


def _pseudo_worker(*args):
    digits = args[0][1]
    result = None
    if digits > 15:
        try:
            import mpmath as mp
            mp.mp.dps = int(digits)
            def ff(x, y): return np.float128(mp.svd(mp.matrix((x+(1j) * y) * np.eye(args[0][-2], args[0][-1]) - args[0][2]), compute_uv=False).tolist()[-1][0])
            result = (args[0][0],
                      [ff(x, y) for x, y in zip(args[0][3], args[0][4])])
        except ImportError:
            warnings.warn('Cannot import mpmath module.\
Precision of computations will be reduced to default value (15 digits).',
                          RuntimeWarning)
    if not result:
        result = (args[0][0], _calc_pseudo(*args[0][2:]))
    return result


def pseudo(A, bbox=gersgorin_bounds, ppd=100, ncpu=1, digits=15):
    ''' calculates pseudospectra of a matrix A via classical grid method.

        .. note::

           It is assumed that :math:`\\varepsilon`-pseudospectra of a matrix is defined as :math:`\\sigma_{\\min}(A-\\lambda I)\\leq\\varepsion\\|A\\|`.

        :param A:  the input matrix as a ``numpy.array`` or 2D list with ``A.shape==(n,m)``.
        :param bbox: the pseudospectra bounding box, a list of size 4 [MIN RE, MAX RE, MIN IM, MAX IM] or a function.
                     (default value: gersgorin_bounds - a function computes bounding box via Gershgorin circle theorem)
        :param ppd: points per dimension, default is 100, i.e. total grid size is 100x100 = 10000.
        :param ncpu: the number of cpu used for calculation, default is 1. If the number of cpu is greater the number of cores, it will be reduced to ncores-1.
        :param digits: the number of digits used for minimal singular value computation. When digits>15, it is assumed that package mpmath is installed.
                       If not, default (double) precision for all calculations will be used. If mpmath is available, the minimal singular value of :math:`(A-\\lambda I)` will
                       be computed with full precision (up to defined value of digits), but returned singular value will be presented as np.float128.
        :type A: numpy.array, 2D list of shape (n,m)
        :type bbox: a list or a function returning list
        :type ppd: int
        :type ncpu: int
        :type digits: int
        :returns: numpy array of epsilon-pseudospectrum values with shape (ppd, ppd), X and Y 2D-arrays where each pseudospectra value was computed (X, Y - are created via numpy.meshgrid function).

        :Example:

        >>> from mpseudo import pseudo
        >>> A = [[-9, 11, -21, 63, -252],
                [70, -69, 141, -421, 1684],
                [-575, 575, -1149, 3451, -13801],
                [3891, -3891, 7782, -23345, 93365],
                [1024, -1024, 2048, -6144, 24572]]
        >>> psa, X, Y = pseudo(A, ncpu=3, digits=100, ppd=100, bbox=[-0.05,0.05,-0.05,0.05])

        You can use contourf function from matplotlib to plot pseudospectra:
        >>> from pylab import *
        >>> contourf(X, Y, psa)
        >>> show()
    '''
    n, m = np.shape(A)
    assert max(n, m) > 1, 'Matricies of size 1x1 not allowed.'
    if hasattr(bbox, '__iter__'):
        bounds = _safe_bbox(bbox, A)
    elif hasattr(bbox, '__call__'):
        try:
            bounds = _safe_bbox(bbox(A), A)
        except:
            bounds = gersgorin_bounds(A)
            warnings.warn('Invalid bbox-function.\
 Gershgorin circles will be used.', RuntimeWarning)
    else:
        bounds = gersgorin_bounds(A)

    _nc = multiprocessing.cpu_count()
    if not ncpu:
        ncpu = 1
        warnings.warn('The number of cpu-cores is not defined.\
 Default (ncpu = 1) value will be used.', RuntimeWarning)
    elif ncpu >= _nc and _nc > 1:
        ncpu = _nc - 1
    else:
        ncpu = 1
    x = np.linspace(bounds[0], bounds[1], ppd)
    y = np.linspace(bounds[2], bounds[3], ppd)
    X, Y = np.meshgrid(x, y)
    yars = np.array_split(Y.ravel(), ncpu)
    xars = np.array_split(X.ravel(), ncpu)
    pool = multiprocessing.Pool(processes=ncpu)
    results = pool.map(_pseudo_worker,
                       [(i, digits, A, xars[i], yars[i], n, m)
                        for i in range(ncpu)]
                       )
    pool.close()
    pool.join()
    pseudo_res = []
    for i in range(ncpu):
        pseudo_res.extend(list(filter(lambda x: x[0] == i, results))[0][1])
    return (np.reshape(pseudo_res, (ppd, ppd))/np.linalg.norm(A), X, Y)


if __name__ == '__main__':
    A = [[-9, 11, -21, 63, -252],
         [70, -69, 141, -421, 1684],
         [-575, 575, -1149, 3451, -13801],
         [3891, -3891, 7782, -23345, 93365],
         [1024, -1024, 2048, -6144, 24572]]
    psa, X, Y = pseudo(A, ncpu=None, digits=100,
                       ppd=100, bbox=[-0.05, 0.05, -0.05, 0.05]
                       )
    print('Pseudospectra of the matrix A ' +
          str(A) + ' was computed successfully.')
