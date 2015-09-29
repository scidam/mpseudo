'''
Parallel computation of pseudospecta of a square matrix by its definition.

Author: Dmitry E. Kislov
E-mail: kislov@easydan.com
Date: 29 Sept. 2015
'''

import numpy as np
import multiprocessing
import warnings
from __future__ import print_function



def gersgorin_bounds(A):
    '''Localize eigenvalues of a matrix in a complex plane.

    The function uses well known S.A. Gersgorin (1931) theorem about
    matrix eigenvalues localization: the eigenvalues lie in the closed region
    of the complex plane consisting of all the rings:

    :param A: the input matrix as a ``numpy.array`` or 2D list with ``A.shape==(n,n)``.

    .. math::
       |z-a_{kk}|\leq R_k - |a_{kk}|, R_k=\sum\limits_{i=1}^n|a_{ki}|

    '''
    n = np.shape(A)[0]
    _A = np.abs(A)
    Rk = np.sum(_A, axis=1)
    rbounds = [-Rk[k] + 2.0*_A[k, k] for k in range(n)]
    rbounds.extend([Rk[k] for k in range(n)])
    return [np.min(rbounds), np.max(rbounds), np.min(rbounds), np.max(rbounds)]


def _calc_pseudo(A, x, y, n):
    ff = lambda x, y: np.linalg.svd((x+(1j)*y)*np.eye(n) - A, compute_uv=False)[-1]
    return [ff(x, y) for x, y in zip(x, y)]


def _pseudo_worker(*args):
    digits = args[0][1]
    result = None
    if digits > 15:
        try:
            import mpmath as mp
            mp.mp.dps = int(digits)
            ff = lambda x, y: np.float128(mp.svd(mp.matrix((x+(1j)*y)*np.eye(args[0][-1]) - args[0][2]), compute_uv=False).tolist()[-1][0])
            result = (args[0][0], [ff(x, y) for x, y in zip(args[0][3], args[0][4])])
        except ImportError:
            warnings.warn('Cannot import mpmath module. Precision of computations will be reduced to default value (15 digits).', RuntimeWarning)
    if not result:
        result = (args[0][0], _calc_pseudo(*args[0][2:]))
    return result


def pseudo(A, bbox=gersgorin_bounds, ppd=100, ncpu=1, digits=15):
    ''' calculates pseudospectrum of a square matrix A via classical grid method.

        :param A:  the input matrix as a ``numpy.array`` or 2D list with ``A.shape==(n,n)``.
        :param bbox: the pseudospectra bounding box, a list of size 4 [MIN RE, MAX RE, MIN IM, MAX IM] or a function.
                     (default value: gersgorin_bounds - a function computes bounding box via Gershgorin circle theorem.
        :param ppd: points per dimension, default is 100. So, pseudospectra grid size is 100x100 = 10000.
        :param ncpu: the number of cpu to use for calculation, default is 1. If the number of cpu exeeds the number of cores, it will be reduced to ncores-1.
        :param digits: the number of digits to use when minimal singular value is computed. It is assumed for digits>15 that package mpmath is installed.
                       If not, default (double) precision for all calculations will be used. If mpmath is available, the minimal singular value of :math:`(A-\\lambda I)` will
                       be computed with full precision (up to defined digits of precision), but returned singular value will be presented as np.float128.
        :type A: numpy.array, 2D list of shape (n,n)
        :type bbox: a list or a function returning list
        :type ppd: int
        :type ncpy: int
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

    if hasattr(bbox, '__iter__'):
        bounds = bbox
    elif hasattr(bbox, '__call__'):
        bounds = bbox(A)
    else:
        bounds = gersgorin_bounds(A)
    try:
        if isinstance(bounds[0]*bounds[1]*bounds[2]*bounds[3], (np.floating, float)):
            pass
        else:
            bounds = gersgorin_bounds(A)
    except TypeError:
        warnings.warn('Boungin box (bbox) should be a function, returning array of size 4 or an array of size 4. Gershgorin estimation used.', RuntimeWarning)
        bounds = gersgorin_bounds(A)
    except:
        warnings.warn('Something wrong in boungin box (bbox) variable. Gershgorin estimation used.', RuntimeWarning)
    _nc = multiprocessing.cpu_count()
    if not ncpu:
        ncpu = 1
        warnings.warn('The number of cpu-cores is not defined. Default (ncpu = 1) value used.', RuntimeWarning)
    elif ncpu >= _nc and _nc > 1:
        ncpu = _nc - 1
    else:
        ncpu = 1
    x = np.linspace(bounds[0], bounds[1], ppd)
    y = np.linspace(bounds[2], bounds[3], ppd)
    X, Y = np.meshgrid(x, y)
    yars = np.array_split(Y.ravel(), ncpu)
    xars = np.array_split(X.ravel(), ncpu)
    n = np.shape(A)[0]
    pool = multiprocessing.Pool(processes=ncpu)
    results = pool.map(_pseudo_worker, [(i, digits, A, xars[i], yars[i], n) for i in range(ncpu)])
    pool.close()
    pool.join()
    pseudo_res = []
    for i in range(ncpu):
        pseudo_res.extend(filter(lambda x: x[0] == i, results)[0][1])
    return (np.reshape(pseudo_res, (ppd, ppd))/np.linalg.norm(A), X, Y)


if __name__ == '__main__':
    A = [[-9, 11, -21, 63, -252],
         [70, -69, 141, -421, 1684],
         [-575, 575, -1149, 3451, -13801],
         [3891, -3891, 7782, -23345, 93365],
         [1024, -1024, 2048, -6144, 24572]]
    psa, X, Y = pseudo(A, ncpu=None, digits=100, ppd=100, bbox=[-0.05, 0.05, -0.05, 0.05])
    print('Pseudospectra of the matrix A '+str(A)+' was computed successfully.')
