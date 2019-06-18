import numpy as np


def r2z(rmat):
    """
    fisher z-transforms the correlation matrix, first setting diagonal to 0
    """
    np.fill_diagonal(rmat, 0)
    zmat = np.log((1.0+rmat) / (1.0-rmat)) / 2
    return zmat


def z2r(zmat):
    """
    applies inverse of fisher z-transform to recover the correlation matrix
    """
    rmat = (np.exp(2*zmat) - 1) / (np.exp(2 * zmat) + 1)
    np.fill_diagonal(rmat, 1)
    return rmat


def get_corrs(traces_stack):
    samps, tps, regs = traces_stack.shape
    z_distro_stack = np.zeros((samps, int((regs**2 - regs)/2)))

    for s in range(samps):
        traces = traces_stack[s, :, :]
        normed = (traces - traces.mean(axis=0)) / traces.std(axis=0)
        rmat = np.dot(normed.T, normed) / tps
        zmat = r2z(rmat)
        z_distro_stack[s, :] = zmat[np.nonzero(np.triu(zmat, k=1))]

    return z_distro_stack
