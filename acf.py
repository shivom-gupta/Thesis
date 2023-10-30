import numpy as np
from numba import njit, prange

@njit(parallel=True)
def acf(series:np.ndarray, kmax:int = 1000):
    Oi = np.mean(series)
    Oi2 = np.mean(series**2)
    Oipk = np.empty((kmax+1))
    Oipk[0] = Oi2
    for k in prange(1,kmax+1):
        Oipk[k] = np.mean(series[:-k] * series[k:])
    return (Oipk - Oi**2) / (Oi2 - Oi**2)
