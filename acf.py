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
        if Oipk[k] < 1e-10:
            break
    denom = Oi2 - np.square(Oi)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    corr = (Oipk - np.square(Oi))/(denom)
    corr = corr[corr > 0]
    corr = corr[corr < 1]
    return corr

def calculate_tau(series:np.ndarray):
    corr = acf(series)
    tau = 1/2
    x = np.arange(corr.shape[0])
    for i in x:
        tau += corr[i]
        if i >= 6 * tau:
            break
        else:
            continue
    return tau
