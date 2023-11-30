import numpy as np

def corr_4(configurations, r):
    R, L = configurations.shape

    shifted_indices = ((np.arange(L) + r) % L).astype(int)

    sum_prod = np.sum(configurations * configurations[:, shifted_indices], axis=1)/L
    sum_sq = (np.sum(configurations, axis=1)/L)**2
    
    return np.sum((sum_prod - sum_sq)/(1-sum_sq)) / R