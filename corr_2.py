import numpy as np
from corr_1 import corr_1

def corr_2(configurations, r):
    R, L = configurations.shape

    shifted_indices = ((np.arange(L) + r) % L).astype(int)

    product_sum = (np.sum(configurations, axis=0)/R) * (np.sum(configurations[:, shifted_indices], axis=0)/R)
    
    second_term = np.sum(product_sum) / L
    cr_1 = corr_1(configurations, r)
    
    return cr_1 - second_term