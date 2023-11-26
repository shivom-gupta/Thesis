import numpy as np


def cr_1(configurations, r):
    c = 0
    R = configurations.shape[0]
    L = configurations.shape[1]
    for k in range(R):
        for i in range(L):
            c += np.sum(configurations[k, i] * configurations[k, int((i + r) % L)])
    return c / (R * L)

def cr_1_optimized(configurations, r):
    R, L = configurations.shape

    shifted_indices = ((np.arange(L) + r) % L).astype(int)

    product_sum = np.sum(configurations * configurations[:, shifted_indices], axis=1)
    return np.sum(product_sum) / (R * L)