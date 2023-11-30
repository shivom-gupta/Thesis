import numpy as np
from corr_1 import corr_1

def corr_5(configurations, r):
    R, L = configurations.shape
    cr_1 = corr_1(configurations, r)
    sum_sq = (np.sum(np.sum(configurations, axis=1)/L)/R)**2
    
    return cr_1 - sum_sq