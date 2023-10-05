from ising import monte_carlo_noconfig
import numpy as np
import concurrent.futures
from scipy.signal import correlate

def calculate_acf(size, n_steps, beta, J, h):
    configuration = np.random.choice([-1,1], size)
    energies, magnetizations = monte_carlo_noconfig(n_steps, beta, J, h, configuration)
    acf = correlate(magnetizations, magnetizations, 'full', 'fft')
    acf = acf[len(acf)//2:][::size]
    acf /= acf.max()
    return acf

def acf_multi(n_sim, size, n_steps, beta, J, h):
    corrs = []
    with concurrent.futures.ProcessPoolExecutor() as p:
        futures = [p.submit(calculate_acf,size, n_steps, beta, J, h) for _ in range(n_sim)]
        for future in concurrent.futures.wait(futures)[0]:
            corrs.append(future.result())
                
    return np.array(corrs).mean(axis=0)
