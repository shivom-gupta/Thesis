from ising import monte_carlo
import numpy as np


size = 1_000
n_steps = 100_000
beta = 1.0
J = 1.0
h = 0.0

def calculate_acf(size=size, n_steps = n_steps, beta=beta, J=J, h=h):
    """Calculate the autocorrelation function using fast fourier transform"""
    configuration = np.random.choice(np.array([-1, 1]), size)
    current_energy, average_spins = monte_carlo(n_steps, beta, J, h, configuration)
    x = average_spins - average_spins.mean()
    N = float(len(x))
    pow2 = int(2**np.ceil(np.log2(len(x))))
    x_new = np.zeros(pow2,float)
    x_new[:len(x)] = x
    FT = np.fft.fft(x_new)
    acf = (np.fft.ifft(FT*np.conjugate(FT)).real)/N
    acf /= acf.max()
    acf = acf[:int(len(acf))//2]
    return acf[::size]