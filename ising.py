import numpy as np
from numba import njit
import sys
sys.setrecursionlimit(10**9)

@njit
def energy_ising(size, configuration, J, h):
    energy = 0.0
    for i in range(size):
        spini = configuration[i]
        ip1 = (i + 1) % size
        spinip1 = configuration[ip1]
        energy = energy - J * (spini * spinip1) - h * spini
    return energy/size

@njit
def energy_difference(configuration, size, spin_to_change, J, h):
    s = configuration[spin_to_change]
    sleft = configuration[(spin_to_change - 1) % size]
    sright = configuration[(spin_to_change + 1) % size]
    return 2 * h * s + 2 * J * s * (sleft + sright)

@njit
def monte_carlo(size, n_sweeps, beta, J, h, configuration, tau, R_max, batch_size=2_500_000):
    if tau is not None:
        if R_max is None:
            R_max = int(int(n_sweeps-20*tau)/int(4*tau))
        else:
            R_max = R_max
        n_sweeps = int(4 * tau * R_max + 20 * tau)
        energies = np.empty(R_max, dtype=np.float32)
        magnetizations = np.empty(R_max, dtype=np.float32)
        configurations = np.empty((R_max, size), dtype=np.bool_)    
    else:
        energies = np.empty(n_sweeps, dtype=np.float32)
        magnetizations = np.empty(n_sweeps, dtype=np.float32)
        configurations = np.empty((n_sweeps, size), dtype=np.bool_)
    current_energy = energy_ising(size, configuration, J, h)
    current_spin = configuration.mean()
    exponentials = np.exp(-beta * np.array([-4, -2, 0, 2, 4]))
    batch_size = min(batch_size, n_sweeps)
    rs = np.random.random((batch_size, size)).astype(np.float32)
    spins_to_change = np.random.randint(0, size, (batch_size, size)).astype(np.int16)
    for sweep in range(n_sweeps):
        if sweep % (5*batch_size) == 0:
            rs = np.random.random((batch_size, size)).astype(np.float32)
            spins_to_change = np.random.randint(0, size, (batch_size, size)).astype(np.int16)
        for spin in range(size):
            spin_to_change = spins_to_change[sweep%batch_size, spin]
            dE = energy_difference(configuration, size, spin_to_change, J, h)
            if dE in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                ex = exponentials[int(dE/2) + 2]
            else:
                ex = np.exp(-beta * dE)
            if rs[sweep%batch_size, spin] < ex:
                configuration[spin_to_change] *= -1
                current_energy += dE/size
                current_spin += 2 * configuration[spin_to_change]/size
        if tau is None:
            energies[sweep] = current_energy
            magnetizations[sweep] = current_spin
            configurations[sweep] = configuration > 0
        else:
            output_loc = int(int(sweep-20*tau)/int(4*tau))
            if output_loc <= R_max:
                if output_loc >= 0:
                    energies[output_loc] = current_energy
                    magnetizations[output_loc] = current_spin
                    configurations[output_loc] = configuration > 0
                else:
                    continue
            else:
                break

    return configurations, energies, magnetizations

def exact_energy(beta, J, N):
    return -N*J*np.tanh(beta*J)

def exact_magnetization(beta, J, h):
    return (np.sinh(beta * h))/np.sqrt(np.exp(2*beta*J)*(np.sinh(2*beta*J))**2 + np.exp(-2*beta*J))

def corr_analytical(beta, J, L, r):
    return (((np.tanh(beta*J))**r + (np.tanh(beta*J))**(L-r))/(1 + (np.tanh(beta*J))**L)).astype(np.half)

if __name__ == "__main__":
    size = 1000
    n_sweeps = 3_000_000
    beta = 1
    J = 1.0
    h = 0.0

    configuration = np.random.choice([-1, 1], size)
    configurations, energies, magnetizations = monte_carlo(size, n_sweeps, beta, J, h, configuration, tau=600, R_max=3000)
    print(configurations.shape, magnetizations.shape)