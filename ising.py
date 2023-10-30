import numpy as np
from numba import njit

def get_configuration(size):
    return np.random.choice([-1, 1], size)

@njit
def energy_ising(J, h, configuration):
    num_spins = len(configuration)
    energy = 0.0
    for i in range(num_spins):
        spini = configuration[i]
        ip1 = (i + 1) % num_spins
        spinip1 = configuration[ip1]
        energy = energy - J * (spini * spinip1) - h * spini
    return energy

@njit
def energy_difference(J, h, spin_to_change, configuration, size):
    s = configuration[spin_to_change]
    sleft = configuration[(spin_to_change - 1) % size]
    sright = configuration[(spin_to_change + 1) % size]
    return 2 * h * s + 2 * J * s * (sleft + sright)

@njit
def monte_carlo_config(n_sweeps, beta, J, h, configuration):
    size = len(configuration)
    energies = np.zeros(n_sweeps, dtype=np.float32)
    magnetizations = np.zeros(n_sweeps, dtype=np.float32)
    configurations = np.zeros((n_sweeps, size), dtype=np.float32)
    current_energy = energy_ising(J, h, configuration)
    current_spin = configuration.mean()
    spins_to_change = np.random.randint(0, size, (n_sweeps, size))
    
    for sweep in range(n_sweeps):
        for spin in range(size):
            spin_to_change = spins_to_change[sweep, spin]
            dE = energy_difference(J, h, spin_to_change, configuration, size)

            r = np.random.random()
            if r < np.exp(-beta * dE):
                configuration[spin_to_change] *= -1
                current_energy += dE
                current_spin += 2 * configuration[spin_to_change]/size
            
        energies[sweep] = current_energy
        magnetizations[sweep] = current_spin
        configurations[sweep] = configuration 
    return energies, magnetizations, configurations

@njit
def monte_carlo_noconfig(n_sweeps, beta, J, h, configuration):
    size = len(configuration)
    energies = np.zeros(n_sweeps, dtype=np.float32)
    magnetizations = np.zeros(n_sweeps, dtype=np.float32)
    current_energy = energy_ising(J, h, configuration)
    current_spin = configuration.mean()
    spins_to_change = np.random.randint(0, size, (n_sweeps, size))
    for sweep in range(n_sweeps):
        for spin in range(size):
            spin_to_change = spins_to_change[sweep, spin]
            dE = energy_difference(J, h, spin_to_change, configuration, size)

            r = np.random.random()
            if r < np.exp(-beta * dE):
                configuration[spin_to_change] *= -1
                current_energy += dE
                current_spin += 2 * configuration[spin_to_change]/size
        energies[sweep] = current_energy
        magnetizations[sweep] = current_spin
    return energies, magnetizations

def exact_energy(beta, J, N):
    return -N*J*np.tanh(beta*J)

def exact_magnetization(beta, J, h):
    return (np.sinh(beta * h))/np.sqrt(np.exp(2*beta*J)*(np.sinh(2*beta*J))**2 + np.exp(-2*beta*J))

if __name__ == "__main__":
    size = 1000
    n_sweeps = 100000
    beta = 1
    J = 1.0
    h = 0.0

    configuration = np.random.choice([-1, 1], size)
    energies, magnetizations = monte_carlo_noconfig(n_sweeps, beta, J, h, configuration)
    print(energies.shape, magnetizations.shape)