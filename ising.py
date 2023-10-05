import numpy as np
import numba

@numba.jit(nopython = True)
def energy_ising(J, h, configuration):
    num_spins = len(configuration)
    energy = 0.0
    for i in range(num_spins):
        spini = configuration[i]
        ip1 = (i + 1) % num_spins
        spinip1 = configuration[ip1]
        energy = energy - J * (spini * spinip1) - h * spini
    return energy

@numba.jit(nopython = True)
def energy_difference(J, h, spin_to_change, configuration, size):
    s = configuration[spin_to_change]
    sleft = configuration[(spin_to_change - 1) % size]
    sright = configuration[(spin_to_change + 1) % size]
    return 2 * h * s + 2 * J * s * (sleft + sright)

@numba.jit(nopython = True)
def monte_carlo_config(n_steps, beta, J, h, configuration):
    size = len(configuration)
    energies = np.zeros(n_steps)
    magnetizations = np.zeros(n_steps)
    configurations = np.zeros((n_steps, size))
    current_energy = energy_ising(J, h, configuration)
    current_spin = configuration.mean()
    spins_to_change = np.random.randint(0, size, n_steps)
    for step in range(n_steps):
        spin_to_change = spins_to_change[step]
        dE = energy_difference(J, h, spin_to_change, configuration, size)

        r = np.random.random()
        if r < np.exp(-beta * dE):
            configuration[spin_to_change] *= -1
            current_energy += dE
            current_spin += 2 * configuration[spin_to_change]/size
        
        energies[step] = current_energy
        magnetizations[step] = current_spin
        configurations[step] = configuration 
    return energies, magnetizations, configurations

@numba.jit(nopython = True)
def monte_carlo_noconfig(n_steps, beta, J, h, configuration):
    size = len(configuration)
    energies = np.zeros(n_steps)
    magnetizations = np.zeros(n_steps)
    current_energy = energy_ising(J, h, configuration)
    current_spin = configuration.mean()
    spins_to_change = np.random.randint(0, size, n_steps)
    for step in range(n_steps):
        spin_to_change = spins_to_change[step]
        dE = energy_difference(J, h, spin_to_change, configuration, size)

        r = np.random.random()
        if r < np.exp(-beta * dE):
            configuration[spin_to_change] *= -1
            current_energy += dE
            current_spin += 2 * configuration[spin_to_change]/size
        
        energies[step] = current_energy
        magnetizations[step] = current_spin
    return energies, magnetizations

def monte_carlo(n_steps, beta, J, h, configuration, return_configurations=False):
    if return_configurations:
        return monte_carlo_config(n_steps, beta, J, h, configuration)
    else:
        return monte_carlo_noconfig(n_steps, beta, J, h, configuration)
    

def exact_energy(beta, J, N):
    return -(N-1)*J*np.tanh(beta*J)

def free_energy(beta, J, h):
    return -J - (np.log(np.cosh(beta*h) + np.sqrt(np.power(np.cosh(beta*h), 2) - 2*np.exp(-2*beta*J)*np.sinh(2*beta*J))))/beta

def exact_magnetization(beta, J, h):
    return (np.sinh(beta * h))/(np.sqrt(np.cosh(beta * h)**2 - (2*np.exp(-2*beta*J)*np.sinh(2*beta*J))))

if __name__ == "__main__":
    size = 1000
    n_steps = 100000*size
    beta = 1.0
    J = 1.0
    h = 0.0

    configuration = np.random.choice([-1, 1], size)
    energies, magnetizations = monte_carlo(n_steps, beta, J, h, configuration)
    print(energies.shape, magnetizations.shape)