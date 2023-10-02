import numpy as np
import matplotlib.pyplot as plt
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
    average_spins = np.zeros(n_steps)
    configurations = np.zeros((n_steps, size))
    current_energy = energy_ising(J, h, configuration)

    for step in range(n_steps):
        spin_to_change = np.random.randint(size)
        dE = energy_difference(J, h, spin_to_change, configuration, size)

        r = np.random.random()
        if r < np.exp(-beta * dE):
            configuration[spin_to_change] *= -1
            current_energy += dE
            
        average_spins[step] = configuration.mean()
        configurations[step] = configuration 
    return current_energy, average_spins, configurations

@numba.jit(nopython = True)
def monte_carlo_noconfig(n_steps, beta, J, h, configuration):
    size = len(configuration)
    average_spins = np.zeros(n_steps)
    current_energy = energy_ising(J, h, configuration)

    for step in range(n_steps):
        spin_to_change = np.random.randint(size)
        dE = energy_difference(J, h, spin_to_change, configuration, size)

        r = np.random.random()
        if r < np.exp(-beta * dE):
            configuration[spin_to_change] *= -1
            current_energy += dE
            
        average_spins[step] = configuration.mean()
    return current_energy, average_spins
    

def monte_carlo(n_steps, beta, J, h, configuration, return_configurations=False)->tuple[float, ]:
    if return_configurations:
        return monte_carlo_config(n_steps, beta, J, h, configuration)
    else:
        return monte_carlo_noconfig(n_steps, beta, J, h, configuration)
    

def energy(beta, J, N):
    return -(N-1)*J*np.tanh(beta*J)

def free_energy(beta, J, h):
    return -J - (np.log(np.cosh(beta*h) + np.sqrt(np.power(np.cosh(beta*h), 2) - 2*np.exp(-2*beta*J)*np.sinh(2*beta*J))))/beta

def magnetization(beta, J, h):
    return (np.sinh(beta * h))/(np.sqrt(np.cosh(beta * h)**2 - (2*np.exp(-2*beta*J)*np.sinh(2*beta*J))))

if __name__ == "__main__":
    size = 1000
    n_steps = 1000000
    beta = 1.0
    J = 1.0
    h = 0.0

    configuration = np.random.choice([-1, 1], size)
    current_energy, average_spins = monte_carlo(n_steps, beta, J, h, configuration)
    
    plt.plot(average_spins)
    plt.xlabel("Monte Carlo Step")
    plt.ylabel("Average Spin")
    plt.show()