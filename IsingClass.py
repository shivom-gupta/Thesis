import numpy as np
from acf import acf
from ising import monte_carlo


class Ising:
    def __init__(self, size, n_sweeps, beta, J, h) -> None:
        self.size = size
        self.n_sweeps = n_sweeps
        self.beta = beta
        self.J = J
        self.h = h
        self.configuration = np.random.choice([-1, 1], self.size)
    
    def monte_carlo(self, tau = None, R_max = None):
        tup = monte_carlo(self.size, self.n_sweeps, self.beta, self.J, self.h, self.configuration, tau, R_max)
        self.configurations = tup[0]
        self.energies = tup[1]
        self.magnetizations = tup[2]
        print("Monte Carlo finished")

    def calculate_tau(self, series:np.ndarray):
        corr = acf(series, series.shape[0])
        count = 0
        while corr.max() > 1:
            count += 1
            corr = acf(series, series.shape[0])
            if count > 10:
                print("Correlation not decaying")
                break
        tau = 1/2
        x = np.arange(corr.shape[0])
        for i in x:
            tau += corr[i]
            if i >= 6 * tau:
                break
            else:
                continue
        with open("taus.txt", "a") as f:
            f.write(f'{self.beta} {tau:0.5f}\n')
        return tau
    
    @staticmethod
    def load_tau():
        with open("taus.txt", "r") as f:
            taus = {}
            for line in f:
                beta, tau = line.split()
                taus[beta] = float(tau)
        return taus
    
    def save_configurations(self, file_name, path = 'data/')->None:
        shape = self.configurations.shape
        configurations = np.packbits(self.configurations)
        magnetizations = self.magnetizations
        energies = self.energies
        np.savez_compressed(path+file_name, configurations = configurations, magnetizations = magnetizations, energies = energies, shape = shape)
        
        print(f'File {file_name} generated successfully')
    
    def read_configurations(self, path_to_file:str):
        files = np.load(path_to_file)
        configurations = np.unpackbits(files['configurations'])
        shape = files['shape']
        try:
            configurations = 2*configurations.reshape(shape).astype(np.int16) - 1
        except:
            configurations = 2*configurations[int(configurations.shape - np.prod(shape)):].reshape(shape).astype(np.int16) - 1
        self.configurations = configurations
        self.magnetizations = files['magnetizations']
        self.energies = files['energies']
        return configurations
