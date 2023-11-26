import numpy as np
from IsingClass import Ising
import argparse

parser = argparse.ArgumentParser(description='Generate configurations for the Ising model')
parser.add_argument('-s', '--size', type=int, default=1000, help='Size of the lattice')
parser.add_argument('-n', '--n_sweeps', type=int, default=1_000_000, help='Number of sweeps in the Monte Carlo simulation')
parser.add_argument('-b', '--beta', type=float, default=1.0, help='Inverse temperature')
parser.add_argument('-J', '--J', type=float, default=1.0, help='Coupling constant')
parser.add_argument('-H', '--H', type=float, default=0.0, help='External magnetic field')
parser.add_argument('-f', '--file_name', type=str, default='configurations', help='Name of the file to save the configurations')


def generate_configurations(size:int, n_sweeps:int, beta:float, J:float, h:float, file_name:str, path:str = 'data/'):
    ising = Ising(size, n_sweeps, beta, J, h)
    taus = ising.load_tau()
    try:
        tau = taus[f'{beta:g}']
    except KeyError:
        tau = None
    ising.monte_carlo(tau)
    shape = ising.configurations.shape
    configurations = np.packbits(ising.configurations)
    magnetizations = ising.magnetizations
    energies = ising.energies
    params = np.array([size, n_sweeps, beta, J, h])
    np.savez_compressed(path+file_name, configurations = configurations, energies = energies, magnetizations = magnetizations, shape = shape, params = params)
 
    print(f'File {file_name} generated successfully')
    
def read_configurations(path_to_file:str)->tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    files = np.load(path_to_file)
    configurations = np.unpackbits(files['configurations'])
    shape = files['shape']
    energies = files['energies']
    magnetizations = files['magnetizations']
    params = files['params']
    params = {
        'size': params[0],
        'n_sweeps': params[1],
        'beta': params[2],
        'J': params[3],
        'h': params[4]}
    try:
        configurations = 2*configurations.reshape(shape).astype(np.int16) - 1
    except:
        configurations = 2*configurations[int(configurations.shape[0] - np.prod(shape)):].reshape(shape).astype(np.int16) - 1
    return configurations, energies, magnetizations, params

def main():
    args = parser.parse_args()
    generate_configurations(args.size, args.n_sweeps, args.beta, args.J, args.H, args.file_name)
    try:
        configurations,_, _, _ = read_configurations('data/'+args.file_name+'.npz')
        print('Shape of configurations created:', configurations.shape)
    except:
        print('Error reading the file')
    
if __name__ == '__main__':
    main()