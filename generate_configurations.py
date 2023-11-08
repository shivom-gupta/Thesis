import numpy as np
from ising import monte_carlo_config, init_configuration
from acf import calculate_tau
import argparse

parser = argparse.ArgumentParser(description='Generate configurations for the Ising model')
parser.add_argument('-s', '--size', type=int, default=1000, help='Size of the lattice')
parser.add_argument('-n', '--n_steps', type=int, default=1_000_000, help='Number of steps in the Monte Carlo simulation')
parser.add_argument('-b', '--beta', type=float, default=1.0, help='Inverse temperature')
parser.add_argument('-J', '--J', type=float, default=1.0, help='Coupling constant')
parser.add_argument('-H', '--H', type=float, default=0.0, help='External magnetic field')
parser.add_argument('-f', '--file_name', type=str, default='configurations', help='Name of the file to save the configurations')

def generate_configurations(size:int, n_steps:int, beta:float=1, J:float = 1, h:float = 0, file_name:str ='configurations', path = 'data/')->None:
    configuration = init_configuration(size)
    energies, magnetizations, configurations = monte_carlo_config(n_sweeps=n_steps, beta=beta, J=J, h=h, configuration=configuration)
    
    tau = calculate_tau(magnetizations)
    configurations = configurations[int(20*tau):][::int(4*tau)].clip(0).astype(bool)
    shape = configurations.shape
    configurations = np.packbits(configurations)
    np.savez_compressed(path+file_name, configurations = configurations, shape = shape)
    
    print(f'File {file_name} generated successfully')
    
def read_configurations(path_to_file:str)->np.ndarray:
    files = np.load(path_to_file)
    configurations = np.unpackbits(files['configurations'])
    shape = files['shape']
    try:
        configurations = 2*configurations.reshape(shape).astype(np.int16) - 1
    except:
        configurations = 2*configurations[int(configurations.shape - np.prod(shape)):].reshape(shape).astype(np.int16) - 1
    return configurations

def main():
    args = parser.parse_args()
    generate_configurations(args.size, args.n_steps, args.beta, args.J, args.H, args.file_name)
    
if __name__ == '__main__':
    main()