import numpy as np
from ising import monte_carlo

def generate_configurations(size, n_steps, beta, J, h, file_name, path = ''):
    configuration = np.random.choice([-1, 1], size=size)
    _, _, configurations = monte_carlo(n_steps=n_steps, beta=beta, J=J, h=h, configuration=configuration, return_configurations=True)
    
    configurations = configurations.clip(0).astype(bool)
    shape = configurations.shape
    configurations = np.packbits(configurations)
    np.savez_compressed(path+file_name, configurations = configurations, shape = shape)
    
    print(f'File {file_name} generated successfully')
    
def read_configurations(path_to_file):
    files = np.load(path_to_file)
    configurations = np.unpackbits(files['configurations'])
    shape = files['shape']
    configurations = configurations.reshape(shape)
    return configurations