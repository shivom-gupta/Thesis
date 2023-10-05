import numpy as np
from ising import monte_carlo_config

def generate_configurations(size:int, n_steps:int, beta:float, J:float, h:float, file_name:str, path = 'data/')->None:
    configuration = np.random.choice([-1, 1], size=size)
    configurations = monte_carlo_config(n_steps=n_steps, beta=beta, J=J, h=h, configuration=configuration)[2]
    
    configurations = configurations.clip(0).astype(bool)
    shape = configurations.shape
    configurations = np.packbits(configurations)
    np.savez_compressed(path+file_name, configurations = configurations, shape = shape)
    
    print(f'File {file_name} generated successfully')
    
def read_configurations(path_to_file:str)->np.ndarray:
    files = np.load(path_to_file)
    configurations = np.unpackbits(files['configurations'])
    shape = files['shape']
    configurations = 2*configurations.reshape(shape).astype(np.int16) - 1
    return configurations