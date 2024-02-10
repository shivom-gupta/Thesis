#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ising
#SBATCH --partition=paul
#SBATCH --mail-user=sg42gofy@studserv.uni-leipzig.de
#SBATCH --mail-type=END
#SBATCH --output=slurm-%j.out

pip install --user numba

`python generate_configurations.py -s 200 -f '200_1'`
`python generate_configurations.py -s 500 -f '500_1'`
`python generate_configurations.py -s 800 -f '800_1'`
`python generate_configurations.py -s 2000 -f '2000_1'`
`python generate_configurations.py -s 5000 -f '5000_1'`
`python generate_configurations.py -s 10000 -f '10000_1'`
`python generate_configurations.py -s 1000 -b 0.1 -f '1000_01'`
`python generate_configurations.py -s 1000 -b 0.25 -f '1000_025'`
`python generate_configurations.py -s 1000 -b 1.5 -f '1000_15'`
