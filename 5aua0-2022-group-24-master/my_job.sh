#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=5:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<email>@student.tue.nl

# activate conda environment
source activate 5aua0

# go into your project folder
cd $TMPDIR/5aua0-project-template-hpc

# make sure on correct branch
git checkout master

# run your code
python train.py
