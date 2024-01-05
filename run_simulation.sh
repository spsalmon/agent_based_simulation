#!/bin/bash
#SBATCH -J agent_based
#SBATCH -o sim.out
#SBATCH -e sim.err
#SBATCH -c 32
#SBATCH -t 1-00:00:00
#SBATCH --mem=128G

cargo run --release