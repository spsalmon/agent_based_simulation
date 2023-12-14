#!/bin/bash
#SBATCH -J agent_based
#SBATCH -o sim.out
#SBATCH -e sim.err
#SBATCH -c 64
#SBATCH -t 1-00:00:00
#SBATCH --mem=128G

cargo run --release