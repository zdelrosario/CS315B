#!/bin/bash -l
#PBS -l nodes=1:ppn=24
#PBS -l walltime=00:05:00
#PBS -m abe
#PBS -q gpu
#PBS -d .

LAUNCHER='mpirun --bind-to none -np 2' regent.py edge.rg
