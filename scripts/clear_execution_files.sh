#!/bin/bash

# This script is used to clear the generated SLURM scripts and their output files.
# It should be run from the root of the project.
# It also clears the generated scripts

rm -f launch_script_*.sh
rm -f *.out
rm -f *.err