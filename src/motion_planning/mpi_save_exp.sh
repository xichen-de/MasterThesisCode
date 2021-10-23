#!/usr/bin/env bash
#
# MIT License
#
# Copyright 2021 Xi Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

NUM_CPUS=20
EXTENSION=pickle

# Where you put the pickled problems
PROBLEM_DIR=/home/chenx/project/master_thesis_xi_chen/data/pickles/problem
# Where you put the pickled meta scenarios
META_DIR=/home/chenx/project/master_thesis_xi_chen/data/pickles/meta_scenario
# Where your planned results is, the trajectory is the `traj` folder of your planned results.
TRAJ_DIR=/home/chenx/project/master_thesis_xi_chen/data/plan/traj
# Where you want to save your transitions (experience)
EXP_DIR=/home/chenx/project/master_thesis_xi_chen/data/exp
# Where you put the temporary folders for MPI, this folder must be empty does not exist!
TMP_DIR=/home/chenx/project/master_thesis_xi_chen/data/exp_tmp

source activate cr36

# Copy trajectory data to subfolders
python ../tools/divide_files.py -i ${TRAJ_DIR} -o ${TMP_DIR} -n ${NUM_CPUS} -e ${EXTENSION}

# Save experience using NUM_CPUS threads
mpirun -np ${NUM_CPUS} python save_exp.py -t ${TMP_DIR} -m ${META_DIR} -p ${PROBLEM_DIR} -e ${EXP_DIR} -mpi

# Delete subfolders
rm -rf ${TMP_DIR}
