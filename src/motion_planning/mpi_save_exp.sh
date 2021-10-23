#!/usr/bin/env bash
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
