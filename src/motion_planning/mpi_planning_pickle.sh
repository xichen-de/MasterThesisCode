#!/usr/bin/env bash
NUM_CPUS=20
EXTENSION=pickle
# Where you put the pickled problems
PROBLEM_DIR=/home/chenx/project/master_thesis_xi_chen/data/pickles/problem
# Where you put the pickled meta scenario
META_DIR=/home/chenx/project/master_thesis_xi_chen/data/pickles/meta_scenario
# Where you put the plan results
RESULT_DIR=/home/chenx/project/master_thesis_xi_chen/data/plan
# Where you put the temporary folders for MPI, this folder must be empty does not exist!
TMP_DIR=/home/chenx/project/master_thesis_xi_chen/data/plan_tmp

source activate cr36

# Copy pickled problems to subfolders
python ../tools/divide_files.py -i ${PROBLEM_DIR} -o ${TMP_DIR} -n ${NUM_CPUS} -e ${EXTENSION}

# Plan using NUM_CPUS threads
mpirun -np ${NUM_CPUS} python plan.py -r ${RESULT_DIR} -p ${TMP_DIR} -m ${META_DIR} -mpi

# Delete subfolders
rm -rf ${TMP_DIR}
