"""
Script for dividing files into subfolders for MPI.
"""
import argparse
import glob
import ntpath
import os
from pathlib import Path
from shutil import copyfile

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


def get_project_root():
    """
    Get the root path of project.
    :return: string of the project root path
    """
    return Path(__file__).parent.parent.parent


ROOT_STR = get_project_root()


def get_args():
    parser = argparse.ArgumentParser(description="Divide files into subfolders for MPI")
    parser.add_argument('--extension', '-e', type=str, default='pickle')
    parser.add_argument('--input-dir', '-i', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    parser.add_argument('--n-cpus', '-n', type=int, default=1)

    return parser.parse_args()


def main():
    args = get_args()
    fns = glob.glob(os.path.join(args.input_dir, f"*.{args.extension}"))
    fns.sort()
    n_files_per_cpu = len(fns) // args.n_cpus
    for n in range(args.n_cpus):
        subdir = os.path.join(args.output_dir, str(n))
        os.makedirs(subdir, exist_ok=True)
        for fn in fns[n * n_files_per_cpu:(n + 1) * n_files_per_cpu]:
            copyfile(fn, os.path.join(subdir, ntpath.basename(fn)))


if __name__ == "__main__":
    main()
