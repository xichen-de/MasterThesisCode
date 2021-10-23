"""
Script for dividing files into subfolders for MPI.
"""
#  MIT License
#
#  Copyright 2021 Xi Chen
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
