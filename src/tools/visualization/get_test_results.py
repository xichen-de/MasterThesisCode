"""
Script for getting test results from log folder.
run_stable_baselines.py save the log file as follows: log_folder_name (method) - algorithm - env_id + experiment_id.
An example can be like this: DDPG_PLAN - ddpgplan - commonroad-v0_1.
However, in order to use OpenAI baselines plot util,
and to average results among random seeds, same experiment with different seeds should be put in separate
folder and be named like this: name (same) + experiment_id.
Therefore, after we extract the test results, we put each test result in separate folder with the name:
log_folder_name + experiment id, for example: DDPG_PLAN-0, DDPG_PLAN-1, etc.
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
import os
from pathlib import Path
from shutil import copyfile

from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


def get_args():
    parser = argparse.ArgumentParser(description="Extract the test result")
    parser.add_argument('--input-dir', '-i', type=str, help="Log folder include logs for different methods",
                        default=f"{ROOT_STR}/logs")
    parser.add_argument('--output-dir', '-o', type=str, help="Output folder", default=f"{ROOT_STR}/test_results")
    return parser.parse_args()


def main():
    args = get_args()
    dest = args.output_dir
    fns = set(glob.glob(os.path.join(args.input_dir, '**/**/**/test/', "*.csv"), recursive=True))
    method_dict = {}
    for f in fns:
        log_path_name = os.path.basename(Path(f).parent.parent.parent.parent)
        algo_name = os.path.basename(Path(f).parent.parent.parent)
        count = method_dict.get(f"{log_path_name}_{algo_name}", 0)
        dest_path = os.path.join(dest, f'{log_path_name}_{algo_name}-{count}')
        os.makedirs(dest_path, exist_ok=True)
        copyfile(f, os.path.join(dest_path, os.path.basename(f)))
        method_dict[f"{log_path_name}_{algo_name}"] = count + 1
        count += 1


if __name__ == '__main__':
    main()
