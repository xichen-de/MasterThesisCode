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
import os

import numpy as np
from src.algorithm.pretrain import prepare_data
from src.algorithm.pretrain.prepare_data import load_data, info_to_status, modify_free_to_success, main
from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

INPUT_DIR = os.path.join(ROOT_STR, "tests", "resources", "pretrain")
OUTPUT_DIR = os.path.join(ROOT_STR, "tests", "pretrain", "results")

argument = {
    "input_dir": INPUT_DIR,
    "output_dir": OUTPUT_DIR,
    "ratio": 0.7,
    "file_name": "data"
}
args = argparse.Namespace()
for attr, val in argument.items():
    setattr(args, attr, val)


def test_handle_imbalance():
    observations, actions, rewards, next_observations, _, infos = load_data(INPUT_DIR)
    status = info_to_status(infos)
    observations, status, num_samples = modify_free_to_success(observations, status)
    assert np.count_nonzero(status == 1) == num_samples
    assert np.count_nonzero(status == 0) >= num_samples
    assert np.count_nonzero(status == 2) >= num_samples


def test_save_data():
    prepare_data.main(args)
