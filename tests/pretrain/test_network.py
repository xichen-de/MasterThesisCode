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

from src.algorithm.pretrain import pretrain_network
from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

DATA_PATH = os.path.join(ROOT_STR, "tests", "resources", "pretrain", "data.npz")
OUTPUT_DIR = os.path.join(ROOT_STR, "tests", "pretrain", "results")


def test_train_network():
    argument = {
        "num_epochs": 100,
        "batch_size": 128,
        "save_interval": 10,
        "transition_network": [100],
        "reward_network": [100],
        "data_path": DATA_PATH,
        "save_path": OUTPUT_DIR
    }
    args = argparse.Namespace()
    for attr, val in argument.items():
        setattr(args, attr, val)
    pretrain_network.main(args)
