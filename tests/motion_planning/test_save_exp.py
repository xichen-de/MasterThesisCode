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

from src.motion_planning import save_exp, plan
from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

argument = {
    "result_dir": os.path.join(ROOT_STR, "tests", "motion_planning", "results"),
    "problem_dir": os.path.join(ROOT_STR, "tests", "resources", "pickles", "problem"),
    "meta_dir": os.path.join(ROOT_STR, "tests", "resources", "pickles", "meta_scenario"),
    "multiprocessing": False,
    "traj_dir": os.path.join(ROOT_STR, "tests", "resources", "plan", "traj"),
    "exp_dir": os.path.join(ROOT_STR, "tests", "motion_planning", "results", "exp"),
    "env_config_path": os.path.join(ROOT_STR, "tests", "resources", "rl", "configs.yaml"),
    "dense_reward": False,
}

args = argparse.Namespace()
for attr, val in argument.items():
    setattr(args, attr, val)


def test_save_exp():
    save_exp.main(args)
