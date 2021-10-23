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
