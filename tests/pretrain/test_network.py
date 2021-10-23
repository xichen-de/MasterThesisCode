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
