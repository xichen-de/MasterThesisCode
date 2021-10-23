"""
Divide problems into training set and test set.
You can choose between dividing randomly or according to the success rate of problems.
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
import random
from shutil import copyfile
from typing import List, Tuple

from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


def divide_with_success_rate(all_failed_scenario: List[str], all_success_scenario: List[str], success_train: float,
                             success_test: float, num_train: int, num_test: int) -> Tuple[List[str], List[str]]:
    """
    Divide problems into training and test set with respective plan success rate.
    :param all_failed_scenario: list of all failed scenarios names
    :param all_success_scenario: list of all successful scenarios names
    :param success_train: success rate of training scenarios
    :param success_test: success rate of test scenarios
    :param num_train: number of training scenarios
    :param num_test: number of test scenarios
    :return: list of training and test scenarios names
    """
    if num_test + num_train > len(all_failed_scenario) + len(all_success_scenario):
        raise ValueError(
            f"You only have {len(all_failed_scenario) + len(all_success_scenario)} scenarios, "
            f"but training and test dataset is {num_test + num_train} scenarios")
    success_train_scenario = random.sample(all_success_scenario,
                                           min(len(all_success_scenario), int(success_train * num_train)))
    remaining_success_scenario = list(set(all_success_scenario) - set(success_train_scenario))
    if remaining_success_scenario:
        success_test_scenario = random.sample(remaining_success_scenario,
                                              min(len(remaining_success_scenario), int(success_test * num_test)))
    else:
        success_test_scenario = []
    selected_failed_scenario = random.sample(all_failed_scenario,
                                             min(len(all_failed_scenario),
                                                 num_train + num_test -
                                                 len(success_train_scenario) -
                                                 len(success_test_scenario)))
    failed_train_scenario = selected_failed_scenario[
                            :min(len(selected_failed_scenario), num_train - len(success_train_scenario))]
    failed_test_scenario = selected_failed_scenario[
                           min(len(selected_failed_scenario), num_train - len(success_train_scenario)):]
    train_scenario = success_train_scenario + failed_train_scenario
    test_scenario = success_test_scenario + failed_test_scenario
    if len(train_scenario) < num_train or len(test_scenario) < num_test:
        rest_scenario = (set(all_success_scenario) - set(success_train_scenario) - set(success_test_scenario)).union(
            set(
                all_failed_scenario) - set(failed_train_scenario) - set(failed_test_scenario))
        sample = random.sample(list(rest_scenario), num_train + num_test - len(train_scenario) - len(test_scenario))
        train_scenario += sample[:num_train - len(train_scenario)]
        test_scenario += sample[num_train - len(train_scenario):]
    return train_scenario, test_scenario


def random_divide(all_scenario: List[str], num_train: int, num_test: int) -> Tuple[List[str], List[str]]:
    """
    Divide problems into training and test set randomly.
    :param all_scenario: all scenarios names
    :param num_train: number of training scenarios
    :param num_test: number of test scenarios
    :return: list of training and test scenarios names
    """
    if num_test + num_train > len(all_scenario):
        raise ValueError(
            f"You only have {len(all_scenario)} scenarios, "
            f"but training and test dataset is {num_test + num_train} scenarios")
    all_selected_scenario = random.sample(all_scenario, num_train + num_test)
    train_scenario = all_selected_scenario[:num_train]
    test_scenario = all_selected_scenario[num_train:]
    return train_scenario, test_scenario


def get_args():
    parser = argparse.ArgumentParser(description="Divide problems into training and test set.")
    parser.add_argument("--exp-dir", type=str, help="directory of experience", default=f"{ROOT_STR}/data/exp")
    parser.add_argument("--train-dir", type=str, help="directory of training problems",
                        default=f"{ROOT_STR}/data/pickles/problem_train")
    parser.add_argument("--test-dir", type=str, help="directory of test problems",
                        default=f"{ROOT_STR}/data/pickles/problem_test")
    parser.add_argument("--backup-dir", type=str, help="directory of all problems",
                        default=f"{ROOT_STR}/data/pickles/problem")
    parser.add_argument("--num-train", type=int, help="number of training problems", default=1400)
    parser.add_argument("--num-test", type=int, help="number of test problems", default=600)
    parser.add_argument("--random", help="divide train and test randomly", default=False, action="store_true")
    parser.add_argument("--success-train", help="planning success rate of training problems", default=0.8, type=float)
    parser.add_argument("--success-test", help="planning success rate of test problems", default=0.2, type=float)
    return parser.parse_args()


def main():
    args = get_args()
    all_scenario = glob.glob1(args.backup_dir, "*.pickle")
    all_success_scenario = list(set(glob.glob1(args.exp_dir, "*.pickle")).intersection(all_scenario))
    all_failed_scenario = list(set(all_scenario) - set(all_success_scenario))

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    if args.random:
        train_scenario, test_scenario = random_divide(all_scenario, args.num_train, args.num_test)
    else:
        train_scenario, test_scenario = divide_with_success_rate(all_failed_scenario, all_success_scenario,
                                                                 args.success_train, args.success_test, args.num_train,
                                                                 args.num_test)

    success_train_count = 0
    success_test_count = 0
    for s in train_scenario:
        if s in all_success_scenario:
            success_train_count += 1
        copyfile(os.path.join(args.backup_dir, s), os.path.join(args.train_dir, s))
    for s in test_scenario:
        if s in all_success_scenario:
            success_test_count += 1
        copyfile(os.path.join(args.backup_dir, s), os.path.join(args.test_dir, s))
    print(f"Train scenarios: {len(train_scenario)}, test scenarios : {len(test_scenario)}, "
          f"success rate: train scenarios: {success_train_count / args.num_train}, "
          f"test scenarios: {success_test_count / args.num_test}")


if __name__ == '__main__':
    main()
