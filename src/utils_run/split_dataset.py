"""
Module for dividing dataset into train and test set
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
import ntpath
import random
from shutil import copyfile


def get_args():
    parser = argparse.ArgumentParser(
        description="Divide pickle files into training and testing"
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, default="/home/chenx/project/pickles/problem"
    )
    parser.add_argument(
        "--output_dir_train",
        "-otrain",
        type=str,
        default="/home/chenx/project/pickles/problem_train",
    )
    parser.add_argument(
        "--output_dir_test",
        "-otest",
        type=str,
        default="/home/chenx/project/pickles/problem_test",
    )
    parser.add_argument("--seed", "-s", type=int, default=5)
    parser.add_argument("--train_ratio", "-tr_r", type=float, default=0.7)

    return parser.parse_args()


def main():
    args = get_args()
    fns = glob.glob(os.path.join(args.input_dir, "*.pickle"))
    random.seed(args.seed)
    random.shuffle(fns)

    train_path = args.output_dir_train
    test_path = args.output_dir_test

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    fns = sorted(glob.glob(os.path.join(args.input_dir, "*.pickle")))
    random.shuffle(fns)

    num_train = int(len(fns) * args.train_ratio)

    print("Copying training data ...")
    for i, fn in enumerate(fns[:num_train]):
        print(f"{i + 1}/{num_train}", end="\r")
        copyfile(fn, os.path.join(train_path, ntpath.basename(fn)))

    print("Copying test data...")
    for i, fn in enumerate(fns[num_train:]):
        print(f"{i + 1}/{len(fns) - num_train}", end="\r")
        copyfile(fn, os.path.join(test_path, ntpath.basename(fn)))


if __name__ == "__main__":
    main()
