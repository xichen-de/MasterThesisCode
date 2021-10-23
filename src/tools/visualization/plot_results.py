"""
Script for plotting learning curves, e.g., success, collision, off-road, and time-out rate.
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
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from baselines.common import plot_util as pu
from baselines.common.plot_util import smooth, Result

from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


def success_rate_xy_fn(results: Result, eval_freq: int, eval_episodes: int) -> Tuple[pd.RangeIndex, np.ndarray]:
    """
    Use the average success rate over eval_episodes at each eval_freq step as y, and eval step as x.
    :param results: OpenAI baselines Result object
    :param eval_freq: evaluation frequency
    :param eval_episodes: evaluation episode
    :return: x, y
    """
    x = pd.RangeIndex(start=0, stop=len(results.monitor.index) * eval_freq / eval_episodes, step=eval_freq)
    y = smooth(results.monitor.is_goal_reached.groupby(np.arange(len(results.monitor.index)) // eval_episodes).mean(),
               radius=10)
    return x, y


def collision_rate_xy_fn(results: Result, eval_freq: int, eval_episodes: int) -> Tuple[pd.RangeIndex, np.ndarray]:
    """
    Use the average collision rate over eval_episodes at each eval_freq step as y, and eval step as x.
    :param results: OpenAI baselines Result object
    :param eval_freq: evaluation frequency
    :param eval_episodes: evaluation episode
    :return: x, y
    """
    x = pd.RangeIndex(start=0, stop=len(results.monitor.index) * eval_freq / eval_episodes, step=eval_freq)
    y = smooth(results.monitor.is_collision.groupby(np.arange(len(results.monitor.index)) // eval_episodes).mean(),
               radius=10)
    return x, y


def off_road_rate_xy_fn(results: Result, eval_freq: int, eval_episodes: int) -> Tuple[pd.RangeIndex, np.ndarray]:
    """
    Use the average off-road rate over eval_episodes at each eval_freq step as y, and eval step as x.
    :param results: OpenAI baselines Result object
    :param eval_freq: evaluation frequency
    :param eval_episodes: evaluation episode
    :return: x, y
    """
    x = pd.RangeIndex(start=0, stop=len(results.monitor.index) * eval_freq / eval_episodes, step=eval_freq)
    y = smooth(results.monitor.is_off_road.groupby(np.arange(len(results.monitor.index)) // eval_episodes).mean(),
               radius=10)
    return x, y


def time_out_rate_xy_fn(results: Result, eval_freq: int, eval_episodes: int) -> Tuple[pd.RangeIndex, np.ndarray]:
    """
    Use the average time-out rate over eval_episodes at each eval_freq step as y, and eval step as x.
    :param results: OpenAI baselines Result object
    :param eval_freq: evaluation frequency
    :param eval_episodes: evaluation episode
    :return: x, y
    """
    x = pd.RangeIndex(start=0, stop=len(results.monitor.index) * eval_freq / eval_episodes, step=eval_freq)
    y = smooth(results.monitor.is_time_out.groupby(np.arange(len(results.monitor.index)) // eval_episodes).mean(),
               radius=10)
    return x, y


def get_args():
    parser = argparse.ArgumentParser(description="Plot learning curves")
    parser.add_argument('--result-dir', '-r', type=str, help="Folder where you put results of different methods",
                        default=f"{ROOT_STR}/test_results")
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--eval-episodes', type=int, default=5)
    return parser.parse_args()


def main():
    args = get_args()
    plt.switch_backend('agg')

    result_path = args.result_dir
    eval_freq = args.eval_freq
    eval_episodes = args.eval_episodes

    # Load results
    results = pu.load_results(result_path)

    # Plot success rate
    try:
        f, axarr = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False,
                                   shaded_err=True,
                                   xlabel='Training steps', ylabel='Success rate',
                                   xy_fn=(lambda r: success_rate_xy_fn(r, eval_freq, eval_episodes)))
        plt.ylim(0, 1)
        plt.xlim(0, 10 ** 6)
        plt.savefig(os.path.join(result_path, "success.svg"), bbox_inches='tight')
    except:
        pass

    # Plot collision rate
    try:
        f, axarr = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False,
                                   shaded_err=True,
                                   xlabel='Training steps', ylabel='Collision rate',
                                   xy_fn=(lambda r: collision_rate_xy_fn(r, eval_freq, eval_episodes)))
        plt.ylim(0, 1)
        plt.xlim(0, 10 ** 6)
        plt.savefig(os.path.join(result_path, "collision.svg"), bbox_inches='tight')
    except:
        pass

    try:
        # Plot off-road rate
        f, axarr = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False,
                                   shaded_err=True,
                                   xlabel='Training steps', ylabel='Off-road rate',
                                   xy_fn=(lambda r: off_road_rate_xy_fn(r, eval_freq, eval_episodes)))
        plt.ylim(0, 1)
        plt.xlim(0, 10 ** 6)
        plt.savefig(os.path.join(result_path, "off-road.svg"), bbox_inches='tight')
    except:
        pass

    try:
        # Plot time-out rate
        f, axarr = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False,
                                   shaded_err=True,
                                   xlabel='Training steps', ylabel='Time-out rate',
                                   xy_fn=(lambda r: time_out_rate_xy_fn(r, eval_freq, eval_episodes)))
        plt.ylim(0, 1)
        plt.xlim(0, 10 ** 6)
        plt.savefig(os.path.join(result_path, "time-out.svg"), bbox_inches='tight')
    except:
        pass


if __name__ == '__main__':
    main()
