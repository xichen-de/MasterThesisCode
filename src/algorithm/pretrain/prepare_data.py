"""
Currently, the experience from running planned trajectories in environment and collected by trained agent is saved
as pickle file. Each pickle file contains a dictionary with the following keys:
'observations' (List[np.ndarray]), 'actions' (List[np.ndarray]), 'rewards' (List[float]),
'next_observations' (List[np.ndarray]), 'dones' (List[bool]), 'infos' (List[dict]).

Each element of the observations and next_observations comes from the commonroad-rl environment. It is the flatten
numpy array of the following dict keys (with dimension of its element):
['a_ego' (1), 'distance_goal_lat' (1), 'distance_goal_long' (1), 'heading' (1),
'is_collision' (1), 'is_goal_reached' (1), 'is_off_road' (1),
'is_time_out' (1), 'lane_circ_p_rel' (6), 'lane_circ_v_rel' (6), 'lat_offset' (1),
'left_marker_distance' (1), 'left_road_edge_distance' (1), 'right_marker_distance' (1),
'right_road_edge_distance' (1), 'steering_angle' (1), 'v_ego' (1)].

For the transition network, we use all observations (all indicators excluded, since they do not contribute to predict
the next status. The excluded indicators are: 'is_collision', 'is_friction_violation', 'is_goal_reached', 'is_off_road',
'is_time_out'.) and actions as input.

The output of the transition network is the status: 'free': 0, 'collision': 1, 'success': 2.

For the reward network, the input is the same as the transition network along with the logits of the status. The output
is the predicted reward.

In this script, we prepare the input (observation + action) and ground truth (label of status for transition network,
reward for reward network). Moreover, we split the data into training set and test set. To avoid that imbalance class,
e.g., free is much more than collision and success samples, we will limit the number of three classes, and keep them
the same in training and test set.
"""
import argparse
import glob
import os
import pickle
from typing import List, Tuple, Union

import numpy as np
import yaml

from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

SELECTED_COLUMN = list(range(4)) + list(range(8, 27))


def load_data(input_dir: str) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[float], List[np.ndarray], List[bool], List[dict]]:
    """
    Load experience data collected by motion planner or trained agent.
    :param input_dir: path to experience data
    :return: list of observations, actions, rewards, next_observations, dones, infos
    """
    data_files = sorted(glob.glob(os.path.join(f"{input_dir}", "*.pickle")))
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    infos = []
    for d in data_files:
        with open(d, 'rb') as f:
            data = pickle.load(f)
        observations += data['observations']
        actions += data['actions']
        rewards += data['rewards']
        next_observations += data['next_observations']
        dones += data['dones']
        infos += data['infos']
    return observations, actions, rewards, next_observations, dones, infos


def info_to_status(infos: List[dict]) -> List[int]:
    """
    Transform info to status.
    Status: 'free': 0, 'collision': 1, 'success': 2.
    :param infos: list of infos
    :return: list of status
    """
    status = []
    for i in infos:
        if i['is_goal_reached']:
            status.append(2)
        else:
            if i['is_collision'] or i['is_off_road']:
                status.append(1)
            else:
                status.append(0)
    return status


def modify_free_to_success(observations: List[np.ndarray], status: List[int], rewards: List[float], random: bool) -> \
        Tuple[
            np.ndarray, np.ndarray, np.ndarray, Union[None, int]]:
    """
    Modify free samples to success samples by changing the longitudinal and lateral distance to goal.
    :param observations: observations
    :param status: status
    :param rewards: rewards
    :param random: no balance
    :return: observations, status, number of samples in each class (equal)
    """
    observations = np.asarray(observations)
    status = np.asarray(status)
    rewards = np.asarray(rewards)
    if random:
        return observations, status, rewards, None
    else:

        idx_free = np.where(status == 0)[0]
        idx_collision = np.where(status == 1)[0]
        idx_success = np.where(status == 2)[0]
        # Since normally, the number of three classes: free > collision > success
        # We can modify free samples to success by changing the longitudinal
        # and lateral distance to goal to close to zero
        # Therefore, we can keep that all three samples have the same number of collision samples.
        num_samples = len(idx_collision)
        num_modify = max(0, num_samples - len(idx_success))
        print(f"Modify {num_modify} free samples to success samples.")

        idx_modify_selected = np.random.choice(idx_free, num_modify, replace=False)
        sample_long = np.random.uniform(low=-1.0, high=1.0, size=(num_modify,))
        sample_lat = np.random.uniform(low=-0.5, high=0.5, size=(num_modify,))

        observations[idx_modify_selected, 2] = sample_long
        observations[idx_modify_selected, 1] = sample_lat
        status[idx_modify_selected] = 2
        with open(f"{ROOT_STR}/src/configs.yaml") as configs_file:
            tmp = yaml.safe_load(configs_file)["env_configs"]
            success_reward = tmp["reward_goal_reached"]
        rewards[idx_modify_selected] = success_reward
    return observations, status, rewards, num_samples


def get_divide_idx(status: np.ndarray, num_samples: int, ratio: float, random: bool) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Divide all selected samples into training set and test set with ratio of the training set
    :param status: status
    :param num_samples: number of samples in each class
    :param ratio: ratio of the training samples of all data
    :param random: random divide, no balance
    :return: index of training set and test set
    """
    if random:
        num_train = int(len(status) * ratio)
        idx = np.arange(len(status))
        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]
    else:
        idx_free = np.where(status == 0)[0]
        idx_collision = np.where(status == 1)[0]
        idx_success = np.where(status == 2)[0]

        idx_free_selected = np.random.choice(idx_free, num_samples, replace=False)
        idx_collision_selected = np.random.choice(idx_collision, num_samples, replace=False)
        idx_success_selected = np.random.choice(idx_success, num_samples, replace=False)

        # divide to train and test according to ratio
        num_train = int(num_samples * ratio)

        np.random.shuffle(idx_free_selected)
        np.random.shuffle(idx_collision_selected)
        np.random.shuffle(idx_success_selected)
        train_idx = np.concatenate(
            (idx_free_selected[:num_train],
             idx_collision_selected[:num_train],
             idx_success_selected[:num_train]))
        np.random.shuffle(train_idx)
        print(f"Number of training samples: {len(train_idx)}, each class has {len(train_idx) // 3} samples.")
        test_idx = np.concatenate(
            (idx_free_selected[num_train:],
             idx_collision_selected[num_train:],
             idx_success_selected[num_train:]))
        np.random.shuffle(test_idx)
        print(f"Number of test samples: {len(test_idx)}, each class has {len(test_idx) // 3} samples.")
    return train_idx, test_idx


def get_network_input(observations: np.ndarray, actions: List[np.ndarray]) -> np.ndarray:
    """
    Return network input: observation + action
    :param observations: observations
    :param actions: actions
    :return: network input
    """
    return np.concatenate((observations[:, SELECTED_COLUMN], np.asarray(actions)), axis=1)


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare data for training reward and transition networks and save npz file.")
    parser.add_argument("-i", "--input-dir", type=str,
                        default=f"{ROOT_STR}/data/input_data")
    parser.add_argument("-o", "--output-dir", type=str,
                        default=f"{ROOT_STR}/data/supervised_data")
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--random", action="store_true", default=False)
    parser.add_argument("-fn", "--file-name", type=str, default='data')
    return parser.parse_args()


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    observations, actions, rewards, next_observations, _, infos = load_data(input_dir)
    status = info_to_status(infos)
    observations, status, rewards, num_samples = modify_free_to_success(observations, status, rewards, args.random)
    network_input = get_network_input(observations, actions)
    train_idx, test_idx = get_divide_idx(status, num_samples, args.ratio, args.random)
    np.savez(os.path.join(output_dir, f"{args.file_name}.npz"),
             network_input=network_input,
             status=status,
             rewards=rewards,
             train_idx=train_idx,
             test_idx=test_idx)


if __name__ == '__main__':
    args = get_args()
    main(args)
