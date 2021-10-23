"""
Script to load list of states planned by Reactive Planner, compute action, and run in the same environment to
collect experience.
"""
import argparse
import glob
import os
import pickle
from shutil import copyfile
from typing import List, Tuple

import gym
import yaml
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv

import numpy as np
from commonroad.scenario.trajectory import State
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

from src.tools.divide_files import ROOT_STR
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

DT = 0.04


def compute_action(state_list: List[State]) -> Tuple[List[float], List[float]]:
    """
    Compute action from planned states.
    :param state_list: list of planned states for a problem
    :return: list of actions (normalized acceleration, normalized steering velocity)
    """
    steer_angle_list = []
    acc_list = []
    vehicle_params = parameters_vehicle2()
    for state in state_list:
        # Normalize acceleration
        acc_list.append(state.acceleration / vehicle_params.longitudinal.a_max)
        # Normalize steering velocity
        if not np.isclose(state.yaw_rate, 0.0):
            radius = state.velocity / np.abs(state.yaw_rate)
            steering_angle = np.arctan(vehicle_params.l / radius)
        else:
            steering_angle = 0.0
        steer_angle_list.append(steering_angle)
    steer_angle_list.append(steer_angle_list[-1])
    steer_vel_list = (np.diff(steer_angle_list) / DT / vehicle_params.steering.max).tolist()
    return acc_list, steer_vel_list


def create_env(problem_dir: str, tmp_dir: str, meta_dir: str, traj_path: str, dense_reward: bool, env_config_path: str):
    """
    Create an environment which only has one problem, which is the same as the
    problem where the trajectory is generated.
    :param env_config_path: path to environment configuration
    :param meta_dir: meta scenario directory
    :param problem_dir: directory of all problems
    :param tmp_dir: an empty directory where we save the to-use problem
    :param traj_path: directory of all planned trajectories
    :param dense_reward: if we use dense reward in environment, then true; otherwise false
    :return: gym env
    """
    # Copy problem to the empty temporary directory

    problem_src = os.path.join(problem_dir, os.path.basename(traj_path))
    problem_dst = os.path.join(tmp_dir, os.path.basename(traj_path))
    copyfile(problem_src, problem_dst)

    # Make environment
    env_kwargs = {}
    with open(PATH_PARAMS["configs"], "r") as config_file:
        configs = yaml.safe_load(config_file)
        env_configs = configs["env_configs"]
        env_kwargs.update(env_configs)

    if os.path.isfile(env_config_path):
        with open(env_config_path, "r") as configs_file:
            tmp = yaml.safe_load(configs_file)["env_configs"]
            env_kwargs.update(tmp)

    env_kwargs.update({
        "train_reset_config_path": tmp_dir,
        "test_reset_config_path": tmp_dir,
        "meta_scenario_path": meta_dir
    })

    if dense_reward:
        env_kwargs["reward_type"] = "dense_reward"
    else:
        env_kwargs["reward_type"] = "sparse_reward"
    env = gym.make('commonroad-v0', **env_kwargs)
    return env


def run_env(env, acc_list, steer_vel_list, benchmark_id, exp_dir):
    """
    Run environment using planned actions. Save all observations, actions, rewards, info.
    :param env: gym env
    :param acc_list: list of actions
    :param steer_vel_list: list of steering velocities
    :param benchmark_id: benchmark id of problem. Same as the name of the problem and trajectory file
    :param exp_dir: direct to save experience
    """
    action_list = []
    obs_list = []
    reward_list = []
    done_list = []
    info_list = []
    obs = env.reset()
    obs_list.append(obs)
    for acc, steer_vel in zip(acc_list, steer_vel_list):
        action = np.asarray([acc, steer_vel])
        obs, reward, done, info = env.step(action)
        action_list.append(action)
        obs_list.append(obs)
        reward_list.append(reward)
        done_list.append(done)
        info_list.append(info)
        if done:
            break
    # only successful experience is saved
    if info_list[-1]["is_goal_reached"]:
        experience = {
            'observations': obs_list[:-1],
            'actions': action_list,
            'rewards': reward_list,
            'next_observations': obs_list[1:],
            'dones': done_list,
            'infos': info_list
        }
        with open(os.path.join(exp_dir, f"{benchmark_id}.pickle"), 'wb') as f:
            pickle.dump(experience, f)


def save_all_exp(traj_dir: str, problem_dir: str, meta_dir: str, experience_dir: str, dense_reward: bool,
                 env_config_path: str):
    """
    Save successful experience of all trajectories.
    :param env_config_path: config path for environment
    :param meta_dir: meta scenario directory
    :param traj_dir: directory of planned trajectories (list of states)
    :param problem_dir: directory of all problems
    :param experience_dir: directory where to save experience
    :param dense_reward: if use dense reward, then true; otherwise, false
    """
    tmp_dir = os.path.join(traj_dir, "temp")
    os.makedirs(tmp_dir, exist_ok=True)
    traj_files = sorted(glob.glob(os.path.join(traj_dir, "*.pickle")))
    for tf in traj_files:
        with open(tf, 'rb') as pf:
            state_list = pickle.load(pf)
        benchmark_id = os.path.basename(tf).split(".")[0]
        acceleration_list, steering_angular_velocity_list = compute_action(state_list)
        env = create_env(problem_dir, tmp_dir, meta_dir, tf, dense_reward, env_config_path)
        run_env(env, acceleration_list, steering_angular_velocity_list, benchmark_id, experience_dir)
        os.remove(os.path.join(tmp_dir, os.path.basename(tf)))
    os.removedirs(tmp_dir)


def get_args():
    parser = argparse.ArgumentParser(description="Save experience from planned trajectories")
    parser.add_argument("-t", "--traj-dir", type=str, default=f"{ROOT_STR}/data/plan/traj",
                        help="directory of saved trajectory from plan.py")
    parser.add_argument("-m", "--meta-dir", type=str, default=f"{ROOT_STR}/data/pickles/meta_scenario",
                        help="directory of meta scenario")
    parser.add_argument("-p", "--problem-dir", type=str, default=f"{ROOT_STR}/data/pickles/problem",
                        help="directory of problems")
    parser.add_argument("-e", "--exp-dir", type=str, default=f"{ROOT_STR}/data/exp",
                        help="directory where you want to save the experience")
    parser.add_argument("--env-config-path", type=str, default=f"{ROOT_STR}/src/configs.yaml",
                        help="Path to file for overwriting environment configurations")
    parser.add_argument('--dense-reward', action="store_true", help="use dense reward in environment",
                        default=False)
    parser.add_argument('--multiprocessing', '-mpi', action="store_true", help="use mpi", default=False)
    return parser.parse_args()


def main(args):
    os.makedirs(args.exp_dir, exist_ok=True)

    # If we use mpi, we have to separate trajectory files into subfolders before
    if args.multiprocessing:
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None
        if MPI is None:
            traj_dir = args.traj_dir
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            traj_dir = os.path.join(args.traj_dir, str(rank))
    else:
        traj_dir = args.traj_dir

    print("=" * 80)
    print(f"Processing traj{traj_dir}")
    print("=" * 80)
    save_all_exp(traj_dir, args.problem_dir, args.meta_dir, args.exp_dir, args.dense_reward, args.env_config_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
