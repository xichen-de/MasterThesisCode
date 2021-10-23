"""
Module to run trained agent and collect transitions.
"""
import argparse
import importlib
import os
import pickle
import sys
import warnings

# numpy warnings because of tensorflow
import yaml
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

from src.tools.divide_files import ROOT_STR
from src.utils_run.utils import ALGOS, StoreDict, get_latest_run_id, get_saved_hyperparams, \
    create_test_env, find_saved_model

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym

import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='commonroad-v0')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ddpg',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=int(1e6),
                        type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (-1: no exp folder, 0: latest)', default=0,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=True,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--load-best', action='store_true', default=False,
                        help='Load best model instead of last model if available')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument("--configs-path", help="Path to file for overwriting environment configurations", type=str,
                        default=f"{ROOT_STR}/src/configs.yaml")
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[],
                        help='Additional external Gym environment package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,
                        help='Optional keyword argument to pass to the env constructor')
    parser.add_argument('--save-trans', action='store_true', default=False, help='Save transition as pickle file')
    parser.add_argument('--trans-path', help='Directory to save transition',
                        default=f"{ROOT_STR}/data/input_data", type=str)
    parser.add_argument('--trans-name', help='Name of transition file', default='pretrain', type=str)
    parser.add_argument('--random-agent', action='store_true', default=False,
                        help='Use random agent instead of trained agent')

    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id, load_best=args.load_best)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        args.n_envs = 1

    set_global_seeds(args.seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    log_dir = args.reward_log if args.reward_log != '' else None

    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs
    if env_id == "commonroad-v0" or env_id == "commonroad-v1":
        # Get environment keyword arguments including observation and reward configurations
        with open(PATH_PARAMS["configs"], "r") as config_file:
            configs = yaml.safe_load(config_file)
            env_configs = configs["env_configs"]
            sampling_setting_reward_configs = configs["sampling_setting_reward_configs"]
            sampling_setting_observation_configs = configs["sampling_setting_observation_configs"]
            env_kwargs.update(env_configs)

    # Overwrite environment configurations if needed, first from file then from command arguments
    if os.path.isfile(args.configs_path):
        with open(args.configs_path, "r") as configs_file:
            tmp = yaml.safe_load(configs_file)["env_configs"]
            env_kwargs.update(tmp)
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                          should_render=not args.no_render,
                          hyperparams=hyperparams, env_kwargs=env_kwargs)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    model = ALGOS[algo].load(model_path, env=load_env)

    obs = env.reset()
    obs_list = [np.squeeze(obs)]
    reward_list = []
    action_list = []
    info_list = []
    done_list = []
    transition = []

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    state = None
    for _ in range(args.n_timesteps):
        if args.random_agent:
            # Random Agent
            action = [env.action_space.sample()]
        else:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        action_list.append(np.squeeze(action))
        obs_list.append(np.squeeze(obs))
        reward_list.append(reward[0])
        done_list.append(done[0])
        info_list.append(infos[0])

        if not args.no_render:
            env.render('human')

        episode_reward += reward[0]
        ep_len += 1

        if args.n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                    print("Atari Episode Length", episode_infos['l'])

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print("Episode Reward: {:.2f}".format(episode_reward))
                print("Episode Length", ep_len)
                state = None
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                episode_reward = 0.0
                ep_len = 0

            # Reset also when the goal is achieved when using HER
            if done or infos[0].get('is_success', False):
                if args.algo == 'her' and args.verbose > 1:
                    print("Success?", infos[0].get('is_success', False))
                # Alternatively, you can add a check to wait for the end of the episode
                # if done:
                obs = env.reset()

                # Add transition to list
                for o, a, r, n, d, i in zip(obs_list[:-1], action_list, reward_list, obs_list[1:], done_list,
                                            info_list):
                    transition.append((o, a, r, n, d, i))
                obs_list = []
                reward_list = []
                action_list = []
                info_list = []
                done_list = []
                obs_list.append(np.squeeze(obs))
                if args.algo == 'her':
                    successes.append(infos[0].get('is_success', False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))

    if args.verbose > 0 and len(episode_lengths) > 0:
        print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()
    if args.save_trans:
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        infos = []
        for o, a, r, n, d, i in transition:
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            next_observations.append(n)
            dones.append(d)
            infos.append(i)

        os.makedirs(args.trans_path, exist_ok=True)
        experience = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones,
            'infos': infos
        }
        with open(os.path.join(args.trans_path, f"{args.trans_name}.pickle"), 'wb') as f:
            pickle.dump(experience, f)


if __name__ == '__main__':
    main()
