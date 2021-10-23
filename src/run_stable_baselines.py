"""
Module for training an agent using stable baselines
"""
import argparse
import copy
import difflib
import importlib
import logging
import os
import sys
import time
import uuid
from typing import TextIO

import gym
import numpy as np
import yaml
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.her import HERGoalEnvWrapper

from src.tools.divide_files import ROOT_STR

os.environ["KMP_WARNINGS"] = "off"
NUM_PARALLEL_EXEC_UNITS = 4
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "none"  # "granularity=fine,verbose,compact,1,0"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

from pprint import pformat

from src.utils_run.callbacks import SaveVecNormalizeCallback
from src.utils_run.hyperparams_opt import optimize_hyperparams
from src.utils_run.observation_configs_opt import optimize_observation_configs
from src.utils_run.reward_configs_opt import optimize_reward_configs
from src.utils_run.noise import LinearNormalActionNoise
from src.utils_run.utils import (
    StoreDict,
    linear_schedule,
    get_wrapper_class,
    get_latest_run_id,
    make_env,
    ALGOS,
)

# numpy warnings because of tensorflow
import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings(action="ignore", category=UserWarning, module="gym")
warnings.simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Optional dependencies
try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import (
    VecFrameStack,
    VecNormalize,
    DummyVecEnv,
    SubprocVecEnv, VecEnv,
)
from stable_baselines.common.noise import (
    AdaptiveParamNoiseSpec,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines.common.schedules import constfn
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback


def run_stable_baselines_argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="commonroad-v0", help="environment ID"
    )
    parser.add_argument(
        "-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        help="Path to a pretrained agent to continue training",
        default="",
        type=str,
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ddpg",
        type=str,
        required=False,
        choices=list(ALGOS.keys()),
    )
    parser.add_argument(
        "-n",
        "--n-timesteps",
        help="Set the number of timesteps",
        default=int(1e6),
        type=int,
    )
    parser.add_argument(
        "--log-interval",
        help="Override log interval (default: -1, no change)",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation)",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--eval-episodes",
        help="Number of episodes to use for evaluation",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--save-freq",
        help="Save the model every n steps (if negative, no checkpoint)",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "-f", "--log-folder", help="Log folder", type=str, default="../logs"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--configs-path",
        help="Path to file for overwriting environment configurations",
        type=str,
        default=f"{ROOT_STR}/src/configs.yaml"
    )
    parser.add_argument(
        "--hyperparams-path",
        help="Path to file for overwriting model hyperparameters",
        type=str,
        default="",
    )
    parser.add_argument(
        "--optimize-observation-configs",
        help="Optimize observation configurations",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--optimize-reward-configs",
        help="Optimize reward configurations",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--optimize-hyperparams",
        help="Optimize model hyperparameters",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n-trials", help="Number of trials for optimization", type=int, default=5
    )
    parser.add_argument(
        "--n-jobs", help="Number of parallel jobs for optimization", type=int, default=1
    )
    parser.add_argument(
        "--sampler",
        help="Sampler for optimization",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner for optimization",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (overrides verbose mode)"
    )
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite model hyperparameters (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument(
        "-uuid",
        "--uuid",
        action="store_true",
        default=False,
        help="Ensure that the run has a unique ID",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help='Overwrite environment configurations (e.g. observe_heading:"True" reward_type:"\'default_reward\'")',
    )
    parser.add_argument(
        "--n-envs", help="Number of parallel training processes", type=int, default=1
    )
    parser.add_argument(
        "--info-keywords",
        type=str,
        nargs="+",
        default=(),
        help="(tuple) extra information to log, from the information return of environment.step, "
             "see stable_baselines/bench/monitor.py",
    )
    return parser


def run_stable_baselines(args):
    """
    Run training with stable baselines
    For help, see README.md
    """
    t1 = time.time()

    if args.env == "cr-monitor-v0":
        pass

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(
            "{} not found in gym registry, you maybe meant {}?".format(
                env_id, closest_match
            )
        )

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = "_{}".format(uuid.uuid4()) if args.uuid else ""
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2 ** 32 - 1)

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        valid_extension = args.trained_agent.endswith(
            ".pkl"
        ) or args.trained_agent.endswith(".zip")
        assert valid_extension and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip/.pkl file"

    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        LOGGER.info(
            "Using MPI for multiprocessing with {} workers".format(
                MPI.COMM_WORLD.Get_size()
            )
        )
        rank = MPI.COMM_WORLD.Get_rank()
        LOGGER.info("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            # Do not log anything for non-master nodes
            LOGGER.setLevel(logging.WARNING)
            args.tensorboard_log = ""

    tensorboard_log = (
        None
        if args.tensorboard_log == ""
        else os.path.join(args.tensorboard_log, env_id)
    )

    is_atari = False
    if "NoFrameskip" in env_id:
        is_atari = True

    log_path = os.path.join(args.log_folder, args.algo)
    save_path = os.path.join(
        log_path,
        "{}_{}{}".format(env_id, get_latest_run_id(log_path, env_id) + 1, uuid_str),
    )
    os.makedirs(save_path, exist_ok=True)

    LOGGER.info("Environment: {}".format(env_id))
    LOGGER.info("Seed: {}".format(args.seed))

    # Get number of environments for parallel training processes
    n_envs = args.n_envs
    LOGGER.debug("Using {} environments".format(n_envs))

    env_kwargs = {}
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

    # Save environment configurations for later inspection
    LOGGER.info("Saving environment configurations into {}".format(save_path))
    with open(
            os.path.join(save_path, "environment_configurations.yml"), "w"
    ) as output_file:
        yaml.dump(env_kwargs, output_file)
    LOGGER.debug(f"Environment configurations:")
    LOGGER.debug(pformat(env_kwargs))

    # Get number of timesteps
    n_timesteps = args.n_timesteps
    LOGGER.debug("Learning with {} timesteps".format(args.n_timesteps))

    # Load model hyperparameters from yaml file
    with open(f"{ROOT_STR}/src/hyperparams/{args.algo}.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(
                "Model hyperparameters not found for {}-{}".format(args.algo, env_id)
            )

    # Overwrite model hyperparameters if needed, first from file then from command arguments
    if os.path.isfile(args.hyperparams_path):
        hyperparams_file: TextIO
        with open(args.hyperparams_path, "r") as hyperparams_file:
            tmp = yaml.safe_load(hyperparams_file)
            hyperparams.update(tmp)
    if args.hyperparams is not None:
        hyperparams.update(args.hyperparams)

    # Save model hyperparameters for later inspection
    LOGGER.info("Saving model hyperparameters into {}".format(save_path))
    with open(os.path.join(save_path, "model_hyperparameters.yml"), "w") as f:
        yaml.dump(hyperparams, f)
    LOGGER.debug(
        "Model hyperparameters loaded and set for {}-{}: ".format(args.algo, env_id)
    )
    LOGGER.debug(pformat(hyperparams))

    # HER is only a wrapper around an algo
    algo_ = args.algo
    if algo_ == "her" or algo_ == "hercr":
        algo_ = hyperparams["model_class"]
        assert algo_ in {
            "sac",
            "ddpg",
            "ddpgplan",
            "dqn",
            "td3",
        }, "{} is not compatible with HER".format(algo_)
        # Retrieve the model class
        hyperparams["model_class"] = ALGOS[hyperparams["model_class"]]
        if hyperparams["model_class"] is None:
            raise ValueError("{} requires MPI to be installed".format(algo_))

    # Create learning rate schedules for ppo2 and sac
    if algo_ in ["ppo2", "sac", "td3"]:
        for key in ["learning_rate", "cliprange", "cliprange_vf"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError(
                    "Invalid value for {}: {}".format(key, hyperparams[key])
                )

    # Check if normalize
    normalize = False
    normalize_kwargs = {}
    if "normalize" in hyperparams.keys():
        normalize = hyperparams["normalize"]
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams["normalize"]

    # Convert to python object if needed
    if "policy_kwargs" in hyperparams.keys() and isinstance(
            hyperparams["policy_kwargs"], str
    ):
        hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])

    # Delete keys so the dict can be pass to the model constructor
    if "n_timesteps" in hyperparams.keys():
        del hyperparams['n_timesteps']

    # Obtain a class object from a wrapper name string in hyperparams and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    callbacks = []
    if args.save_freq > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq // n_envs, 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.save_freq,
                save_path=save_path,
                name_prefix="rl_model",
                verbose=1,
            )
        )

    def create_env(n_envs, eval_env=False, **kwargs):
        """
        Create the environment and wrap it if necessary
        :param n_envs: (int)
        :param eval_env: (bool) Whether is it an environment used for evaluation or not
        :return: (Union[gym.Env, VecEnv])
        :return: (gym.Env)
        """
        # Update environment keyword arguments from optimization sampler if specified
        env_kwargs.update(kwargs)

        # Differentiate keyword arguments for test environments
        env_kwargs_test = copy.deepcopy(env_kwargs)
        if env_id == "commonroad-v0" or env_id == "commonroad-v1":
            env_kwargs_test["test_env"] = True

        # Do not log eval env (issue with writing the same file)
        # log_dir = None if eval_env else save_path
        log_dir = os.path.join(save_path, "test") if eval_env else save_path

        if is_atari:
            LOGGER.debug("Using Atari wrapper")
            new_env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
            # Frame-stacking with 4 frames
            new_env = VecFrameStack(new_env, n_stack=4)
        elif algo_ in ["dqn", "ddpg", "ddpgplan"]:
            if hyperparams.get("normalize", False):
                LOGGER.warning("WARNING: normalization not supported yet for DDPG/DQN/DDPGPlan")

            if eval_env:
                new_env = DummyVecEnv(
                    [make_env(env_id, rank, args.seed, wrapper_class=env_wrapper, log_dir=log_dir,
                              env_kwargs=env_kwargs_test, info_keywords=tuple(args.info_keywords))])
            else:
                new_env = DummyVecEnv(
                    [make_env(env_id, rank, args.seed, wrapper_class=env_wrapper, log_dir=log_dir,
                              env_kwargs=env_kwargs, info_keywords=tuple(args.info_keywords))])
            new_env.seed(args.seed)
            if env_wrapper is not None:
                new_env = env_wrapper(new_env)
        else:
            if n_envs == 1:
                if eval_env:
                    new_env = DummyVecEnv(
                        [
                            make_env(
                                env_id,
                                0,
                                args.seed,
                                wrapper_class=env_wrapper,
                                log_dir=log_dir,
                                env_kwargs=env_kwargs_test,
                                info_keywords=tuple(args.info_keywords),
                            )
                        ]
                    )
                else:
                    new_env = DummyVecEnv(
                        [
                            make_env(
                                env_id,
                                0,
                                args.seed,
                                wrapper_class=env_wrapper,
                                log_dir=log_dir,
                                env_kwargs=env_kwargs,
                                info_keywords=tuple(args.info_keywords),
                            )
                        ]
                    )
            else:
                # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
                # On most env, SubprocVecEnv does not help and is quite memory hungry
                if eval_env:
                    new_env = SubprocVecEnv(
                        [
                            make_env(
                                env_id,
                                i,
                                args.seed + n_envs,
                                log_dir=log_dir,
                                wrapper_class=env_wrapper,
                                env_kwargs=env_kwargs_test,
                                subproc=True,
                                info_keywords=tuple(args.info_keywords),
                            )
                            for i in range(n_envs)
                        ],
                        start_method="spawn",
                    )
                else:
                    new_env = SubprocVecEnv(
                        [
                            make_env(
                                env_id,
                                i,
                                args.seed,
                                log_dir=log_dir,
                                wrapper_class=env_wrapper,
                                env_kwargs=env_kwargs,
                                subproc=True,
                                info_keywords=tuple(args.info_keywords),
                            )
                            for i in range(n_envs)
                        ],
                        start_method="spawn",
                    )
            if normalize:
                if len(normalize_kwargs) > 0:
                    LOGGER.debug("Normalization activated: {}".format(normalize_kwargs))
                else:
                    LOGGER.debug("Normalizing input and reward")
                new_env = VecNormalize(new_env, **normalize_kwargs)
        # Optional Frame-stacking
        if hyperparams.get("frame_stack", False):
            n_stack = hyperparams["frame_stack"]
            new_env = VecFrameStack(new_env, n_stack)
            LOGGER.info("Stacking {} frames".format(n_stack))
            del hyperparams["frame_stack"]
            output_file.close()
        if args.algo == 'her' or args.algo == 'hercr':
            # Wrap the env if need to flatten the dict obs
            if isinstance(new_env, VecEnv):
                new_env = _UnvecWrapper(new_env)
            new_env = HERGoalEnvWrapper(new_env)
        return new_env

    # Create training environments
    env = create_env(n_envs)

    # Create testing environments if needed, do not normalize reward
    if args.eval_freq > 0:
        # Account for the number of parallel environments
        args.eval_freq = max(args.eval_freq // n_envs, 1)

        # Do not normalize the rewards of the eval env
        old_kwargs = None
        if normalize:
            if len(normalize_kwargs) > 0:
                old_kwargs = normalize_kwargs.copy()
                normalize_kwargs["norm_reward"] = False
            else:
                normalize_kwargs = {"norm_reward": False}

        LOGGER.debug("Creating test environment")

        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=save_path)
        eval_callback = EvalCallback(
            create_env(1, eval_env=True),
            callback_on_new_best=save_vec_normalize,
            best_model_save_path=save_path,
            n_eval_episodes=args.eval_episodes,
            log_path=save_path,
            eval_freq=args.eval_freq,
        )
        callbacks.append(eval_callback)

        # Restore original kwargs
        if old_kwargs is not None:
            normalize_kwargs = old_kwargs.copy()

    LOGGER.info(f"Elapsed time for preparing steps: {time.time() - t1} s")

    # Stop env processes to free memory
    # if args.optimize_hyperparams and n_envs > 1:
    #     env.close()

    # Parse noise string for DDPG and SAC and TD3
    if algo_ in ["ddpg", "ddpgplan", "sac", "td3"] and hyperparams.get("noise_type") is not None:
        noise_type = hyperparams["noise_type"].strip()
        noise_std = hyperparams["noise_std"]
        n_actions = env.action_space.shape[0]
        if "adaptive-param" in noise_type:
            assert algo_ == "ddpg" or algo_ == "ddpgplan", "Parameter is not supported by SAC"
            hyperparams["param_noise"] = AdaptiveParamNoiseSpec(
                initial_stddev=noise_std, desired_action_stddev=noise_std
            )
        elif "normal" in noise_type:
            if "lin" in noise_type:
                hyperparams["action_noise"] = LinearNormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=noise_std * np.ones(n_actions),
                    final_sigma=hyperparams.get("noise_std_final", 0.0)
                                * np.ones(n_actions),
                    max_steps=n_timesteps,
                )
            else:
                hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
                )
        elif "ornstein-uhlenbeck" in noise_type:
            hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
            )
        else:
            raise RuntimeError("Unknown noise type {}".format(noise_type))
        LOGGER.info("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams["noise_type"]
        del hyperparams["noise_std"]
        if "noise_std_final" in hyperparams:
            del hyperparams["noise_std_final"]

    if ALGOS[args.algo] is None:
        raise ValueError("{} requires MPI to be installed".format(args.algo))

    # HINT: Three main options to be chosen from
    # HINT: 1. Continue training with a pretrained agent
    # HINT: 2. Optimize model hyperparameters and/or configurations of observations and rewards
    # HINT: 3. Start training from scratch
    if os.path.isfile(args.trained_agent):
        LOGGER.debug("Loading pretrained agent")

        # Policy should not be changed
        del hyperparams["policy"]
        if 'n_envs' in hyperparams.keys():
            del hyperparams['n_envs']

        model = ALGOS[args.algo].load(
            args.trained_agent,
            env=env,
            tensorboard_log=tensorboard_log,
            verbose=args.verbose + args.debug,
            **hyperparams,
        )

        exp_folder = os.path.dirname(args.trained_agent)

        # TODO: The following statements have no effect, probably the model definition should be moved after these lines
        if normalize:
            LOGGER.info("Loading saved running average")
            if os.path.exists(os.path.join(exp_folder, "vecnormalize.pkl")):
                env = VecNormalize.load(
                    os.path.join(exp_folder, "vecnormalize.pkl"), env
                )
            else:
                # Legacy:
                env.load_running_average(exp_folder)

    elif (
            args.optimize_hyperparams
            or args.optimize_observation_configs
            or args.optimize_reward_configs
    ):

        def create_model(hyperparams, configs):
            """
            Helper to create a model with different hyperparameters
            """
            if 'n_envs' in hyperparams.keys():
                del hyperparams['n_envs']
            return ALGOS[args.algo](
                env=create_env(n_envs, eval_env=False, **configs),
                tensorboard_log=tensorboard_log,
                verbose=1,
                **hyperparams,
            )

        if args.optimize_hyperparams:
            LOGGER.debug("Optimizing model hyperparameters")

            log_path = os.path.join(save_path, "model_hyperparameter_optimization")
            best_model_save_path = log_path
            optimized_hyperparams = optimize_hyperparams(
                args.algo,
                env_id,
                create_model,
                create_env,
                n_trials=args.n_trials,
                n_timesteps=n_timesteps,
                hyperparams=hyperparams,
                configs=env_kwargs,
                n_jobs=args.n_jobs,
                seed=args.seed,
                sampler_method=args.sampler,
                pruner_method=args.pruner,
                verbose=args.verbose + args.debug,
                log_path=log_path,
                best_model_save_path=best_model_save_path,
            )

            LOGGER.info("Saving optimized model hyperparameters to {}".format(log_path))
            report_name = "report_{}_{}-{}-trials-{}-steps-{}-{}.yml".format(
                args.algo, env_id, args.n_trials, n_timesteps, args.sampler, args.pruner
            )
            with open(os.path.join(log_path, report_name), "w") as f:
                yaml.dump(optimized_hyperparams, f)

        if args.optimize_reward_configs:
            LOGGER.debug("Optimizing reward configurations")

            log_path = os.path.join(save_path, "reward_configuration_optimization")
            best_model_save_path = log_path
            os.makedirs(log_path, exist_ok=True)
            optimized_reward_configs = optimize_reward_configs(
                args.algo,
                env_id,
                create_model,
                create_env,
                sampling_setting=sampling_setting_reward_configs,
                n_trials=args.n_trials,
                n_timesteps=n_timesteps,
                hyperparams=hyperparams,
                configs=env_kwargs,
                n_jobs=args.n_jobs,
                seed=args.seed,
                sampler_method=args.sampler,
                pruner_method=args.pruner,
                verbose=args.verbose + args.debug,
                log_path=log_path,
                best_model_save_path=best_model_save_path,
            )

            LOGGER.info("Saving optimized reward configurations to {}".format(log_path))
            report_name = "report_{}_{}-{}-trials-{}-steps-{}-{}.yml".format(
                args.algo, env_id, args.n_trials, n_timesteps, args.sampler, args.pruner
            )
            with open(os.path.join(log_path, report_name), "w") as f:
                yaml.dump(optimized_reward_configs, f)

        if args.optimize_observation_configs:
            LOGGER.debug("Optimizing observation configurations")

            log_path = os.path.join(save_path, "observation_configuration_optimization")
            best_model_save_path = log_path
            optimized_observation_configs = optimize_observation_configs(
                args.algo,
                env_id,
                create_model,
                create_env,
                sampling_setting=sampling_setting_observation_configs,
                n_trials=args.n_trials,
                n_timesteps=n_timesteps,
                hyperparams=hyperparams,
                configs=env_kwargs,
                n_jobs=args.n_jobs,
                seed=args.seed,
                sampler_method=args.sampler,
                pruner_method=args.pruner,
                verbose=args.verbose + args.debug,
                log_path=log_path,
                best_model_save_path=best_model_save_path,
            )

            LOGGER.info(
                "Saving optimized observation configurations to {}".format(log_path)
            )
            report_name = "report_{}_{}-{}-trials-{}-steps-{}-{}.yml".format(
                args.algo, env_id, args.n_trials, n_timesteps, args.sampler, args.pruner
            )
            with open(os.path.join(log_path, report_name), "w") as f:
                yaml.dump(optimized_observation_configs, f)

        return
    else:
        # Train an agent from scratch
        if 'n_envs' in hyperparams.keys():
            del hyperparams['n_envs']
        model = ALGOS[args.algo](
            env=env,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            verbose=args.verbose + args.debug,
            **hyperparams,
        )

    # Arguments to the learn function
    kwargs = {}
    if args.log_interval > -1:
        kwargs = {"log_interval": args.log_interval}
    if len(callbacks) > 0:
        kwargs["callback"] = callbacks

    try:
        model.learn(n_timesteps, **kwargs)
    except KeyboardInterrupt:
        pass

    # Only save worker of rank 0 when using mpi
    if rank == 0:
        LOGGER.info("Saving model to {}".format(save_path))
        model.save(os.path.join(save_path, str(env_id)))
    if normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(save_path, "vecnormalize.pkl"))

    LOGGER.info(f"Elapsed time: {time.time() - t1} s")


if __name__ == "__main__":
    args = run_stable_baselines_argsparser().parse_args(sys.argv[1:])
    if args.verbose:
        LOGGER.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)
    if args.debug:
        args.verbose = True
        LOGGER.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)

    run_stable_baselines(args)
