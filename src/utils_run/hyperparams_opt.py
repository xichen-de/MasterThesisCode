"""
A utility function to be called when optimizing model hyperparameters
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

import numpy as np
import optuna
import os
import yaml
from pprint import pprint
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.ddpg import (
    AdaptiveParamNoiseSpec,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines.her import HERGoalEnvWrapper
from commonroad_rl.utils_run.callbacks import HyperparamsTrialEvalCallback

__author__ = "Brian Liao"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"


def optimize_hyperparams(
    algo,
    env,
    model_fn,
    env_fn,
    n_trials=10,
    n_timesteps=5000,
    hyperparams=None,
    configs=None,
    n_jobs=1,
    sampler_method="random",
    pruner_method="halving",
    seed=13,
    verbose=1,
    log_path=None,
    best_model_save_path=None,
):
    """
    :param algo: (str)
    :param env: (str)
    :param model_fn: (func) function that is used to instantiate the model
    :param env_fn: (func) function that is used to instantiate the env
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param hyperparams: (dict) model hyperparameters to be optimized
    :param configs: (dict) environment configurations
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param seed: (int)
    :param verbose: (int)
    :param log_path: (str) folder for saving evaluation results during optimization
    :param best_model_save_path (str) folder for saving the best model
    :return: (dict) detailed result of the optimization
    """
    # TODO: eval each hyperparams several times to account for noisy evaluation
    # TODO: take into account the normalization (also for the test env -> sync obs_rms)
    print("[hyperparams_opt.py] Optimizing model hyperparameters for {}".format(algo))

    # Model hyperparameters such as learning rate, minibatch size, ... etc
    if hyperparams is None:
        hyperparams = {}

    n_startup_trials = 10
    # Set number of episodes to be run for one evaluation
    n_eval_episodes = 5
    # Set number of evaluations for one trial
    n_evaluations = 20
    # Evaluations to be spanned over the trial (ie. conduct one evaluation every eval_freq learning time steps)
    eval_freq = int(n_timesteps / n_evaluations)

    if verbose > 0:
        print(
            "[hyperparams_opt.py] Optimizing with {} trials using {} parallel jobs, each with {} maximal time steps and {} evaluations".format(
                n_trials, n_jobs, n_timesteps, n_evaluations
            )
        )

    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if sampler_method == "random":
        sampler = RandomSampler(seed=seed)
    elif sampler_method == "tpe":
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == "skopt":
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        sampler = SkoptSampler(
            skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"}
        )
    else:
        raise ValueError(
            "[hyperparams_opt.py] Unknown sampler: {}".format(sampler_method)
        )

    if pruner_method == "halving":
        pruner = SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=4, min_early_stopping_rate=0
        )
    elif pruner_method == "median":
        pruner = MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3
        )
    elif pruner_method == "none":
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    else:
        raise ValueError(
            "[hyperparams_opt.py] Unknown pruner: {}".format(pruner_method)
        )

    if verbose > 0:
        print(
            "[hyperparams_opt.py] Sampler: {} - Pruner: {}".format(
                sampler_method, pruner_method
            )
        )

    # Create study object on model hyperparameters from Optuna
    hyperparams_study = optuna.create_study(sampler=sampler, pruner=pruner)

    # Prepare the sampler
    algo_sampler = HYPERPARAMS_SAMPLER[algo]

    def objective_hyperparams(trial):
        """
        Optimization objective for model hyperparameters
        """
        trial.model_class = None

        if algo == "her":
            trial.model_class = hyperparams["model_class"]

        # Hack to use DDPG/TD3 noise sampler
        if algo in ["ddpg", "td3"] or trial.model_class in ["ddpg", "td3"]:
            trial.n_actions = env_fn(n_envs=1).action_space.shape[0]

        # Get keyword arguments for model hyperparameters and environment configurations
        kwargs_hyperparams = hyperparams.copy()
        kwargs_configs = configs.copy()

        # Sample model hyperparameters and keep environment configurations
        sampled_hyperparams = algo_sampler(trial)
        kwargs_hyperparams.update(sampled_hyperparams)

        # Save data for later inspection
        tmp_path = os.path.join(log_path, "trial_" + str(trial.number))
        os.makedirs(tmp_path, exist_ok=True)
        with open(
            os.path.join(tmp_path, "sampled_model_hyperparameters.yml"), "w"
        ) as f:
            yaml.dump(kwargs_hyperparams, f)
        if verbose > 0:
            print("[hyperparams_opt.py] Sampled model hyperparameters:")
            pprint(kwargs_hyperparams)

        # Create model and environments for optimization
        model = model_fn(kwargs_hyperparams, kwargs_configs)
        eval_env = env_fn(n_envs=1, eval_env=True, **kwargs_configs)

        # Account for parallel envs
        eval_freq_ = eval_freq
        if isinstance(model.get_env(), VecEnv):
            eval_freq_ = max(eval_freq // model.get_env().num_envs, 1)

        if verbose > 0:
            print(
                "[hyperparams_opt.py] Evaluating with {} episodes after every {} time steps".format(
                    n_eval_episodes, eval_freq_
                )
            )

        # TODO: use non-deterministic eval for Atari?
        hyperparams_eval_callback = HyperparamsTrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq_,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=True,
            verbose=verbose,
        )
        if algo == "her":
            # Wrap the env if need to flatten the dict obs
            if isinstance(eval_env, VecEnv):
                eval_env = _UnvecWrapper(eval_env)
            eval_env = HERGoalEnvWrapper(eval_env)

        try:
            model.learn(n_timesteps, callback=hyperparams_eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()

        is_pruned = hyperparams_eval_callback.is_pruned
        cost = hyperparams_eval_callback.cost
        del model.env, eval_env
        del model
        if is_pruned:
            raise optuna.exceptions.TrialPruned()
        return cost

    try:
        hyperparams_study.optimize(
            objective_hyperparams, n_trials=n_trials, n_jobs=n_jobs
        )
    except KeyboardInterrupt:
        pass

    if verbose > 0:
        print(
            "[hyperparams_opt.py] Number of finished trials: ",
            len(hyperparams_study.trials),
        )
        print("[hyperparams_opt.py] Best value: ", hyperparams_study.best_trial.value)
        print("[hyperparams_opt.py] Best model hyperparameters: ")
        for key, value in hyperparams_study.best_trial.params.items():
            print("[hyperparams_opt.py]     {}: {}".format(key, value))

    return hyperparams_study.best_trial.params


def sample_ppo2_params(trial):
    """
    Sampler for PPO2 hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps = trial.suggest_categorical(
        "n_steps", [16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    cliprange = trial.suggest_categorical("cliprange", [0.1, 0.2, 0.3, 0.4])
    noptepochs = trial.suggest_categorical("noptepochs", [1, 5, 10, 20, 30, 50])
    lam = trial.suggest_categorical("lamdba", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

    if n_steps < batch_size:
        nminibatches = 1
    else:
        nminibatches = int(n_steps / batch_size)

    return {
        "n_steps": n_steps,
        "nminibatches": nminibatches,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "cliprange": cliprange,
        "noptepochs": noptepochs,
        "lam": lam,
    }


def sample_a2c_params(trial):
    """
    Sampler for A2C hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # normalize = trial.suggest_categorical('normalize', [True, False])
    # TODO: take into account the normalization (also for the test env)

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "lr_schedule": lr_schedule,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
    }


def sample_acktr_params(trial):
    """
    Sampler for ACKTR hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    n_steps = trial.suggest_categorical(
        "n_steps", [16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "lr_schedule": lr_schedule,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
    }


def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(1e5), int(1e6)]
    )
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 10000, 20000]
    )
    train_freq = trial.suggest_categorical("train_freq", [1, 10, 100, 300])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    ent_coef = trial.suggest_categorical(
        "ent_coef", ["auto", 0.5, 0.1, 0.05, 0.01, 0.0001]
    )
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    net_arch = {"small": [64, 64], "medium": [256, 256], "big": [400, 300],}[net_arch]

    target_entropy = "auto"
    if ent_coef == "auto":
        target_entropy = trial.suggest_categorical(
            "target_entropy", ["auto", -1, -10, -20, -50, -100]
        )

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(layers=net_arch),
    }


def sample_td3_params(trial):
    """
    Sampler for TD3 hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 100, 128, 256, 512]
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(1e5), int(1e6)]
    )
    train_freq = trial.suggest_categorical("train_freq", [1, 10, 100, 1000, 2000])
    gradient_steps = train_freq
    noise_type = trial.suggest_categorical(
        "noise_type", ["ornstein-uhlenbeck", "normal"]
    )
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    return hyperparams


def sample_trpo_params(trial):
    """
    Sampler for TRPO hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    timesteps_per_batch = trial.suggest_categorical(
        "timesteps_per_batch", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    )
    max_kl = trial.suggest_loguniform("max_kl", 0.000001, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    lam = trial.suggest_categorical("lamdba", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    # cg_damping = trial.suggest_loguniform('cg_damping', 1e-5, 1)
    cg_damping = 0.1
    cg_iters = trial.suggest_categorical("cg_iters", [10, 15, 20, 30])
    vf_stepsize = trial.suggest_loguniform("vf_stepsize", 1e-5, 1)
    vf_iters = trial.suggest_categorical("vf_iters", [1, 3, 5, 10, 20])

    return {
        "gamma": gamma,
        "timesteps_per_batch": timesteps_per_batch,
        "max_kl": max_kl,
        "entcoeff": ent_coef,
        "lam": lam,
        "cg_damping": cg_damping,
        "cg_iters": cg_iters,
        "vf_stepsize": vf_stepsize,
        "vf_iters": vf_iters,
    }


def sample_ddpg_params(trial):
    """
    Sampler for DDPG hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    # actor_lr = trial.suggest_loguniform('actor_lr', 1e-5, 1)
    # critic_lr = trial.suggest_loguniform('critic_lr', 1e-5, 1)
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    buffer_size = trial.suggest_categorical(
        "memory_limit", [int(1e4), int(1e5), int(1e6)]
    )
    noise_type = trial.suggest_categorical(
        "noise_type", ["ornstein-uhlenbeck", "normal", "adaptive-param"]
    )
    noise_std = trial.suggest_uniform("noise_std", 0, 1)
    normalize_observations = trial.suggest_categorical(
        "normalize_observations", [True, False]
    )
    normalize_returns = trial.suggest_categorical("normalize_returns", [True, False])

    hyperparams = {
        "gamma": gamma,
        "actor_lr": learning_rate,
        "critic_lr": learning_rate,
        "batch_size": batch_size,
        "memory_limit": buffer_size,
        "normalize_observations": normalize_observations,
        "normalize_returns": normalize_returns,
    }

    if noise_type == "adaptive-param":
        hyperparams["param_noise"] = AdaptiveParamNoiseSpec(
            initial_stddev=noise_std, desired_action_stddev=noise_std
        )
        # Apply layer normalization when using parameter perturbation
        hyperparams["policy_kwargs"] = dict(layer_norm=True)
    elif noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    return hyperparams


def sample_her_params(trial):
    """
    Sampler for HER hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    if trial.model_class == SAC:
        hyperparams = sample_sac_params(trial)
    elif trial.model_class == DDPG:
        hyperparams = sample_ddpg_params(trial)
    elif trial.model_class == TD3:
        hyperparams = sample_td3_params(trial)
    else:
        raise NotImplementedError(
            f"Trial model class {trial.model_class} not supported"
        )

    hyperparams["random_exploration"] = trial.suggest_uniform(
        "random_exploration", 0, 1
    )
    hyperparams["n_sampled_goal"] = trial.suggest_categorical(
        "n_sampled_goal", [1, 2, 4, 6, 8]
    )

    return hyperparams


HYPERPARAMS_SAMPLER = {
    "ppo2": sample_ppo2_params,
    "sac": sample_sac_params,
    "a2c": sample_a2c_params,
    "trpo": sample_trpo_params,
    "ddpg": sample_ddpg_params,
    "her": sample_her_params,
    "acktr": sample_acktr_params,
    "td3": sample_td3_params,
}
