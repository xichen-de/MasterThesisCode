#  MIT License
#
#  Copyright 2021 Xi Chen
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import numpy as np
from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines.common.vec_env import sync_envs_normalization
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecEnv

__author__ = "Brian Liao"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"


class HyperparamsTrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial during model hyperparameter optimization.
    """

    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=None,
        best_model_save_path=None,
        deterministic=True,
        verbose=1,
    ):

        if best_model_save_path is not None:
            best_model_save_path = os.path.join(
                best_model_save_path, "trial_" + str(trial.number)
            )
        if log_path is not None:
            log_path = os.path.join(log_path, "trial_" + str(trial.number))

        super(HyperparamsTrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.cost = 0.0

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(HyperparamsTrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report last?
            # report num_timesteps or elasped time?
            self.cost = -1 * self.best_mean_reward
            self.trial.report(self.cost, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune(self.eval_idx):
                self.is_pruned = True
                return False
        return True


class ObservationConfigsTrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial during observation configuration optimization.
    """

    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=None,
        best_model_save_path=None,
        deterministic=True,
        verbose=1,
    ):

        if best_model_save_path is not None:
            best_model_save_path = os.path.join(
                best_model_save_path, "trial_" + str(trial.number)
            )
        if log_path is not None:
            log_path = os.path.join(log_path, "trial_" + str(trial.number))

        super(ObservationConfigsTrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.cost = 0.0

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(ObservationConfigsTrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            self.cost = -1 * self.best_mean_reward
            self.trial.report(self.cost, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune(self.eval_idx):
                self.is_pruned = True
                return False
        return True


class RewardConfigsTrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial during reward configaration optimization.
    """

    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=None,
        best_model_save_path=None,
        deterministic=True,
        verbose=1,
    ):

        super(RewardConfigsTrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.lowest_mean_cost = np.inf
        self.last_mean_cost = np.inf
        self.cost = 0.0

        # Save best model into `($best_model_save_path)/trial_($trial_number)/best_model.zip`
        if best_model_save_path is not None:
            self.best_model_save_path = os.path.join(
                best_model_save_path, "trial_" + str(trial.number), "best_model"
            )
            os.makedirs(self.best_model_save_path, exist_ok=True)
        else:
            self.best_model_save_path = best_model_save_path

        # Log evaluation information into `($log_path)/trial_($trial_number)/evaluations.npz`
        self.evaluation_timesteps = []
        self.evaluation_costs = []
        self.evaluation_lengths = []
        if log_path is not None:
            self.log_path = os.path.join(
                log_path, "trial_" + str(trial.number), "evaluations"
            )
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        else:
            self.log_path = log.path

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            def evaluate_policy_configs(
                model,
                env,
                n_eval_episodes=10,
                render=False,
                deterministic=True,
                callback=None,
            ):
                """
                Runs policy for `n_eval_episodes` episodes and returns cost for optimization.
                This is made to work only with one env.

                :param model: (BaseRLModel) The RL agent you want to evaluate.
                :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`, this must contain only one environment.
                :param n_eval_episodes: (int) Number of episode to evaluate the agent
                :param deterministic: (bool) Whether to use deterministic or stochastic actions
                :param render: (bool) Whether to render the environment or not
                :param callback: (callable) callback function to do additional checks, called after each step.
                :return: ([float], [int]) list of episode costs and lengths
                """
                if isinstance(env, VecEnv):
                    assert (
                        env.num_envs == 1
                    ), "You must pass only one environment when using this function"

                episode_costs = []
                episode_lengths = []
                for _ in range(n_eval_episodes):
                    obs = env.reset()
                    done, info, state = False, None, None

                    # Record required information
                    # Since vectorized environments get reset automatically after each episode,
                    # we have to keep a copy of the relevant states here.
                    # See https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html for more details.
                    episode_length = 0
                    episode_cost = 0.0
                    episode_is_time_out = []
                    episode_is_collision = []
                    episode_is_off_road = []
                    episode_is_goal_reached = []
                    episode_is_friction_violation = []
                    while not done:
                        action, state = model.predict(
                            obs, state=state, deterministic=deterministic
                        )
                        obs, reward, done, info = env.step(action)

                        episode_length += 1
                        episode_is_time_out.append(info[-1]["is_time_out"])
                        episode_is_collision.append(info[-1]["is_collision"])
                        episode_is_off_road.append(info[-1]["is_off_road"])
                        episode_is_goal_reached.append(info[-1]["is_goal_reached"])
                        episode_is_friction_violation.append(info[-1]["is_friction_violation"])

                        if callback is not None:
                            callback(locals(), globals())
                        if render:
                            env.render()

                    # Calculate cost for optimization from state information
                    normalized_episode_length = (
                        episode_length / info[-1]["max_episode_time_steps"]
                    )
                    if episode_is_time_out[-1]:
                        episode_cost += 10.0  # * (1 / normalized_episode_length)
                    if episode_is_collision[-1]:
                        episode_cost += 10.0  # * (1 / normalized_episode_length)
                    if episode_is_off_road[-1]:
                        episode_cost += 10.0  # * (1 / normalized_episode_length)
                    if episode_is_friction_violation[-1]:
                        episode_cost += 10.0 * episode_is_friction_violation[-1] / episode_length  # * (1 / normalized_episode_length)
                    if episode_is_goal_reached[-1]:
                        episode_cost -= 10.0  # * normalized_episode_length

                    episode_costs.append(episode_cost)
                    episode_lengths.append(episode_length)

                return episode_costs, episode_lengths

            sync_envs_normalization(self.training_env, self.eval_env)
            episode_costs, episode_lengths = evaluate_policy_configs(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
            )

            mean_cost, std_cost = np.mean(episode_costs), np.std(episode_costs)
            mean_length, std_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_cost = mean_cost

            if self.verbose > 0:
                print(
                    "[callbacks.py] Evaluating at learning time step: {}".format(
                        self.num_timesteps
                    )
                )
                print(
                    "[callbacks.py] Cost mean: {:.2f}, std: {:.2f}".format(
                        mean_cost, std_cost
                    )
                )
                print(
                    "[callbacks.py] Length mean: {:.2f}, std: {:.2f}".format(
                        mean_length, std_length
                    )
                )

            if self.log_path is not None:
                self.evaluation_timesteps.append(self.num_timesteps)
                self.evaluation_costs.append(episode_costs)
                self.evaluation_lengths.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluation_timesteps,
                    episode_costs=self.evaluation_costs,
                    episode_lengths=self.evaluation_lengths,
                )

            if mean_cost < self.lowest_mean_cost:
                self.lowest_mean_cost = mean_cost
                if self.best_model_save_path is not None:
                    self.model.save(self.best_model_save_path)
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            self.eval_idx += 1
            self.cost = self.lowest_mean_cost
            self.trial.report(self.cost, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune(self.eval_idx):
                self.is_pruned = True
                return False
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix=None, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(
                    self.save_path,
                    "{}_{}_steps.pkl".format(self.name_prefix, self.num_timesteps),
                )
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("Saving VecNormalize to {}".format(path))
        return True
