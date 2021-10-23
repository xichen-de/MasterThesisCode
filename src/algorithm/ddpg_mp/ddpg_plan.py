"""
This module implements the DDPG algorithm along with motion planner and actor model update.
"""
import glob
import os
import pickle
import time
from collections import deque
from copy import deepcopy
from functools import reduce

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import yaml
from mpi4py import MPI
from src.algorithm.ddpg_mp.replay_buffer import ReplayBufferInfo, PrioritizedReplayBufferInfo, \
    EmphasizeRecentExperienceBufferInfo
from src.algorithm.pretrain.prepare_data import info_to_status, SELECTED_COLUMN
from src.algorithm.pretrain.pretrain_network import PretrainNetwork
from stable_baselines import logger, DDPG
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.math_util import unscale_action, scale_action
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.ddpg.ddpg import denormalize, normalize
from stable_baselines.ddpg.policies import DDPGPolicy

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


def compute_termination_from_status(status_logits: tf.Tensor) -> tf.Tensor:
    """
    Compute the probability of termination of an episode. It is 1.0 - P(free).
    :param status_logits: logits of three status: free (1. column), collision (2. column), success (3. column)
    :return: probability of termination
    """
    softmax_output = tf.nn.softmax(status_logits, axis=1)
    termination_probability = 1.0 - tf.split(softmax_output, 3, axis=1)[0]
    return termination_probability


class DDPGPlan(DDPG):
    """
    DDPG algorithm with motion planner and actor model update.
    Now it support replay strategy: uniform sampling, Prioritized Experience Replay, and
    Emphasize Recent Experience.
    """

    def __init__(self, policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1,
                 use_plan=False, plan_path=None, use_model_update=False, pretrain_path=None, pretrain_lr=1e-4,
                 train_pretrain=False,
                 use_split_buffer=False, split_factor_initial=0.8, split_factor_final=0.1,
                 use_per=False, per_alpha=0.6, per_beta0=0.4, per_beta_iters=None, per_eps=1e-6,
                 use_ere=False):
        """
        Deep Deterministic Policy Gradient (DDPG) model with motion planner and actor model update.
        :param policy: the policy model to use
        :param env: the environment to learn from
        :param gamma: the discount factor
        :param memory_policy: the replay buffer
        :param eval_env: the evaluation environment (can be None)
        :param nb_train_steps: the number of training steps
        :param nb_rollout_steps: the number of rollout steps
        :param nb_eval_steps: the number of evaluation steps
        :param param_noise: the parameter noise type (can be None)
        :param action_noise: the action noise type (can be None)
        :param normalize_observations: should the observation be normalized
        :param tau: the soft update coefficient (keep old values, between 0 and 1)
        :param batch_size: the size of the batch for learning the policy
        :param param_noise_adaption_interval: apply param noise every N steps
        :param normalize_returns: should the critic output be normalized
        :param enable_popart: enable pop-art normalization of the critic output
        (https://arxiv.org/pdf/1602.07714.pdf), normalize_returns must be set to True.
        :param observation_range: the bounding values for the observation
        :param critic_l2_reg: l2 regularizer coefficient
        :param return_range: the bounding values for the critic output
        :param actor_lr: the actor learning rate
        :param critic_lr: the critic learning rate
        :param clip_norm: clip the gradients (disabled if None)
        :param reward_scale: the value the reward should be scaled by
        :param render: enable rendering of the environment
        :param render_eval: enable rendering of the evaluation environment
        :param memory_limit: the max number of transitions to store, size of the replay buffer
        :param buffer_size: the max number of transitions to store, size of the replay buffer
        :param random_exploration: probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for DDPG normally but can help exploring when using HER + DDPG.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
        :param verbose: the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :param tensorboard_log: the log location for tensorboard (if None, no logging)
        :param _init_setup_model: whether or not to build the network at the creation of the instance
        :param policy_kwargs: additional arguments to be passed to the policy on creation
        :param full_tensorboard_log: enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
        :param seed: seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
        :param n_cpu_tf_sess: the number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
        :param use_plan: whether or not use motion planner results in replay buffer
        :param plan_path: path to pre-computed motion planner results
        :param pretrain_lr: the pretrain network learning rate
        :param use_model_update: whether or not use actor model update
        :param pretrain_path: path to pretrain transition and reward network
        :param train_pretrain: update pretrain network online
        :param use_split_buffer: whether or not use separate buffer for motion planner results
        :param split_factor_initial: how much should results from separate replay buffer for motion planner should take
        up when sampling, at the beginning of training
        :param split_factor_final: how much should results from separate replay buffer for motion planner should take
        up when sampling, at the end of training
        :param use_per: whether or not use Prioritized Experience Replay
        :param per_alpha: alpha parameter for prioritized replay buffer
        :param per_beta0: initial value of beta for prioritized replay buffer
        :param per_beta_iters: number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps
        :param per_eps: epsilon to add to the TD errors when updating priorities
        :param use_ere: whether or not use Emphasize Recent Experience
        """

        super(DDPGPlan, self).__init__(policy, env, gamma=gamma, memory_policy=memory_policy, eval_env=eval_env,
                                       nb_train_steps=nb_train_steps,
                                       nb_rollout_steps=nb_rollout_steps, nb_eval_steps=nb_eval_steps,
                                       param_noise=param_noise, action_noise=action_noise,
                                       normalize_observations=normalize_observations, tau=tau, batch_size=batch_size,
                                       param_noise_adaption_interval=param_noise_adaption_interval,
                                       normalize_returns=normalize_returns, enable_popart=enable_popart,
                                       observation_range=observation_range, critic_l2_reg=critic_l2_reg,
                                       return_range=return_range, actor_lr=actor_lr, critic_lr=critic_lr,
                                       clip_norm=clip_norm, reward_scale=reward_scale,
                                       render=render, render_eval=render_eval, memory_limit=memory_limit,
                                       buffer_size=buffer_size, random_exploration=random_exploration,
                                       verbose=verbose, tensorboard_log=tensorboard_log, _init_setup_model=False,
                                       policy_kwargs=policy_kwargs,
                                       full_tensorboard_log=full_tensorboard_log, seed=seed,
                                       n_cpu_tf_sess=n_cpu_tf_sess)

        # Actor model update
        self.use_model_update = use_model_update
        self.pretrain_path = pretrain_path
        self.pretrain_lr = pretrain_lr
        self.pretrain_network = None
        self.train_pretrain = train_pretrain

        # Motion planner for smart exploration
        self.use_plan = use_plan
        self.plan_path = plan_path
        self.planned_scenarios = None
        self.use_split_buffer = use_split_buffer
        self.split_factor_initial = split_factor_initial
        self.split_factor_final = split_factor_final
        self.split_factor_scheduler = None

        # Prioritized replay
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta0 = per_beta0
        self.per_beta_iters = per_beta_iters
        self.per_eps = per_eps
        self.beta_schedule = None
        self.importance_weights_ph = None
        self.td_error = None

        # Emphasize recent experience
        self.use_ere = use_ere
        self.current_train_step = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        """
        Set up actor and critic network. If use pretrain, also set up pretrain network and restore parameters.
        Initialize optimizers.
        """
        with SetVerbosity(self.verbose):
            assert isinstance(self.action_space, gym.spaces.Box), \
                "Error: DDPG cannot output a {} action space, only spaces.Box is supported.".format(self.action_space)
            assert issubclass(self.policy, DDPGPolicy), "Error: the input policy for the DDPG model must be " \
                                                        "an instance of DDPGPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                # When use actor model update, we have to reconstruct the pretrained network
                if self.use_model_update:
                    with open(os.path.join(self.pretrain_path, 'config.yml'), "r") as config_file:
                        # Load config file to decide the architecture of the pretrained network
                        configs = yaml.safe_load(config_file)
                        transition_network = configs["transition_network"]
                        reward_network = configs["reward_network"]
                    self.pretrain_network: PretrainNetwork = PretrainNetwork(transition_network, reward_network)

                with tf.variable_scope("input", reuse=False):
                    # Observation normalization.
                    if self.normalize_observations:
                        with tf.variable_scope('obs_rms'):
                            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
                    else:
                        self.obs_rms = None

                    # Return normalization.
                    if self.normalize_returns:
                        with tf.variable_scope('ret_rms'):
                            self.ret_rms = RunningMeanStd()
                    else:
                        self.ret_rms = None

                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None,
                                                 **self.policy_kwargs)

                    # Create target networks.
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None,
                                                     **self.policy_kwargs)
                    self.obs_target = self.target_policy.obs_ph
                    self.action_target = self.target_policy.action_ph

                    normalized_obs = tf.clip_by_value(normalize(self.policy_tf.processed_obs, self.obs_rms),
                                                      self.observation_range[0], self.observation_range[1])
                    normalized_next_obs = tf.clip_by_value(normalize(self.target_policy.processed_obs, self.obs_rms),
                                                           self.observation_range[0], self.observation_range[1])

                    if self.param_noise is not None:
                        # Configure perturbed actor.
                        self.param_noise_actor = self.policy(self.sess, self.observation_space, self.action_space, 1, 1,
                                                             None, **self.policy_kwargs)
                        self.obs_noise = self.param_noise_actor.obs_ph
                        self.action_noise_ph = self.param_noise_actor.action_ph

                        # Configure separate copy for stddev adoption.
                        self.adaptive_param_noise_actor = self.policy(self.sess, self.observation_space,
                                                                      self.action_space, 1, 1, None,
                                                                      **self.policy_kwargs)
                        self.obs_adapt_noise = self.adaptive_param_noise_actor.obs_ph
                        self.action_adapt_noise = self.adaptive_param_noise_actor.action_ph

                    # Inputs.
                    self.obs_train = self.policy_tf.obs_ph
                    self.action_train_ph = self.policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name='actions')
                    self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
                    self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
                if self.use_per:
                    self.importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

                # Create networks and core TF parts that are shared across setup parts.
                with tf.variable_scope("model", reuse=False):
                    self.actor_tf = self.policy_tf.make_actor(normalized_obs)
                    self.normalized_critic_tf = self.policy_tf.make_critic(normalized_obs, self.actions)
                    self.normalized_critic_with_actor_tf = self.policy_tf.make_critic(normalized_obs,
                                                                                      self.actor_tf,
                                                                                      reuse=True)
                # Noise setup
                if self.param_noise is not None:
                    self._setup_param_noise(normalized_obs)

                with tf.variable_scope("target", reuse=False):
                    critic_target = self.target_policy.make_critic(normalized_next_obs,
                                                                   self.target_policy.make_actor(normalized_next_obs))

                with tf.variable_scope("loss", reuse=False):
                    self.critic_tf = denormalize(
                        tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]),
                        self.ret_rms)

                    self.critic_with_actor_tf = denormalize(
                        tf.clip_by_value(self.normalized_critic_with_actor_tf,
                                         self.return_range[0], self.return_range[1]),
                        self.ret_rms)

                    q_next_obs = denormalize(critic_target, self.ret_rms)
                    self.target_q = self.rewards + (1. - self.terminals_ph) * self.gamma * q_next_obs

                    tf.summary.scalar('critic_target', tf.reduce_mean(self.critic_target))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('critic_target', self.critic_target)

                    # Set up parts.
                    if self.normalize_returns and self.enable_popart:
                        self._setup_popart()
                    self._setup_stats()
                    self._setup_target_network_updates()

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('rewards', tf.reduce_mean(self.rewards))
                    tf.summary.scalar('param_noise_stddev', tf.reduce_mean(self.param_noise_stddev))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('rewards', self.rewards)
                        tf.summary.histogram('param_noise_stddev', self.param_noise_stddev)
                        if len(self.observation_space.shape) == 3 and self.observation_space.shape[0] in [1, 3, 4]:
                            tf.summary.image('observation', self.obs_train)
                        else:
                            tf.summary.histogram('observation', self.obs_train)

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self._setup_actor_optimizer()
                    self._setup_critic_optimizer()
                    tf.summary.scalar('actor_loss', self.actor_loss)
                    tf.summary.scalar('critic_loss', self.critic_loss)

                self.params = tf_util.get_trainable_vars("model") \
                              + tf_util.get_trainable_vars('noise/') + tf_util.get_trainable_vars('noise_adapt/')

                self.target_params = tf_util.get_trainable_vars("target")
                self.obs_rms_params = [var for var in tf.global_variables()
                                       if "obs_rms" in var.name]
                self.ret_rms_params = [var for var in tf.global_variables()
                                       if "ret_rms" in var.name]

                if self.use_model_update:
                    with tf.variable_scope("Pretrain_mpi", reuse=False):
                        self._setup_pretrain_optimizer()
                        tf.summary.scalar('pretrain_loss', self.pretrain_total_loss)
                        tf.summary.scalar('pretrain_accuracy', self.pretrain_accuracy)

                with self.sess.as_default():
                    self._initialize(self.sess)
                    # Load parameters for pretrained network
                    if self.use_model_update:
                        self.restore_pretrain_network(self.pretrain_path)

                self.summary = tf.summary.merge_all()

    def restore_pretrain_network(self, pretrain_path: str):
        """
        Restore pretrain network weights.
        :param pretrain_path:
        """
        self.pretrain_network.restore_model(self.sess, pretrain_path)

    def _setup_actor_optimizer(self):
        """
        Setup the optimizer for the actor.
        """

        if self.verbose >= 2:
            logger.info('setting up actor optimizer')

        if self.use_model_update:
            # Use pretrained network to predict the status and reward, then compute the actor loss
            predicted_status = self.pretrain_network.status
            predicted_reward = self.pretrain_network.reward

            episode_termination_prob = compute_termination_from_status(predicted_status)
            include_next_state = (1.0 - episode_termination_prob)
            self.actor_loss = -tf.reduce_mean(
                predicted_reward + self.gamma * self.critic_with_actor_tf * include_next_state)
        else:
            self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        if self.verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))

        self.actor_grads = tf_util.flatgrad(self.actor_loss, tf_util.get_trainable_vars('model/pi/'),
                                            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/pi/'), beta1=0.9, beta2=0.999,
                                       epsilon=1e-08)

    def _setup_critic_optimizer(self):
        """
        Setup the optimizer for the critic.
        """
        if self.verbose >= 2:
            logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        if self.use_per:
            # When use PER, the critic loss is weighted by the importance sampling weight
            self.critic_loss = tf.reduce_mean(
                tf.square(self.normalized_critic_tf - normalized_critic_target_tf) * self.importance_weights_ph)
            self.td_error = tf.math.abs(self.normalized_critic_tf - normalized_critic_target_tf)
        else:
            self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in tf_util.get_trainable_vars('model/qf/')
                               if 'bias' not in var.name and 'qf_output' not in var.name and 'b' not in var.name]
            if self.verbose >= 2:
                for var in critic_reg_vars:
                    logger.info('  regularizing: {}'.format(var.name))
                logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/qf/')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        if self.verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = tf_util.flatgrad(self.critic_loss, tf_util.get_trainable_vars('model/qf/'),
                                             clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/qf/'), beta1=0.9, beta2=0.999,
                                        epsilon=1e-08)

    def _setup_pretrain_optimizer(self):
        """
        Set up optimizer for the pretrain network.
        """
        self.pretrain_total_loss = self.pretrain_network.total_loss
        self.pretrain_accuracy = self.pretrain_network.accuracy
        self.pretrain_grads = tf_util.flatgrad(self.pretrain_total_loss, self.pretrain_network.variables,
                                               clip_norm=self.clip_norm)
        self.pretrain_optimizer = MpiAdam(var_list=self.pretrain_network.variables, beta1=0.9, beta2=0.999,
                                          epsilon=1e-08)

    def _train_step(self, step: int, writer, log: bool = False):
        """
        Run a step of training from batch

        :param step: the current step iteration
        :param writer: the writer for tensorboard
        :param log: whether or not to log to metadata
        :return: critic loss, actor loss
        """
        # Sample a batch
        (obs, actions, rewards, next_obs, terminals,
         infos, weights, batch_idxes, batch_idxes1, batch_idxes2) = self._sample_batch(
            **{"num_timesteps": self.num_timesteps, "k": self.current_train_step})

        # Reshape to match previous behavior and placeholder shape
        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_q],
                                                        feed_dict={
                                                            self.obs_target: next_obs,
                                                            self.rewards: rewards,
                                                            self.terminals_ph: terminals
                                                        })
            self.ret_rms.update(target_q.flatten())
            self.sess.run(self.renormalize_q_outputs_op, feed_dict={
                self.old_std: np.array([old_std]),
                self.old_mean: np.array([old_mean]),
            })

        else:
            target_q = self.sess.run(self.target_q, feed_dict={
                self.obs_target: next_obs,
                self.rewards: rewards,
                self.terminals_ph: terminals
            })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]

        td_map = {
            self.obs_train: obs,
            self.actions: actions,
            self.action_train_ph: actions,
            self.rewards: rewards,
            self.critic_target: target_q,
            self.param_noise_stddev: 0 if self.param_noise is None else self.param_noise.current_stddev
        }

        if self.use_model_update:
            # If we use actor model update, we have to feed the input placeholder and status and reward ground truth
            # place holder
            status = np.asarray(info_to_status(infos))
            status_onehot = np.zeros((status.size, 3))
            status_onehot[np.arange(status.size), status.astype(int)] = 1
            td_map.update({self.pretrain_network.status_ph: status_onehot,
                           self.pretrain_network.reward_ph: rewards,
                           self.pretrain_network.input_ph: np.concatenate((obs[:, SELECTED_COLUMN], actions),
                                                                          axis=1)})
        if self.use_per:
            # If we use PER, we have feed the weights palace holder for importance sampling
            td_map.update({self.importance_weights_ph: weights})

        if writer is not None:
            # run loss backprop with summary if the step_id was not already logged (can happen with the right
            # parameters as the step value is only an estimate)
            if self.full_tensorboard_log and log and step not in self.tb_seen_steps:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, actor_grads, actor_loss, critic_grads, critic_loss = \
                    self.sess.run([self.summary] + ops, td_map, options=run_options, run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%d' % step)
                self.tb_seen_steps.append(step)
            else:
                summary, actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run([self.summary] + ops,
                                                                                            td_map)
            writer.add_summary(summary, step)
        else:
            actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, td_map)

        self.actor_optimizer.update(actor_grads, learning_rate=self.actor_lr)
        self.critic_optimizer.update(critic_grads, learning_rate=self.critic_lr)

        if self.use_model_update and self.train_pretrain:
            # Update pretrain network
            pretrain_ops = [self.pretrain_grads, self.pretrain_total_loss]
            pretrain_grads, pretrain_loss = self.sess.run(pretrain_ops, td_map)
            self.pretrain_optimizer.update(pretrain_grads, learning_rate=self.pretrain_lr)

        if self.use_per:
            # If we use PER, we have to update the priorities of each sample according to the absolute
            # value of td error after each sampling
            td_errors = np.asarray(self.sess.run([self.td_error], td_map)).flatten()
            new_priorities = np.abs(td_errors) + self.per_eps
            assert isinstance(self.replay_buffer, PrioritizedReplayBufferInfo)
            if self.use_split_buffer:
                self.replay_buffer.update_priorities(batch_idxes1, new_priorities[:len(batch_idxes1)])
                self.replay_buffer_planner.update_priorities(batch_idxes2, new_priorities[len(batch_idxes1):])
            else:
                self.replay_buffer.update_priorities(batch_idxes, new_priorities)

        return critic_loss, actor_loss

    def _get_stats(self):
        """
        Get the mean and standard deviation of the model's inputs and outputs
        :return: the means and stds
        """
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.

            # ignoring multiple values
            # *(variable) used to assign multiple value to a variable as list while unpacking
            # it's called "Extended Unpacking", only available in Python 3.x
            obs, actions, rewards, next_obs, terminals, *_ = self._sample_batch()

            self.stats_sample = {
                'obs': obs,
                'actions': actions,
                'rewards': rewards,
                'next_obs': next_obs,
                'terminals': terminals
            }

        feed_dict = {
            self.actions: self.stats_sample['actions']
        }

        for placeholder in [self.action_train_ph, self.action_target, self.action_adapt_noise, self.action_noise_ph]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['actions']

        for placeholder in [self.obs_train, self.obs_target, self.obs_adapt_noise, self.obs_noise]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['obs']

        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def _adapt_param_noise(self):
        """
        Calculate the adaptation for the parameter noise.
        :return: the mean distance for the parameter noise
        """
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        obs, *_ = self._sample_batch()

        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs_adapt_noise: obs, self.obs_train: obs,
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def _sample_batch(self, **kwargs):
        """
        Sample a batch of transitions.
        :return: obs, actions, rewards, next_obs, terminals, infos, weights, batch_idxes, batch_idxes1, batch_idxes2
        weights and batch_idxes are only for PER. batch_idxes1 and batch_idxes2 are for PER when replay buffer is
        separate for RL and motion planner
        """
        if self.use_split_buffer:
            current_split_factor = self.split_factor_scheduler.value(self.num_timesteps)
            n_planner_rb = int(self.batch_size * current_split_factor)
            n_normal_rb = self.batch_size - n_planner_rb
        else:  # When we use the same replay buffer
            n_normal_rb = self.batch_size

        if self.use_per:
            experience = self.replay_buffer.sample(
                batch_size=n_normal_rb, beta=self.beta_schedule.value(self.num_timesteps), env=self._vec_normalize_env)
            (obs, actions, rewards, next_obs, terminals, infos, weights, batch_idxes) = experience
            batch_idxes1, batch_idxes2 = batch_idxes, None
        else:
            obs, actions, rewards, next_obs, terminals, infos = self.replay_buffer.sample(
                batch_size=n_normal_rb, env=self._vec_normalize_env, **kwargs)
            weights, batch_idxes = np.ones_like(rewards), None
            batch_idxes1, batch_idxes2 = None, None

        if self.use_split_buffer:
            if self.use_per:
                experience2 = self.replay_buffer_planner.sample(batch_size=n_planner_rb,
                                                                beta=self.beta_schedule.value(self.num_timesteps),
                                                                env=self._vec_normalize_env)
                (obs2, actions2, rewards2, next_obs2, terminals2, infos2, weights2, batch_idxes2) = experience2
            else:
                obs2, actions2, rewards2, next_obs2, terminals2, infos2 = self.replay_buffer_planner.sample(
                    batch_size=n_planner_rb,
                    env=self._vec_normalize_env, **kwargs)
                weights2, batch_idxes2 = np.ones_like(rewards2), None

            obs = np.concatenate((obs, obs2))
            actions = np.concatenate((actions, actions2))
            rewards = np.concatenate((rewards, rewards2))
            next_obs = np.concatenate((next_obs, next_obs2))
            terminals = np.concatenate((terminals, terminals2))
            infos = infos + infos2
            weights = np.concatenate((weights, weights2))
            if batch_idxes1 is not None and batch_idxes2 is not None:
                batch_idxes = np.concatenate((batch_idxes1, batch_idxes2))
            else:
                batch_idxes = None

        return obs, actions, rewards, next_obs, terminals, infos, weights, batch_idxes, batch_idxes1, batch_idxes2

    def _setup_replay_buffer(self, total_timesteps, use_per: bool, use_ere: bool, split_buffer: bool):
        """
        Set up for replay buffer. You can choose from default (uniform sampling), PER, and ERE, with same or separate
        replay buffer.
        :param total_timesteps: total training time steps
        :param use_per: whether or not use PER
        :param use_ere: whether or not use ERE
        :param split_buffer: whether or not use separate buffer
        """
        assert not (use_per and use_ere), "You can either use normal replay buffer, PER, or ERE."
        if use_per:
            if self.per_beta_iters:
                prioritized_replay_beta_iters = self.per_beta_iters
            else:
                prioritized_replay_beta_iters = total_timesteps
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                initial_p=self.per_beta0,
                                                final_p=1.0)
            self.replay_buffer = PrioritizedReplayBufferInfo(self.buffer_size,
                                                             alpha=self.per_alpha)
        elif use_ere:
            self.replay_buffer = EmphasizeRecentExperienceBufferInfo(size=self.buffer_size,
                                                                     total_timesteps=total_timesteps,
                                                                     nb_train_steps=self.nb_train_steps)
        else:
            self.replay_buffer = ReplayBufferInfo(self.buffer_size)

        if split_buffer:
            self.replay_buffer_planner = deepcopy(self.replay_buffer)
            self.split_factor_scheduler = LinearSchedule(total_timesteps,
                                                         initial_p=self.split_factor_initial,
                                                         final_p=self.split_factor_final)

    def _setup_plan_result(self):
        """
        Load pre-computed planning results.
        """
        self.planned_scenarios = dict()
        fns = glob.glob(os.path.join(self.plan_path, "*.pickle"))
        for fn in fns:
            with open(fn, 'rb') as f:
                self.planned_scenarios[os.path.basename(fn).split(".")[0]] = pickle.load(f)

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DDPG",
              reset_num_timesteps=True, replay_wrapper=None):
        """
        Train the DDPGPlan agent.
        :param total_timesteps: the total number of samples to train on
        :param callback: function called at every steps with state of the algorithm
        :param log_interval: the number of timesteps before logging
        :param tb_log_name: the name of the run for tensorboard log
        :param reset_num_timesteps: whether or not to reset the current time step number (used in logging)
        :param replay_wrapper: replay wrapper
        :return: the trained model
        """
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        self._setup_replay_buffer(total_timesteps, self.use_per, self.use_ere, self.use_split_buffer)

        if self.use_plan:
            self._setup_plan_result()

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)
            if self.use_split_buffer:
                self.replay_buffer_planner = replay_wrapper(self.replay_buffer_planner)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # A list for tensorboard logging, to prevent logging with the same step number, if it already occurred
            self.tb_seen_steps = []

            rank = MPI.COMM_WORLD.Get_rank()

            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            episode_successes = []

            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything
                self._reset()
                obs = self.env.reset()
                # Retrieve unnormalized observation for saving into the buffer
                if self._vec_normalize_env is not None:
                    obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                eval_obs = None
                if self.eval_env is not None:
                    eval_obs = self.eval_env.reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                eval_episode_rewards = []
                eval_qs = []
                epoch_actions = []
                epoch_qs = []
                epoch_episodes = 0
                epoch = 0

                callback.on_training_start(locals(), globals())

                while True:
                    for _ in range(log_interval):
                        callback.on_rollout_start()
                        # Perform rollouts.
                        for _ in range(self.nb_rollout_steps):

                            if total_steps >= total_timesteps:
                                callback.on_training_end()
                                return self

                            # Predict next action.
                            action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                            assert action.shape == self.env.action_space.shape

                            # Execute next action.
                            if rank == 0 and self.render:
                                self.env.render()

                            # Randomly sample actions from a uniform distribution
                            # with a probability self.random_exploration (used in HER + DDPG)
                            if np.random.rand() < self.random_exploration:
                                # actions sampled from action space are from range specific to the environment
                                # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                                unscaled_action = self.action_space.sample()
                                action = scale_action(self.action_space, unscaled_action)
                            else:
                                # inferred actions need to be transformed to environment action_space before stepping
                                unscaled_action = unscale_action(self.action_space, action)

                            new_obs, reward, done, info = self.env.step(unscaled_action)

                            self.num_timesteps += 1

                            if callback.on_step() is False:
                                callback.on_training_end()
                                return self

                            step += 1
                            total_steps += 1
                            if rank == 0 and self.render:
                                self.env.render()

                            # Book-keeping.
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)

                            # Store only the unnormalized version
                            if self._vec_normalize_env is not None:
                                new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                                reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                            else:
                                # Avoid changing the original ones
                                obs_, new_obs_, reward_ = obs, new_obs, reward

                            self._store_transition(obs_, action, reward_, new_obs_, done, info)

                            obs = new_obs
                            # Save the unnormalized observation
                            if self._vec_normalize_env is not None:
                                obs_ = new_obs_

                            episode_reward += reward_
                            episode_step += 1

                            if writer is not None:
                                ep_rew = np.array([reward_]).reshape((1, -1))
                                ep_done = np.array([done]).reshape((1, -1))
                                tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                                    writer, self.num_timesteps)

                            if done:
                                # Episode done.
                                epoch_episode_rewards.append(episode_reward)
                                episode_rewards_history.append(episode_reward)
                                epoch_episode_steps.append(episode_step)
                                episode_reward = 0.
                                episode_step = 0
                                epoch_episodes += 1
                                episodes += 1

                                maybe_is_success = info.get('is_success')
                                if maybe_is_success is not None:
                                    episode_successes.append(float(maybe_is_success))

                                if self.use_plan and not info.get('is_goal_reached', False):
                                    # If the episode is not successful, we store the successful plan results
                                    self.store_plan_result(info)

                                self._reset()
                                if not isinstance(self.env, VecEnv):
                                    obs = self.env.reset()

                        callback.on_rollout_end()
                        # Train
                        epoch_actor_losses = []
                        epoch_critic_losses = []
                        epoch_adaptive_distances = []
                        for t_train in range(self.nb_train_steps):
                            self.current_train_step = t_train
                            # Not enough samples in the replay buffer
                            if self.use_split_buffer:
                                current_split_factor = self.split_factor_scheduler.value(self.num_timesteps)
                                n_planner_rb = int(self.batch_size * current_split_factor)
                                n_normal_rb = self.batch_size - n_planner_rb
                                if not self.replay_buffer.can_sample(
                                        n_normal_rb) or not self.replay_buffer_planner.can_sample(n_planner_rb):
                                    break
                            else:
                                n_normal_rb = self.batch_size
                                if not self.replay_buffer.can_sample(self.batch_size):
                                    break

                            # Adapt param noise, if necessary.
                            if len(self.replay_buffer) >= self.batch_size and \
                                    t_train % self.param_noise_adaption_interval == 0:
                                distance = self._adapt_param_noise()
                                epoch_adaptive_distances.append(distance)

                            # weird equation to deal with the fact the nb_train_steps will be different
                            # to nb_rollout_steps
                            step = (int(t_train * (self.nb_rollout_steps / self.nb_train_steps)) +
                                    self.num_timesteps - self.nb_rollout_steps)

                            critic_loss, actor_loss = self._train_step(step, writer, log=t_train == 0)
                            epoch_critic_losses.append(critic_loss)
                            epoch_actor_losses.append(actor_loss)
                            self._update_target_net()

                        # Evaluate.
                        eval_episode_rewards = []
                        eval_qs = []
                        if self.eval_env is not None:
                            eval_episode_reward = 0.
                            for _ in range(self.nb_eval_steps):
                                if total_steps >= total_timesteps:
                                    return self

                                eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                                unscaled_action = unscale_action(self.action_space, eval_action)
                                eval_obs, eval_r, eval_done, _ = self.eval_env.step(unscaled_action)
                                if self.render_eval:
                                    self.eval_env.render()
                                eval_episode_reward += eval_r

                                eval_qs.append(eval_q)
                                if eval_done:
                                    if not isinstance(self.env, VecEnv):
                                        eval_obs = self.eval_env.reset()
                                    eval_episode_rewards.append(eval_episode_reward)
                                    eval_episode_rewards_history.append(eval_episode_reward)
                                    eval_episode_reward = 0.

                    mpi_size = MPI.COMM_WORLD.Get_size()

                    # Not enough samples in the replay buffer
                    if self.use_split_buffer:
                        n_planner_rb = int(self.batch_size * self.split_factor_initial)
                        n_normal_rb = self.batch_size - n_planner_rb
                        if not self.replay_buffer.can_sample(
                                n_normal_rb) or not self.replay_buffer_planner.can_sample(n_planner_rb):
                            continue
                    else:
                        if not self.replay_buffer.can_sample(self.batch_size):
                            continue

                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                    combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                    combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                    combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                    combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                    combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                    if len(epoch_adaptive_distances) != 0:
                        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rollout/episodes'] = epoch_episodes
                    combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                    # Evaluation statistics.
                    if self.eval_env is not None:
                        combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                        combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                        combined_stats['eval/Q'] = np.mean(eval_qs)
                        combined_stats['eval/episodes'] = len(eval_episode_rewards)

                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/epochs'] = epoch + 1
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.dump_tabular()
                    logger.info('')
                    logdir = logger.get_dir()
                    if rank == 0 and logdir:
                        if hasattr(self.env, 'get_state'):
                            with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.env.get_state(), file_handler)
                        if self.eval_env and hasattr(self.eval_env, 'get_state'):
                            with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.eval_env.get_state(), file_handler)

    def store_plan_result(self, info):
        """
        Store successful plan results in replay buffer.
        :param info: info from the saved motion planner results
        """
        plan = self.planned_scenarios.get(info["scenario_name"], None)
        if plan is not None and plan["infos"][-1]["is_goal_reached"]:
            observations = plan["observations"]
            actions = plan["actions"]
            rewards = plan["rewards"]
            next_observations = plan["next_observations"]
            dones = plan["dones"]
            infos = plan["infos"]
            for o, a, r, no, d, i in zip(observations, actions, rewards, next_observations,
                                         dones, infos):
                self._store_transition(obs=o, action=a, reward=r, next_obs=no, done=d,
                                       info=i, is_plan=True)

    def _store_transition(self, obs, action, reward, next_obs, done, info, **kwargs):
        """
        Store a transition in the replay buffer
        :param obs: the last observation
        :param action: the action
        :param reward: the reward
        :param next_obs: the current observation
        :param done: whether the episode is over
        :param info: extra values used to compute reward when using HER
        """
        is_plan = kwargs.get("is_plan", False)
        reward *= self.reward_scale
        if self.use_split_buffer and is_plan:
            self.replay_buffer_planner.add(obs, action, reward, next_obs, float(done), info=info)
        else:
            self.replay_buffer.add(obs, action, reward, next_obs, float(done), info=info)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs]))

    def save(self, save_path, cloudpickle=False):
        """
        Save parameters and data. If use pretrain, also save pretrain network and weights.
        :param save_path: path to save
        :param cloudpickle: whether or not use cloud pickle
        """
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "nb_eval_steps": self.nb_eval_steps,
            "param_noise_adaption_interval": self.param_noise_adaption_interval,
            "nb_train_steps": self.nb_train_steps,
            "nb_rollout_steps": self.nb_rollout_steps,
            "verbose": self.verbose,
            "param_noise": self.param_noise,
            "action_noise": self.action_noise,
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_returns": self.normalize_returns,
            "enable_popart": self.enable_popart,
            "normalize_observations": self.normalize_observations,
            "batch_size": self.batch_size,
            "observation_range": self.observation_range,
            "return_range": self.return_range,
            "critic_l2_reg": self.critic_l2_reg,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "clip_norm": self.clip_norm,
            "reward_scale": self.reward_scale,
            "memory_limit": self.memory_limit,
            "buffer_size": self.buffer_size,
            "random_exploration": self.random_exploration,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "use_plan": self.use_plan,
            "plan_path": self.plan_path,
            "use_model_update": self.use_model_update,
            "pretrain_path": self.pretrain_path,
            "use_split_buffer": self.use_split_buffer,
            "split_factor_initial": self.split_factor_initial,
            "split_factor_final": self.split_factor_final,
            "use_per": self.use_per,
            "per_alpha": self.per_alpha,
            "per_beta0": self.per_beta0,
            "per_beta_iters": self.per_beta_iters,
            "per_eps": self.per_eps,
            "use_ere": self.use_ere,
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path,
                           data=data,
                           params=params_to_save,
                           cloudpickle=cloudpickle)
        if self.use_model_update:
            self.pretrain_network.save_model(self.sess, "pretrain", os.path.dirname(save_path))

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Load trained agent.
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()
        # Patch for version < v2.6.0, duplicated keys where saved
        if len(params) > len(model.get_parameter_list()):
            n_params = len(model.params)
            n_target_params = len(model.target_params)
            n_normalisation_params = len(model.obs_rms_params) + len(model.ret_rms_params)
            # Check that the issue is the one from
            # https://github.com/hill-a/stable-baselines/issues/363
            assert len(params) == 2 * (n_params + n_target_params) + n_normalisation_params, \
                "The number of parameter saved differs from the number of parameters" \
                " that should be loaded: {}!={}".format(len(params), len(model.get_parameter_list()))
            # Remove duplicates
            params_ = params[:n_params + n_target_params]
            if n_normalisation_params > 0:
                params_ += params[-n_normalisation_params:]
            params = params_
        model.load_parameters(params)

        if data["use_model_update"]:
            # If use actor model update, load pretrain network from last training
            last_pretrain_path = os.path.join(os.path.dirname(load_path), "pretrain")
            model.restore_pretrain_network(last_pretrain_path)
        return model
