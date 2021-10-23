"""
This module implements three experience replay strategies: uniform sampling, Prioritized Experience Replay,
and Emphasize Recent Experience.
"""
import random
from typing import Optional, List, Union

import numpy as np
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from stable_baselines.common.vec_env import VecNormalize

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


class ReplayBufferInfo(ReplayBuffer):
    """
    Default replay buffer with info buffer, it uses FIFO.
    """

    def __init__(self, size: int):
        """
        Initialize reply buffer.
        :param size: maximal number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped
        """
        super().__init__(size)

    def add(self, obs_t: np.ndarray, action: np.ndarray, reward: float, obs_tp1: np.ndarray, done: bool,
            **kwargs: dict):
        """
        Add a new transition to the buffer.
        :param obs_t: previous observation
        :param action: action
        :param reward: reward
        :param obs_tp1: current observation
        :param done: is the episode done
        """
        info = kwargs.get("info", None)
        data = (obs_t, action, reward, obs_tp1, done, info)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t: np.ndarray, action: np.ndarray, reward: np.ndarray, obs_tp1: np.ndarray, done: np.ndarray,
               **kwargs: dict):
        """
        Add a new batch of transitions to the buffer.
        :param obs_t: the last batch of observations
        :param action: the batch of actions
        :param reward: the batch of the rewards of the transition
        :param obs_tp1: the current batch of observations
        :param done: terminal status of the batch
        """
        info = kwargs.get("info", None)
        for data in zip(obs_t, action, reward, obs_tp1, done, info):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: Union[List[int], np.ndarray], env: Optional[VecNormalize] = None):
        """
        Sample a batch of transitions.
        :param idxes: indexes of transition
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :return: transitions
        """
        obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, info = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            infos.append(info)
        return (self._normalize_obs(np.array(obses_t), env),
                np.array(actions),
                self._normalize_reward(np.array(rewards), env),
                self._normalize_obs(np.array(obses_tp1), env),
                np.array(dones),
                infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, **_kwargs):
        """
        Sample a batch of experiences.
        :param batch_size: how many transitions to sample
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :return: batch of transitions
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, env=env)


class EmphasizeRecentExperienceBufferInfo(ReplayBufferInfo):
    """
    Boosting Soft Actor-Critic: Emphasizing Recent Experience without Forgetting the Past.
    Link: https://arxiv.org/abs/1906.04009
    """

    def __init__(self, size: int, total_timesteps: int, nb_train_steps: int, initial_eta: float = 0.996,
                 final_eta: float = 1.0, c_k_min=100000):
        """
        Initialize replay buffer.
        :param size: maximal number of transitions to store in the buffer
        :param total_timesteps: total time steps of training
        :param nb_train_steps: the number of training steps
        :param initial_eta: initial eta value
        :param final_eta: final eta value
        :param c_k_min: minimal value of transitions to draw
        """
        super().__init__(size)
        self.eta_scheduler = LinearSchedule(total_timesteps,
                                            initial_p=initial_eta,
                                            final_p=final_eta)
        self.K = nb_train_steps
        self.c_k_min = c_k_min

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, num_timesteps: int = None, k: int = None,
               **_kwargs):
        """
        Sample a batch of transitions.
        :param batch_size: how many transitions to sample
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :param num_timesteps: current time step
        :param k: parameter (see paper)
        :return: batch of transitions
        """
        if k and num_timesteps:
            current_eta = self.eta_scheduler.value(num_timesteps)
            c_k = max(int(self.buffer_size * current_eta ** (k * 1000 / self.K)), self.c_k_min)
            if c_k > len(self._storage) - 1:
                c_k = len(self._storage) - 1
            idxes = [random.randint(len(self._storage) - 1 - c_k, len(self._storage) - 1) for _ in range(batch_size)]
        else:
            idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, env=env)


class PrioritizedReplayBufferInfo(ReplayBufferInfo):
    """
    Prioritized replay buffer with info.
    Link: https://arxiv.org/abs/1511.05952
    """

    def __init__(self, size: int, alpha: float):
        """
        Initialize replay buffer.
        :param size: max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBufferInfo, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t: np.ndarray, action: np.ndarray, reward: float, obs_tp1: np.ndarray, done: bool,
            **kwargs: dict):
        """
        Add a new transition to the buffer.
        :param obs_t: previous observation
        :param action: action
        :param reward: reward
        :param obs_tp1: current observation
        :param done: is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def extend(self, obs_t: np.ndarray, action: np.ndarray, reward: np.ndarray, obs_tp1: np.ndarray, done: np.ndarray,
               **kwargs: dict):
        """
        Add a new batch of transitions to the buffer.
        :param obs_t: the last batch of observations
        :param action: the batch of actions
        :param reward: the batch of the rewards of the transition
        :param obs_tp1: the current batch of observations
        :param done: terminal status of the batch
        """
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done, **kwargs)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha
            idx = (idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0, env: Optional[VecNormalize] = None, **kwargs):
        """
        Sample a batch of experiences, it also returns importance weights and idxes of sampled experiences.
        :param batch_size: how many transitions to sample
        :param beta: to what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :return: a batch of transitions along with weights and idxes
        """
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes, env=env)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray):
        """
        Update priorities of sampled transitions.
        :param idxes: list of idxes of sampled transitions
        :param priorities: list of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))
