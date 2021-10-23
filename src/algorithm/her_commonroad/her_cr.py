#  MIT License
#
#  Copyright 2021 Xi Chen
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import functools

from src.algorithm.her_commonroad.replay_buffer import HERCrBufferWrapper
from stable_baselines import HER
from stable_baselines.her import HERGoalEnvWrapper

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


class HERCommonroad(HER):
    """
    Hindsight Experience Replay (HER). Link: https://arxiv.org/abs/1707.01495
    This module allows to use HER with CommonroadEnv. Default HER cannot be used with CommonroadEnv,
    because the observation space is not complied with stable-baselines API (not dict).
    Also, to use HER, when sampled intermediate goal, we have to update the goal-related observation.
    """

    def __init__(self, policy, env, model_class, n_sampled_goal=4,
                 goal_selection_strategy='future', *args, **kwargs):
        super().__init__(policy, env, model_class, n_sampled_goal, goal_selection_strategy, *args, **kwargs)

    def _create_replay_wrapper(self, env):
        """
        Wrap the environment in a HERGoalEnvWrapper, if needed and create the replay buffer wrapper.
        """
        if not isinstance(env, HERGoalEnvWrapper):
            env = HERGoalEnvWrapper(env)

        self.env = env

        self.replay_wrapper = functools.partial(HERCrBufferWrapper,
                                                n_sampled_goal=self.n_sampled_goal,
                                                goal_selection_strategy=self.goal_selection_strategy,
                                                wrapped_env=self.env)
