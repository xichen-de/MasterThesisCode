#  MIT License
#
#  Copyright 2021 Xi Chen
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import gym
import numpy as np
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.trajectory import State
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


class CommonroadGoalEnvWrapper(gym.GoalEnv):
    """
    A wrapper that allow to use CommonroadEnv with HER.
    It uses dict observation space (coming from GoalEnv) with the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.
    """

    def __init__(self, **kwargs):
        """
        Initialize wrapper.
        """
        self.env = CommonroadEnv(**kwargs)
        # To use CommonroadEnv with HER, we have to set the observation space as dict.
        # achieved_goal: [ego_vehicle_position_x, ego_vehicle_position_y, ego_vehicle_orientation]
        # desired_goal: [goal_position_x, goal_position_y, goal_orientation]
        self.observation_space = gym.spaces.Dict(dict(
            observation=self.env.observation_space,
            achieved_goal=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype='float32'),
            desired_goal=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype='float32')
        ))
        self.action_space = self.env.action_space

    def reset(self):
        """
        Reset environment and return observation dict.
        :return: observation_dict
        """
        state_vector = self.env.reset()
        achieved_goal, desired_goal = self.get_achieved_desired_goal()

        observation_dict = {'observation': state_vector,
                            'achieved_goal': achieved_goal,
                            'desired_goal': desired_goal}

        return observation_dict

    def get_achieved_desired_goal(self):
        """
        Generate achieved and desired goal.
        :return:
        """
        ego_vehicle_state = self.env.ego_vehicle.state
        goal_state = self.env.planning_problem.goal.state_list[0]
        achieved_goal = np.asarray([ego_vehicle_state.position[0],
                                    ego_vehicle_state.position[1],
                                    ego_vehicle_state.orientation])
        desired_goal = np.asarray([goal_state.position.center[0],
                                   goal_state.position.center[1],
                                   0.5 * (goal_state.orientation.start + goal_state.orientation.end)])
        return achieved_goal, desired_goal

    def step(self, action: np.ndarray):
        """
        Step the environment with action.
        :param action: the action
        :return: observation, reward, done, info
        """
        state_vector, reward, done, info = self.env.step(action)
        achieved_goal, desired_goal = self.get_achieved_desired_goal()

        info.update({
            "distance_goal_long": self.env.observation_dict['distance_goal_long'][0],
            "distance_goal_lat": self.env.observation_dict['distance_goal_lat'][0],
            "previous_state": self.env.ego_vehicle.previous_state,
            "current_state": self.env.ego_vehicle.state
        })

        observation_dict = {'observation': state_vector,
                            'achieved_goal': achieved_goal,
                            'desired_goal': desired_goal}
        return observation_dict, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment.
        :param mode: render mode
        """
        self.env.render()

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict):
        """
        Compute the reward. Only support sparse reward.
        :param achieved_goal: the achieved goal
        :param desired_goal: the desired goal
        :param info: info dict
        :return: reward value
        """
        reward = 0.0

        # Generate goal region using desired goal
        original_goal_state = self.env.planning_problem.goal.state_list[0]
        position_rect = Rectangle(length=original_goal_state.position.length,
                                  width=original_goal_state.position.width,
                                  center=np.array([desired_goal[0], desired_goal[1]]),
                                  orientation=desired_goal[2])
        current_goal_distance = np.sqrt(info["distance_goal_long"] ** 2 + info["distance_goal_lat"] ** 2)

        # Modify time and velocity interval according to covered distance
        distance_frac = max(self.env.initial_goal_dist - current_goal_distance, 0) / self.env.initial_goal_dist
        time_step_interval = Interval(
            distance_frac * self.env.planning_problem.goal.state_list[0].time_step.start,
            distance_frac * self.env.planning_problem.goal.state_list[0].time_step.end)
        velocity_interval = Interval(
            (1 - distance_frac) * self.env.planning_problem.initial_state.velocity + (
                original_goal_state.velocity.start) * distance_frac,
            (1 - distance_frac) * self.env.planning_problem.initial_state.velocity + (
                original_goal_state.velocity.end) * distance_frac,
        )
        goal_region_state = State(position=position_rect,
                                  orientation=original_goal_state.orientation,
                                  velocity=velocity_interval,
                                  time_step=time_step_interval)
        sampled_goal_region = GoalRegion(state_list=[goal_region_state])

        reached_state = info['current_state']
        reach_goal = sampled_goal_region.is_reached(reached_state)

        # Reach goal
        if reach_goal:
            reward += self.env.reward_goal_reached

        # Collision
        if info["is_collision"]:
            reward += self.env.reward_collision

        # Off-road
        if info["is_off_road"]:
            reward += self.env.reward_off_road

        # Time-out
        if info["is_time_out"]:
            reward += self.env.reward_time_out
        return reward
