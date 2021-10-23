"""
This module implements the experience replay strategy for HER. In order to use HER with CommonroadEnv,
we have to update the goal-related observation after sampling additional goal. Also we remove the invalid goal,
i.e., transition that is collision or off-road.
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

import copy

import numpy as np
from stable_baselines.her import HindsightExperienceReplayWrapper, GoalSelectionStrategy

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


def update_goal_distance(obs_dict, next_obs_dict, goal, info):
    """
    Update the goal-related observation: longitudinal and lateral distance to goal region.
    Because in the highD dataset, the lanelets are approximately straight, we use the x, y distance difference to
    replace calculation in the curvilinear cosy. The function in CommonroadEnv is much slower than this, therefore,
    in order to use this function for other dataset, another method need to be come up with.
    :param obs_dict: previous observation dict
    :param next_obs_dict: current observation dict
    :param goal: desired goal
    :param info: info buffer
    :return: updated observation dict
    """
    (new_goal_distance_lat,
     new_goal_distance_long) = (goal[1] - info["previous_state"].position[1],
                                goal[0] - info["previous_state"].position[0])
    (new_goal_distance_lat_next,
     new_goal_distance_long_next) = (goal[1] - info["current_state"].position[1],
                                     goal[0] - info["current_state"].position[0])
    obs_dict["observation"][1] = new_goal_distance_lat
    obs_dict["observation"][2] = new_goal_distance_long
    next_obs_dict["observation"][1] = new_goal_distance_lat_next
    next_obs_dict["observation"][2] = new_goal_distance_long_next
    return obs_dict, next_obs_dict


class HERCrBufferWrapper(HindsightExperienceReplayWrapper):
    """
    Wrapper around a replay buffer in order to use HERCommonroad.
    """

    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, wrapped_env):
        super().__init__(replay_buffer, n_sampled_goal, goal_selection_strategy,
                         wrapped_env)

    def add(self, obs_t, action, reward, obs_tp1, done, info):
        """
        Add a new transition to the buffer
        :param obs_t: the last observation
        :param action: the action
        :param reward: the reward of the transition
        :param obs_tp1: the new observation
        :param done: is the episode done
        :param info: extra values used to compute reward
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append((obs_t, action, reward, obs_tp1, done, info))
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, action, reward, obs_tp1, done, info = transition

            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.episode_transitions) - 1 and
                    self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                # Copy transition to avoid modifying the original one
                obs, action, reward, next_obs, done, info = copy.deepcopy(transition)

                # Convert concatenated obs to dict, so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # Update the desired goal in the transition
                obs_dict['desired_goal'] = goal
                next_obs_dict['desired_goal'] = goal

                # Update the reward according to the new desired goal
                reward = self.env.compute_reward(next_obs_dict['achieved_goal'], goal, info)

                # Update goal-related observations
                obs_dict, next_obs_dict = update_goal_distance(obs_dict, next_obs_dict,
                                                               goal, info)

                done = False

                # Transform back to np.ndarray
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add artificial transition to the replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.
        :param episode_transitions: a list of all the transitions in the current episode
        :param transition_idx: the transition to start sampling from
        :return: an achieved goal and if it is a free sample (not collision or off-road)
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return (self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal'],
                not (selected_transition[5]["is_collision"] or selected_transition[5]["is_off_road"]))

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.

        :param episode_transitions: list of the transitions in the current episode
        :param transition_idx: the transition to start sampling from
        :return: an achieved goal
        """
        sampled_goals = [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]
        # Remove collision and off-road samples
        valid_goals = [g[0] for g in sampled_goals if g[1]]
        return valid_goals
