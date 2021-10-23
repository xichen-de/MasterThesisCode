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

import gym
from src.algorithm.her_commonroad.her_cr import HERCommonroad
from src.tools.divide_files import ROOT_STR
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from stable_baselines import DDPG

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

META_SCENARIO_PATH = os.path.join(ROOT_STR, "tests", "resources", "pickles", "meta_scenario")
PROBLEM_PATH = os.path.join(ROOT_STR, "tests", "resources", "pickles", "problem")
VIZ_PATH = os.path.join(ROOT_STR, "tests", "rl")
CONFIG = os.path.join(ROOT_STR, "tests", "resources", "rl", "configs.yaml")


def test_her_cr():
    env = gym.make("commonroad-v1", **{"meta_scenario_path": META_SCENARIO_PATH,
                                       "train_reset_config_path": PROBLEM_PATH,
                                       "test_reset_config_path": PROBLEM_PATH,
                                       "visualization_path": VIZ_PATH,
                                       "config_file": CONFIG})
    model = HERCommonroad('MlpPolicy', env, model_class=DDPG).learn(1000)
    model.save('hercr.zip')
    model = HERCommonroad.load('hercr.zip', env=env)
    model.learn(1000)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
