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
