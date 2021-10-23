import os

import gym
from src.algorithm.ddpg_mp.ddpg_plan import DDPGPlan
from src.tools.divide_files import ROOT_STR
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

META_SCENARIO_PATH = os.path.join(ROOT_STR, "tests", "resources", "pickles", "meta_scenario")
PROBLEM_PATH = os.path.join(ROOT_STR, "tests", "resources", "pickles", "problem")
VIZ_PATH = os.path.join(ROOT_STR, "tests", "rl")
CONFIG = os.path.join(ROOT_STR, "tests", "resources", "rl", "configs.yaml")
PLAN_PATH = os.path.join(ROOT_STR, "tests", "resources", "plan", "exp")
PRETRAIN_PATH = os.path.join(ROOT_STR, "tests", "resources", "pretrain", "2020_10_08_15_10_25")


def test_ddpg_plan():
    env = gym.make("commonroad-v0", **{"meta_scenario_path": META_SCENARIO_PATH,
                                       "train_reset_config_path": PROBLEM_PATH,
                                       "test_reset_config_path": PROBLEM_PATH,
                                       "visualization_path": VIZ_PATH,
                                       "config_file": CONFIG})
    model = DDPGPlan('MlpPolicy', env, use_plan=True, plan_path=PLAN_PATH, use_model_update=True,
                     pretrain_path=PRETRAIN_PATH).learn(1000)
    model.save('ddpgplan.zip')
    model = DDPGPlan.load('ddpgplan.zip', env=env)
    model.learn(1000)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
