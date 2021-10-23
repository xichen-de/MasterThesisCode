import gym

try:
    print('[src/__init__.py] Trying to register "commonroad-v1 (CommonroadGoalEnvWrapper)" ...')
    gym.envs.register(
        id="commonroad-v1",
        entry_point="src.algorithm.her_commonroad.env_wrapper:CommonroadGoalEnvWrapper",
        kwargs=None,
    )
except gym.error.Error:
    print("[src/__init__.py] Error occurs while registering commonroad-v1")
    pass
