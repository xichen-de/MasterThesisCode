# Brief Guide to Major Files

## run_stable_baselines.py

The main file to run DDPGPlan or HER with commonroad-rl environment. Here we only have used the training from scratch function, but the script also provides continual training, hyperparameters/configurations optimizing function. For these two functions, please refer to [guide](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl/-/tree/development/commonroad_rl).

For training from scratch, the often-used flags are:

- `--env` (str): in this repository, we use `commonroad-v0` (works with `ddpg` and `ddpgplan`), and `commonroad-v1` (works with `hercr`)
- `--algo` (str): in this repository, we use `ddpg` for DDPG, `ddpgplan` for DDPGPlan, and `hercr` for modified HER
- `--n-timesteps` (int): set the number of training time steps, default value is 1e6
- `--eval-freq` (int): evaluate the agent every n steps (if negative, no evaluation), default value is 1000
- `--eval-episodes` (int): number of episodes to use for evaluation, default value is 5
- `--save-freq` (int): save the model every n steps (if negative, no checkpoint), default value is -1
- `--log-folder` (int): log folder, default value is "../logs"
- `--seed` (int): random seed, default value is 0

-  `--hyperparams-path` (str): path to file for overwriting model hyperparameters. The default hyperparameters are saved as `./hyperparams/<model>.yml`
- `--configs-path` (str): Path to file for overwriting environment configurations. The default configuration file is saved as `./configs.yaml`
- `--info-keywords` (tuple): extra information to log, from the information return of environment step, see `stable_baselines/bench/monitor.py`. In this repository, usually we set `--info-keywords` as `is_goal_reached is_collision is_off_road is_time_out` to log the episode outcome for plotting

## play_stable_baselines.py

For evaluation and visualization of a trained agent. We can also use a trained agent to save transitions for the pretrain network. The often-used flags are:

- `--env` (str): in this repository, we use `commonroad-v0` (works with `ddpg` and `ddpgplan`), and `commonroad-v1` (works with `hercr`)
- `--algo` (str): in this repository, we use `ddpg` for DDPG, `ddpgplan` for DDPGPlan, and `hercr` for modified HER
- `--folder` (str): log folder
- `--n-timesteps` (int): number of time steps
- `--exp-id` (int): experiment ID (-1: no separate experiment folder, 0: latest), default is 0
- `--no-render`: do not render the environment (useful for tests), default is true
- `--configs-path` (str): path to file for overwriting environment configurations. The default configuration file is saved as `./configs.yaml`
- `--reward-log` (str): where to log reward, when empty, no log
- `--save-trans`: save transition as pickle file, default is False
- `--trans-path` (str): directory to save transition, default is `$PROJECT_ROOT/data/input_data`
- `--trans-name` (str): name of transition file, default is pretrain
- `--random-agent`: use a random agent instead of a trained agent, default is False

## configs.yaml

In this repository, we keep the environment configuration mostly fixed, because we use the same observation for all experiments. The only thing we have changed is the path configuration to overwrite the default commonroad-rl settings:

- `train_reset_config_path` (str): path for training pickles
- `test_reset_config_path`(str): path for testing pickles
- `meta_scenario_path`(str): path for meta scenario pickles