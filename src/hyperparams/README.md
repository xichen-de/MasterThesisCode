# Hyperparameters Settings

In this folder, we save the default hyperparameters for three algorithms, namely, DDPG, DDPGPlan, and HERCommonroad. The file name is the name of the algorithm, and the content is `KEY_WORD:VALUE`. In this project, we only overwrite a few hyperparameters, they are presented in each *.yml file. 

For more detailed information about algorithms, please check out the algorithm or the [Stable-Baselines Documentation](https://stable-baselines.readthedocs.io/en/master/).

To use DDPGPlan, we have the following additional flags for hyperparameters:

## Motion Planner

- `use_plan` (bool): whether or not use motion planner results in replay buffer, default is False
- `plan_path` (str): the path to pre-computed motion planner results, default is None

## Actor Model Update

- `use_model_update` (bool): whether or not use actor model update, default is False
- `pretrain_path` (str): path to pretrain transition and reward network, default is None
- `pretrain_lr` (float): the pretrain network learning rate, default is 1e-3

## Experience Replay

### Separate Replay Buffer for Motion Planner

We use a separate buffer for results from the motion planner. The fraction of samples from this buffer is linearly scheduled. 
```math
  f_{\mathrm{PLAN}}=f_{0}+\left(f_{T}-f_{0}\right)\cdot\frac{t}{T} \\
  f_{\mathrm{DDPG}}=1-f_{\mathrm{PLAN}}                                      
```
where $`T`$ is the total number of time steps in training, $`t`$ denotes the current training time step, $`f_{0}`$ and $`f_{T}`$ are the initial and final fraction value.

The flags are:

- `use_split_buffer` (bool): whether or not use a separate buffer for motion planner results, default is False
- `split_factor_initial` (float): how much should result from separate replay buffer for motion planner should take
          up when sampling, at the beginning of training, default is 0.8
- `split_factor_final` (float): how much should result from separate replay buffer for motion planner should take
          up when sampling, at the end of the training, default is 0.1

### Prioritized Experience Replay

For more detail, please refer to the paper: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

- `use_per` (bool): whether or not use Prioritized Experience Replay, default is False
- `per_alpha` (float): alpha parameter for prioritized replay buffer, default is 0.6
- `per_beta0` (float): initial value of beta for prioritized replay buffer, default is 0.4
- `per_beta_iters` (float): number of iterations over which beta will be annealed from initial
              value to 1.0. If set to None equals to max_timesteps, default is None
- `per_eps` (float): epsilon to add to the TD errors when updating priorities, default is 1e-6

### Emphasize Recent Experience

For more detail, please refer to the paper: [Boosting Soft Actor-Critic: Emphasizing Recent Experience without Forgetting the Past](https://arxiv.org/abs/1906.04009).

- `use_ere` (bool): whether or not use Emphasize Recent Experience, default is False