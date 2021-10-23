# Master Thesis Xi Chen

This repository contains the documents and software package the author wrote or implemented during her master thesis "Learning Driving Policies Using Reinforcement Learning Combined with Motion Planning". 

A method combining [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) (DDPG) with a sampling-based trajectory planner, the [Reactive Planner](https://gitlab.lrz.de/cps/reactive-planner) is developed. This repository also includes the first attempt of applying modified [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER) on [commonroad-rl](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl) environment. The DDPG and HER algorithms are based on the implementation of [Stable-Baselines](https://github.com/hill-a/stable-baselines).

## Folder Structure

- **documents**: Thesis-related documents (exposé, presentation, master thesis)
- **src**
  - **algorithm**
    - **ddpg_mp**: DDPG with motion planner and actor model update
      - **ddpg_plan.py**: Main script of DDPGPlan
      - **replay_buffer.py**: Replay strategies of DDPGPlan
    - **her_commonroad**: Modified HER 
      - **env_wrapper.py**: Environment wrapper for commonroad-rl environment to use HER
      - **her_cr.py**: Main script of HER
      - **replay_buffer.py**: Replay strategies of HER
    - **pretrain**: Pretrain network for actor model update
      - **prepare_data.py**: Prepare training samples for pretrain
      - **pretrain_network.py**: Main script of pretrain network
  - **hyperparams**: Config files for default hyperparameters for ddpg, ddpg_plan, and her_cr
  - **motion_planning**: Usage of Reactive Planner
    - **mpi_planning_pickle.sh**: Script to use Reactive Planner on pickled scenarios with multiprocessing tool MPI
    - **mpi_save_exp.sh**: Script to save planning experience with multiprocessing tool MPI
    - **plan.py**: Main script of motion planning
    - **save_exp.py**: Main script of saving experience
  - **tools**: Tools for visualization and data processing 
    - **visualization**: Utility for plotting learning curves 
    - **divide_files.py**: Utility for dividing files for MPI
    - **divide_train_test.py**: Utility for dividing dataset 
  - **utils_run**: Utility functions to run training, tuning and evaluating files
  - **configs.yaml**: Default config file for defining train and test dataset path, observation space, reward definition, and termination conditions 
  - **play_stable_baselines.py**: Script to evaluate a trained RL agent and save transitions   
  - **run_stable_baselines.py**: Script to train RL model 
- **tests**: Tests for this repository
  - **motion_planning**
  - **pretrain**
  - **resources**
  - **rl**
- **README.md**

## Installation

### Prerequisites

- **This project utilizes the [commonroad-rl](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl) environment. Please refer to the installation guide of this environment and finish the installation in advance.**
- **The visualization model use the plotting function of [OpenAI Baselines](https://github.com/openai/baselines), please check out the installation guide of this repository as well.**
- **We use the [Reactive Planner](https://gitlab.lrz.de/cps/reactive-planner) to do motion planning. Please run `git clone git@gitlab.lrz.de:cps/reactive-planner.git ` to clone this repo, add `$PROJECT_ROOT/src` to your Python interpreter, and run `git checkout feature_highd_chen ` to use `feature_highd_chen` branch.** 
- **The following steps requiring activating the conda environment.**

### Install Repository

Add the project root to your Python interpreter.

## Usage

### Data Preprocessing

This project uses highD dataset for the environment. Please refer to the [tutorial](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl/-/blob/development/tutorials/Tutorial%2001%20-%20Data%20Preprocessing.ipynb) for acquiring, converting the data to *.xml, and preprocessing data to *.pickle.

### Motion Planning

After preprocessing, we now have two folders for storing data: the road networks are stored in `meta_scenario` folder, and problems are stored in `problem` folder. For simplification and use the default parameters for the command line, we suppose the path to each folder is `$PROJECT_ROOT/data/pickles/meta_scenario` and `$PROJECT_ROOT/data/pickles/problem`.

Change into `$PROJECT_ROOT/src/motion_planning` and run:

```
python plan.py
```

to plan all problems in `problem` folder. The resulting plots and trajectories are saved in `$PROJECT_ROOT/data/plan`. 

We can also change the `NUM_CPUS`, `PROBLEM_DIR`, `META_DIR`, `RESULT_DIR`, and `TMP_DIR` in `mpi_planning_pickle.sh` to run the planning in parallel using MPI:

```bash
bash mpi_planning_pickle.sh
```

After planning, we can compute the planning results and collect the experience for training the RL agent. Simply run:

```bash
python save_exp.py
```

The actions suggested by planned trajectories are computed and run in the corresponding scenarios. The transitions (**experience**) are saved into `PROJECT_ROOT/data/exp`. 

We can also use `mpi_save_exp.sh` to run in parallel after changing `NUM_CPUS`, `PROBLEM_DIR`, `META_DIR`, `TRAJ_DIR`, `EXP_DIR`, and `TMP_DIR`. Simply run:

```bash
bash mpi_save_exp.sh
```

### Divide Training and Test Dataset

To train the RL agent, we will now prepare the training and test dataset. The training and test set can be divided randomly, or according to the planning results. 

Change into `$PROJECT_ROOT/src/tools` and run:

```bash
python divide_train_test.py --random
```

to divide the dataset randomly. They are saved in `PROJECT_ROOT/data/pickles/problem_train` and `PROJECT_ROOT/data/pickles/problem_test`.

Or run: 

```bash
python divide_train_test.py --success-train 0.8 --success-test 0.2
```

to keep the success rate of planning in the training set and test set are approximately 0.8 and 0.2. Or 

### Train DDPG Agent for Baseline Performance

After diving the dataset, we have to modify the training and test path in `$PROJECT_ROOT/src/configs.yaml`. Modify the `train_reset_config_path`, `test_reset_config_path`, and `meta_scenario_path` to where you save the training, test, and meta scenarios.

Change into `$PROJECT_ROOT/src` and run: 

```bash
python run_stable_baselines.py --algo ddpg --info-keywords is_goal_reached is_collision is_off_road is_time_out --seed 1 -f ../logs -tb ../logs
```

It will train a DDPG agent for 1e6 steps. The log files and trained model are saved in `$PROJECT_ROOT/logs/ddpg/commonroad-v0_1`.

### Use DDPG Agent to Collect Pretrain Samples

We can now use the trained agent to run in the environment to collect transitions. Run:

```bash
python play_stable_baselines.py -f $PROJECT_ROOT/logs --save-trans --trans-name trained --seed 1
```

It will run the trained agent for `1e6` steps and save the transitions in `$PROJECT_ROOT/data/input_data`.

In order to get more diverse samplesm, we can run the same command with different samples. Moreover, usually, using a trained agent will get more successful samples than collision samples. Therefore, we could also use a random agent like this:

```bash
python play_stable_baselines.py -f $PROJECT_ROOT/logs --save-trans --trans-name random --random-agent --seed 1
```

Also, this can be run with different seeds. 

The experience collected with motion planner can also be used. Simply copy them to the `input_data` folder:

```bash
cp $PROJECT_ROOT/data/exp/* $PROJECT_ROOT/data/input_data
```

### Pretrain Network

The pretrain network is used for actor model update. To train the network, we have to preprocess the collected transitions. 

Change into `$PROJECT_ROOT/src/algorithm/pretrain`. Run:

```bash
python prepare_data.py
```

It will divide the data into a 70% and 30% ratio for training and testing of pretrain network, and save the data as `$PROJECT_ROOT/data/supervised_data/data.npz`.

To train the network, run:

```bash
python pretrain_network.py
```

It now trains a pretrain network with the `data.npz` and saves it in `$PROJECT_ROOT/pretrain_network`. The model folder name is the training time stamp, e.g., `2020_10_08_15_10_25`.

### Train DDPG Agent with Motion Planner and Actor Model Update

Now we can modify the `$PROJECT_ROOT/src/hyperparams/ddpgplan.yml` file. We have to change the `plan_path` to `$PROJECT_ROOT/data/plan`, and `pretrain_path`to the absolute path of our pretrain model, for example the path to `2020_10_08_15_10_25`.

Now we can run the training with these hyperparameters. Change to `$PROJECT_ROOT/src` and run:

```bash
python run_stable_baselines.py --algo ddpgplan --info-keywords is_goal_reached is_collision is_off_road is_time_out --seed 1 -f ../logs -tb ../logs
```

It will train the agent for `1e6` steps.

### Train DDPG Agent with HER

We can also use the HER with DDPG on commonroad-rl environment. Simply run:

```bash
python run_stable_baselines.py --algo hercr --env commonroad-v1 --info-keywords is_goal_reached is_collision is_off_road is_time_out --seed 1 -f ../logs -tb ../logs
```

### Visualize Testing Results

After training, the log files are saved in `$PROJECT_ROOT/logs/`. We can extract the testing results and plot them.

Change to `$PROJECT_ROOT/src/tools/visualization` and run:

```bash
python get_test_results.py && python plot_results.py
```

The figures are saved in `$PROJECT_ROOT/test_results`. We can now see the test results: success rate (`success.svg`), collision rate (`collision.svg`), off-road rate (`off-road.svg`), and time-out rate (`time-out.svg`) of the agent.

## Author

- Xi Chen - xi.chen@tum.de

 