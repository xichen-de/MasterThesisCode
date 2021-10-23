#!/usr/bin/env bash
trap "exit" INT
source activate cr36

NUM_SEED=5
for i in $(seq 1 ${NUM_SEED}); do
  python run_stable_baselines.py --algo ddpgplan --info-keywords is_goal_reached is_collision is_off_road is_time_out --seed "${i}" -f /home/chenx/project/master_thesis_xi_chen/logs_random_scenario &&
    python run_stable_baselines.py --algo ddpg --info-keywords is_goal_reached is_collision is_off_road is_time_out --seed "${i}" -f /home/chenx/project/master_thesis_xi_chen/logs_random_scenario
done
