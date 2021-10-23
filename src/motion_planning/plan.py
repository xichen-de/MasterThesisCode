"""
Script for planning trajectory for CommonRoad scenarios (.pickle or .xml).
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

import argparse
import glob
import os
import pickle
from copy import deepcopy
from shutil import copyfile
from typing import List, Tuple

import commonroad_dc
import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import StaticObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rp.reactive_planner import ReactivePlanner
from route_planner import RoutePlanner
from scenario_helpers import create_road_boundary

# Visualization parameters
from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

DRAW_PARAMS = {
    'draw_shape': True,
    'draw_icon': False,
    'draw_bounding_box': True,
    'trajectory_steps': 2,
    'show_label': False,
    'occupancy': {
        'draw_occupancies': 0, 'shape': {
            'rectangle': {
                'opacity': 0.2,
                'facecolor': '#fa0200',
                'edgecolor': '#0066cc',
                'linewidth': 0.5,
                'zorder': 18}}},
    'shape': {'rectangle': {'opacity': 1.0,
                            'facecolor': '#fa0200',
                            'edgecolor': '#831d20',
                            'linewidth': 0.5,
                            'zorder': 20}}}


def plan_scenario(scenario: Scenario, planning_problem: PlanningProblem) -> Tuple[
    List[State], List[np.ndarray], StaticObstacle]:
    """
    Given one scenario and its planning problem, plan the feasible trajectory.
    :param scenario: scenario to plan
    :param planning_problem: planning problem to tackle
    :return: list of planned states, reference paths, road boundary
    """
    # Define initial state
    problem_init_state = planning_problem.initial_state
    if not hasattr(problem_init_state, 'acceleration'):
        problem_init_state.acceleration = 0.
    # Create road boundary
    road_boundary_sg, _ = create_road_boundary(scenario, draw=False)
    # Initialize route planner
    route_planner = RoutePlanner(scenario.benchmark_id, scenario.lanelet_network, planning_problem)
    dt = scenario.dt
    t_h = 1
    planned_states = None
    ref_path_list = None
    print(f"Planning horizon: {t_h}s")
    # Define desired speed
    if hasattr(planning_problem.goal.state_list[0], 'velocity'):
        desired_velocity = (planning_problem.goal.state_list[0].velocity.start
                            + planning_problem.goal.state_list[0].velocity.end) / 2
    else:
        desired_velocity = problem_init_state.velocity
    # Initialize planner and plan
    try:
        planner = ReactivePlanner(scenario, planning_problem=planning_problem, route_planner=route_planner, dt=dt,
                                  t_h=t_h, N=int(t_h / dt), v_desired=desired_velocity)
        planner.set_desired_velocity(desired_velocity)
        x_0 = deepcopy(problem_init_state)
        planned_states, ref_path_list, _ = planner.re_plan(x_0)
    except:
        print(f"Planner has error for {scenario.benchmark_id}.")
    if planned_states:
        # If planning is successful
        print(f"Plan successfully for {scenario.benchmark_id}.")
    else:
        print(f"Plan fails for {scenario.benchmark_id}.")
    return planned_states, ref_path_list, road_boundary_sg


def visualize_plan(scenario: Scenario, planning_problem: PlanningProblem, plot_dir: str, **kwargs):
    """
    Visualize scenario, planning problem, planned trajectory, reference path.
    :param scenario: to-plan scenario
    :param planning_problem: to-plan planning problem
    :param plot_dir: directory to save figures
    :param kwargs: planned_states, ref_path_list, road_boundary_sg, if they exist
    """
    plt.figure(figsize=(20, 10))
    draw_object(scenario, draw_params=DRAW_PARAMS)
    draw_object(planning_problem)

    planned_states = kwargs.get('planned_states', None)
    ref_path_list = kwargs.get('ref_path_list', None)
    road_boundary_sg = kwargs.get('road_boundary_sg', None)

    if planned_states:
        plt.plot([s.position[0] for s in planned_states],
                 [s.position[1] for s in planned_states],
                 color='k', marker='o', markersize=1, zorder=20, linewidth=0.5, label='Planned route'
                 )
    if ref_path_list:
        for rp in ref_path_list:
            plt.plot(rp[:, 0], rp[:, 1], color='g', marker='*', markersize=1, zorder=19, linewidth=0.5,
                     label='Reference route')

    if road_boundary_sg:
        commonroad_dc.collision.visualization.draw_dispatch.draw_object(road_boundary_sg,
                                                                        draw_params={
                                                                            'collision': {'facecolor': 'yellow'}})

    plt.gca().set_aspect('equal')
    plt.autoscale()
    plt.savefig(os.path.join(plot_dir, scenario.benchmark_id + '.svg'), dpi=300,
                bbox_inches='tight')
    plt.close()


def plan_all_scenarios(meta_scenario_dir: str, problem_dir: str, plot_dir: str, no_solution_dir: str, traj_dir: str,
                       use_xml: bool):
    """
    Plan all scenarios in problem_dir.
    :param meta_scenario_dir: directory of meta scenario
    :param problem_dir: directory of problem pickles or xml files
    :param plot_dir: directory to save figures
    :param no_solution_dir: directory to save failed problem
    :param traj_dir: directory to save planned states for each successful scenario
    :param use_xml: choose to use pickle or xml, if true, use xml, otherwise, use pickle
    """
    # Load planning problems and scenarios
    if use_xml:
        problem_files = sorted(glob.glob(os.path.join(problem_dir, "*.xml")))
    else:
        problem_meta_scenario_dict_path = os.path.join(meta_scenario_dir, "problem_meta_scenario_dict.pickle")
        with open(problem_meta_scenario_dict_path, 'rb') as pf:
            problem_meta_scenario_dict = pickle.load(pf)
        problem_files = sorted(glob.glob(os.path.join(problem_dir, "*.pickle")))

    for pf in problem_files:
        if use_xml:
            scenario, planning_problem_set = CommonRoadFileReader(pf).open()
            planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
        else:
            with open(pf, 'rb') as f:
                problem_dict = pickle.load(f)
            meta_scenario = problem_meta_scenario_dict[os.path.basename(pf).split(".")[0]]
            obstacle_list = problem_dict["obstacle"]
            scenario = restore_scenario(meta_scenario, obstacle_list)
            scenario.benchmark_id = os.path.basename(pf).split(".")[0]
            planning_problem = list(problem_dict["planning_problem_set"].planning_problem_dict.values())[0]

        # Plan and visualize
        planned_states, ref_path_list, road_boundary_sg = plan_scenario(scenario, planning_problem)
        visualize_plan(scenario, planning_problem, plot_dir, planned_states=planned_states, ref_path_list=ref_path_list,
                       road_boundary_sg=road_boundary_sg)
        if not planned_states:
            # Save failed planning problem
            copyfile(pf, os.path.join(no_solution_dir, os.path.basename(pf)))
        else:
            # Save trajectory
            with open(os.path.join(traj_dir, f"{scenario.benchmark_id}.pickle"), 'wb') as f:
                pickle.dump(planned_states, f)


def get_args():
    parser = argparse.ArgumentParser(description="Plan routes for CommonRoad scenarios (pickles or xml)")
    parser.add_argument("-r", "--result-dir", type=str, default=f"{ROOT_STR}/data/plan",
                        help='Where you put the results (trajectories, plot, and failed scenario')
    parser.add_argument("-p", "--problem-dir", type=str, default=f"{ROOT_STR}/data/pickles/problem",
                        help='If you use xml, it is the xml directory; otherwise, it is the pickled problem directory')
    parser.add_argument("-m", "--meta-dir", type=str, default=f"{ROOT_STR}/data/pickles/meta_scenario",
                        help='If you use pickle, it is the meta-scenario directory')
    parser.add_argument("--use-xml", action="store_true", default=False,
                        help='If true, then use xml, otherwise, use pickle')
    parser.add_argument('--multiprocessing', '-mpi', action="store_true", help="use mpi", default=False)
    return parser.parse_args()


def main(args):
    meta_dir = args.meta_dir
    plot_dir = os.path.join(args.result_dir, "plot")
    no_solution_dir = os.path.join(args.result_dir, "no_solution")
    traj_dir = os.path.join(args.result_dir, "traj")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(no_solution_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    # If we use mpi, we have to separate problem files into subfolders in advance
    if args.multiprocessing:
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None
        if MPI is None:
            problem_dir = args.problem_dir
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            problem_dir = os.path.join(args.problem_dir, str(rank))
    else:
        problem_dir = args.problem_dir

    print("=" * 80)
    print(f"Processing {problem_dir}")
    print("=" * 80)
    plan_all_scenarios(meta_dir, problem_dir, plot_dir, no_solution_dir, traj_dir, args.use_xml)


if __name__ == "__main__":
    args = get_args()
    main(args)
