import glob
import os
import pickle

import pytest
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario

from src.motion_planning.plan import plan_scenario
from src.tools.divide_files import ROOT_STR

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"

META_DIR = os.path.join(ROOT_STR, "tests", "resources", "pickles", "meta_scenario")
PROBLEM_DIR = os.path.join(ROOT_STR, "tests", "resources", "pickles", "problem")
PLOT_DIR = os.path.join(ROOT_STR, "tests", "motion_planning", "results")


@pytest.fixture
def get_scenario_problem():
    scenarios = []
    planning_problems = []
    problem_meta_scenario_dict_path = os.path.join(META_DIR, "problem_meta_scenario_dict.pickle")
    with open(problem_meta_scenario_dict_path, 'rb') as pf:
        problem_meta_scenario_dict = pickle.load(pf)
    problem_files = sorted(glob.glob(os.path.join(PROBLEM_DIR, "*.pickle")))
    for pf in problem_files:
        with open(pf, 'rb') as f:
            problem_dict = pickle.load(f)
            meta_scenario = problem_meta_scenario_dict[os.path.basename(pf).split(".")[0]]
            obstacle_list = problem_dict["obstacle"]
            scenario = restore_scenario(meta_scenario, obstacle_list)
            scenario.benchmark_id = os.path.basename(pf).split(".")[0]
            planning_problem = list(problem_dict["planning_problem_set"].planning_problem_dict.values())[0]
            scenarios.append(scenario)
            planning_problems.append(planning_problem)
    return scenarios, planning_problems


def test_plan_scenario(get_scenario_problem):
    scenarios, planning_problems = get_scenario_problem
    results = []
    for s, p in zip(scenarios, planning_problems):
        result = plan_scenario(s, p)
        results.append(True if result else False)
    assert all(results)
