import deampy.plots.histogram as hist
import deampy.plots.sample_paths as path
import numpy as np
import pandas as pd

from enum import Enum
import deampy.econ_eval as econ
import deampy.statistics as stat
import numpy as np
from deampy.markov import MarkovJumpProcess
from deampy.plots.sample_paths import PrevalencePathBatchUpdate

# simulation settings
POP_SIZE = 5000      # cohort population size
SIM_TIME_STEPS = 50  # length of simulation (years)
ALPHA = 0.05         # alpha for our confidence interval
DISCOUNT = 0.03      # annual discount rate
'''NOTE: THESE ALL SHOULD BE ADJUSTED BASED ON OUR THINKING'''


class HealthStates(Enum):
    """ health states of patients """
    WELL = 0
    PRE_DIABETES = 1
    DIABETES = 2
    DEAD = 3


# annual health utility of each health state
ANNUAL_STATE_UTILITY = [
    1,      # WELL
    0.9,    # PRE-DIABETES
    0.8,    # DIABETES
    0       # DEAD
]
'''NOTE: NEED TO DECIDE UTILITIES'''

ANNUAL_STATE_COST = [
    0,      # WELL
    200,    # PRE-DIABETES
    1000,   # DIABETES
    0,      # DEAD
]
'''NOTE: NEED TO DECIDE ANNUAL COSTS'''

# Screening individual costs:
SCREENING_COST = 3000
'''NOTE: NEED TO DECIDE SCREENING COST'''


'''THE FOLLOWING ARE SUGGESTED UPDATES FOR RUN_SIMULATION'''


def run_simulation(population, num_years, selected_guideline):
    results = []
    costs = {'USPSTF': 0, 'ADA': 0, 'AACE': 0}
    screened_counts = {'USPSTF': 0, 'ADA': 0, 'AACE': 0}

    for year in range(num_years):
        current_year = year

        for individual in population:
            guideline = selected_guideline
            if is_eligible_for_screening(individual, guideline, current_year):
                screened_counts[guideline] += 1
                individual = screen_individual(individual, guideline, current_year)  # Pass current_year as an argument
                costs[guideline] += individual['cost']

        results.append(screened_counts)

    return results, costs


'''THE FOLLOWING IS WHAT WE HAD IN OUR HW SOLUTION'''


def simulate(self, n_time_steps):
    # random number generator
    rng = np.random.RandomState(seed=self.id)  # keep
    # Markov jump process
    markov_jump = MarkovJumpProcess(transition_prob_matrix=self.params.probMatrix)  # utilize our existing matrix

    k = 0  # simulation time step

    # while the patient is alive and simulation length is not yet reached
    while self.stateMonitor.get_if_alive() and k < n_time_steps:
        # sample from the Markov jump process to get a new state
        # (returns an integer from {0, 1, 2, ...})
        new_state_index = markov_jump.get_next_state(
            current_state_index=self.stateMonitor.currentState.value,
            rng=rng)

        # update health state
        self.stateMonitor.update(time_step=k, new_state=HealthStates(new_state_index))

        # increment time
        k += 1