import deampy.plots.histogram as hist
import deampy.plots.sample_paths as path
import numpy as np
import pandas as pd

from enum import Enum

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

