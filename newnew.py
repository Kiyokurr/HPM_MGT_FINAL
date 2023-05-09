import numpy as np
import pandas as pd

# Define the transition probabilities between different health states
transition_probabilities = {
    'Healthy': {'Healthy': 0.791, 'Uncaught PD': 0.186, 'Diabetes': 0.033, 'Caught PD': 0},
    'Uncaught PD': {'Healthy': 0.513, 'Uncaught PD': 0.315, 'Diabetes': 0.172, 'Caught PD': 0},
    'Diabetes': {'Healthy': 0.078, 'Uncaught PD': 0.197, 'Diabetes': 0.725, 'Caught PD': 0},
    'Caught PD': {'Healthy': 0.513, 'Uncaught PD': 0, 'Diabetes': 0.172, 'Caught PD': 0.315},
}

# Define the costs associated with different health states
state_costs = {
    'USPSTF': {
        'Healthy': 0,
        'Uncaught PD': 100,
        'Diabetes': 500,
        'Caught PD': 100,
    },
    'ADA': {
        'Healthy': 0,
        'Uncaught PD': 100,
        'Diabetes': 600,
        'Caught PD': 100,
    },
    'AACE': {
        'Healthy': 0,
        'Uncaught PD': 150,
        'Diabetes': 700,
        'Caught PD': 150,
    }
}

# Add the screen_success variable
screen_success = 0.9

# The rest of the code remains the same until the screen_individual function

def screen_individual(individual, guideline, current_year):  # Add current_year as an argument
    health_state = individual['health_state']
    cost = state_costs[guideline][health_state]
    if guideline == "USPSTF" and health_state in ["Uncaught PD", "Diabetes"]:
        if np.random.random() < screen_success:
            health_state = "Caught PD"
        else:
            health_state = "Healthy"
    elif guideline == "ADA" and health_state == "Diabetes":
        if np.random.random() < screen_success:
            health_state = "Caught PD"
        else:
            health_state = "Uncaught PD"
    elif guideline == "AACE" and health_state == "Uncaught PD":
        if np.random.random() < screen_success:
            health_state = "Caught PD"
        else:
            health_state = "Diabetes"
    individual['health_state'] = health_state
    individual['last_screened'] = current_year  # Update the last_screened value
    individual['cost'] = cost
    return individual

# Update the run_simulation function

def run_simulation(population, num_years, selected_guideline):
    results = []
    costs = {'USPSTF': 0, 'ADA': 0, 'AACE': 0}
    screened_counts = {'USPSTF': 0, 'ADA': 0
