import numpy as np
import pandas as pd
import random


# Define the transition probabilities between different health states
transition_probabilities = {
    'Healthy': {'Healthy': 0.791, 'Uncaught PD': 0.186, 'Caught PD': 0.0, 'Diabetes': 0.033},
    'Uncaught PD': {'Healthy': 0.513, 'Uncaught PD': 0.315, 'Caught PD': 0.0, 'Diabetes': 0.172},
    'Caught PD':  {'Healthy': 0.513, 'Uncaught PD': 0.315, 'Caught PD': 0.0, 'Diabetes': 0.172},
    'Diabetes': {'Healthy': 0.078, 'Uncaught PD': 0.197, 'Caught PD': 0.0, 'Diabetes': 0.725},
}

# Define the costs associated with different health states
state_costs = {
    'USPSTF': {
        'Healthy': 0,
        'Uncaught PD': 100,
        'Caught PD': 100,
        'Diabetes': 500,
    },
    'ADA': {
        'Healthy': 0,
        'Uncaught PD': 100,
        'Caught PD': 100,
        'Diabetes': 600,
    },
    'AACE': {
        'Healthy': 0,
        'Uncaught PD': 150,
        'Caught PD': 150,
        'Diabetes': 700,
    }
}

def generate_hypothetical_population(n):
    np.random.seed(42)
    population = []
    for _ in range(n):
        gender = np.random.choice(['male', 'female'])
        age = np.random.randint(18, 90)
        pregnant = False
        if gender == 'female' and 18 <= age <= 45:
            pregnant = np.random.choice([False, True], p=[0.95, 0.05])
        individual = {
            'age': age,
            'BMI': np.random.normal(loc=23, scale=4),
            'gender': gender,
            'pregnant': pregnant,
            'gestational_diabetes': 0,
            'family_history': np.random.choice([0, 1], p=[0.8, 0.2]),
            'race_ethnicity': np.random.choice(['low_risk', 'high_risk'], p=[0.7, 0.3]),
            'physical_inactivity': int(np.random.choice([0, 1], p=[0.6, 0.4])),
            'hypertension': int(np.random.choice([0, 1], p=[0.7, 0.3])),
            'glucose_tolerance': int(np.random.choice([0, 1], p=[0.85, 0.15])),
            'PCOS': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'history_of_CVD': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'baby_weight_risk': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'antipsychotic_therapy': int(np.random.choice([0, 1], p=[0.95, 0.05])),
            'sleep_disorder_risk': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'health_state': np.random.choice(['Healthy', 'Uncaught PD', 'Caught PD', 'Diabetes'], p=[0.7, 0.25, 0, 0.05]),
            'last_screened': -999, # -999 means this person has never screen before
            'TG': np.random.normal(loc=129, scale=50),
            'cost': 0,
        }
        HDL_below_40 = np.random.choice([True, False], p=[0.17, 0.83])
        if HDL_below_40:
            individual['HDL'] = np.random.uniform(10, 39)
        else:
            individual['HDL'] = np.random.uniform(40, 100)
        if gender == 'female':
            individual['PCOS'] = np.random.choice([0, 1], p=[1 - 0.09, 0.09])  # Use the higher estimate of 12%
        else:
            individual['PCOS'] = 0  # Male individuals don't have PCOS
        population.append(individual)
    return population
screening_ages = {
    'USPSTF': {'Healthy': 40, 'Uncaught PD': 40},
    'ADA': {'Healthy': 45, 'Uncaught PD': 45},
    'AACE': {'Healthy': 45, 'Uncaught PD': 45},
}

screening_success = {
    'USPSTF': {'Healthy': 0.1, 'Uncaught PD': 0.5},
    'ADA': {'Healthy': 0.1, 'Uncaught PD': 0.5},
    'AACE': {'Healthy': 0.1, 'Uncaught PD': 0.5},
}

screening_costs = {
    'USPSTF': 100,
    'ADA': 100,
    'AACE': 150,
}

discount_rate = 0.03
# Define a function to determine if an individual is eligible for screening based on
# their risk factors and the recommended guideline
def is_eligible_for_screening(individual, recommendation, current_year):
    age = individual['age']
    BMI = individual['BMI']
    gender = individual['gender']
    race_ethnicity = individual['race_ethnicity']
    last_screened = individual['last_screened']
    screening_interval = 3

    if current_year - last_screened < screening_interval:
        return False

    risk_factors = {
        'physical_inactivity': individual['physical_inactivity'],
        'hypertension': individual['hypertension'],
        'hdl_risk': int(individual['HDL'] < 40),
        'TG_risk': int(individual['TG'] >= 150),
        'glucose_tolerance': individual['glucose_tolerance'],
        'gestational_diabetes': individual['gestational_diabetes'],
        'family_history': individual['family_history'],
        'race_ethnicity': 1 if individual['race_ethnicity'] == 'high_risk' else 0,
        'PCOS': individual['PCOS'],
        'history_of_CVD': individual['history_of_CVD'],
        'baby_weight_risk': individual['baby_weight_risk'],
        'antipsychotic_therapy': individual['antipsychotic_therapy'],
        'sleep_disorder_risk': individual['sleep_disorder_risk'],
    }

    risk_factor_count = sum(risk_factors.values())

    if recommendation == "USPSTF":
        return 40 <= age <= 70 and BMI >= 25
    elif recommendation == "ADA":
        return age >= 45 or (BMI >= 25 and risk_factor_count >= 1)
    elif recommendation == "AACE":
        return age >= 45 and risk_factor_count >= 2
    else:
        return False
# has issue here
'''def select_screening_guideline(individual, current_year):  # Add current_year as an argument
    eligible_guidelines = []
    if is_eligible_for_screening(individual, "USPSTF", current_year):
        eligible_guidelines.append("USPSTF")
    if is_eligible_for_screening(individual, "ADA", current_year):
        eligible_guidelines.append("ADA")
    if is_eligible_for_screening(individual, "AACE", current_year):
        eligible_guidelines.append("AACE")
    if len(eligible_guidelines) > 0:
        selected_guideline = np.random.choice(eligible_guidelines)
    else:
        selected_guideline = None
    return selected_guideline'''


#  If they are eligible and not in one of the excluded health states, the screening is performed.
#  If the individual is healthy, the screening has no effect on their health state,
#  but the screening cost is added to their current cost.
#  If the individual is in the "Uncaught PD" state, there is a probability that they will be caught during screening and their health state will change to "Caught PD".
#  In this case, the screening cost is also added to their current cost, discounted based on the number of years since their birth year.
def screen_individual(individual, guideline, current_year):
    health_state = individual['health_state']
    cost = 0

    # Check if individual is eligible for screening
    if health_state in ["Healthy", "Uncaught PD"]:
        # Perform screening
        catch_prob = screening_success[guideline][health_state]
        if random.random() < catch_prob:
            health_state = "Caught PD"
        cost += screening_costs[guideline] * (1 / (1 + discount_rate) ** (current_year - individual['last_screened']))

    individual['health_state'] = health_state
    individual['last_screened'] = current_year

    return cost


# Update the run_simulation function:

def apply_chosen_guideline_to_population(population, guideline, current_year):
    screened_individuals = []
    for individual in population:
        if is_eligible_for_screening(individual, guideline, current_year):
            screened_individuals.append(individual)
    return screened_individuals

def run_simulation(population, num_years, selected_guideline):
    costs = {'USPSTF': 0, 'ADA': 0, 'AACE': 0}
    screened_counts = {'USPSTF': 0, 'ADA': 0, 'AACE': 0}

    for year in range(num_years):
        current_year = year

        screened_individuals = apply_chosen_guideline_to_population(population, selected_guideline, current_year)
        screened_counts[selected_guideline] += len(screened_individuals)

        for individual in screened_individuals:
            additional_cost = screen_individual(individual, selected_guideline, current_year)
            individual['cost'] += additional_cost
            costs[selected_guideline] += additional_cost

    return screened_counts, costs



# Run the modified code
population_USPSTF = generate_hypothetical_population(1000)
results_USPSTF, costs_USPSTF = run_simulation(population_USPSTF, 10, "USPSTF")

population_ADA = generate_hypothetical_population(1000)
results_ADA, costs_ADA = run_simulation(population_ADA, 10, "ADA")

population_AACE = generate_hypothetical_population(1000)
results_AACE, costs_AACE = run_simulation(population_AACE, 10, "AACE")

#df_results_USPSTF = pd.DataFrame(results_USPSTF)
#df_results_ADA = pd.DataFrame(results_ADA)
#df_results_AACE = pd.DataFrame(results_AACE)

print("Screened counts for USPSTF guideline:")
#print(df_results_USPSTF)
print("\nTotal costs for USPSTF guideline:")
print(costs_USPSTF)

print("\nScreened counts for ADA guideline:")
#print(df_results_ADA)
print("\nTotal costs for ADA guideline:")
print(costs_ADA)

print("\nScreened counts for AACE guideline:")
#print(df_results_AACE)
print("\nTotal costs for AACE guideline:")
print(costs_AACE)

# cost effective analysis part
# ... [Keep the previous parts of the code]

# Calculate effectiveness for each guideline
def calculate_effectiveness(population, guideline):
    caught_pd_count = 0
    uncaught_pd_count = 0
    for individual in population:
        if individual['health_state'] == 'Caught PD' and individual['selected_guideline'] == guideline:
            caught_pd_count += 1
        if individual['health_state'] == 'Uncaught PD':
            uncaught_pd_count += 1
    return caught_pd_count / (caught_pd_count + uncaught_pd_count)

effectiveness_USPSTF = calculate_effectiveness(population_USPSTF, "USPSTF")
effectiveness_ADA = calculate_effectiveness(population_ADA, "ADA")
effectiveness_AACE = calculate_effectiveness(population_AACE, "AACE")

# Calculate incremental cost-effectiveness ratio (ICER)
def calculate_ICER(costs_base, costs_comparator, effectiveness_base, effectiveness_comparator):
    delta_cost = costs_comparator - costs_base
    delta_effectiveness = effectiveness_comparator - effectiveness_base
    if delta_effectiveness == 0:
        return float('inf')
    return delta_cost / delta_effectiveness

ICER_ADA_vs_USPSTF = calculate_ICER(costs_USPSTF['USPSTF'], costs_ADA['ADA'], effectiveness_USPSTF, effectiveness_ADA)
ICER_AACE_vs_USPSTF = calculate_ICER(costs_USPSTF['USPSTF'], costs_AACE['AACE'], effectiveness_USPSTF, effectiveness_AACE)
ICER_AACE_vs_ADA = calculate_ICER(costs_ADA['ADA'], costs_AACE['AACE'], effectiveness_ADA, effectiveness_AACE)

print("\nCost-Effectiveness:")
print("USPSTF: Effectiveness =", effectiveness_USPSTF)
print("ADA: Effectiveness =", effectiveness_ADA)
print("AACE: Effectiveness =", effectiveness_AACE)

print("\nIncremental Cost-Effectiveness Ratios (ICERs):")
print("ADA vs. USPSTF:", ICER_ADA_vs_USPSTF)
print("AACE vs. USPSTF:", ICER_AACE_vs_USPSTF)
print("AACE vs. ADA:", ICER_AACE_vs_ADA)

# sensitivity analysis
# Sensitivity analysis
screening_success_values = [0.1, 0.3, 0.5, 0.7, 0.9]

print("\nSensitivity Analysis:")

for success_value in screening_success_values:
    screening_success_temp = {
        'USPSTF': {'Healthy': success_value, 'Uncaught PD': success_value},
        'ADA': {'Healthy': success_value, 'Uncaught PD': success_value},
        'AACE': {'Healthy': success_value, 'Uncaught PD': success_value},
    }
    screening_success = screening_success_temp

    # Run the modified code
    population_USPSTF = generate_hypothetical_population(1000)
    results_USPSTF, costs_USPSTF = run_simulation(population_USPSTF, 10, "USPSTF")

    population_ADA = generate_hypothetical_population(1000)
    results_ADA, costs_ADA = run_simulation(population_ADA, 10, "ADA")

    population_AACE = generate_hypothetical_population(1000)
    results_AACE, costs_AACE = run_simulation(population_AACE, 10, "AACE")

    effectiveness_USPSTF = calculate_effectiveness(population_USPSTF, "USPSTF")
    effectiveness_ADA = calculate_effectiveness(population_ADA, "ADA")
    effectiveness_AACE = calculate_effectiveness(population_AACE, "AACE")

    ICER_ADA_vs_USPSTF = calculate_ICER(costs_USPSTF['USPSTF'], costs_ADA['ADA'], effectiveness_USPSTF,
                                        effectiveness_ADA)
    ICER_AACE_vs_USPSTF = calculate_ICER(costs_USPSTF['USPSTF'], costs_AACE['AACE'], effectiveness_USPSTF,
                                         effectiveness_AACE)
    ICER_AACE_vs_ADA = calculate_ICER(costs_ADA['ADA'], costs_AACE['AACE'], effectiveness_ADA, effectiveness_AACE)

    print("\nScreening success rate: {:.1f}".format(success_value))
    print("Incremental Cost-Effectiveness Ratios (ICERs):")
    print("ADA vs. USPSTF:", ICER_ADA_vs_USPSTF)
    print("AACE vs. USPSTF:", ICER_AACE_vs_USPSTF)
    print("AACE vs. ADA:", ICER_AACE_vs_ADA)


