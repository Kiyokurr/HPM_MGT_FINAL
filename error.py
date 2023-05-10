import numpy as np
import pandas as pd
import random
from scipy import stats

discount_rate = 0.03
def calculate_ICER(cost1, cost2, utility1, utility2):
    delta_cost = cost2 - cost1
    delta_utility = utility2 - utility1
    if delta_utility == 0:
        return np.inf
    return delta_cost / delta_utility
# Define the transition probabilities between different health states
transition_probabilities = {
    'Healthy': {'Healthy': 0.791, 'Uncaught PD': 0.186, 'Caught PD': 0.0, 'Diabetes': 0.033},
    'Uncaught PD': {'Healthy': 0.513, 'Uncaught PD': 0.315, 'Caught PD': 0.0, 'Diabetes': 0.172},
    'Caught PD':  {'Healthy': 0.186, 'Uncaught PD': 0.0, 'Caught PD': 0.791, 'Diabetes': 0.033},
    'Diabetes': {'Healthy': 0.0, 'Uncaught PD': 0.0, 'Caught PD': 0.0, 'Diabetes': 1.0},
}

# Define the annual costs associated with different health states
state_costs = {
    'USPSTF': {
        'Healthy': 0,
        'Uncaught PD': 500,
        'Caught PD': 500,
        'Diabetes': 9600,
    },
    'ADA': {
        'Healthy': 0,
        'Uncaught PD': 500,
        'Caught PD': 500,
        'Diabetes': 9600,
    },
    'AACE': {
        'Healthy': 0,
        'Uncaught PD': 500,
        'Caught PD': 500,
        'Diabetes': 9600,
    }
}

utilities = {
    'Healthy': 1,
    'Uncaught PD': 0.85,
    'Caught PD': 0.85,
    'Diabetes': 0.8
}

def generate_hypothetical_population(n):
    np.random.seed(42)
    population = []
    for _ in range(n):
        gender = np.random.choice(['male', 'female'])
        age = np.random.randint(18, 64)
        pregnant = False
        if gender == 'female' and 18 <= age <= 45:
            pregnant = np.random.choice([False, True], p=[0.95, 0.05])
        individual = {
            'age': age,
            'BMI': np.random.normal(loc=26.5, scale=4),
            'gender': gender,
            'pregnant': pregnant,
            'gestational_diabetes': 0,
            'family_history': np.random.choice([0, 1], p=[0.887, 0.113]),
            'race_ethnicity': np.random.choice(['low_risk', 'high_risk'], p=[0.601, 0.399]),
            'physical_inactivity': int(np.random.choice([0, 1], p=[0.75, 0.25])),
            'hypertension': int(np.random.choice([0, 1], p=[0.53, 0.47])),
            'glucose_tolerance': int(np.random.choice([0, 1], p=[0.888, 0.112])),
            'PCOS': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'history_of_CVD': int(np.random.choice([0, 1], p=[0.799, 0.201])),
            'baby_weight_risk': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'antipsychotic_therapy': int(np.random.choice([0, 1], p=[0.984, 0.016])),
            'sleep_disorder_risk': int(np.random.choice([0, 1], p=[0.875, 0.125])),
            'health_state': np.random.choice(['Healthy', 'Uncaught PD', 'Caught PD', 'Diabetes'], p=[0.51, 0.304, 0.076,0.11]),
            'last_screened': -999, # -999 means this person has never screened before
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

screening_success = {
    'USPSTF': {'Healthy': 1, 'Uncaught PD': 0.98},
    'ADA': {'Healthy': 1, 'Uncaught PD': 0.98},
    'AACE': {'Healthy': 1, 'Uncaught PD': 0.98},
}

screening_costs = {
    'USPSTF': 65,
    'ADA': 65,
    'AACE': 65,
}

def calculate_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    stderr = np.std(data, ddof=1) / np.sqrt(len(data))
    margin_of_error = stderr * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return (mean - margin_of_error, mean + margin_of_error)

# Define a function to determine if an individual is eligible for screening based on
# their risk factors and the recommended guideline
def is_eligible_for_screening(individual, recommendation, current_year, screening_interval, min_age, max_age):
    age = individual['age']
    BMI = individual['BMI']
    last_screened = individual['last_screened']

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
        if 40 <= age <= 70 and BMI >= 25:
            if current_year - last_screened < screening_interval:
                return False
            else:
                return True
    elif recommendation == "ADA":
        if age >= 45 or (BMI >= 25 and risk_factor_count >= 1):
            if current_year - last_screened < screening_interval:
                return False
            else:
                return True
    elif recommendation == "AACE":
            if risk_factor_count >= 2:
                if current_year - last_screened < 1:
                    return False
                else:
                    return True
            else:
                if current_year - last_screened < screening_interval:
                    return False
                else:
                    return True
    else:
        return False

def screen_individual(individual, guideline, current_year):
    health_state = individual['health_state']
    cost = 0

    # Check if individual is eligible for screening
    # If the screening is successful and someone has “Healthy,” nothing happens. Screening is successful 100% of the time when “Healthy.”
    if health_state in ["Healthy", "Uncaught PD"]:
        # Perform screening
        catch_prob = screening_success[guideline][health_state]
        #If the screening is successful and someone has “Uncaught PD,” their health state is changed to “Caught PD.”
        if random.random() < catch_prob:
            health_state = "Caught PD"
        cost += screening_costs[guideline] * (1 / (1 + discount_rate) ** (current_year - individual['last_screened']))

    individual['health_state'] = health_state
    individual['last_screened'] = current_year

    return cost

        # Update the run_simulation function:
def run_simulation(population, num_years, selected_guideline, screening_interval, min_age, max_age):
    costs = 0
    screened_counts = 0
    utilities_qalys = 0

    for year in range(num_years):
        current_year = year

        for individual in population:
            individual['cost'] += state_costs[selected_guideline][individual['health_state']] * (
                    1 / (1 + discount_rate) ** current_year)  # Add annual costs and apply the discount rate
            if is_eligible_for_screening(individual, selected_guideline, current_year, screening_interval, min_age,
                                         max_age):
                screened_counts += 1
                additional_cost = (screen_individual(individual, selected_guideline, current_year))
                individual['cost'] += additional_cost * (
                        1 / (1 + discount_rate) ** (current_year - individual['last_screened']))
                costs += additional_cost
                utilities_qalys += utilities[individual['health_state']] * (
                        1 / (1 + discount_rate) ** (current_year - individual['last_screened']))

    return screened_counts, costs, utilities_qalys
def sensitivity_analysis(input_params, num_simulations=10):
    discount_rates = input_params['discount_rates']
    screening_intervals = input_params['screening_intervals']

    results = {}
    for guideline in ["USPSTF", "ADA", "AACE"]:
        results[guideline] = {}
        for dr in discount_rates:
            discount_rate = dr
            for si in screening_intervals:
                costs_samples = []
                utilities_samples = []

                for _ in range(num_simulations):
                    population = generate_hypothetical_population(1000)
                    results_, costs_, utilities_ = run_simulation(
                        population, 10, guideline, si, 18, 90
                    )
                    costs_samples.append(costs_)
                    utilities_samples.append(utilities_)

                cost_ci = calculate_confidence_interval(costs_samples)
                utility_ci = calculate_confidence_interval(utilities_samples)
                results[guideline][(dr, si)] = {
                    "cost_mean": np.mean(costs_samples),
                    "cost_ci": cost_ci,
                    "utility_mean": np.mean(utilities_samples),
                    "utility_ci": utility_ci,
                }

    return results
input_params = {
    "discount_rates": [0.01, 0.03, 0.05],
    "screening_intervals": [1, 2, 3, 4, 5],
}
sensitivity_results = sensitivity_analysis(input_params)
# Run the modified code
num_simulations = 100
costs_samples = {'USPSTF': [], 'ADA': [], 'AACE': []}
utilities_samples = {'USPSTF': [], 'ADA': [], 'AACE': []}

for _ in range(num_simulations):
    population_USPSTF = generate_hypothetical_population(1000)
    results_USPSTF, costs_USPSTF, utilities_USPSTF = run_simulation(population_USPSTF, 30, "USPSTF", 3, 40, 70)
    costs_samples['USPSTF'].append(costs_USPSTF)
    utilities_samples['USPSTF'].append(utilities_USPSTF)

    population_ADA = generate_hypothetical_population(1000)
    results_ADA, costs_ADA, utilities_ADA = run_simulation(population_ADA, 30, "ADA", 3, 18, 90)
    costs_samples['ADA'].append(costs_ADA)
    utilities_samples['ADA'].append(utilities_ADA)

    population_AACE = generate_hypothetical_population(1000)
    results_AACE, costs_AACE, utilities_AACE = run_simulation(population_AACE, 30, "AACE", 3, 18, 90)
    costs_samples['AACE'].append(costs_AACE)
    utilities_samples['AACE'].append(utilities_AACE)

cost_ci = {key: calculate_confidence_interval(costs_samples[key]) for key in costs_samples}
utility_ci = {key: calculate_confidence_interval(utilities_samples[key]) for key in utilities_samples}

ICER_USPSTF_ADA = calculate_ICER(costs_USPSTF, costs_ADA, utilities_USPSTF, utilities_ADA)
ICER_USPSTF_AACE = calculate_ICER(costs_USPSTF, costs_AACE, utilities_USPSTF, utilities_AACE)
ICER_ADA_AACE = calculate_ICER(costs_ADA, costs_AACE, utilities_ADA, utilities_AACE)
pop_size = 1000
total_diabetes = sum([1 for individual in generate_hypothetical_population(pop_size) if individual['health_state'] == 'Diabetes'])
percent_diabetes = (total_diabetes / pop_size) * 100

print("\nTotal costs and utilities for each guideline with confidence intervals:")
print("USPSTF: \nCost =", costs_USPSTF, "\nUtility =", utilities_USPSTF, "\nCI for cost =", cost_ci['USPSTF'], "\nCI for utility =", utility_ci['USPSTF'], "\nNumber of screenings =", results_USPSTF)
print("\nADA: Cost =", costs_ADA, "\nUtility =", utilities_ADA, "\nCI for cost =", cost_ci['ADA'], "\nCI for utility =", utility_ci['ADA'], "\nNumber of screenings =", results_ADA)
print("\nAACE: Cost =", costs_AACE, "\nUtility =", utilities_AACE, "\nCI for cost =", cost_ci['AACE'], "\nCI for utility =", utility_ci['AACE'], "\nNumber of screenings =", results_AACE)
print("\nPercentage of total population with diabetes: {:.2f}%".format(percent_diabetes))
print("\nSensitivity Analysis Results:")
for guideline in sensitivity_results:
    print(f"\n{guideline}:")
    for (dr, si), values in sensitivity_results[guideline].items():
        print(
            f"  Discount Rate: {dr}, Screening Interval: {si} years\n"
            f"    Cost: {values['cost_mean']} (CI: {values['cost_ci']})\n"
            f"    Utility: {values['utility_mean']} (CI: {values['utility_ci']})"
        )

def calculate_effectiveness(population, guideline):
    caught_pd_count = 0
    uncaught_pd_count = 0
    for individual in population:
        if individual['health_state'] == 'Caught PD':
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

ICER_ADA_vs_USPSTF = calculate_ICER(costs_USPSTF, costs_ADA, effectiveness_USPSTF, effectiveness_ADA)
ICER_AACE_vs_USPSTF = calculate_ICER(costs_USPSTF, costs_AACE, effectiveness_USPSTF, effectiveness_AACE)
ICER_AACE_vs_ADA = calculate_ICER(costs_ADA, costs_AACE, effectiveness_ADA, effectiveness_AACE)

print("\nCost-Effectiveness:")
print("USPSTF: Effectiveness =", effectiveness_USPSTF)
print("ADA: Effectiveness =", effectiveness_ADA)
print("AACE: Effectiveness =", effectiveness_AACE)

print("\nIncremental Cost-Effectiveness Ratios (ICERs): ")
print("ADA vs. USPSTF:", ICER_ADA_vs_USPSTF)
print("AACE vs. USPSTF:", ICER_AACE_vs_USPSTF)
print("AACE vs. ADA:", ICER_AACE_vs_ADA)

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
    results_USPSTF, costs_USPSTF, utilities_USPSTF  = run_simulation(population_USPSTF, 30, "USPSTF", 3, 18, 90)

    population_ADA = generate_hypothetical_population(1000)
    results_ADA, costs_ADA, utilities_ADA = run_simulation(population_ADA, 30, "ADA",  3, 18, 90)

    population_AACE = generate_hypothetical_population(1000)
    results_AACE, costs_AACE, utilities_AACE= run_simulation(population_AACE, 30, "AACE",  3, 18, 90)

    effectiveness_USPSTF = calculate_effectiveness(population_USPSTF, "USPSTF")
    effectiveness_ADA = calculate_effectiveness(population_ADA, "ADA")
    effectiveness_AACE = calculate_effectiveness(population_AACE, "AACE")

    ICER_ADA_vs_USPSTF = calculate_ICER(costs_USPSTF, costs_ADA, effectiveness_USPSTF,
                                        effectiveness_ADA)
    ICER_AACE_vs_USPSTF = calculate_ICER(costs_USPSTF, costs_AACE, effectiveness_USPSTF,
                                         effectiveness_AACE)
    ICER_AACE_vs_ADA = calculate_ICER(costs_ADA, costs_AACE, effectiveness_ADA, effectiveness_AACE)

    print("\nScreening success rate: {:.1f}".format(success_value))
    print("Incremental Cost-Effectiveness Ratios (ICERs):")
    print("ADA vs. USPSTF:", ICER_ADA_vs_USPSTF)
    print("AACE vs. USPSTF:", ICER_AACE_vs_USPSTF)
    print("AACE vs. ADA:", ICER_AACE_vs_ADA)