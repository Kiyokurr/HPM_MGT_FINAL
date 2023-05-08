import numpy as np
import pandas as pd
# Define the transition probabilities between different health states
transition_probabilities = {
    'Healthy': {'Healthy': 0.95, 'Pre-diabetes': 0.04, 'Diabetes': 0.01},
    'Pre-diabetes': {'Healthy': 0.1, 'Pre-diabetes': 0.85, 'Diabetes': 0.05},
    'Diabetes': {'Healthy': 0, 'Pre-diabetes': 0, 'Diabetes': 1},
}
# Define the costs associated with different health states
state_costs = {
    'Healthy': 0,
    'Pre-diabetes': 100,
    'Diabetes': 500,
}
# Define a function to generate a hypothetical population
def generate_hypothetical_population(n):
    np.random.seed(42)
    population = []
    for _ in range(n):
        individual = {
            'age': np.random.randint(18, 90),
            'BMI': np.random.uniform(18.5, 45),
            'gender': np.random.choice(['male', 'female']),
            'gestational_diabetes': np.random.choice([0, 1], p=[0.9, 0.1]),
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
            'health_state': np.random.choice(['Healthy', 'Pre-diabetes', 'Diabetes'], p=[0.7, 0.25, 0.05]),
            'last_screened': -999,
            'TG': np.random.randint(50, 300),
            'HDL': np.random.randint(30, 100) # Add HDL key with random integer value
        }
        population.append(individual)
    return population
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

# Each individual in the hypothetical population
# will only choose one guideline to see if they are eligible for screening.
def select_screening_guideline(individual):
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
    return selected_guideline

def screen_individual(individual, guideline):
    health_state = individual['health_state']
    if guideline == "USPSTF" and health_state in ["Pre-diabetes", "Diabetes"]:
        health_state = "Healthy"
    elif guideline == "ADA" and health_state == "Diabetes":
        health_state = "Pre-diabetes"
    elif guideline == "AACE" and health_state == "Pre-diabetes":
        health_state = "Diabetes"
    individual['health_state'] = health_state
    individual['last_screened'] = current_year
    return individual

def run_simulation(population, num_years):
    results = []
    for year in range(num_years):
        global current_year
        current_year = year
        screened_counts = {'USPSTF': 0, 'ADA': 0, 'AACE': 0}
        for individual in population:
            guideline = select_screening_guideline(individual)
            if guideline is not None:
                screened_counts[guideline] += 1
                individual = screen_individual(individual, guideline)
        results.append(screened_counts)
    return results

# 10 years:
population = generate_hypothetical_population(1000)
results = run_simulation(population, 10)
df_results = pd.DataFrame(results)
print(df_results)
