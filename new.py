#UNFINISHED
import numpy as np
import pandas as pd

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
            'physical_inactivity': np.random.choice([0, 1], p=[0.6, 0.4]),
            'hypertension': np.random.choice([0, 1], p=[0.7, 0.3]),
            'HDL': np.random.uniform(20, 100),
            'TG': np.random.uniform(50, 400),
            'glucose_tolerance': np.random.choice([0, 1], p=[0.85, 0.15]),
            'PCOS': np.random.choice([0, 1], p=[0.9, 0.1]),
            'history_of_CVD': np.random.choice([0, 1], p=[0.9, 0.1]),
            'baby_weight_risk': np.random.choice([0, 1], p=[0.9, 0.1]),
            'antipsychotic_therapy': np.random.choice([0, 1], p=[0.95, 0.05]),
            'sleep_disorder_risk': np.random.choice([0, 1], p=[0.9, 0.1]),
            'last_screened': -999
        }
        population.append(individual)
    return population


def is_eligible_for_screening(individual, recommendation):
    age = individual['age']
    BMI = individual['BMI']
    gender = individual['gender']
    race_ethnicity = individual['race_ethnicity']
    last_screened = individual['last_screened']

    risk_factors = {
        'physical_inactivity': individual['physical_inactivity'],
        'hypertension': individual['hypertension'],
        'hdl_risk': individual['HDL'] < 35,
        'tg_risk': individual['TG'] > 250,
        'family_history': individual['family_history'],
        'gestational_diabetes': individual['gestational_diabetes'],
        'glucose_tolerance_risk': individual['glucose_tolerance'],
        'pcos': individual['PCOS'],
        'history_of_cvd': individual['history_of_CVD'],
        'baby_weight_risk': individual['baby_weight_risk'],
        'antipsychotic_therapy': individual['antipsychotic_therapy'],
        'sleep_disorder_risk': individual['sleep_disorder_risk']
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

# (Other parts of the code remain the same)



def apply_screening(population, recommendation, test_cost):
    cost = 0
    num_screened = 0
    for individual in population:
        if is_eligible_for_screening(individual, recommendation):
            cost += test_cost[recommendation]
            num_screened += 1
            individual['screened'] = 1
    return cost, num_screened

# Change these cost values according to your research data
test_cost = {
    'USPSTF': 50,  # Adjust cost for USPSTF screening
    'ADA': 50,  # Adjust cost for ADA screening
    'AACE': 50  # Adjust cost for AACE screening
}

# Generate a hypothetical population of 1000 individuals
population_size = 1000  # You can adjust the population size
population = generate_hypothetical_population(population_size)

# Apply the screening for each recommendation and calculate the cost
uspstf_cost, uspstf_screened = apply_screening(population, "USPSTF", test_cost)
ada_cost, ada_screened = apply_screening(population, "ADA", test_cost)
aace_cost, aace_screened = apply_screening(population, "AACE", test_cost)

print("USPSTF cost:", uspstf_cost, "Number of people screened:", uspstf_screened)
print("ADA cost:", ada_cost, "Number of people screened:", ada_screened)
print("AACE cost:", aace_cost, "Number of people screened:", aace_screened)
