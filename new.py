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
    risk_factor_count = sum([individual['physical_inactivity'],
                             individual['hypertension'],
                             individual['hdl'] < 35 or individual['tg'] > 250,
                             individual['family_history'],
                             individual['gestational_diabetes'],
                             individual['A1c'] >= 5.7 or individual['IFG'] >= 100 or individual['IGT'] >= 140,
                             individual['other_risk_factors'],
                             individual['delivered_baby_over_9lbs'],
                             individual['antipsychotic_therapy'],
                             individual['sleep_disorders']])

    if recommendation == "USPSTF":
        return 40 <= age <= 70 and BMI >= 25
    elif recommendation == "ADA":
        return age >= 45 or (BMI >= 25 and risk_factor_count >= 1)
    elif recommendation == "AACE":
        return age >= 45 and risk_factor_count >= 2
    else:
        return False

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
