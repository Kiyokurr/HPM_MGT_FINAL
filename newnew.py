# begin code

import numpy as np
import pandas as pd

# Define the transition probabilities between different health states
transition_probabilities = {
    'Healthy': {'Healthy': 0.791, 'Uncaught PD': 0.186, 'Diabetes': 0.033, 'death': 0.01},
    'Uncaught PD': {'Healthy': 0.513, 'Uncaught PD': 0.315, 'Diabetes': 0.172, 'death': 0.01},
    'Diabetes': {'Healthy': 0.078, 'Uncaught PD': 0.197, 'Diabetes': 0.725, 'death': 0.01},
    'Caught PD': {'Healthy': 0.513, 'Caught PD': 0.315, 'Diabetes': 0.172, 'death': 0.01},
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

screen_success = 0.9

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
            'health_state': np.random.choice(['Healthy', 'Uncaught PD', 'Diabetes'], p=[0.7, 0.25, 0.05]),
            'last_screened': -999,  # -999 means this person has never screened before
            'TG': np.random.normal(loc=129, scale=50),
            'is_alive': True,
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

    # Update the run_simulation function:

def run_simulation(population, num_years, selected_guideline):
    results = []
    total_costs = {'USPSTF': 0, 'ADA': 0, 'AACE': 0}
    screened_counts = {'USPSTF': 0, 'ADA': 0, 'AACE': 0}
    screen_success = 0.9

    for year in range(num_years):
        current_year = year

        for individual in population:
            if not individual['is_alive']:
                continue

            # Transition probabilities and death
            health_state = individual['health_state']
            next_health_state = np.random.choice(list(transition_probabilities[health_state].keys()),
                                                 p=list(transition_probabilities[health_state].values()))
            if np.random.random() < 0.02:  # 2% probability of death each year
                individual['is_alive'] = False
                continue
            else:
                individual['health_state'] = next_health_state

            guideline = selected_guideline
            if is_eligible_for_screening(individual, guideline, current_year):
                screened_counts[guideline] += 1
                # Apply screening with screen_success rate
                if np.random.random() < screen_success:
                    if health_state == "Uncaught PD":
                        individual['health_state'] = "Caught PD"
                        individual['last_screened'] = current_year
                        cost = state_costs[guideline][health_state]
                        total_costs[guideline] += cost

        results.append(screened_counts)

    return results, total_costs

    # Run the modified code
population_USPSTF = generate_hypothetical_population(1000)
results_USPSTF, total_costs_USPSTF = run_simulation(population_USPSTF, 10, "USPSTF")

population_ADA = generate_hypothetical_population(1000)
results_ADA, total_costs_ADA = run_simulation(population_ADA, 10, "ADA")

population_AACE = generate_hypothetical_population(1000)
results_AACE, total_costs_AACE = run_simulation(population_AACE, 10, "AACE")

df_results_USPSTF = pd.DataFrame(results_USPSTF)
df_results_ADA = pd.DataFrame(results_ADA)
df_results_AACE = pd.DataFrame(results_AACE)

print("Screened counts for USPSTF guideline:")
print(df_results_USPSTF)
print("\nTotal costs for USPSTF guideline:")
print(total_costs_USPSTF)

print("\nScreened counts for ADA guideline:")
print(df_results_ADA)
print("\nTotal costs for ADA guideline:")
print(total_costs_ADA)

print("\nScreened counts for AACE guideline:")
print(df_results_AACE)
print("\nTotal costs for AACE guideline:")
print(total_costs_AACE)
