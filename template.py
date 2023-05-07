import numpy as np

# Example base transition matrix (fake data)
base_transition_matrix = np.array([
    [0.85, 0.10, 0.05, 0.00],
    [0.00, 0.80, 0.20, 0.00],
    [0.00, 0.00, 0.90, 0.10],
    [0.00, 0.00, 0.00, 1.00]
])

costs = {
    'normal': 100,
    'prediabetes': 200,
    'diabetes': 500
}

screening_costs = {
    'uspstf': 10,
    'ada': 15,
    'aace': 20
}

utilities = {
    'normal': 1.0,
    'prediabetes': 0.9,
    'diabetes': 0.7
}

risk_factors = {
    'physical_inactivity': 0.3,
    'hypertension': 0.25,
    'low_hdl': 0.2,
    'high_tg': 0.15,
    'family_history': 0.1,
    'high_risk_race': 0.05,
    'gestational_diabetes': 0.01,
    'history_of_cvd': 0.1  # Add this line
}


n_cycles = 10
n_patients = 1000


def generate_fake_population(n_patients, risk_factors):
    population = []
    for _ in range(n_patients):
        patient = {
            'age': np.random.randint(18, 80),
            'bmi': np.random.uniform(18, 45),
            'hdl': np.random.randint(25, 75),
            'tg': np.random.randint(100, 400),
            'gestational_diabetes': np.random.choice([0, 1], p=[0.9, 0.1]),
            'family_history': np.random.choice([0, 1], p=[0.8, 0.2]),
            'race_ethnicity': np.random.choice(['low_risk', 'high_risk'], p=[0.7, 0.3]),
            'physical_inactivity': np.random.choice([0, 1], p=[0.6, 0.4]),
            'hypertension': np.random.choice([0, 1], p=[0.7, 0.3]),
            'pcos': np.random.choice([0, 1], p=[0.9, 0.1]),
            'history_of_cvd': np.random.choice([0, 1], p=[0.9, 0.1]),
            'screened': 0
        }

        risk_score = (
            patient['physical_inactivity'] * risk_factors['physical_inactivity']
            + patient['hypertension'] * risk_factors['hypertension']
            + patient['family_history'] * risk_factors['family_history']
            + patient['history_of_cvd'] * risk_factors['history_of_cvd']
        )

        if patient['age'] <= 45:
            risk_score *= 1.2
        if patient['bmi'] >= 30:
            risk_score *= 1.3
        if patient['hdl'] < 35 or patient['tg'] > 250:
            risk_score *= 1.3

        patient['risk_score'] = risk_score
        population.append(patient)
    return population


population = generate_fake_population(n_patients, risk_factors)


def apply_screening_strategy(patient, strategy):
    age = patient['age']
    bmi = patient['bmi']
    family_history = patient['family_history']
    history_of_cvd = patient['history_of_cvd']

    if strategy == 'uspstf':
        return 40 <= age <= 70 and bmi >= 25
    elif strategy == 'ada':
        return age <= 45
    elif strategy == 'aace':
        return (age <= 45 and (family_history or history_of_cvd)) or bmi >= 25
    else:
        return False


def run_simulation(population, strategy, n_cycles, base_transition_matrix, costs, utilities):
    total_costs = np.zeros(n_patients)
    total_utilities = np.zeros(n_patients)
    initial_state = np.zeros(n_patients, dtype=int)

    for cycle in range(n_cycles):
        for patient in range(n_patients):
            if not population[patient]['screened'] and apply_screening_strategy(population[patient], strategy):
                population[patient]['screened'] = 1

            current_state = initial_state[patient]

            if current_state == 0:
                total_costs[patient] += costs['normal'] + screening_costs[strategy]
                total_utilities[patient] += utilities['normal']
            elif current_state == 1:
                total_costs[patient] += costs['prediabetes'] + screening_costs[strategy]
                total_utilities[patient] += utilities['prediabetes']
            elif current_state == 2:
                total_costs[patient] += costs['diabetes'] + screening_costs[strategy]
                total_utilities[patient] += utilities['diabetes']

            initial_state[patient] = np.random.choice(4, p=base_transition_matrix[current_state])

    return total_costs, total_utilities


strategies = ['uspstf', 'ada', 'aace']

for strategy in strategies:
    total_costs, total_utilities = run_simulation(population, strategy, n_cycles, base_transition_matrix, costs,
                                                  utilities)
    print(f"Average cost and utility for {strategy} strategy:")
    print(f"Cost: {np.mean(total_costs):.2f}, Utility: {np.mean(total_utilities):.2f}")

