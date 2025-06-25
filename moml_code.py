import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product


data = pd.read_csv('/content/drive/My Drive/Berlin_solar_regression.csv')


features = ['Year','Month','Day','Hour','Minute','Temperature','Clearsky.DHI','Clearsky.DNI',
            'Clearsky.GHI','Cloud.Type','Dew.Point','DHI','DNI','Fill.Flag','GHI','Ozone',
            'Relative.Humidity','Solar.Zenith.Angle','Surface.Albedo','Pressure',
            'Precipitable.Water','Wind.Direction','Wind.Speed']
target = 'X50Hertz..MW.'

# dropping rows with missing target values
data = data.dropna(subset=[target])


X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# expanding search space for XGBoost hyperparameters
param_space = {
    'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1, 2]
}

# function to calculate model complexity
def calculate_complexity(params):
    # complexity metric based on model parameters
    # we used a combination of tree depth and number of trees as a proxy for complexity
    complexity = params['n_estimators'] * (2 ** params['max_depth'])
    return complexity

# function to evaluate a parameter set
def evaluate_params(params):
    start_time = time.time()
    
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params.get('gamma', 0),
        reg_alpha=params.get('reg_alpha', 0),
        reg_lambda=params.get('reg_lambda', 1),
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    
    mse = mean_squared_error(y_test, y_pred)
    mbe = np.mean(y_pred - y_test)  # Mean Bias Error
    complexity = calculate_complexity(params)
    
    return {
        'mse': mse,
        'mbe': abs(mbe),  # using absolute value to minimize bias in either direction
        'complexity': complexity,
        'params': params
    }

# random search implementation (more efficient than grid search for larger spaces)
def random_search(param_space, n_iter=300): 
    results = []
    keys, values = zip(*param_space.items())
    
    for _ in range(n_iter):
        # randomly sample parameters
        params = dict(zip(keys, [np.random.choice(v) for v in values]))
        results.append(evaluate_params(params))
    
    return pd.DataFrame(results)


search_results = random_search(param_space, n_iter=300)

# Pareto frontier identification
def identify_pareto_frontier(results_df):
    pareto_front = []
    data = results_df.to_dict('records')
    
    for candidate in data:
        dominated = False
        
        # Comparing with all points in current Pareto front
        for pf_point in pareto_front:
            if (pf_point['mse'] <= candidate['mse'] and 
                pf_point['mbe'] <= candidate['mbe'] and 
                pf_point['complexity'] <= candidate['complexity']):
                dominated = True
                break
        
        if not dominated:
            # removing any points in the current front that this candidate dominates
            pareto_front = [pf_point for pf_point in pareto_front 
                          if not (candidate['mse'] <= pf_point['mse'] and 
                                 candidate['mbe'] <= pf_point['mbe'] and 
                                 candidate['complexity'] <= pf_point['complexity'])]
            pareto_front.append(candidate)
    
    return pd.DataFrame(pareto_front)


pareto_frontier = identify_pareto_frontier(search_results)


while len(pareto_frontier) < 10:
    # running additional iterations if we don't have enough points
    additional_results = random_search(param_space, n_iter=50)
    search_results = pd.concat([search_results, additional_results])
    pareto_frontier = identify_pareto_frontier(search_results)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(search_results['mse'], search_results['mbe'], search_results['complexity'],
           c='blue', alpha=0.3, label='All Evaluations')


ax.scatter(pareto_frontier['mse'], pareto_frontier['mbe'], pareto_frontier['complexity'],
           c='red', s=100, label='Pareto Frontier')

ax.set_xlabel('MSE')
ax.set_ylabel('MBE')
ax.set_zlabel('Complexity')
ax.set_title('3D Pareto Frontier: MSE vs MBE vs Complexity')
plt.legend()
plt.tight_layout()
plt.show()


print(f"Found {len(pareto_frontier)} Pareto Optimal Solutions:")
print(pareto_frontier.sort_values('mse')[['mse', 'mbe', 'complexity', 'params']])