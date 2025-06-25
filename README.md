# 🌞 Pareto-Optimal Solar Power Forecasting

This repository contains our **Machine Learning for Modern Optimization (MOML)** project on forecasting solar power output using meteorological data from Berlin. We approach this as a **multi-objective regression problem** using **XGBoost**, optimizing simultaneously for:

- **Accuracy** (Mean Squared Error - MSE)
- **Bias** (Mean Bias Error - MBE)
- **Model Complexity** (proxy: number and depth of trees)

Our final model set achieves **33 Pareto-optimal configurations**, offering a spectrum of trade-offs for real-world deployment needs such as high accuracy, low bias, or minimal model size.

---

## 📊 Problem Statement

Forecasting solar power is essential for:

- Grid reliability  
- Energy trading  
- Renewable resource integration  

Single-objective models often neglect important trade-offs. Our goal is to offer a **Pareto frontier** of model choices balancing accuracy, bias, and complexity.

---

## 🧠 Key Contributions

- 🔍 Performed **randomized search** over an expanded hyperparameter space of XGBoost  
- ⚖️ Identified **Pareto-optimal** configurations across 3 objectives: MSE, |MBE|, and model complexity  
- 📈 Visualized the Pareto frontier to guide model selection  
- 🛠️ Demonstrated trade-offs between complexity and performance  
  - Smallest model MSE ≈ 38,543  
  - Best model MSE ≈ 4,518  

---

## 📁 Dataset

The dataset `Berlin_solar_regression.csv` contains half-hourly readings of:

- Meteorological features (Temperature, Humidity, Wind, etc.)
- Solar irradiance measures (DHI, DNI, GHI, etc.)
- Target: **50Hertz solar power output (MW)**

**Preprocessing Steps**:
- Removed rows with missing target values  
- Train-test split: 80/20 (random state = 42)  
- Standardized features using `StandardScaler`  

---

## 🔧 Methodology

1. **Model**: `xgboost.XGBRegressor`  
2. **Hyperparameter Space**:
   - `n_estimators`: 50 to 400
   - `max_depth`: 2 to 9
   - `learning_rate`: 0.001 to 0.2
   - `subsample`, `colsample_bytree`: 0.5 to 1.0
   - Regularization: `gamma`, `reg_alpha`, `reg_lambda`
3. **Evaluation Metrics**:
   - `MSE`, `|MBE|`, `Complexity = n_estimators × 2^max_depth`
4. **Optimization**: Random search (300+50 iterations)
5. **Pareto Frontier Detection**: Maintained non-dominated configurations
6. **Visualization**: 3D scatter plot of MSE vs. MBE vs. Complexity

---

## 📌 Results

- **33 Pareto-optimal points** discovered from 350 hyperparameter configurations  
- MBE consistently **< 2.0 MW** across all Pareto points  

**Trade-offs Observed**:
- 🔁 **Accuracy vs. Complexity**:
  - MSE ≈ 4,518 → complexity = 153,600
  - MSE ≈ 38,543 → complexity = 400
- ⚡ Lightweight models (complexity ≈ 3,200) still achieve MSE ≈ 6,000 with low MBE
- 📉 Lowest bias observed: MBE = 0.0837 MW

---

## 💻 Tech Stack

- Python 3.8  
- XGBoost 1.6.1  
- Scikit-learn 1.0.2  
- NumPy, Pandas, Matplotlib  

---

## 📈 Visualizations

> The repository includes a 3D Pareto frontier visualization.

