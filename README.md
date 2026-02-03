# Boston-Housing-Price-Prediction
This repository contains the first-place winning solution for the Boston Housing prediction competition on Kaggle. 
The goal is to accurately model and predict the median value (medv) of homes based on socio-economic and environmental features using a robust ensemble learning approach.

## Description

The dataset contains housing information collected from the U.S. Census Service and includes **14 columns**:

- **13 numerical features**, such as crime rate, number of rooms, tax rate, etc.
- **1 target variable**: `medv` (median house value)

Files used in this project:

- `train.csv`: Training data with labels  
- `test.csv`: Test data without labels

Evaluation Metric:

- valuation metric: RMSE (Root Mean Squared Error)

## Strategy
This solution adopts a **classical ensemble learning approach**, combining multiple regression models with complementary strengths.

Rather than maximizing model complexity, the focus is on:
- Robust generalization
- Careful validation
- Well-balanced ensemble weighting

## Ensemble Models
The final prediction is produced by a **weighted voting ensemble**.  

| Model | Description |
|------|------------|
| Ridge Regression | Linear model with L2 regularization to reduce overfitting |
| Lasso Regression | Linear model with L1 regularization for feature sparsity |
| Support Vector Regression (SVR) | Captures non-linear relationships using kernel methods |
| Random Forest Regressor | Bagging-based tree ensemble to reduce variance |
| XGBoost Regressor | Gradient boosting trees with strong predictive performance |

##  Project Structure
```
Boston-Housing-Price-Prediction/
├── boston_housing_competition.py       # Main training & ensemble script
├── submission_example.csv              # Example Kaggle submission format
└── data/
    ├── train.csv                       # Training dataset
    └── test.csv                        # Test dataset
```

## Final Result

- **Final Leaderboard Rank**: 🥇 **1st Place**
- **Best Score (RMSE)**: **2.94700**
- **Total Submissions**: 187
