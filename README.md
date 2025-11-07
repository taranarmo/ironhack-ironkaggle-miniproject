# IronKaggle House Price Prediction Project

## Overview
This project predicts house prices in King County, USA using machine learning techniques. The [dataset](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa) contains 21,613 records with 20 features describing various aspects of residential homes.

## Requirements

The project uses [uv](https://docs.astral.sh/uv/) to handle python dependencies and Nix flake to handle the rest.
If you have [Nix](https://nixos.org/) then `nix develop` will create working environment.

## Workflow

1. Data Loading & Exploration
   - Loaded the King County house sales dataset
   - Analyzed data structure, features, and target variable
   - Identified data types and shape of the dataset

2. Data Cleaning
   - Checked for missing values (none found)
   - Handled duplicates if any
   - Created new feature 'is_renovated' based on yr_renovated > 2005
   - Dropped 'date' column as it contained post dates from a single year

3. Feature Engineering
   - Identified categorical and numerical features
   - Performed correlation analysis between features and target variable
   - Addressed multicollinearity by removing highly correlated features (>0.8)
   - Applied standardization where appropriate

4. Model Development
   - Split data into train/test sets
   - Trained baseline models (Linear Regression, KNN Regressor)
   - Implemented advanced models (Random Forest, Gradient Boosting, SVR)
   - Performed hyperparameter tuning using GridSearchCV

5. Model Evaluation & Selection
   - Evaluated models using R² and MSE metrics
   - Compared all models to select the best performer
   - Analyzed feature importance from the best model

## Key Decisions & Reasoning

1. *Feature Engineering*: Created 'is_renovated' binary feature to capture the impact of recent renovations on house prices.

2. *Data Preprocessing*: Dropped the 'date' column as it had limited temporal dynamics (only one year of data).

3. *Multicollinearity Handling*: Identified and removed highly correlated features (>0.8) by keeping the one with higher correlation to the target variable in each pair, to improve model stability and interpretability.

4. *Feature Selection*: Removed features with low correlation (<0.2) with the target variable to reduce noise and improve model performance.

5. *Model Selection*: Started with baseline models and progressively tried more complex algorithms, including ensemble methods, to find the best performing model.

6. *Hyperparameter Tuning*: Used GridSearchCV to optimize model parameters and improve performance.

7. *Feature Filtering*: Applied both multicollinearity removal and low-correlation feature removal simultaneously to create a cleaner feature set for model training.

## Deliverables

- Jupyter Notebook (`notebook.ipynb` or `notebook.py`) containing all analysis, code, and results
- Complete Python script (`run_analysis.py`) that executes the full analysis pipeline
- Visualizations saved as images (correlation_matrix.png, feature_distribution.png, feature_importance.png, etc.)
- Presentation slides summarizing the ML process, insights, and final model  
- This README file explaining the workflow, reasoning, and key decisions

## Key Findings

- House size and location influencing house prices in King County the most
- The most singificant feature is `grade` though it's a synthetic value based on other parameters
- Among chosen models Gradient boosting ones performed the best (on as raw as possible dataset, with only removed outliers)
- Model performance metrics (R² and MSE) comparing different approaches
- Impact of feature selection and filtering on model performance
- Insights about house price determinants in the region
