# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import catppuccin
from pathlib import Path
import sklearn

mpl.style.use(catppuccin.PALETTE.mocha.identifier)

# %%
try:
    import kagglehub
    dataset_path = kagglehub.dataset_download("minasameh55/king-country-houses-aa")
    dataset_path = Path(dataset_path)
except ModuleNotFoundError:
    dataset_path = ""

# %%
if not dataset_path:
    dataset_path = Path(".")
df = pd.read_csv(dataset_path / "king_ country_ houses_aa.csv")
df.set_index("id", inplace=True)

# %%
df.dtypes
df.max()

# %%
df.shape
df.dropna().shape

# %% [markdown]
# target variable is price
# date can be probably dropped (post dates, just one year thus temporal dynamics isn't available)
# no NaNs
# half of yr_renovated values are 0 which is probably "not known"
# categorical columns are already encoded

# %%
df.eval("is_renovated=yr_renovated>2005", inplace=True)

# %%
df.drop(columns="date", inplace=True)
categorical_columns = ["waterfront", "view", "condition", "grade", "zipcode", "is_renovated"]
numerical_columns = [col for col in df.columns if col not in categorical_columns]

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.close()
# several features show strong correlation and might be excluded
# price correlates with size related features and not so strongly with the rest

# %%
df.plot.box(figsize=(15, 8))
plt.title("Distribution of Features")
plt.tight_layout()
plt.savefig("feature_distribution.png", dpi=150)
plt.close()
# most columnss have densly distributed values except price and sqft_lot which have more dispersion to the right, especially price

# %%
scaler = sklearn.preprocessing.StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])
df_scaled.plot.box(figsize=(15, 8))
plt.title("Distribution of Features After Scaling")
plt.tight_layout()
plt.savefig("feature_distribution_scaled.png", dpi=150)
plt.close()

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def print_report(model, X_train, X_test, y_train, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(
        f"R² on train is {r2_score(y_train, y_pred_train):.4f}",
        f"MSE: {mean_squared_error(y_train, y_pred_train):.4f}",
    )
    print(
        f"R² on test is  {r2_score(y_test, y_pred_test):.4f}",
        f"MSE: {mean_squared_error(y_test, y_pred_test):.4f}",
    )

# %%
X = df.drop(columns="price")
X_scaled = df_scaled.drop(columns="price")
target = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    target,
    shuffle=True,
    random_state=42, # we will reuse the split for scaled inputs
)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled,
    target,
    shuffle=True,
    random_state=42,
)
unscaled_features = {
    "X_train": X_train,
    "X_test":  X_test,
    "y_train": y_train,
    "y_test":  y_test,
}
scaled_features = {
    "X_train": X_train_scaled,
    "X_test":  X_test_scaled,
    "y_train": y_train,
    "y_test":  y_test
}

# %% [markdown]
# ## Baseline models

# %%
print("Baseline models")
from sklearn.linear_model import LinearRegression

lm = LinearRegression().fit(X_train, y_train)
print("LinearRegression")
print_report(lm, **unscaled_features)
print_report(lm, **scaled_features)

# %%
from sklearn.neighbors import KNeighborsRegressor

print("KNN Regressor")
knn = KNeighborsRegressor().fit(X_train, y_train)
print_report(knn, **unscaled_features)
print_report(knn, **scaled_features)

# %% [markdown]
# ## Model Improvement

# %% [markdown]
# ### Handle multicollinearity by removing highly correlated features

# %%
# Calculate the correlation matrix
corr_matrix = X_train.corr().abs()
print("Correlation matrix:")
print(corr_matrix)

# %% [markdown]
# Find pairs of features with high correlation (>0.8)
# Select upper triangle of correlation matrix

# %%
upper_triangle = corr_matrix.where(
    np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
)

# Find features with correlation greater than 0.8
high_corr_pairs = []
for i in range(len(upper_triangle.columns)):
    for j in range(len(upper_triangle.columns)):
        if i != j and upper_triangle.iloc[i, j] > 0.8:
            high_corr_pairs.append((upper_triangle.index[i], upper_triangle.columns[j], upper_triangle.iloc[i, j]))

print("Highly correlated feature pairs (correlation > 0.8):")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

# %% [markdown]
# Identify features to drop based on high correlation (one from each pair, keeping the one with higher correlation to target)

# %%
features_to_drop = set()
for feat1, feat2, corr_value in high_corr_pairs:
    # Drop the feature with lower correlation to target (price)
    corr_with_target_1 = abs(df[feat1].corr(df["price"]))
    corr_with_target_2 = abs(df[feat2].corr(df["price"]))
    if corr_with_target_1 < corr_with_target_2:
        features_to_drop.add(feat1)
    else:
        features_to_drop.add(feat2)  # If equal, drop the second one

print(f"\nFeatures to be dropped due to high correlation (>0.8): {list(features_to_drop)}")

# %% [markdown]
# Create new datasets without highly correlated features

# %%
X_train_no_multicoll = X_train.drop(columns=list(features_to_drop))
X_test_no_multicoll = X_test.drop(columns=list(features_to_drop))

print(f"Shape of X_train after removing multicollinear features: {X_train_no_multicoll.shape}")
print(f"Shape of X_test after removing multicollinear features: {X_test_no_multicoll.shape}")

# %% [markdown]
# Train models with multicollinearity handled

# %%
lm_no_multicoll = LinearRegression().fit(X_train_no_multicoll, y_train)
print("\nLinear Regression after handling multicollinearity")
print_report(lm_no_multicoll, X_train_no_multicoll, X_test_no_multicoll, y_train, y_test)

# %% [markdown]
# Feature selection based on correlation with target variable
# Let's select features with higher correlation to price

# Calculate correlation with target variable

# %%
correlation_with_target = df.corr()["price"].abs().sort_values(ascending=False)
print("Features correlation with price:")
print(correlation_with_target)

# %% [markdown]
# Select features with correlation above threshold and remove low-correlation features

# %%
feature_threshold = 0.2
selected_features = correlation_with_target[correlation_with_target > feature_threshold].index.tolist()
selected_features = [feat for feat in selected_features if feat != "price"]  # Remove price from features
print(f"\nFeatures selected with correlation > {feature_threshold}: {selected_features}")

# %% [markdown]
# Apply both multicollinearity removal and low-correlation feature removal
# Start with all features, remove multicollinearity, then remove low-correlation features

# %%
all_features = set(X_train.columns)
features_after_multicoll = all_features - features_to_drop
features_after_multicoll_and_low_corr = [f for f in selected_features if f in features_after_multicoll]

print(f"\nFeatures after removing both multicollinearity and low-correlation features: {features_after_multicoll_and_low_corr}")

# %% [markdown]
# Train models with features that survived both filters
X_train_filtered = X_train[features_after_multicoll_and_low_corr]
X_test_filtered = X_test[features_after_multicoll_and_low_corr]

# Train Linear Regression with filtered features
lm_filtered = LinearRegression().fit(X_train_filtered, y_train)
print("\nLinear Regression with Features Filtered (Multicollinearity + Low Correlation)")
print_report(lm_filtered, X_train_filtered, X_test_filtered, y_train, y_test)

# %% [markdown]
# Handling outliers

# %%
def remove_outliers(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from the target variable 'price'
df_no_outliers = remove_outliers(df, 'price')
print(f"Shape before outlier removal: {df.shape}")
print(f"Shape after outlier removal: {df_no_outliers.shape}")

# %% [markdown]
# Re-split the data without outliers

# %%
X_no_outliers = df_no_outliers.drop(columns="price")
target_no_outliers = df_no_outliers["price"]

X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(
    X_no_outliers,
    target_no_outliers,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

print(f"Training set shape after outlier removal: {X_train_out.shape}")
print(f"Test set shape after outlier removal: {X_test_out.shape}")

# %% [markdown]
# Try ensemble methods

# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# %% [markdown]
# Random Forest Regressor with filtered features

# %%
rf_filtered = RandomForestRegressor(random_state=42)
rf_filtered.fit(X_train_filtered, y_train)
print("\nRandom Forest Regressor (with feature filtering)")
print_report(rf_filtered, X_train_filtered, X_test_filtered, y_train, y_test)

# %% [markdown]
# Hyperparameter tuning for Random Forest with filtered features

# %%
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_rf,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search_rf.fit(X_train_filtered, y_train)
best_rf_filtered = grid_search_rf.best_estimator_

print("\nTuned Random Forest Regressor (with feature filtering)")
print(f"Best parameters: {grid_search_rf.best_params_}")
print_report(best_rf_filtered, X_train_filtered, X_test_filtered, y_train, y_test)

# %% [markdown]
# Gradient Boosting Regressor with filtered features

# %%
gb_filtered = GradientBoostingRegressor(random_state=42)
gb_filtered.fit(X_train_filtered, y_train)
print("\nGradient Boosting Regressor (with feature filtering)")
print_report(gb_filtered, X_train_filtered, X_test_filtered, y_train, y_test)

# %% [markdown]
# Hyperparameter tuning for Gradient Boosting with filtered features

# %%
param_grid_gb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search_gb = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid_gb,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search_gb.fit(X_train_filtered, y_train)
best_gb_filtered = grid_search_gb.best_estimator_

print("\nTuned Gradient Boosting Regressor (with feature filtering)")
print(f"Best parameters: {grid_search_gb.best_params_}")
print_report(best_gb_filtered, X_train_filtered, X_test_filtered, y_train, y_test)

# %% [markdown]
# Support Vector Regressor with filtered and scaled features

# %%
X_train_filtered_scaled = X_train_scaled[features_after_multicoll_and_low_corr]
X_test_filtered_scaled = X_test_scaled[features_after_multicoll_and_low_corr]

svr_filtered = SVR()
svr_filtered.fit(X_train_filtered_scaled, y_train)
print("\nSupport Vector Regressor (with feature filtering and scaling)")
print_report(svr_filtered, X_train_filtered_scaled, X_test_filtered_scaled, y_train, y_test)

# %% [markdown]
# Hyperparameter tuning for SVR with filtered features

# %%
param_grid_svr = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'epsilon': [0.1, 0.2, 0.5]
}

grid_search_svr = GridSearchCV(
    SVR(),
    param_grid_svr,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search_svr.fit(X_train_filtered_scaled, y_train)
best_svr_filtered = grid_search_svr.best_estimator_

print("\nTuned Support Vector Regressor (with feature filtering and scaling)")
print(f"Best parameters: {grid_search_svr.best_params_}")
print_report(best_svr_filtered, X_train_filtered_scaled, X_test_filtered_scaled, y_train, y_test)

# %% [markdown]
# ## Model Comparison

# %%
# Store models, features, and results together for easier lookup
models_and_results = {
    "Linear Regression": {
        "model": lm,
        "features": X.columns,
        "R2_test": r2_score(y_test, lm.predict(X_test)),
        "MSE_test": mean_squared_error(y_test, lm.predict(X_test)),
        "X_test": X_test
    },
    "Linear Regression (No Multicoll)": {
        "model": lm_no_multicoll,
        "features": X_train_no_multicoll.columns,
        "R2_test": r2_score(y_test, lm_no_multicoll.predict(X_test_no_multicoll)),
        "MSE_test": mean_squared_error(y_test, lm_no_multicoll.predict(X_test_no_multicoll)),
        "X_test": X_test_no_multicoll
    },
    "Linear Regression (Filtered)": {
        "model": lm_filtered,
        "features": X_train_filtered.columns,
        "R2_test": r2_score(y_test, lm_filtered.predict(X_test_filtered)),
        "MSE_test": mean_squared_error(y_test, lm_filtered.predict(X_test_filtered)),
        "X_test": X_test_filtered
    },
    "Random Forest (Filtered)": {
        "model": rf_filtered,
        "features": X_train_filtered.columns,
        "R2_test": r2_score(y_test, rf_filtered.predict(X_test_filtered)),
        "MSE_test": mean_squared_error(y_test, rf_filtered.predict(X_test_filtered)),
        "X_test": X_test_filtered
    },
    "Tuned Random Forest (Filtered)": {
        "model": best_rf_filtered,
        "features": X_train_filtered.columns,
        "R2_test": r2_score(y_test, best_rf_filtered.predict(X_test_filtered)),
        "MSE_test": mean_squared_error(y_test, best_rf_filtered.predict(X_test_filtered)),
        "X_test": X_test_filtered
    },
    "Gradient Boosting (Filtered)": {
        "model": gb_filtered,
        "features": X_train_filtered.columns,
        "R2_test": r2_score(y_test, gb_filtered.predict(X_test_filtered)),
        "MSE_test": mean_squared_error(y_test, gb_filtered.predict(X_test_filtered)),
        "X_test": X_test_filtered
    },
    "Tuned Gradient Boosting (Filtered)": {
        "model": best_gb_filtered,
        "features": X_train_filtered.columns,
        "R2_test": r2_score(y_test, best_gb_filtered.predict(X_test_filtered)),
        "MSE_test": mean_squared_error(y_test, best_gb_filtered.predict(X_test_filtered)),
        "X_test": X_test_filtered
    },
    "SVR (Filtered)": {
        "model": svr_filtered,
        "features": X_train_filtered.columns,
        "R2_test": r2_score(y_test, svr_filtered.predict(X_test_filtered_scaled)),
        "MSE_test": mean_squared_error(y_test, svr_filtered.predict(X_test_filtered_scaled)),
        "X_test": X_test_filtered_scaled
    },
    "Tuned SVR (Filtered)": {
        "model": best_svr_filtered,
        "features": X_train_filtered.columns,
        "R2_test": r2_score(y_test, best_svr_filtered.predict(X_test_filtered_scaled)),
        "MSE_test": mean_squared_error(y_test, best_svr_filtered.predict(X_test_filtered_scaled)),
        "X_test": X_test_filtered_scaled
    }
}

# Create a DataFrame for better visualization
results_data = {}
for name, data in models_and_results.items():
    results_data[name] = {k: v for k, v in data.items() if k in ['R2_test', 'MSE_test']}

results_df = pd.DataFrame(results_data).T
results_df = results_df.sort_values(by='R2_test', ascending=False)
print("Model Comparison (sorted by R2 on test set):")
print(results_df)

# %% [markdown]
# ## Feature Importance Analysis (using the best performing model)

# %%
# Determine best model based on R2 score
best_model_name = results_df.index[0]

# Use models_and_results dictionary to get model and features directly
if best_model_name.startswith("SVR"):
    # SVR doesn't have feature importance, so use the best performing tree-based model for importance analysis
    tree_models = [name for name in models_and_results.keys() if "Random Forest" in name or "Gradient Boosting" in name]
    if tree_models:
        # Find the best tree model among available ones
        best_tree_model_name = None
        best_r2 = -float('inf')
        for name in tree_models:
            if name in results_df.index and results_df.loc[name, 'R2_test'] > best_r2:
                best_tree_model_name = name
                best_r2 = results_df.loc[name, 'R2_test']

        if best_tree_model_name:
            best_model_for_importance = models_and_results[best_tree_model_name]['model']
            feature_names = models_and_results[best_tree_model_name]['features']
            final_model_name = f"{best_tree_model_name} (for feature importance)"
        else:
            # If no tree models, skip importance analysis
            best_model_for_importance = None
            feature_names = []
            final_model_name = best_model_name
    else:
        # If no tree models in the available models, skip importance analysis
        best_model_for_importance = None
        feature_names = []
        final_model_name = best_model_name
else:
    # For non-SVR models, use the actual best model
    best_model_for_importance = models_and_results[best_model_name]['model']
    feature_names = models_and_results[best_model_name]['features']
    final_model_name = best_model_name

print(f"Best performing model: {final_model_name}")

# %% [markdown]
# Extract feature importance based on model type

# %%
if best_model_for_importance is not None and hasattr(best_model_for_importance, 'feature_importances_'):
    # For tree-based models
    feature_importance = best_model_for_importance.feature_importances_

    # Create a dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importances from Best Model:")
    print(importance_df)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
    plt.title(f'Top 10 Feature Importances - {final_model_name}')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

elif best_model_for_importance is not None and hasattr(best_model_for_importance, 'coef_'):
    # For linear models, use coefficients
    feature_importance = np.abs(best_model_for_importance.coef_)

    # Create a dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient_Magnitude': feature_importance
    }).sort_values('Coefficient_Magnitude', ascending=False)

    print("\nFeature Importance (Absolute Coefficient Values) from Linear Model:")
    print(importance_df)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(10), x='Coefficient_Magnitude', y='Feature')
    plt.title(f'Top 10 Feature Importance (Linear Model) - {final_model_name}')
    plt.tight_layout()
    plt.savefig("feature_importance_linear.png", dpi=150)
    plt.close()
else:
    print(f"Feature importance not available for {final_model_name}")
    importance_df = pd.DataFrame()  # This will be handled in the final analysis

# %% [markdown]
# ## Final Analysis and Conclusions

# %%
print("Key findings:")
print(f"1. Best model: {best_model_name}")
print(f"2. Best R2 score on test set: {results_df.iloc[0]['R2_test']:.4f}")
print(f"3. Best MSE on test set: {results_df.iloc[0]['MSE_test']:.2f}")

# Top 5 features that influence house prices
if 'importance_df' in locals() or 'importance_df' in globals():
    top_features = importance_df.head(5)['Feature'].tolist()
    print(f"4. Top 5 features influencing house prices: {top_features}")

