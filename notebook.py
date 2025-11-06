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
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()
# several features show strong correlation and might be excluded
# price correlates with size related features and not so strongly with the rest

# %%
df.plot.box()
plt.show()
# most columnss have densly distributed values except price and sqft_lot which have more dispersion to the right, especially price

# %%
scaler = sklearn.preprocessing.StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])
df_scaled.plot.box()
plt.show()

# %% [markdown]
# ## Baseline models

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

# %%
from sklearn.linear_model import LinearRegression

lm = LinearRegression().fit(X_train, y_train)
print("LinearRegression")
print_report(lm, **unscaled_features)

# %%
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor().fit(X_train, y_train)
print_report(knn, **unscaled_features)

# %%

