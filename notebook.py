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

dataset_path = ""

# %%
import kagglehub

dataset_path = kagglehub.dataset_download("minasameh55/king-country-houses-aa")
dataset_path = Path(dataset_path)

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
# most columnss have densly distributed values except price and sqft_lot which have more dispersion to the right

# %%
scaler = sklearn.preprocessing.StandardScaler()
numerical_scaled = df[numerical_columns].copy()
numerical_scaled[:] = scaler.fit_transform(df[numerical_columns])
numerical_scaled.plot.box()
plt.show()
