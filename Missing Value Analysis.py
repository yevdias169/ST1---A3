import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the data
df = pd.read_csv("Medical_insurance.csv")

# Function to calculate percentage of missing values
def missing_percentage(df):
    return df.isnull().sum().sort_values(ascending=False) / len(df) * 100

# Display missing value percentages
print("Percentage of missing values:")
print(missing_percentage(df))

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

# 1. Delete rows with missing values
df_deleted = df.dropna()
print("\nShape after deleting rows with missing values:", df_deleted.shape)

# 2. Impute continuous variables with median
continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_median_imputed = df.copy()
for col in continuous_columns:
    df_median_imputed[col] = df_median_imputed[col].fillna(df_median_imputed[col].median())

# 3. Impute categorical variables with mode
categorical_columns = df.select_dtypes(include=['object']).columns
df_mode_imputed = df.copy()
for col in categorical_columns:
    df_mode_imputed[col] = df_mode_imputed[col].fillna(df_mode_imputed[col].mode()[0])

# 4. Insert values based on nearby values (for time series or sequential data)
# Assuming 'date' column exists and is in datetime format
if 'date' in df.columns:
    df_interpolated = df.copy()
    df_interpolated = df_interpolated.sort_values('date')
    df_interpolated = df_interpolated.set_index('date')
    df_interpolated = df_interpolated.interpolate(method='time')
else:
    print("No 'date' column found for time-based interpolation.")

# 5. Insert values based on business logic (example)
def impute_based_on_logic(row):
    if pd.isnull(row['bmi']):
        if row['age'] < 30:
            return 25  # Example: assume average BMI for young adults
        else:
            return 28  # Example: assume slightly higher BMI for older adults
    return row['bmi']

df_logic_imputed = df.copy()
df_logic_imputed['bmi'] = df_logic_imputed.apply(impute_based_on_logic, axis=1)

# Advanced imputation methods
# MICE (Multiple Imputation by Chained Equations)
mice_imputer = IterativeImputer(random_state=0)
df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)

# Comparison of imputation methods
methods = {
    'Original': df,
    'Deleted Rows': df_deleted,
    'Median Imputed': df_median_imputed,
    'Mode Imputed': df_mode_imputed,
    'Logic Imputed': df_logic_imputed,
    'MICE Imputed': df_mice_imputed
}

# Visualize the effect of different imputation methods on a numeric column
numeric_col = continuous_columns[0]  # Choose the first numeric column for visualization

plt.figure(figsize=(15, 10))
for i, (name, data) in enumerate(methods.items(), 1):
    plt.subplot(3, 2, i)
    sns.histplot(data[numeric_col].dropna(), kde=True)
    plt.title(f'{name} - {numeric_col}')
    plt.xlabel(numeric_col)
plt.tight_layout()
plt.show()

# Print summary statistics for each method
for name, data in methods.items():
    print(f"\nSummary statistics for {name}:")
    print(data.describe())

# Save the imputed datasets
for name, data in methods.items():
    data.to_csv(f'Medical_insurance_{name.lower().replace(" ", "_")}.csv', index=False)

print("\nImputed datasets have been saved.")