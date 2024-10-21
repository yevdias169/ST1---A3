import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
df = pd.read_csv("Medical_insurance_cleaned.csv")

# Identify the target variable (assuming 'charges' is the target)
target_variable = 'charges'

# Separate features and target
X = df.drop(columns=[target_variable])
y = df[target_variable]

# Identify continuous and categorical columns
continuous_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Function to plot scatter plots for continuous predictors
def plot_scatter(df, x, y):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f'Scatter plot of {y} vs {x}')
    plt.show()

# Function to plot box plots for categorical predictors
def plot_box(df, x, y):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(f'Box plot of {y} by {x}')
    plt.xticks(rotation=45)
    plt.show()

# Visualize relationships
for col in continuous_cols:
    if col != target_variable:
        plot_scatter(df, col, target_variable)

for col in categorical_cols:
    plot_box(df, col, target_variable)

# Calculate correlations
correlation_matrix = df.corr()
correlation_with_target = correlation_matrix[target_variable].sort_values(ascending=False)

print("Correlations with target variable:")
print(correlation_with_target)

# Visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.show()

# Function to calculate correlation ratio (for categorical variables)
def correlation_ratio(categories, measurements):
    categories = pd.Categorical(categories)
    measurements = np.array(measurements)
    ssw = sum(sum((measurements[categories == category] - np.mean(measurements[categories == category]))**2) 
              for category in categories.categories)
    sst = np.sum((measurements - np.mean(measurements))**2)
    return 1 - ssw/sst

# Calculate correlation ratios for categorical variables
categorical_correlations = {}
for col in categorical_cols:
    correlation = correlation_ratio(df[col], df[target_variable])
    categorical_correlations[col] = correlation

print("\nCorrelation ratios for categorical variables:")
print(pd.Series(categorical_correlations).sort_values(ascending=False))

# Feature selection based on correlation threshold
correlation_threshold = 0.5  # Adjust this value as needed
selected_features = correlation_with_target[abs(correlation_with_target) > correlation_threshold].index.tolist()
selected_features.remove(target_variable)  # Remove target variable from the list

print("\nSelected features based on correlation:")
print(selected_features)

# Visualize selected features
plt.figure(figsize=(12, 6))
sns.heatmap(df[selected_features + [target_variable]].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Selected Features')
plt.show()

# Print final selected features
print("\nFinal selected features:")
print(selected_features)

# Create a new dataframe with selected features
df_selected = df[selected_features + [target_variable]]

# Save the selected features dataset
df_selected.to_csv('Medical_insurance_selected_features.csv', index=False)
print("\nDataset with selected features has been saved.")