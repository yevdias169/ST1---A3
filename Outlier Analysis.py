import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv("Medical_insurance.csv")

# Remove duplicates
df_cleaned = df.drop_duplicates()

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Identify numeric columns
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

# Remove outliers from numeric columns
for col in numeric_cols:
    df_cleaned = remove_outliers(df_cleaned, col)

# Remove rows with missing values
df_cleaned = df_cleaned.dropna()

# Function to plot histogram before and after outlier removal
def plot_histogram(df_original, df_cleaned, column):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_original[column], kde=True)
    plt.title(f'Distribution of {column} (Before)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df_cleaned[column], kde=True)
    plt.title(f'Distribution of {column} (After)')
    
    plt.tight_layout()
    plt.show()

# Plot histograms for numeric columns
for col in numeric_cols:
    plot_histogram(df, df_cleaned, col)

# Print summary statistics
print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", df_cleaned.shape)
print("\nMissing values in cleaned dataset:")
print(df_cleaned.isnull().sum())

# Save the cleaned DataFrame
df_cleaned.to_csv('Medical_insurance_cleaned_no_outliers.csv', index=False)