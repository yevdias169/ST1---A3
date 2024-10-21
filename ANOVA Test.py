import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("Medical_insurance.csv")

# Function to perform ANOVA test
def perform_anova(df, categorical_var, continuous_var):
    categories = df[categorical_var].unique()
    samples = [df[df[categorical_var] == category][continuous_var].dropna() for category in categories]
    
    f_statistic, p_value = stats.f_oneway(*samples)
    
    return f_statistic, p_value

# Identify categorical and continuous variables
categorical_vars = df.select_dtypes(include=['object', 'category']).columns
continuous_vars = df.select_dtypes(include=['float64', 'int64']).columns

# Perform ANOVA for each combination of categorical and continuous variables
results = []

for cat_var in categorical_vars:
    for cont_var in continuous_vars:
        f_statistic, p_value = perform_anova(df, cat_var, cont_var)
        results.append({
            'Categorical Variable': cat_var,
            'Continuous Variable': cont_var,
            'F-statistic': f_statistic,
            'p-value': p_value
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Sort results by p-value
results_df = results_df.sort_values('p-value')

# Print results
print("ANOVA Test Results:")
print(results_df)

# Visualize results
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Categorical Variable', y='Continuous Variable', size='F-statistic', 
                hue='p-value', data=results_df, sizes=(20, 200), hue_norm=(0, 0.05))
plt.xticks(rotation=45, ha='right')
plt.title('ANOVA Test Results')
plt.tight_layout()
plt.show()

# Function to create box plots for significant relationships
def plot_significant_relationships(df, results_df, p_value_threshold=0.05):
    significant_results = results_df[results_df['p-value'] < p_value_threshold]
    
    for _, row in significant_results.iterrows():
        cat_var = row['Categorical Variable']
        cont_var = row['Continuous Variable']
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_var, y=cont_var, data=df)
        plt.title(f'{cont_var} vs {cat_var} (p-value: {row["p-value"]:.4f})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Plot significant relationships
plot_significant_relationships(df, results_df)

# Print interpretation
print("\nInterpretation:")
print("- The null hypothesis is that there is no relationship between the categorical and continuous variables.")
print("- If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis.")
print("- A low p-value suggests a significant relationship between the variables.")
print("- The F-statistic represents the ratio of variance between the groups to the variance within the groups.")
print("- A larger F-statistic indicates a stronger relationship between the variables.")