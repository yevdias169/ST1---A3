import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv("Medical_insurance.csv")

# Remove duplicates
df_cleaned = df.drop_duplicates()

# Save the cleaned DataFrame back to the CSV file
df_cleaned.to_csv('Medical_insurance_cleaned.csv', index=False)

# Set a consistent style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Distribution of Charges (Histogram)
plt.figure()
sns.histplot(data=df_cleaned, x='charges', kde=True)
plt.title('Distribution of Medical Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# 2. Box Plot of Charges by Smoker Status
plt.figure()
sns.boxplot(x='smoker', y='charges', data=df_cleaned)
plt.title('Distribution of Charges by Smoker Status')
plt.show()

# 3. Scatter Plot of Age vs Charges
plt.figure()
sns.scatterplot(x='age', y='charges', hue='smoker', data=df_cleaned)
plt.title('Age vs Charges (Color: Smoker Status)')
plt.show()

# 4. Violin Plot of BMI by Sex
plt.figure()
sns.violinplot(x='sex', y='bmi', data=df_cleaned)
plt.title('Distribution of BMI by Sex')
plt.show()

# 5. Pair Plot of Numerical Variables
sns.pairplot(df_cleaned[['age', 'bmi', 'children', 'charges']], hue='sex')
plt.suptitle('Pair Plot of Numerical Variables', y=1.02)
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = df_cleaned[['age', 'bmi', 'children', 'charges']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 7. Bar Plot of Average Charges by Region
plt.figure()
sns.barplot(x='region', y='charges', data=df_cleaned)
plt.title('Average Charges by Region')
plt.show()

# 8. Count Plot of Smokers by Sex
plt.figure()
sns.countplot(x='sex', hue='smoker', data=df_cleaned)
plt.title('Count of Smokers by Sex')
plt.show()

# 9. Regression Plot of BMI vs Charges
plt.figure()
sns.regplot(x='bmi', y='charges', data=df_cleaned)
plt.title('BMI vs Charges with Regression Line')
plt.show()

# 10. Box Plot of Charges by Number of Children
plt.figure()
sns.boxplot(x='children', y='charges', data=df_cleaned)
plt.title('Distribution of Charges by Number of Children')
plt.show()

# Print summary statistics
print(df_cleaned.describe())

# Print information about the dataset
print(df_cleaned.info())

# Check for missing values
print(df_cleaned.isnull().sum())

# Print the first few rows of the dataset
print(df_cleaned.head())