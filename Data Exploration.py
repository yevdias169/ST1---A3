import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv("Medical_insurance.csv")

# Remove duplicates
df_cleaned = df.drop_duplicates()

# Save the cleaned DataFrame back to the CSV file
df_cleaned.to_csv('Medical_insurance_cleaned.csv', index=False)

## Data Exploration

# Gauge volume of data
print("Dataset shape:", df_cleaned.shape)
print("\nNumber of rows:", df_cleaned.shape[0])
print("Number of columns:", df_cleaned.shape[1])

# Display data types
print("\nData types:")
print(df_cleaned.dtypes)

# Identify column types
numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns

print("\nNumeric columns:")
print(numeric_cols.tolist())

print("\nCategorical columns:")
print(categorical_cols.tolist())

# Further categorize numeric columns into quantitative and qualitative
quantitative_cols = []
qualitative_cols = []

for col in numeric_cols:
    if df_cleaned[col].nunique() > 10:  # Adjust this threshold as needed
        quantitative_cols.append(col)
    else:
        qualitative_cols.append(col)

print("\nQuantitative columns:")
print(quantitative_cols)

print("\nQualitative columns (including categorical):")
print(qualitative_cols + categorical_cols.tolist())

# Identify potentially unwanted columns
low_variance_cols = []
for col in df_cleaned.columns:
    if df_cleaned[col].nunique() == 1:
        low_variance_cols.append(col)

print("\nPotentially unwanted columns (low variance):")
print(low_variance_cols)

# Display summary statistics
print("\nSummary statistics:")
print(df_cleaned.describe())

# Display information about non-numeric columns
print("\nNon-numeric column information:")
print(df_cleaned[categorical_cols].describe())

# Visualize the distribution of a quantitative variable
plt.figure(figsize=(10, 6))
sns.histplot(data=df_cleaned, x=quantitative_cols[0], kde=True)
plt.title(f'Distribution of {quantitative_cols[0]}')
plt.show()

# Visualize the distribution of a categorical variable
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x=categorical_cols[0])
plt.title(f'Distribution of {categorical_cols[0]}')
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap for quantitative variables
plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned[quantitative_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Quantitative Variables')
plt.show()