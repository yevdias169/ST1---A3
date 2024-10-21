import pandas as pd

# Load the data
df = pd.read_csv("Medical_insurance.csv")

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Use get_dummies() to convert categorical variables to numeric
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, prefix_sep='_')

# Print the first few rows of the encoded DataFrame
print(df_encoded.head())

# Save the encoded DataFrame to a new CSV file
df_encoded.to_csv('Medical_insurance_encoded.csv', index=False)

print("Encoded dataset has been saved to 'Medical_insurance_encoded.csv'.")