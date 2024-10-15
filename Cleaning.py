import pandas as pd 

df = pd.read_csv("Medical_insurance.csv")

df_cleaned = df.drop_duplicates()

# Save the cleaned DataFrame back to the CSV file
df_cleaned.to_csv('Medical_insurance.csv', index=False)




