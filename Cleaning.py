import pandas as pd 

df = pd.read_csv("Medical_insurance.csv")

df_cleaned = df.drop_duplicates()

df_cleaned.to_csv('Medical_insurance.csv', index=False)




