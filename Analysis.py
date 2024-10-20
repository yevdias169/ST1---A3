import pandas as pd 

df = pd.read_csv("Medical_insurance.csv")

#Basic information about the dataset and its numerical columns
print(df.shape)
print(df.size)
print(df.info())
print(df.describe())

#Produce value counts for the catergorical data
for col in df.select_dtypes(include=['object']).columns:
    print(f"Value counts for {col}:")
    print(df[col].value_counts())
    print("\n")   
