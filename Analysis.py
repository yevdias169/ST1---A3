import pandas as pd 

df = pd.read_csv("Medical_insurance.csv")

'''print(df.shape)
print(df.size)
print(df.info())'''
print(df.describe())