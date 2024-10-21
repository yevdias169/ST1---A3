import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Load the encoded data
df = pd.read_csv('Medical_insurance_encoded.csv')

# Assuming 'charges' is your target variable
target = 'charges'

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of testing set:", X_test.shape)

# Function to perform standardization
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Function to perform normalization
def normalize_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized, scaler

# Standardization (optional)
X_train_std, X_test_std, std_scaler = standardize_data(X_train, X_test)

# Normalization (optional)
X_train_norm, X_test_norm, norm_scaler = normalize_data(X_train, X_test)

# Print sample of raw and transformed data
print("\nSample of raw training data:")
print(X_train.head())

print("\nSample of standardized training data:")
print(pd.DataFrame(X_train_std, columns=X_train.columns).head())

print("\nSample of normalized training data:")
print(pd.DataFrame(X_train_norm, columns=X_train.columns).head())

# Save the splits (optional)
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save the transformed data (optional)
np.save('X_train_std.npy', X_train_std)
np.save('X_test_std.npy', X_test_std)
np.save('X_train_norm.npy', X_train_norm)
np.save('X_test_norm.npy', X_test_norm)

print("\nData splits and transformations have been saved.")
