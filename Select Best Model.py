import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load the data
df = pd.read_csv('Medical_insurance_encoded.csv')

# Assuming 'charges' is your target variable
X = df.drop('charges', axis=1)
y = df['charges']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'SVM': SVR()
}

# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    return test_r2, cv_scores.mean()

# Evaluate each model
results = {}
for name, model in models.items():
    test_r2, cv_r2 = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    avg_r2 = (test_r2 + cv_r2) / 2
    results[name] = {'Test R2': test_r2, 'CV R2': cv_r2, 'Avg R2': avg_r2}

# Print results
print("Model Evaluation Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Test R2 Score: {metrics['Test R2']:.4f}")
    print(f"  Cross-Validation R2: {metrics['CV R2']:.4f}")
    print(f"  Average R2: {metrics['Avg R2']:.4f}")

# Find the best model based on average R2 score
best_model = max(results, key=lambda x: results[x]['Avg R2'])
print(f"\nBest performing model based on average R2 score: {best_model}")
print(f"Average R2 score: {results[best_model]['Avg R2']:.4f}")

# Optional: Feature importance for tree-based models
if best_model in ['Decision Tree', 'Random Forest', 'XGBoost']:
    model = models[best_model]
    model.fit(X_train_scaled, y_train)
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    print("\nTop 10 important features:")
    print(feature_imp.head(10))