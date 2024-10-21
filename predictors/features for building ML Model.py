import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load the data
try:
    df = pd.read_csv("Medical_insurance.csv")
except FileNotFoundError:
    print("Error: The file 'Medical_insurance.csv' was not found.")
    exit()

# Identify the target variable
target = 'charges'
if target not in df.columns:
    print(f"Error: Target variable '{target}' not found in the dataset.")
    exit()

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove target from features if present
features = [col for col in df.columns if col != target]
numeric_features = [col for col in numeric_features if col != target]

# 1. Correlation Analysis (for numeric features only)
correlation_matrix = df[numeric_features + [target]].corr()
correlation_with_target = correlation_matrix[target].abs().sort_values(ascending=False)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap (Numeric Features)')
plt.tight_layout()
plt.show()

print("Top correlations with target variable:")
print(correlation_with_target)

# 2. ANOVA for categorical variables
anova_results = {}
for feature in categorical_features:
    groups = [group for _, group in df.groupby(feature)[target]]
    if len(groups) > 1:  # ANOVA requires at least two groups
        f_value, p_value = stats.f_oneway(*groups)
        anova_results[feature] = {'F-statistic': f_value, 'p-value': p_value}

anova_df = pd.DataFrame.from_dict(anova_results, orient='index')
print("\nANOVA results for categorical variables:")
print(anova_df)

# 3. Prepare data for Mutual Information and Random Forest
# Create a preprocessor with imputation
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor and transform the features
X = preprocessor.fit_transform(df[features])
y = df[target]

# Get feature names after preprocessing
onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = onehot_encoder.get_feature_names(categorical_features).tolist()
feature_names = numeric_features + cat_feature_names

# 4. Mutual Information
mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
mi_scores.plot(kind='bar')
plt.title('Mutual Information Scores')
plt.ylabel('Mutual Information')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print("\nMutual Information Scores:")
print(mi_scores)

# 5. Random Forest Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title('Random Forest Feature Importance')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print("\nRandom Forest Feature Importance:")
print(importances)

# 6. Combine all metrics
feature_metrics = pd.DataFrame({
    'Mutual_Information': mi_scores,
    'RF_Importance': importances
})

# Add correlation for numeric features
for feature in numeric_features:
    feature_metrics.loc[feature, 'Correlation'] = correlation_with_target.get(feature, 0)

# Add ANOVA p-values for categorical features
for feature in categorical_features:
    if feature in anova_results:
        feature_metrics.loc[feature, 'ANOVA_p_value'] = anova_results[feature]['p-value']

feature_metrics = feature_metrics.sort_values('RF_Importance', ascending=False)

print("\nCombined Feature Metrics:")
print(feature_metrics)

# 7. Select final features
# You can adjust these thresholds based on your specific needs
correlation_threshold = 0.1
mi_threshold = 0.01
importance_threshold = 0.01
p_value_threshold = 0.05

selected_features = feature_metrics[
    (feature_metrics['Correlation'].abs() > correlation_threshold) |
    (feature_metrics['Mutual_Information'] > mi_threshold) |
    (feature_metrics['RF_Importance'] > importance_threshold) |
    ((feature_metrics['ANOVA_p_value'] < p_value_threshold) & (feature_metrics['ANOVA_p_value'].notna()))
].index.tolist()

print("\nSelected Features:")
print(selected_features)

# 8. Visualize relationships between selected features and target
for feature in selected_features:
    if feature in numeric_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature, y=target)
        plt.title(f'Scatter plot: {feature} vs {target}')
    elif feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=feature, y=target)
        plt.title(f'Box plot: {target} by {feature}')
    plt.tight_layout()
    plt.show()

# 9. Save the dataset with selected features
df_selected = df[[col for col in selected_features if col in df.columns] + [target]]
df_selected.to_csv('Medical_insurance_selected_features.csv', index=False)
print("\nDataset with selected features has been saved.")