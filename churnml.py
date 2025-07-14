import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/New folder/Bank Customer Churn Prediction.csv")
df
df.head()
df.describe()
df.info()

df.drop(columns=['customer_id'], inplace=True)

# Plot churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='churn', palette='Set2')
plt.title("Churn Distribution")
plt.xlabel("Churn (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

#EDA

# Gender distribution by churn
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='gender', hue='churn', palette='coolwarm')
plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.show()

# Country distribution by churn
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='country', hue='churn', palette='pastel')
plt.title("Churn by Country")
plt.xlabel("Country")
plt.ylabel("Count")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.show()

# Age distribution by churn
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='age', hue='churn', bins=30, kde=True, palette='muted')
plt.title("Age Distribution by Churn")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Balance distribution by churn
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='balance', hue='churn', bins=30, kde=True, palette='Set2')
plt.title("Balance Distribution by Churn")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()



# Encode categorical columns
df_encoded = df.copy()
le_country = LabelEncoder()
le_gender = LabelEncoder()

df_encoded['country'] = le_country.fit_transform(df_encoded['country'])
df_encoded['gender'] = le_gender.fit_transform(df_encoded['gender'])

# CORRELATION HEATMAP
plt.figure(figsize=(10, 6))
corr = df_encoded.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# FEATURE IMPORTANCE (Random Forest)
X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()


# 1. Encode categorical features
df_model = df.copy()
df_model['country'] = LabelEncoder().fit_transform(df_model['country'])
df_model['gender'] = LabelEncoder().fit_transform(df_model['gender'])

# 2. Split into features and target
X = df_model.drop('churn', axis=1)
y = df_model['churn']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4a. Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

# 4b. Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# 5. Confusion Matrix for Random Forest
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize base model
rf = RandomForestClassifier(random_state=42)

# Setup GridSearch
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, scoring='f1', n_jobs=-1, verbose=1)

# Fit on training data
grid_search.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate best model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report (Best Random Forest):")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Tuned Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Save the best (tuned) Random Forest model
joblib.dump(best_rf, 'rf_model.pkl')


# Predictions already available:
# y_pred_log from logistic regression
# y_pred_rf from random forest default
# y_pred_best from tuned random forest

models = {
    "Logistic Regression": y_pred_log,
    "Random Forest (Default)": y_pred_rf,
    "Random Forest (Tuned)": y_pred_best
}

print("Model Performance Comparison:\n")
for name, y_pred in models.items():
    print(f"{name}:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_pred):.4f}")
    print("-" * 40)
    
