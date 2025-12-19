"""
Train and save the medical appointment no-show prediction model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('project cloud/KaggleV2-May-2016.csv')

# Feature engineering
print("Engineering features...")
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['days_in_advance'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# Extract month and day of week
df['month_appointment'] = df['AppointmentDay'].dt.month
df['day_of_week_appointment'] = df['AppointmentDay'].dt.dayofweek

# Create dummy variables
df = pd.get_dummies(df, columns=['month_appointment', 'day_of_week_appointment'], drop_first=True)

# Encode Gender
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

# Encode target variable
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

# Count previous appointments and no-shows
df = df.sort_values(['PatientId', 'ScheduledDay'])
df['number_of_previous_apptms'] = df.groupby('PatientId').cumcount()
df['number_of_previous_noshows'] = df.groupby('PatientId')['No-show'].cumsum() - df['No-show']

# Prepare features
features_to_drop = ['Neighbourhood', 'ScheduledDay', 'AppointmentDay', 'PatientId', 'AppointmentID', 'No-show']
X = df.drop(features_to_drop, axis=1)
y = df['No-show']

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
print("Training Logistic Regression model...")
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Evaluate
y_pred = logreg.predict(X_test)
print("\n=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Train Random Forest model
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf.predict(X_test)
print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Save the better model (Random Forest typically performs better)
print("\nSaving Random Forest model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Save feature names for later use
feature_names = list(X.columns)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\nModel saved successfully!")
print(f"Features used: {feature_names}")
