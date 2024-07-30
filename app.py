import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('/workspaces/deployment/week_approach_maskedID_timeseries.csv')

# Data Preprocess
dfc = df.set_index('Athlete ID').drop_duplicates()

# Create lagged features and rolling averages
for lag in range(1, 4):
    for col in df.columns:
        if col.startswith('total kms'):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

df['avg_exertion_rolling'] = df['avg exertion'].rolling(window=3).mean()
df['avg_training_success_rolling'] = df['avg training success'].rolling(window=3).mean()
df.dropna(inplace=True)

# Define target and features for classification
y_class = df['injury']
X_class = df.drop(['injury', 'Date'], axis=1)

# Handle class imbalance
X_class, y_class = SMOTE().fit_resample(X_class, y_class)
sc = StandardScaler()
X_class = sc.fit_transform(X_class)

# Split the data for classification models
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.3, random_state=0)

xgb_params = {
    # 'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [2],
    'n_estimators': [30]
}

xgb_grid = GridSearchCV(XGBClassifier(), xgb_params, cv=5)
xgb_grid.fit(X_train_class, y_train_class)
y_pred_xgb = xgb_grid.predict(X_test_class)

rf_regressor_params = {
    'n_estimators': [10],
    'criterion': ['squared_error']
}

import pickle

# Save the best classification model
best_classification_model = xgb_grid  # Assuming xgb_grid is the best classification model
with open('best_classification_model.pkl', 'wb') as file:
    pickle.dump(best_classification_model, file)

# Define target and features for regression
y_performance = df['total kms']
X_performance = df.drop(['total kms', 'Date'], axis=1)

# Scale the features
sc = StandardScaler()
X_performance = sc.fit_transform(X_performance)

# Split the data for regression models
X_train_performance, X_test_performance, y_train_performance, y_test_performance = train_test_split(X_performance, y_performance, test_size=0.3, random_state=0)

rf_regressor_grid = GridSearchCV(RandomForestRegressor(), rf_regressor_params, cv=5)
rf_regressor_grid.fit(X_train_performance, y_train_performance)

# Save the best regression model
best_regression_model = rf_regressor_grid  # Assuming rf_regressor_grid is the best regression model
with open('best_regression_model.pkl', 'wb') as file:
    pickle.dump(best_regression_model, file)

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved models
with open('best_classification_model.pkl', 'rb') as file:
    xgb_classifier_model = pickle.load(file)

with open('best_regression_model.pkl', 'rb') as file:
    rf_regressor_model = pickle.load(file)

# Feature names
feature_names = ['nr. sessions', 'nr. rest days', 'total kms', 'max km one day', 'total km Z3-Z4-Z5-T1-T2', 'nr. tough sessions (effort in Z5, T1 or T2)', 'nr. days with interval session', 'total km Z3-4', 'max km Z3-4 one day', 'total km Z5-T1-T2', 'max km Z5-T1-T2 one day', 'total hours alternative training', 'nr. strength trainings', 'avg exertion', 'min exertion', 'max exertion', 'avg training success', 'min training success', 'max training success', 'avg recovery', 'min recovery', 'max recovery', 'nr. sessions.1', 'nr. rest days.1', 'total kms.1', 'max km one day.1', 'total km Z3-Z4-Z5-T1-T2.1', 'nr. tough sessions (effort in Z5, T1 or T2).1', 'nr. days with interval session.1', 'total km Z3-4.1', 'max km Z3-4 one day.1', 'total km Z5-T1-T2.1', 'max km Z5-T1-T2 one day.1', 'total hours alternative training.1', 'nr. strength trainings.1', 'avg exertion.1', 'min exertion.1', 'max exertion.1', 'avg training success.1', 'min training success.1', 'max training success.1', 'avg recovery.1', 'min recovery.1', 'max recovery.1', 'nr. sessions.2', 'nr. rest days.2', 'total kms.2', 'max km one day.2', 'total km Z3-Z4-Z5-T1-T2.2', 'nr. tough sessions (effort in Z5, T1 or T2).2', 'nr. days with interval session.2', 'total km Z3-4.2', 'max km Z3-4 one day.2', 'total km Z5-T1-T2.2', 'max km Z5-T1-T2 one day.2', 'total hours alternative training.2', 'nr. strength trainings.2', 'avg exertion.2', 'min exertion.2', 'max exertion.2', 'avg training success.2', 'min training success.2', 'max training success.2', 'avg recovery.2', 'min recovery.2', 'max recovery.2', 'Athlete ID', 'rel total kms week 0_1', 'rel total kms week 0_2', 'rel total kms week 1_2', 'total kms_lag_1', 'total kms.1_lag_1', 'total kms.2_lag_1', 'total kms_lag_2', 'total kms.1_lag_2', 'total kms.2_lag_2', 'total kms_lag_1_lag_2', 'total kms.1_lag_1_lag_2', 'total kms.2_lag_1_lag_2', 'total kms_lag_3', 'total kms.1_lag_3', 'total kms.2_lag_3', 'total kms_lag_1_lag_3', 'total kms.1_lag_1_lag_3', 'total kms.2_lag_1_lag_3', 'total kms_lag_2_lag_3', 'total kms.1_lag_2_lag_3', 'total kms.2_lag_2_lag_3', 'total kms_lag_1_lag_2_lag_3', 'total kms.1_lag_1_lag_2_lag_3', 'total kms.2_lag_1_lag_2_lag_3', 'avg_exertion_rolling', 'avg_training_success_rolling']


# Streamlit app
st.title("Injury Risk Evaluation and Performance Forecast for Runners")

# Collect user input for all features
st.header("Input Features")
features = {}
for feature in feature_names:
    features[feature] = st.number_input(feature, value=0.0)

# Create the input feature vector
input_vector = np.array([features[feature] for feature in feature_names]).reshape(1, -1)

# Standardize the input features
sc = StandardScaler()
input_vector = sc.fit_transform(input_vector)

# Make predictions
if st.button("Predict"):
    # Classification prediction
    injury_risk = xgb_classifier_model.predict(input_vector)[0]
    injury_probability = xgb_classifier_model.predict_proba(input_vector)[0][1]
    
    # Regression prediction
    performance_forecast = rf_regressor_model.predict(input_vector)[0]
    
    # Display the predictions
    st.subheader("Prediction Results")
    st.write(f"Injury Risk: {'Yes' if injury_risk == 1 else 'No'}")
    st.write(f"Injury Probability: {injury_probability:.2f}")
    st.write(f"Performance Forecast (Total Kms): {performance_forecast:.2f}")


