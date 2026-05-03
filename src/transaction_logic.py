import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Path to save the model
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'transaction_rf_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'transaction_scaler.pkl')

def train_model(csv_path):
    """
    Trains the Random Forest model on the provided dataset.
    Saves the model and scaler for later use.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    df = pd.read_csv(csv_path)
    
    # Feature Engineering (consistent with notebook)
    X = df.drop(['Time', 'Class'], axis=1)
    y = df['Class']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling 'Amount' (recommended for prediction consistency)
    scaler = StandardScaler()
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount'] = scaler.transform(X_test[['Amount']])
    
    # Handling class imbalance
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    
    # Model: n_estimators=100, max_depth=3 (identified as best performer)
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_res, y_res)
    
    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return "Model trained and saved successfully."

def predict_fraud(data_dict):
    """
    Predicts fraud for a single transaction.
    data_dict: Dictionary with keys V1-V28 and Amount.
    """
    if not os.path.exists(MODEL_PATH):
        return None, "Model not trained yet."
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Convert dict to DataFrame
    df_input = pd.DataFrame([data_dict])
    
    # Ensure columns are in correct order (V1...V28, Amount)
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    df_input = df_input[feature_cols]
    
    # Scale Amount
    df_input['Amount'] = scaler.transform(df_input[['Amount']])
    
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]
    
    return prediction, probability
