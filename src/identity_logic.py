import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# Path to save artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'identity_xgb_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'identity_encoders.pkl')
COLS_PATH = os.path.join(MODEL_DIR, 'identity_features.pkl')

def preprocess_data(df, is_training=False, encoders=None):
    # Drop columns with > 75% missing
    threshold = 0.75
    if is_training:
        missing_fraction = df.isnull().mean()
        cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
        df = df.drop(columns=cols_to_drop)
        # Store features for prediction consistency
        joblib.dump(df.columns.tolist(), COLS_PATH)
    else:
        keep_cols = joblib.load(COLS_PATH)
        # Add isFraud if it exists (for training but not prediction)
        if 'isFraud' in df.columns and 'isFraud' not in keep_cols:
            keep_cols.append('isFraud')
        df = df[[c for c in keep_cols if c in df.columns]]

    # Separate column types
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include='object').columns

    # Impute
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean() if is_training else 0)
    
    new_encoders = {}
    for col in categorical_cols:
        if is_training:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            new_encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels by mapping them to 'nan' or a default
                df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                df[col] = 0
                
    return df, new_encoders

def train_identity_model(transaction_csv, identity_csv):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    t_df = pd.read_csv(transaction_csv)
    i_df = pd.read_csv(identity_csv)
    df = pd.merge(t_df, i_df, how='left', on='TransactionID')
    
    df_proc, encoders = preprocess_data(df, is_training=True)
    
    X = df_proc.drop('isFraud', axis=1)
    y = df_proc['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42)
    model.fit(X_res, y_res)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    return "Identity Fraud model trained successfully."

def predict_identity_fraud(data_dict):
    if not os.path.exists(MODEL_PATH):
        return None, "Model not trained yet."
    
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    
    df_input = pd.DataFrame([data_dict])
    df_input, _ = preprocess_data(df_input, is_training=False, encoders=encoders)
    
    # Ensure columns match training (excluding target)
    train_cols = joblib.load(COLS_PATH)
    if 'isFraud' in train_cols: train_cols.remove('isFraud')
    
    # Fill missing columns with 0
    for col in train_cols:
        if col not in df_input.columns:
            df_input[col] = 0
            
    df_input = df_input[train_cols]
    
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]
    
    return prediction, probability
