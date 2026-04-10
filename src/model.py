import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from pipeline import load_data, clean_data, engineer_features

def prepare_features(df):
    le = LabelEncoder()
    df['Type_encoded'] = le.fit_transform(df['Type'])
    
    features = ['Type_encoded', 'Air_temperature_K', 'Process_temperature_K',
                'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min',
                'temp_diff', 'power']
    
    X = df[features]
    y = df['Machine_failure']
    return X, y

def train_model(X, y):
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")
    
    return model, X_test, y_test

if __name__ == "__main__":
    df = load_data('../data/raw/ai4i2020.csv')
    df = clean_data(df)
    df = engineer_features(df)
    
    X, y = prepare_features(df)
    print(f"Failure rate: {y.mean()*100:.2f}%")
    
    model, X_test, y_test = train_model(X, y)

    import os
model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
model = joblib.load(model_path)