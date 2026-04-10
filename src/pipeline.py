import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Drop columns we don't need
    df = df.drop(columns=['UDI', 'Product ID'])
    
    # Rename columns for easier use
    df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
    
    return df

def engineer_features(df):
    # Temperature difference between process and air
    df['temp_diff'] = df['Process_temperature_K'] - df['Air_temperature_K']
    
    # Power load
    df['power'] = df['Rotational_speed_rpm'] * df['Torque_Nm']
    
    return df

if __name__ == "__main__":
    df = load_data('../data/raw/ai4i2020.csv')
    print("Raw data shape:", df.shape)
    
    df = clean_data(df)
    print("After cleaning:", df.shape)
    print("Columns:", df.columns.tolist())
    
    df = engineer_features(df)
    print("After feature engineering:", df.shape)
    print(df.head())