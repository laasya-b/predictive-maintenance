import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import load_data, clean_data, engineer_features
from model import prepare_features, train_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def plot_failure_rate_by_type(df):
    failure_by_type = df.groupby('Type')['Machine_failure'].mean() * 100
    plt.figure(figsize=(8, 5))
    failure_by_type.plot(kind='bar', color=['#1D9E75', '#3A7EC6', '#E05C2A'])
    plt.title('Failure Rate by Machine Type (%)')
    plt.xlabel('Machine Type')
    plt.ylabel('Failure Rate (%)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../data/failure_by_type.png')
    plt.show()
    print("Saved: failure_by_type.png")

def plot_sensor_distributions(df):
    sensors = ['Air_temperature_K', 'Torque_Nm', 'Tool_wear_min', 'power']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(sensors):
        df[df['Machine_failure'] == 0][col].hist(
            ax=axes[i], alpha=0.6, label='No Failure', color='steelblue', bins=30)
        df[df['Machine_failure'] == 1][col].hist(
            ax=axes[i], alpha=0.6, label='Failure', color='red', bins=30)
        axes[i].set_title(col)
        axes[i].legend()
    
    plt.suptitle('Sensor Distributions: Failure vs No Failure')
    plt.tight_layout()
    plt.savefig('../data/sensor_distributions.png')
    plt.show()
    print("Saved: sensor_distributions.png")

def plot_feature_importance(model, feature_names):
    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)
    
    plt.figure(figsize=(8, 5))
    importance.plot(kind='barh', color='#1D9E75')
    plt.title('Feature Importance — Random Forest')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('../data/feature_importance.png')
    plt.show()
    print("Saved: feature_importance.png")

if __name__ == "__main__":
    df = load_data('../data/raw/ai4i2020.csv')
    df = clean_data(df)
    df = engineer_features(df)
    
    print("Generating visualizations...")
    plot_failure_rate_by_type(df)
    plot_sensor_distributions(df)
    
    X, y = prepare_features(df)
    model, X_test, y_test = train_model(X, y)
    
    features = ['Type_encoded', 'Air_temperature_K', 'Process_temperature_K',
                'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min',
                'temp_diff', 'power']
    plot_feature_importance(model, features)
    print("\nAll visualizations saved to data/ folder.")