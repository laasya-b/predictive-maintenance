import pandas as pd
import sys
import os
sys.path.append('../src')
from pipeline import load_data, clean_data, engineer_features
from model import prepare_features, train_model

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6

# Load and prepare data
df = load_data('../data/raw/ai4i2020.csv')
df = clean_data(df)
df = engineer_features(df)
X, y = prepare_features(df)
model, X_test, y_test = train_model(X, y)

# ── Chart 1: Failure Rate by Machine Type ──
failure_by_type = df.groupby('Type')['Machine_failure'].mean().reset_index()
failure_by_type.columns = ['Type', 'Failure_Rate']
failure_by_type['Failure_Rate'] = (failure_by_type['Failure_Rate'] * 100).round(2)
source1 = ColumnDataSource(failure_by_type)

p1 = figure(
    x_range=failure_by_type['Type'].tolist(),
    title="Failure Rate by Machine Type (%)",
    height=300, width=400,
    toolbar_location=None
)
p1.vbar(
    x='Type', top='Failure_Rate', source=source1,
    width=0.5,
    color=factor_cmap('Type',
                      palette=['#1D9E75', '#3A7EC6', '#E05C2A'],
                      factors=failure_by_type['Type'].tolist())
)
p1.add_tools(HoverTool(tooltips=[("Type", "@Type"), ("Failure Rate", "@Failure_Rate%")]))
p1.yaxis.axis_label = "Failure Rate (%)"
p1.xaxis.axis_label = "Machine Type"

# ── Chart 2: Torque vs Tool Wear ──
df['failure_color'] = df['Machine_failure'].map({0: 'steelblue', 1: 'red'})
df['failure_label'] = df['Machine_failure'].map({0: 'No Failure', 1: 'Failure'})
source2 = ColumnDataSource(df)

p2 = figure(
    title="Torque vs Tool Wear (Red = Failure)",
    height=300, width=400,
    toolbar_location=None
)
p2.circle(
    x='Torque_Nm', y='Tool_wear_min',
    color='failure_color', alpha=0.4, size=4,
    source=source2
)
p2.add_tools(HoverTool(tooltips=[
    ("Torque", "@Torque_Nm"),
    ("Tool Wear", "@Tool_wear_min"),
    ("Status", "@failure_label")
]))
p2.xaxis.axis_label = "Torque (Nm)"
p2.yaxis.axis_label = "Tool Wear (min)"

# ── Chart 3: Feature Importance ──
import numpy as np
features = ['Type_encoded', 'Air_temperature_K', 'Process_temperature_K',
            'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min',
            'temp_diff', 'power']
importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False).reset_index()
importance.columns = ['Feature', 'Importance']
source3 = ColumnDataSource(importance)

p3 = figure(
    x_range=importance['Feature'].tolist(),
    title="Feature Importance — Random Forest",
    height=300, width=600,
    toolbar_location=None
)
p3.vbar(
    x='Feature', top='Importance', source=source3,
    width=0.6, color='#1D9E75'
)
p3.add_tools(HoverTool(tooltips=[("Feature", "@Feature"), ("Importance", "@Importance{0.000}")]))
p3.xaxis.major_label_orientation = 0.8
p3.yaxis.axis_label = "Importance Score"

output_file("../dashboard/maintenance_dashboard.html")
save(column(row(p1, p2), p3))
print("Dashboard saved to dashboard/maintenance_dashboard.html")