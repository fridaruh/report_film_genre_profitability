import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('employee_attrition_dataset.csv')

# Prepare the data
y = df['Employee_Attrition']
X = df[['Job_Satisfaction', 'Performance_Rating', 'Overtime_Flag', 'Distance_From_Home']]

print("Dataset Overview:")
print("=" * 70)
print(f"Total samples: {len(df)}")
print(f"Features: {X.columns.tolist()}")
print(f"Target (Employee_Attrition): {y.value_counts().to_dict()}")
print()

# Fit Decision Tree with default parameters (no hyperparameter tuning)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importances (Decision Tree):")
print("=" * 70)
for idx, row in feature_importances.iterrows():
    print(f"{row['Feature']:25s}: {row['Importance']:.6f} ({row['Importance']*100:.2f}%)")

print()
print("Most Important Feature:")
print("=" * 70)
most_important = feature_importances.iloc[0]
print(f">>> {most_important['Feature']} <<<")
print(f"Importance: {most_important['Importance']:.6f} ({most_important['Importance']*100:.2f}%)")

# Additional model information
print()
print("Decision Tree Model Information:")
print("=" * 70)
print(f"Max depth reached: {dt_model.get_depth()}")
print(f"Number of leaves: {dt_model.get_n_leaves()}")
print(f"Number of features: {dt_model.n_features_in_}")

# Calculate accuracy on training data (for reference)
train_accuracy = dt_model.score(X, y)
print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")