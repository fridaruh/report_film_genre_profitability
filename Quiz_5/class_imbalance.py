import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/fridaruh/Documents/Sussex/Data_Science_Research_Methods/Quiz_5/employee_attrition_dataset.csv')

counts = df['Overtime_Flag'].value_counts()

plt.figure(figsize=(5,4))
counts.plot(kind='bar')
plt.title("Overtime_Flag Distribution")
plt.xlabel("Flag")
plt.ylabel("Count")
plt.tight_layout()

#Employee_Attrition
counts = df['Employee_Attrition'].value_counts()

plt.figure(figsize=(5,4))
counts.plot(kind='bar')
plt.title("Attrition Class Distribution")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.tight_layout()