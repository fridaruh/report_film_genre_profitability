import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/fridaruh/Documents/Sussex/Data_Science_Research_Methods/Quiz_5/employee_attrition_dataset.csv')

y = df['Employee_Attrition']
X = df[['Performance_Rating']]
X = sm.add_constant(X)

model = sm.Logit(y, X).fit(disp=False)
probs = model.predict(X)
preds = (probs >= 0.5).astype(int)

print(accuracy_score(y, preds))
