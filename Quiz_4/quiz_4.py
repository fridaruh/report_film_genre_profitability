import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Leer dataset
df = pd.read_csv("building_heights_dataset.csv")

# 2. Coeficiente de correlación de Pearson entre floors y height
pearson_corr = df["height"].corr(df["floors"])
print("Pearson correlation coefficient:", pearson_corr)

# 3. Ajustar modelo lineal y obtener beta_1
X = sm.add_constant(df["floors"])
y = df["height"]
model = sm.OLS(y, X).fit()

beta_0 = model.params["const"]
beta_1 = model.params["floors"]
print("Beta_0:", beta_0)
print("Beta_1:", beta_1)

# 4. Calcular RMSE
pred = model.predict(X)
rmse = np.sqrt(np.mean((y - pred)**2))
print("RMSE:", rmse)

# 5. Predicción para un edificio con 5 pisos
prediction_5floors = beta_0 + beta_1 * 5
print("Predicted height for 5 floors:", prediction_5floors)
