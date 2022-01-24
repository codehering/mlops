import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
print('Parameters: ', results.params)
print('R2: ', results.rsquared)
with open ("metrics.txt", "w") as f:
  f.write(f"{results.summary().as_text()}")
