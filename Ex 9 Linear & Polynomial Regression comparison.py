import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
# Add some noise
y[::5] += 3 * (0.5 - np.random.rand(16))


lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)


poly = PolynomialFeatures(degree=4) # Degree 4 allows for complexity
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)


plt.scatter(X, y, color='gray', s=10, label='Data Points')
plt.plot(X, y_pred_lin, color='red', label='Linear Regression')
plt.plot(X, y_pred_poly, color='blue', label='Polynomial Regression (Deg=4)')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.show()