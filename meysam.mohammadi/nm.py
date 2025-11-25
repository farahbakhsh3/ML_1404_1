import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(42)  # For reproducibility
x = np.arange(0, 501).reshape(-1, 1)  # x from 0 to 500

# Larger noise for visible randomness
noise = np.random.uniform(-50, 50, size=x.shape)
y = 2 * x + noise  # y = 2x ± noise

# Linear regression model
model = LinearRegression()
model.fit(x, y)

# Line coefficients
c = model.intercept_[0]  # intercept
b = model.coef_[0][0]    # slope

# Prediction for plotting
y_pred = model.predict(x)

# Print coefficients
print(f"Slope (b): {b}")
print(f"Intercept (c): {c}")

# Plotting
plt.figure(figsize=(10,6))
plt.scatter(x, y, color='blue', s=10, label='Random data')
plt.plot(x, y_pred, color='red', label='Linear regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Random Data')
plt.legend()
plt.show()
