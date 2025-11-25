import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(500)
n = 200
x = np.random.uniform(0, 10, n).reshape(-1, 1)
noise = np.random.uniform(-0.5 , 0.5, n)
y = 3 * x.flatten() + 5 + noise

model = LinearRegression()
model.fit(x, y)

print("shib:", model.coef_[0])
print("arz:", model.intercept_)

plt.scatter(x, y, color='green', label="data")
plt.plot(x, model.predict(x), color='red', label="line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()