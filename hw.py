import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
n_samples = 500

x = np.random.normal(0, 2, n_samples)
noise = np.random.uniform(-0.5, 0.5, n_samples) 
y = 2 * x + noise


x_mean = np.mean(x)
y_mean = np.mean(y)
slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
intercept = y_mean - slope * x_mean
y_pred = slope * x + intercept



plt.figure(figsize=(12, 8))
plt.scatter(x, y, alpha=0.5, label='data with noise', color='blue')
plt.plot(x, y_pred, color='red', linewidth=3, label=f'y = {slope:.4f}x + {intercept:.4f}')
plt.plot(x, 2*x, '--', color='green', linewidth=2, label='real model : y = 2x', alpha=0.8)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear regression with [-0.5,0.5] noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



