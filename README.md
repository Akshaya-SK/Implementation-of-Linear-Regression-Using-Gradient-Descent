# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as pandas, numpy, and matplotlib. Load the dataset (50_startups.csv) using pandas.
2. Extract independent variables (e.g., R&D Spend) and the dependent variable (Profit). Normalize the data for faster convergence.
3. Set initial values for the slope (m) and intercept (b). Choose a learning rate and number of iterations.
4. For each iteration, Predict output using the current values of m and b, Calculate gradients of the loss function (MSE), Update m and b using gradients and learning rate.
5. Display the final regression equation. Plot the dataset and regression line.

## Program:
```
/*
# Program to implement the linear regression using gradient descent.
# Developed by: Akshaya S K
# Register Number: 212223040011

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv('50_Startups.csv')

# 2. Use only R&D Spend to predict Profit
X = data['R&D Spend'].values
y = data['Profit'].values

# 3. Normalize data
X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)

# 4. Initialize parameters
m = 0  # slope
b = 0  # intercept
learning_rate = 0.01
iterations = 1000
n = len(X)

# 5. Gradient Descent Loop
for i in range(iterations):
    y_pred = m * X + b
    error = y_pred - y
    cost = (1/n) * sum(error ** 2)
    dm = (2/n) * sum(error * X)
    db = (2/n) * sum(error)
    m = m - learning_rate * dm
    b = b - learning_rate * db
    if i % 100 == 0:
        print(f"Iteration {i}, Cost: {cost:.4f}, m: {m:.4f}, b: {b:.4f}")

# 6. Final equation
print(f"\nFinal equation: y = {m:.4f}x + {b:.4f}")

# 7. Plotting
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, m*X + b, color='red', label='Best Fit Line')
plt.xlabel("R&D Spend (normalized)")
plt.ylabel("Profit (normalized)")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

*/
```

## Output:
![image](https://github.com/user-attachments/assets/c4cfe520-939a-4201-b5b1-f835f62e0131)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
