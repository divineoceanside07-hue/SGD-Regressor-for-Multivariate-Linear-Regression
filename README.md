# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Read the dataset, strip column names, select features, and scale inputs.

2.Initialize models: Create two SGDRegressor instances with chosen learning rate and iterations.

3.Train models: Fit the scaled features to predict Price and Occupants.

4.Predict outputs: Take user input, scale it, and use trained models to estimate price and occupants.


## Program:

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

Developed by: SRI RAM M

RegisterNumber:  212225040423

```
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])


model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')


model.fit(X, y)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

y_pred = model.predict(X)

plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  
plt.show()
```
## Output:
<img width="806" height="621" alt="image" src="https://github.com/user-attachments/assets/68a9b0b0-1146-43ee-a996-5f8ee5074bb8" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
