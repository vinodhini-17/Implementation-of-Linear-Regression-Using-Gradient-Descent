# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
To train, initialize theta and iteratively update it using gradient descent.
For preprocessing, read and scale data.
For modeling, train the linear regression model.
To predict data values, scale a new data and predict.
Print the prediction.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: vinodhini k
RegisterNumber:  212223230245
*/
```
````
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
`````
![image](https://github.com/user-attachments/assets/a04ea2db-1604-4d93-9386-7b1bf3714877)
`````
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
``````
![image](https://github.com/user-attachments/assets/22ca3667-703f-4f43-9e5f-6f79b599b561)

````
x.shape
`````

![image](https://github.com/user-attachments/assets/e87a3106-ee1a-46f9-82e5-6aaad6992542)
````
`y.shape
``````
![image](https://github.com/user-attachments/assets/2a7c8327-f405-462a-9387-a9b2cc59f8ba)

```````
m=0
c=0
L=0.001 # learning rate
epochs=5000 # No.of iterations to be performed
n=float(len(x))
error=[]
# Performing Gradient Descent
for i in range(epochs):
  y_pred = m*x + c
  D_m = (-2/n)*sum(x*(y-y_pred))
  D_c = (-2/n)*sum(y-y_pred)
  m = m-L*D_m
  c = c-L*D_c
  error.append(sum(y-y_pred)**2)
print(m,c)
type(error)
print(len(error))
plt.plot(range(0,epochs),error)
``````````
`![image](https://github.com/user-attachments/assets/02e33472-9eb4-4d37-9fde-db473fbc77de)
![image](https://github.com/user-attachments/assets/1b5352b6-a4f0-4039-843e-bf029bdad278)

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
