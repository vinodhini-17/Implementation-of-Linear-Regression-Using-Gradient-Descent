# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To train, initialize theta and iteratively update it using gradient descent.
2.For preprocessing, read and scale data.
3.For modeling, train the linear regression model.
4.To predict data values, scale a new data and predict.
5.Print the prediction.

## Program & output:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: vinodhini k
RegisterNumber:  212223230245
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
````
````
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
````
![image](https://github.com/user-attachments/assets/7bd875db-d66b-4462-b8ac-8ea5320dc008)

````
x.shape

``````````
![image](https://github.com/user-attachments/assets/48974605-22e6-4cbe-9986-ae3ebed6e816)
````
y.shape
`````
![image](https://github.com/user-attachments/assets/908be5e9-401a-41a8-a505-cdd94c1ef425)

````
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
````
![image](https://github.com/user-attachments/assets/1ffdacb7-1362-4da6-abe5-aaeedfa78fd9)

![image](https://github.com/user-attachments/assets/67ccf769-65a7-4a12-88fe-1487f3595314)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
