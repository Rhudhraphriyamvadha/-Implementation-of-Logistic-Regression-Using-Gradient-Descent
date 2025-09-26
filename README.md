# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries: Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.

2. Define the Linear Regression Function: Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.

3. Load and Preprocess the Data: Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.

4. Perform Linear Regression: Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.

5. Make Predictions on New Data: Prepare new data, scale it, and use the trained model to predict the target variable, transforming predictions back to the original scale.

6. Print the Predicted Value

## Program:
```python

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Rhudhra phriyamvadha K S
RegisterNumber: 212224040275

```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the columns
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

#selecting the features and labels
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
#display independent variable
Y

#initialize the model parameters
theta=np.random.randn(X.shape[1])
y=Y
#define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))
#define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

#defining the gradient descent algorithm.
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
#train the model
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
#makeprev \dictions
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)


accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

#### Read the file and display
<img width="1227" height="466" alt="image" src="https://github.com/user-attachments/assets/a793faf2-1d21-41a3-b62e-974d2d2ed6d8" />

#### Categorizing columns
<img width="313" height="314" alt="image" src="https://github.com/user-attachments/assets/71c77479-f660-46b5-902e-71cda7b4b928" />

#### Labelling columns and displaying dataset
<img width="966" height="466" alt="image" src="https://github.com/user-attachments/assets/db20d07d-2303-4750-a74a-d4a867973423" />

#### Display dependent variable
<img width="689" height="233" alt="image" src="https://github.com/user-attachments/assets/0284d113-7d4a-4c3d-9491-bdf649e5cfb5" />

#### Printing accuracy & Y
<img width="771" height="163" alt="image" src="https://github.com/user-attachments/assets/4b72171e-9a33-4cae-a21b-edbb65a36441" />

#### Printing y_prednew
<img width="437" height="38" alt="image" src="https://github.com/user-attachments/assets/d3527b02-0506-41e9-aac9-26c6540bf9ae" />
<img width="287" height="42" alt="image" src="https://github.com/user-attachments/assets/7cadb35b-2043-48d2-8ae7-d9f8232d3621" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

