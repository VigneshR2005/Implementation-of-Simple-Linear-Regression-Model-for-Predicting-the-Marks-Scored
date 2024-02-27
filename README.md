# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VIGNESH R
RegisterNumber:  212222230172
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
df.head()

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/6841e4bc-ff99-4213-a8ad-8a6441f37d7c)

df.tail()

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/94115a23-443b-4519-8f75-7780ba483bd1)

Array value of X

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/e7bf5922-bf77-4b93-8ed7-c2f976c8d22f)

Array value of Y

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/37a15163-2824-4714-948c-ac808bba0b8a)


Values of Y prediction

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/304b8472-a8f4-4c3e-9295-dd03946e2133)


Array values of Y test

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/a370b651-6d7a-4881-b06e-e436080ef20a)


Training Set Graph

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/146f3d20-c50e-4d95-98dd-c6079237e040)


Test Set Graph

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/73488fda-d231-4a9a-ab53-56521755291e)


Values of MSE, MAE and RMSE

![image](https://github.com/VigneshR2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401246/fce1f6ea-5f71-4953-a66c-8c7ddb0ac765)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
