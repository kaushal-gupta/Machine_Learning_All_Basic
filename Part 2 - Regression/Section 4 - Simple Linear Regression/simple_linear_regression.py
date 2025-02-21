import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset=pd.read_csv("Salary_Data.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

#viusalizing the training set results
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="b")
plt.title("Salary vs Experience (training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#visualing the test set 
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,y_pred,color="b")
plt.title("Salary vs Experience (Testing set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

regressor.coef_

regressor.intercept_

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)