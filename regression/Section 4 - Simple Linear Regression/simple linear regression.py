import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=1/3,random_state=0)

"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_x=sc_x.fit_transform(train_x)
test_x=sc_x.transform(test_x)"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(train_x,train_y)

y_pred=regressor.predict(test_x)

plt.scatter(train_x,train_y,color="red")
plt.plot(train_x,regressor.predict(train_x),color="green")
plt.title("sr")


plt.scatter(test_x,test_y,color="red")
plt.plot(train_x,regressor.predict(train_x),color="green")
plt.title("srtest")
plt.show()