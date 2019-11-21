import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(train_x,train_y)

ypred=regressor.predict(test_x)

"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_x=sc_x.fit_transform(train_x)
test_x=sc_x.transform(test_x)"""