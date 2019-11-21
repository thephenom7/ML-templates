# Classification template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
testdataset=pd.read_csv('test.csv')
dataset=dataset.set_index('PassengerId')
testdataset=testdataset.set_index('PassengerId')
"""dataset['Age'].fillna(25,inplace=True)
testdataset['Age'].fillna(25,inplace=True)"""
testdataset.isna().any()
"""na=pd.isnull(dataset['Age'])
miss=dataset[na]"""
X = dataset.iloc[:, [1,3,5,6]].values
y = dataset.iloc[:, 0].values

tX = testdataset.iloc[:, [0,2,4,5]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
labelencoder_X1 = LabelEncoder()
tX[:, 1] = labelencoder_X1.fit_transform(tX[:, 1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

sc1 = StandardScaler()
tX = sc1.fit_transform(tX)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(tX)
classifier.score(X_test,y_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = pd.DataFrame(y_pred)
passangerid = pd.read_csv('test.csv').iloc[:,0].to_frame()
res = pd.concat([passangerid,y_pred],axis=1)
res.columns=['PassangerID','Survive']
res.to_csv('Result.csv',index=False)