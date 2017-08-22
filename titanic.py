import pandas as pd
from sklearn import linear_model as lm
import numpy as np

# ************************************************
# * File Parsing Functions
# ************************************************
def read_files():
    return pd.read_csv('train.csv',sep=','), pd.read_csv('test.csv',sep=',')

train_data, test_data = read_files();

# basic

train_data = train_data.drop(["Name","Cabin","Age","Embarked","Ticket"],axis=1)
test_data = test_data.drop(["Name","Cabin","Age","Embarked","Ticket"],axis=1)

# train_data.info()
test_data.info()
sex_mat = train_data["Sex"]
train_data = train_data.drop("Sex",axis=1)
train_data = pd.concat([train_data, pd.get_dummies(sex_mat)],axis=1)

train_y = train_data["Survived"].values
train_X = train_data.drop("Survived",axis=1).values

clf = lm.LogisticRegressionCV(Cs=1,cv=5,penalty="l2",n_jobs=-1)
clf.fit(train_X,train_y)

sex_mat = test_data["Sex"]
test_data = test_data.drop("Sex",axis=1)
test_data = pd.concat([test_data, pd.get_dummies(sex_mat)],axis=1).fillna(0)

result = pd.concat([test_data["PassengerId"],pd.DataFrame(clf.predict(test_data))],axis=1).values

np.savetxt("simple_output.csv", result, delimiter=",",fmt='%i',header="PassengerId,Survived")
# output = pd.DataFrame(clf.predict(test_data))
# output.to_csv('simple_output.csv')
