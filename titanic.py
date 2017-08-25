import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
import numpy as np

# ************************************************
# * File Parsing
# ************************************************
def read_files():
    return pd.read_csv('train.csv',sep=','), pd.read_csv('test.csv',sep=',')

train, test = read_files();


# ************************************************
# * Analyze Data
# ************************************************
# ~ 61.6 % did not survive
# print("training examples survived distribution: " + str(train["Survived"].value_counts()[0] / (train["Survived"].value_counts()[0] + train["Survived"].value_counts()[1])))

# the lowest class (3) seems to be the least likely to survive
# plt_economic_class = train.pivot_table(values="PassengerId",index="Pclass",columns="Survived",aggfunc="count").plot(kind="bar")

# females were more likely to survive than males
# plt_sex = train.pivot_table(values="PassengerId",index="Sex",columns="Survived",aggfunc="count").plot(kind="bar")

# normally distributed with a mean of ~30
# plt_age = train.pivot_table(values="PassengerId",index="Age",columns="Survived",aggfunc="count").plot(kind="bar")
avg_age = train["Age"].mean()

# majority of people did not have siblings, but the highest percentage of survivors had 1 sibling
# plt_sib = train.pivot_table(values="PassengerId",index="SibSp",columns="Survived",aggfunc="count").plot(kind="bar")

# majority of people embarked from Southhampton (S)
# plt_embarked = train.pivot_table(values="PassengerId",index="Embarked",columns=None,aggfunc="count").plot(kind="bar")

# for now, drop the cabin column for both
# plt_embarked = train.pivot_table(values="PassengerId",index="Cabin",columns="Survived",aggfunc="count").plot(kind="bar")

# parch and sibsip seem to be very heavily correlated
# plt_parch = train.pivot_table(values="PassengerId",index="Parch",columns="Survived",aggfunc="count").plot(kind="bar")

# plt_parch = train.pivot_table(values="PassengerId",index="Ticket",columns="Survived",aggfunc="count").plot(kind="bar")

# plt.show()

# ************************************************
# * Missing Values
# ************************************************
# Age: 177
# Cabin: 687
# Embarked: 2
# print(train.isnull().sum(axis=0))

# Age: 86
# Fare: 1
# Cabin: 327
# print(test.isnull().sum(axis=0))

# ************************************************
# * Feature Engineering
# ************************************************
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)
train.drop("Ticket",axis=1,inplace=True)
test.drop("Ticket",axis=1,inplace=True)
# TODO - See what can be done with the title?
train.drop("Name",axis=1,inplace=True)
test.drop("Name",axis=1,inplace=True)

# fill the age with the mean age
# TODO - I can make this better by running a regression or MICE
train["Age"].fillna(value=avg_age,inplace=True)
test["Age"].fillna(value=test["Age"].mean(),inplace=True)

# replace with k-1 categorical variables
train_sex = pd.get_dummies(data=train["Sex"],drop_first=True)
test_sex = pd.get_dummies(data=test["Sex"],drop_first=True)
train.drop("Sex",axis=1,inplace=True)
test.drop("Sex",axis=1,inplace=True)
train = pd.concat([train, train_sex],axis=1)
test = pd.concat([test, test_sex],axis=1)

train_embarked = pd.get_dummies(data=train["Embarked"],drop_first=True)
test_embarked = pd.get_dummies(data=test["Embarked"],drop_first=True)
train.drop("Embarked",axis=1,inplace=True)
test.drop("Embarked",axis=1,inplace=True)
train = pd.concat([train, train_embarked],axis=1)
test = pd.concat([test, test_embarked],axis=1)


test["Fare"].fillna(value=test["Fare"].mean(),inplace=True)

train_y = train["Survived"]
train.drop("Survived",axis=1,inplace=True)

log_clf = lm.LogisticRegressionCV(Cs=10,cv=5,dual=False,penalty="l2",scoring="roc_auc",n_jobs=-1)
log_clf.fit(train,train_y)

print(log_clf.scores_[1].mean(axis=0).max())

prediction = log_clf.predict(test)
output = pd.concat([test["PassengerId"], pd.DataFrame(prediction)] ,axis=1)

np.savetxt("simple_output.csv", output, delimiter=",",fmt='%i',header="PassengerId,Survived")

