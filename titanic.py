import pandas as pd
from sklearn import linear_model as lm
import numpy as np

# ************************************************
# * File Parsing Functions
# ************************************************
def read_files():
    return pd.read_csv('train.csv',sep=','), pd.read_csv('test.csv',sep=',')

def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip().lower()
    if title == "miss" or title == "mme" or title == "mlle":
        title = "Ms"
    return title
train_data, test_data = read_files();

train_data = train_data.drop(["Cabin","Embarked","Ticket"],axis=1)
test_data = test_data.drop(["Cabin","Embarked","Ticket"],axis=1)

# one hot encode sex variable
train_data = pd.concat([train_data, pd.get_dummies(train_data["Sex"])],axis=1)
train_data = train_data.drop("Sex",axis=1)

# TODO impute age
# TODO extract title from name
# title_df = pd.DataFrame({"Title":train_data["Name"].apply(extract_title)})
# # print(title_df["Title"].value_counts())
# title_df = pd.get_dummies(title_df["Title"])
train_data = train_data.drop("Name",axis=1)
# train_data = pd.concat([train_data, title_df],axis=1)

# TODO simple mean imputation
train_data["Age"].fillna(train_data["Age"].mean(),inplace=True)


# train_data = pd.concat([train_data, pd.get_dummies(title_df["Title"])],axis=1)
# train_data.drop("Name",axis=1)

test_data["Age"].fillna(test_data["Age"].mean(),inplace=True)
test_data["Fare"].fillna(test_data["Fare"].mean(),inplace=True)


train_y = train_data["Survived"]
train_data = train_data.drop("Survived",axis=1)
train_X = train_data.as_matrix()

clf = lm.LogisticRegressionCV(Cs=1,cv=5,penalty="l2",n_jobs=-1)
clf.fit(train_data,train_y)

sex_mat = test_data["Sex"]
test_data = test_data.drop("Sex",axis=1)
test_data = pd.concat([test_data, pd.get_dummies(sex_mat)],axis=1).fillna(0)

# extract title from name
# title_df = pd.DataFrame({"Title":test_data["Name"].apply(extract_title)})
test_data = test_data.drop("Name",axis=1)
# print(title_df["Title"].value_counts())
# title_df = pd.get_dummies(title_df["Title"])
# test_data = pd.concat([test_data, title_df],axis=1)
# padding = pd.DataFrame(np.zeros((418,6)))
# test_data = pd.concat([test_data, padding],axis=1)
# padding.info()
# train_data.info()
# test_data = pd.concat([ test_data, padding],axis=1)


result = pd.concat([test_data["PassengerId"],pd.DataFrame(clf.predict(test_data))],axis=1).values

np.savetxt("simple_output.csv", result, delimiter=",",fmt='%i',header="PassengerId,Survived")
