import numpy as np
import pandas as pd

# ************************************************
# * File Parsing Functions
# ************************************************
def read_files():
    return pd.read_csv('train.csv',sep=','), pd.read_csv('test.csv',sep=',')

def split_solution(train_file):
    train_y = train_file[train_file.columns[1]]
    train_X = train_file.drop('Survived', axis=1)
    return train_X, train_y

def convertCategoricalColumns(train_X):
    # parses the name of the person and replaces according to these codes
    #   * 0 - catch-all 
    #   * 1 - Mr.
    #   * 2 - Mrs.
    #   * 3 - Miss or Ms.

def impute_data(train_X):
    return train_X

def transform_data(train_X):
    return impute_data(convertCategoricalColumns(train_X))

train_file, test_file = read_files();

train_X, train_y = split_solution(train_file)

train_X = train_X.as_matrix()
train_y = train_y.as_matrix()

train_X = transform_data(train_X)
