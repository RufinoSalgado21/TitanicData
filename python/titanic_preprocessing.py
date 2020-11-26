import os
import pandas as pd
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dropout, Dense

def read_file(directory, filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + directory + os.path.sep + filename
    df = pd.read_csv(path)
    return df

def select_features(df,index,features):
    x = df[features]
    x = x.set_index(index)
    return x

def encode_column(df,column):
    temp = pd.get_dummies(df[column])
    df.drop(columns=[column], inplace=True)
    return pd.merge(df,temp, on='PassengerId')

'''
    Returns a dataframe of preprocessed features from the given Titanic dataset.
    Any labels with categorical values have been replaced with encoded columns,
    one per category being represented.
'''
def preprocess(directory,train_data, test_data):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df_train = read_file(directory,train_data)
    df_test = read_file(directory, test_data)

    # Trying to lump cabin values by letters, presumably areas of ship
    '''
    simplify_cabin(df_train)
    simplify_cabin(df_test)
    '''

    ###Prepping Training set **Includes column of class labels for use in model tgraining
    features = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Survived']
    index = 'PassengerId'

    train = select_features(df_train, index, features)

    #Replace NaN values
    train['Cabin'] = train['Cabin'].replace(np.NaN,'Unknown')
    train = train.replace(np.NaN, 0)

    #Encoding
    need_to_encode = ['Pclass', 'Sex','Cabin','Embarked']
    for label in need_to_encode:
        train = encode_column(train, label)

    ###Prepping Testing set
    test_features = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
    test = select_features(df_test, index, test_features)

    #Replacing NaN values
    test['Cabin'] = test['Cabin'].replace(np.NaN, 'Unknown')
    test = test.replace(np.NaN, 0)

    #Adding missing cabin values from testing data as new columns in training data
    additional_cabin_labels = np.unique(test['Cabin'].values)

    for label in additional_cabin_labels:
        if label not in train.columns.values:
            train[label] = [0]*len(df_train['Cabin'])

    #Returning all unique cabin values from training data
    all_cabin_values = np.delete(train.columns.values,range(10))

    #Encoding testing data
    need_to_encode = ['Pclass', 'Sex', 'Cabin', 'Embarked']
    for label in need_to_encode:
        test = encode_column(test, label)

    # Adding missing cabin values from training data as new columns in testing data
    for label in all_cabin_values:
        if label not in test.columns.values:
            test[label] = [0]*len(df_test['Cabin'])

    return train, test


def simplify_cabin(df):
    df['Cabin'] = df['Cabin'].replace(np.NaN, 'Unknown')
    for index, i in enumerate(df['Cabin']):
        if 'U' in i:
            df['Cabin'][index] = 'U'
        elif 'A' in i:
            df['Cabin'][index] = 'A'
        elif 'C' in i:
            df['Cabin'][index] = 'C'
        elif 'B' in i:
            df['Cabin'][index] = 'B'
        elif 'D' in i:
            df['Cabin'][index] = 'D'
        elif 'E' in i:
            df['Cabin'][index] = 'E'
        elif 'F' in i:
            df['Cabin'][index] = 'F'
        elif 'G' in i:
            df['Cabin'][index] = 'G'


if __name__ == '__main__':
    preprocess(directory='datasets',train_data='train.csv', test_data='test.csv')

