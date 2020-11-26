import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from python import titanic_preprocessing
from keras import Sequential
import numpy as np
import os

def build_model(n_inputs):
    model = Sequential()
    model.add(Dense(n_inputs*1, input_dim=n_inputs, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_inputs*2, input_dim=n_inputs, activation='relu'))
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(num_of_inputs,input_values, output_values):
    model = build_model(num_of_inputs)
    model.fit(x=input_values, y=output_values, epochs=200, batch_size=100,verbose=True)
    model.summary()
    return model

# Prints out the given dataframe to a csv file.
def write_to_csv(cols_list, csv_lists,directory, output_filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + directory + os.path.sep + output_filename
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(cols_list)
        for c in csv_lists:
            try:
                writer.writerow(c)
            except:
                print(c)


if __name__ == '__main__':
    #Loading training and testing dataframes through preprocessing method
    training_data, testing_data = titanic_preprocessing.preprocess('datasets', 'train.csv', 'test.csv')
    print(testing_data)
    #Storing number of features expected
    n_inputs = len(testing_data.columns)

    #expected_classes => class labels
    expected_classes = training_data['Survived'].copy().values
    training_data.drop(columns=['Survived'], inplace=True)

    x_scalar = StandardScaler()
    x_scalar.fit(training_data.values)
    values = x_scalar.transform(training_data.values)

    #Training the model
    model = train_model(num_of_inputs=n_inputs, input_values=values, output_values=expected_classes)

    #Predicting class labels based on test data
    test_values = testing_data.values
    result = model.predict(test_values)
    result = np.round(result)

    new_df = pd.DataFrame(np.array(testing_data.index), columns=['PassengerId'])
    new_df['Survived'] = result

    new_df['PassengerId'] = new_df['PassengerId'].astype(int)
    new_df['Survived'] = new_df['Survived'].astype(int)
    print(new_df.dtypes)

    cols = new_df.columns.values
    csv_values = new_df.values
    write_to_csv(cols_list=cols, csv_lists=csv_values, directory='datasets', output_filename='she_let_go.csv')