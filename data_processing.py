# -*- coding: utf-8 -*-

def data_preprocessing():

    #IMPORT DATASET
    import pandas as pd
    
    train_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
    training_set = train_dataset.iloc[:, 1].values
    
    test_dataset = pd.read_csv('Google_Stock_Price_Test.csv')
    test_set = test_dataset.iloc[:,1].values
    
    #FEATURE SCALING
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    test_set = sc.transform(test_set)
    
    #INPUT AND OUTPUT FOR THE RNN
    X_train = training_set[0:-1]
    y_train = training_set[1:]
    
    #RESHAPE TO 3D ARRAY TO ACCOUNT FOR TIME
    X_train = X_train.reshape(len(X_train), 1, 1)
    X_test = test_set.reshape(len(test_set), 1, 1)
    
    return X_train, y_train, X_test, sc, test_set