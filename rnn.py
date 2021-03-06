# -*- coding: utf-8 -*-

#IMPORT PREPROCESSED DATA
from data_processing import data_preprocessing

X_train, y_train, X_test, sc, test_set, real_data, X_real_data = data_preprocessing()

#BUILDING RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM

def build_model():
    model = Sequential()
    
    model.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
    model.add(Dense(units = 1))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    return model

#FITTING MODEL TO DATA
model = build_model()
model.fit(X_train, y_train, batch_size = 32, epochs = 200)

#TESTING PREDICTIONs
prediction = model.predict(X_test)

#RESCALING PREDICTION AND TEST SET
prediction = sc.inverse_transform(prediction)
test_set = sc.inverse_transform(test_set)

#PLOTTING STOCK PRICE
from matplotlib import pyplot as plt
plt.plot(prediction, color = 'red', label = 'Test_set Google Stock Price')
plt.plot(test_set, color = 'blue', label = 'Real Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.show

#PREDICT ON GOOGLE STOCK PRICE BETWEEN 2012 AND 2016
real_data_prediction = model.predict(X_real_data)
real_data_prediction = sc.inverse_transform(real_data_prediction)
real_data = sc.inverse_transform(real_data)

from matplotlib import pyplot as plt
plt.plot(real_data_prediction, color = 'red', label = 'Predicted Google Stock Price 2012-2016')
plt.plot(real_data, color = 'blue', label = 'Real Google Stock Price 2012-2016')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.show

#EVALUATING RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_data, real_data_prediction))
accuracy = rmse/real_data.mean()
