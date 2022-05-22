import pandas as pdr

data=pdr.read_csv('NSE-TATAGLOBAL11(1).csv')
# Reversing DataRow
data=data.iloc[::-1]
data.reset_index(drop=True, inplace=True)
data

# By taking close price
data=data['Close']

import matplotlib.pyplot as plt
plt.plot(data)

# Scaling price
from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler=MinMaxScaler(feature_range=(0,1))
data=scaler.fit_transform(np.array(data).reshape(-1,1))

data.shape

data

#Splitting dataset into train and test


x=int((len(data))*0.65)
training_data=data[0:x,:]
testing_data=data[x:len(data),:1]

# Create Data Set
def dataset(dataset,step):
  X, Y = [] , []
  for i in range(len(dataset)-step-1):
    Z = dataset[i:(i+step), 0]
    X.append(Z)
    Y.append(dataset[i + step, 0])
  return np.array(X), np.array(Y)
x_train, y_train = dataset(training_data, 150)
x_test, y_test = dataset(testing_data, 150)

x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

x_train.shape

#LSTM Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(150,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(units=1))

model.summary()

model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train,y_train,batch_size=64,epochs=100,validation_data=(x_test, y_test))

# Prediction and Error
test_predict=model.predict(x_test)
#Test Data RMSE
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_test,test_predict))

y_predicated=model.predict(x_test)
y_predicated.shape

y_predicated=scaler.inverse_transform(y_predicated)
y_test_1=scaler.inverse_transform(np.array(y_test).reshape(-1,1))

plt.figure(figsize=(12,6))
plt.plot(y_test_1,label='Original stock price')
plt.plot(y_predicated,label='Predicated stock price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

#forecasting stock prices for next 60 days

len(testing_data)-150

initial_price=testing_data[283:].reshape(1,-1)
temp_prices=list(initial_price)
temp_prices=temp_prices[0].tolist()
from numpy import array

output=[]
steps=150
i=0

while(i<60):

  if(len(temp_prices)>150):
      initial_price=np.array(temp_prices[1:])
      initial_price=initial_price.reshape(1,-1)
      initial_price = initial_price.reshape((1, steps, 1))
      yhat = model.predict(initial_price, verbose=0)
      temp_prices.extend(yhat[0].tolist())
      temp_prices=temp_prices[1:]
      output.extend(yhat.tolist())
      i=i+1
  else:
      initial_price = initial_price.reshape((1, steps,1))
      yhat = model.predict(initial_price, verbose=0)
      temp_prices.extend(yhat[0].tolist())
      output.extend(yhat.tolist())
      i=i+1


stock_price_next_60days=scaler.inverse_transform(output)
stock_price_next_60days=pdr.DataFrame(stock_price_next_60days)
stock_price_next_60days

plt.plot(np.arange(1,151),scaler.inverse_transform(data[1085:]))
plt.plot(np.arange(151,211),stock_price_next_60days[0])