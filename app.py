from flask import Flask, request

# First we will import the necessary Library 
import pandas as pd
import numpy as np
import datetime as dt

# For Evalution we will use these library
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

#getting data
import json
import requests

app = Flask(__name__)

@app.route('/hello', methods=['POST'])
def hello():
    crypt=['BTC',"ETH"]
    curr=['USD','INR']
    data = request.get_json()
    fsym=crypt[data['crypt']]
    tsym=curr[data['cur']]
    #########################################
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    req=endpoint + '?fsym='+fsym+'&tsym='+tsym+'&limit=500'
    res = requests.get(req)
    maindf = pd.DataFrame(json.loads(res.content)['Data'])
    maindf['time'] = pd.to_datetime(maindf['time'], unit='s')
    maindf.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)


    # Lets First Take all the Close Price 
    closedf = maindf[['time','close']]
    

    del closedf['time']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

    training_size=int(len(closedf)*0.65)
    
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    model=Sequential()
    model.add(LSTM(100,input_shape=(None,1),activation="linear"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")

    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=32,verbose=1)

    loss = history.history['loss']
    

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict.shape, test_predict.shape

    # Transform back to original form

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
     

    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    from numpy import array

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 5
    while(i<pred_days):
        
        if(len(temp_input)>time_step):
            
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1
            
    last_days=np.arange(1,time_step+1)
    
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]
    next_predicted_days_value = temp_mat

    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    q=str( round(next_predicted_days_value[time_step+1],2))


    ##########################################
    return {'message': q}

if __name__ == '__main__':
    app.run(debug=True)
