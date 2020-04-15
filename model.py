# Import libraries
import os
import numpy as np
from numpy import concatenate
from math import sqrt
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM
from datetime import datetime, timedelta
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

# convert series to supervised learning & normalize input variables
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

company_list = [1, 2, 6, 9, 17]
#company_list = [2]
#predict_list = [7, 30, 90]
predict_list = [1]

for company in company_list:
    
    for n_predict in predict_list:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, """reports/company_report_{0}.csv""".format(company))
        # load dataset
        df = read_csv(filename, header=0, index_col="time")
        df = df[['volume_tests', 'date', 'month', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]
        df = df.dropna(axis='columns')
        values = df.values
            
        # specify the number of days, features
        n_days = 7
        n_features = df.shape[1]
        
        #slice first n_days (no prediction made)
        df = df[n_days:]
        
        # ensure all data is float
        values = values.astype('float32')
        
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        #scaled = values
        
        # frame as supervised learning
        reframed = series_to_supervised(scaled, n_days, n_predict)
        
        # split into train and test sets
        values = reframed.values
        
        train_size = int(len(values) * 0.80)
        test_size = len(values) - train_size
        train, test = values[:train_size,:], values[train_size:,:]
        
        # split into input and outputs
        n_obs = n_days * n_features
        n_predict_obs = n_predict * n_features
        train_X, train_y = train[:, :n_obs], train[:, -n_predict_obs::n_features]
        test_X, test_y = test[:, :n_obs], test[:, -n_predict_obs::n_features]
        
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
    
        # design LSTM Model
        model = Sequential()
        
        model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.1))
        model.add(Dense(n_predict, kernel_initializer='lecun_uniform', activation='hard_sigmoid'))
        optimizer = Adam(lr=0.001)
        model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
        
        # fit Model
        model.fit(train_X, train_y, epochs=100, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        
        score, acc = model.evaluate(test_X, test_y)
        print('Test score:', score)
        print('Test accuracy:', acc)
        
        # make a prediction
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
            
        pred_arr = np.empty((test_X.shape[0], 1))
        actual_arr = np.empty((test_X.shape[0], 1))
            
        for i in range(0, yhat.shape[1]):
            yhat_col = yhat[:, i].reshape(len(yhat[:, i]), 1)
            inv_yhat = concatenate((yhat_col, test_X[:, -(n_features-1):]), axis=1)
            # invert scaling for forecast
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:, 0]
            inv_yhat = inv_yhat.reshape(len(inv_yhat), 1)
            pred_arr = np.append(pred_arr, inv_yhat, axis=1)
        
        pred_arr = pred_arr[:,1:]
        
        for i in range(0, test_y.shape[1]):
            # invert scaling for actual
            test_y_col = test_y[:, i].reshape(len(test_y[:, i]), 1)
            #test_y = test_y.reshape((len(test_y), 1))
            inv_y = concatenate((test_y_col, test_X[:, -(n_features-1):]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,0]
            
            inv_y = inv_y.reshape(len(inv_y), 1)
            actual_arr = np.append(actual_arr, inv_y, axis=1)
            
        actual_arr = actual_arr[:,1:]
        
        # calculate RMSE
        for i in range(0, actual_arr.shape[1]):
            act = actual_arr[:, i]
            pred = pred_arr[:, i]
            rmse = sqrt(mean_squared_error(actual_arr[:, i], pred_arr[:, i]))
            print('t+{} RMSE: {:.3f}'.format(i+1, rmse))
            
        # Save the model
        filename = os.path.join(dirname, """models/model_{0}_n_{1}.h5""".format(company, n_predict))
        model.save(filename)
                    
# =============================================================================
#         # calculate Accuracy Score
#         for i in range(0, actual_arr.shape[1]):
#             actual_arr = actual_arr.astype('int')
#             pred_arr = pred_arr.astype('int')
#             act = actual_arr[:, i].tolist()
#             pred = pred_arr[:, i].tolist()
#             acc_score = accuracy_score(actual_arr[:, i].tolist(), pred_arr[:, i].tolist())
#             print('t+{} Acc Score: {:.3f}'.format(i+1, acc_score))
# =============================================================================
        
# =============================================================================
#     # plot actual vs prediction
#     num_days = 90
#     pyplot.plot(list(inv_y[:num_days]), label='actual')
#     pyplot.plot(inv_yhat[:num_days], label='prediction')
#     pyplot.legend()
#     pyplot.show()
# =============================================================================