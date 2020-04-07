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
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM
from datetime import datetime, timedelta
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

# def reframe_data(data, n_days):
#     n_features = data.shape[1]
#     cols = []
#     arr = []
#     np_cols = []
        
#     for i in range(0, df.shape[0]):
#         for j in range(i, i+((n_days * 7) + 1), n_days):
#             if j < data.shape[0]:
#                 arr.append(data[j])
#             else:
#                 return DataFrame(np_cols)
        
#         cols.append(arr)
#         np_cols = np.array(cols)
#         np_cols = np_cols.reshape(np_cols.shape[0], (n_days+1)*n_features)
#         arr = []

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

#get previous inputs from date
def get_input_features(df, date, company_id, input_data, prediction_data):

    # specify the number of days and features 
    n_days = 7
    n_features = df.shape[1]

    # numpy array for previous inputs
    input_features = np.array([])

    # target_date & cur_date
    target_date = datetime.strptime(date, "%Y-%m-%d")
    cur_date = target_date - timedelta(days=n_days)
        
    # get end_date stored in dataframe
    end_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")
    
    # while loop until target_date
    while cur_date < target_date:
        if cur_date <= end_date:
            input_features = np.append(input_features, [df.loc[cur_date.strftime("%Y-%m-%d")]])
            input_data[cur_date.strftime("%Y-%m-%d")] = df.loc[cur_date.strftime("%Y-%m-%d")]['volume_tests']
            cur_date += timedelta(days=1)       
        else:
            weekday = [0, 1, 2, 3, 4]
            # check if cur_date is weekend
            if cur_date.weekday() in weekday:
                isWeekend = 0
            else:
                isWeekend = 1
                
            # get last_date in df on that day
            last_date = cur_date
            while last_date >= end_date:
                last_date -= timedelta(days=7)
            last_date = last_date.strftime('%Y-%m-%d')
            
            if 'min_commit' in df.columns:
                input_features = np.append(input_features, [prediction_data[cur_date.strftime('%Y-%m-%d')], isWeekend, df.loc[last_date]['quality_too_poor'], df.loc[last_date]['number_busy'],
                                                              df.loc[last_date]['temporarily_unable_test'], df.loc[last_date]['outage_hrs'], df.loc[last_date]['number_test_types'], df.loc[last_date]['numbers_tested'], df.loc[last_date]['min_commit']])
            else:
                input_features = np.append(input_features, [prediction_data[cur_date.strftime('%Y-%m-%d')], isWeekend, df.loc[last_date]['quality_too_poor'], df.loc[last_date]['number_busy'],
                                                              df.loc[last_date]['temporarily_unable_test'], df.loc[last_date]['outage_hrs'], df.loc[last_date]['number_test_types'], df.loc[last_date]['numbers_tested']])
            cur_date += timedelta(days=1)
        
    # reshape input_features
    input_features = input_features.reshape((n_days, n_features))
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    #normalize previous_input features
    input_features = scaler.fit_transform(input_features)
    input_features = input_features.reshape((1, n_days, n_features))
    
    return input_features, scaler, input_data

#invert scale prediction
def invert_scailing(input_features, prediction, scaler):
    
    # specify the number of days and features 
    n_days = 7
    n_features = input_features.shape[2]
    
    # reshape input_features
    input_features = input_features.reshape((1, n_days*n_features))
    
    # invert scaling for forecast
    inv_prediction = concatenate((prediction, input_features[:, -(n_features-1):][0:1]), axis=1)
    inv_prediction = scaler.inverse_transform(inv_prediction)
    inv_prediction = inv_prediction[:,0]
    
    return inv_prediction

company_list = [1, 2, 6, 9, 17]
#company_list = [2]
#company_list = []
#company_list = [17]

for company in company_list:
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, """reports/company_report_{0}.csv""".format(company))
    # load dataset
    df = read_csv(filename, header=0, index_col="time")
    df = df[['volume_tests', 'date', 'month', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]
    df = df.dropna(axis='columns')
    values = df.values
        
    # specify the number of days and features 
    n_days = 7
    n_features = df.shape[1]
    
    #slice first n_days (no prediction made)
    df = df[n_days:]
    
    # ensure all data is float
    values = values.astype('float32')
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_days, 1)
    #reframed = reframe_data(scaled, n_days)
    
    # split into train and test sets
    values = reframed.values
    
    train_size = int(len(values) * 0.80)
    test_size = len(values) - train_size
    train, test = values[:train_size,:], values[train_size:,:]
    
    # split into input and outputs
    n_obs = n_days * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
    
    # design LSTM Model
    model = Sequential()
    
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='lecun_uniform', activation='hard_sigmoid'))
    optimizer = Adam(lr=0.001)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    
    # fit Model
    history = model.fit(train_X, train_y, epochs=100, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
    
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
    # Save the model
    filename = os.path.join(dirname, """models/model_{0}.h5""".format(company))
    model.save(filename)
    
    print(company)

    # plot actual vs prediction
    num_days = 210
    pyplot.plot(list(inv_y[:num_days]), label='actual')
    pyplot.plot(inv_yhat[:num_days], label='prediction')
    pyplot.legend()
    pyplot.show()

# =============================================================================
# # Make a prediction
# 
# # inputs
# date = '2020-04-07'
# company_id = '2'
# 
# # store all input data and prediction data
# input_data = {}
# prediction_data = {}
# 
# # get model for that company
# model = keras.models.load_model(f"./models/model_{company_id}.h5")
# 
# # load dataset
# df = read_csv(f'./reports/company_report_' + company_id + '.csv', header=0, index_col="time")
# df = df[['volume_tests', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]
# df = df.dropna(axis='columns')
# 
# # get last date in dataframe
# cur_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")
# 
# if cur_date < datetime.strptime(date, '%Y-%m-%d'):
# 
#     # while loop until date selected
#     while cur_date < datetime.strptime(date, '%Y-%m-%d'):
#     
#         # iterate cur_date
#         cur_date += timedelta(days=1)
#     
#         input_features, scaler, input_data = get_input_features(df, cur_date.strftime('%Y-%m-%d'), company_id, input_data, prediction_data)
#         prediction = model.predict(input_features)
#                 
#         inv_prediction = invert_scailing(input_features, prediction, scaler)
#         prediction_data[cur_date.strftime('%Y-%m-%d')] = int(round(inv_prediction[0]))
#         
#         print(input_data)
#         print(prediction_data)
# 
# else:
#     # iterate cur_date
#     cur_date += timedelta(days=1)
#     
#     input_features, scaler, input_data = get_input_features(df, cur_date.strftime('%Y-%m-%d'), company_id, input_data, prediction_data)
#     print(input_data)
# =============================================================================
    
# =============================================================================
# if not os.path.isdir('models'):
#     os.mkdir('models') 
# =============================================================================