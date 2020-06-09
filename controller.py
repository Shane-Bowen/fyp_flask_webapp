# Import libraries
import numpy as np
from math import sqrt
from numpy import concatenate
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

#invert scale prediction
def invert_scailing(input_features, prediction, scaler):
    
    # specify the number of days and features 
    n_days = 7
    n_features = input_features.shape[2]
    
    # reshape input features and create array to store predictions
    input_features = input_features.reshape((n_days, n_features))
    pred_arr = np.empty((1, n_features))
        
    # loop each prediction and append to array
    for i in range(0, prediction.shape[1]):
        pred_col = prediction[:, i].reshape(len(prediction[:, i]), 1)
        inv_prediction = concatenate((pred_col, input_features[:, -(n_features-1):][0:1]), axis=1)
        inv_prediction = scaler.inverse_transform(inv_prediction)
        inv_prediction = inv_prediction[:, 0]
        inv_prediction = inv_prediction.reshape(len(inv_prediction), 1)
        pred_arr = np.append(pred_arr, inv_prediction, axis=1)
    
    # only keep predictions we need
    pred_arr = pred_arr[:, n_features:]
    
    return pred_arr

def history_accuracy(df, model, n_predict, scaler):

    # store variables
    values = df.values
    
    # specify the number of days, features
    n_days = 7
    n_features = df.shape[1]

    # ensure all data is float
    values = values.astype('float32')

    # normalize features
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_days, n_predict)
    n_reframed = series_to_supervised(values, n_days, n_predict)

    # set values
    values = reframed.values
    n_values = n_reframed.values

    # split into input and outputs
    n_obs = n_days * n_features
    n_predict_obs = n_predict * n_features
    test_X,= values[:, :n_obs], 
    test_y = n_values[:, -n_predict_obs::n_features]

    # normalized test y values
    inv_test_y = values[:, -n_predict_obs::n_features] 

    # reshape input to be 3D [samples, timesteps, features]
    test_X = test_X.reshape((test_X.shape[0], n_days, n_features))

    # make prediction
    prediction = model.predict(test_X)
    
    # reshape input to 2D
    test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
    
    # get inverted prediction in array
    pred_arr = np.empty((test_X.shape[0], 1))
    for i in range(0, prediction.shape[1]):
        # invert scaling for forecast
        prediction_col = prediction[:, i].reshape(len(prediction[:, i]), 1)
        inv_prediction = concatenate((prediction_col, test_X[:, -(n_features-1):]), axis=1)
        inv_prediction = scaler.inverse_transform(inv_prediction)
        inv_prediction = inv_prediction[:, 0]
        inv_prediction = inv_prediction.reshape(len(inv_prediction), 1)
        pred_arr = np.append(pred_arr, inv_prediction, axis=1)

    # slice prediction we need  
    pred_arr = pred_arr[:,1:]

    # calculate RMSE
    rmse_list = []
    for i in range(0, pred_arr.shape[1]):
        rmse = sqrt(mean_squared_error(test_y[:, i], pred_arr[:, i]))
        rmse_list.append(rmse)

    # calculate MAE
    mae_list = []
    for i in range(0, pred_arr.shape[1]):
        mae = mean_absolute_error(test_y[:, i], pred_arr[:, i])
        mae_list.append(mae)
        
    # calculate accuracy score based on expected and predicted value
    accuracy_scores = []
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i][j] > inv_test_y[i][j]:
                score = inv_test_y[i][j] / prediction[i][j] * 100
                if np.isnan(score) == False and score > 0:
                    accuracy_scores.append(score)
            else:
                score = prediction[i][j] / inv_test_y[i][j] * 100
                if np.isnan(score) == False and score > 0:
                    accuracy_scores.append(score)

    return round(np.mean(accuracy_scores), 2), round(np.mean(rmse_list), 2), round(np.mean(mae_list), 2), test_X, scaler

def get_prediciton(company_id, n_predict):

    # store all input data and prediction data
    input_data = {}
    prediction_data = {}

    # get model for that company
    model = keras.models.load_model(f"./models/model_{company_id}_n_{n_predict}.h5")

    # load dataset
    df = read_csv(f'./reports/company_report_' + company_id + '.csv', header=0, index_col="time")
    df = df[['volume_tests', 'date', 'month', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]

    # drop columns where nan or replace nan with mean
    df = df.dropna(axis='columns', how='all')
    df.iloc[:, -1] = df.iloc[:, -1].fillna(df.iloc[:, -1].mean())

    # scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # history accuracy
    accuracy_score, avg_rmse, avg_mae, test_X, scaler = history_accuracy(df, model, int(n_predict), scaler)

    # specify the number of days and features 
    n_days = 7
    n_features = df.shape[1]

    # set input features
    input_features = test_X[-1:]
    input_features = input_features.reshape((1, n_days, n_features))

    # make prediction
    prediction = model.predict(input_features)
    inv_prediction = invert_scailing(input_features, prediction, scaler)

    # specify cur_date and end_date
    cur_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")
    end_date = cur_date + timedelta(days=int(n_predict))
    
    i = 0
    prediction_data_sum = 0
    # append predictions to dictionary
    while cur_date < end_date:
        cur_date += timedelta(days=1)
        prediction_data[cur_date.strftime('%Y-%m-%d')] = int(round(inv_prediction[0][i]))
        prediction_data_sum += int(round(inv_prediction[0][i]))
        i += 1
        
    # specify cur_date and end_date
    cur_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")
    end_date = cur_date
    cur_date -= timedelta(days=n_days)

    # append inputs to dictionary
    while cur_date < end_date:
        cur_date += timedelta(days=1)
        input_data[cur_date.strftime("%Y-%m-%d")] = df.loc[cur_date.strftime("%Y-%m-%d")]['volume_tests']

    # specify cur_date and end_date
    cur_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")
    end_date = cur_date
    cur_date -= timedelta(days=int(n_predict))
    
    # calcuate input data sum in relation to n_prediction
    input_data_sum = 0
    while cur_date < end_date:
        cur_date += timedelta(days=1)
        input_data_sum += df.loc[cur_date.strftime("%Y-%m-%d")]['volume_tests']
        
    percent_change = ((prediction_data_sum - input_data_sum) / input_data_sum) * 100

    return input_data, prediction_data, accuracy_score, avg_rmse, avg_mae, round(percent_change, 2)