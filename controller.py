# Import libraries
import numpy as np
import os
from math import sqrt
from numpy import concatenate
from datetime import datetime, timedelta
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from model import series_to_supervised, train_model, evaluate_model

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
    
    # return pred_arr
    return pred_arr

def get_prediciton(company_id, n_predict):

    # store all input data and prediction data
    input_data = {}
    prediction_data = {}

    # load dataset
    df = read_csv(f'./reports/company_report_sorted.csv', header=0, index_col="time")

    # if model exists, then load model
    if os.path.isfile(f"./models/model_{company_id}_n_{n_predict}.h5"):
        model = keras.models.load_model(f"./models/model_{company_id}_n_{n_predict}.h5")
    # else train new model
    else:
        model = train_model(df, company_id, n_predict)
    
    df_subset = df[df['company_id'] == company_id]
    df_subset = df_subset[['volume_tests', 'date', 'month', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]
    #df = df[['volume_tests', 'date', 'month', 'year', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit', 'has_min_commit', 'is_testing']]

    # fill nan with 0
    df_subset['min_commit'] = df_subset['min_commit'].fillna(0)

    # scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # history accuracy
    accuracy_score, avg_rmse, avg_mae, test_X, scaler = evaluate_model(df, company_id, n_predict)

    # specify the number of days and features 
    n_days = 7
    n_features = df_subset.shape[1]

    # set input features
    input_features = test_X[-1:]
    input_features = input_features.reshape((1, n_days, n_features))

    # make prediction
    prediction = model.predict(input_features)
    inv_prediction = invert_scailing(input_features, prediction, scaler)

    # specify cur_date and end_date
    cur_date = datetime.strptime(df_subset.tail(1).index.item(), "%Y-%m-%d")
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
    cur_date = datetime.strptime(df_subset.tail(1).index.item(), "%Y-%m-%d")
    end_date = cur_date
    cur_date -= timedelta(days=n_days)

    # append inputs to dictionary
    while cur_date < end_date:
        cur_date += timedelta(days=1)
        input_data[cur_date.strftime("%Y-%m-%d")] = df_subset.loc[cur_date.strftime("%Y-%m-%d")]['volume_tests']

    # specify cur_date and end_date
    cur_date = datetime.strptime(df_subset.tail(1).index.item(), "%Y-%m-%d")
    end_date = cur_date
    cur_date -= timedelta(days=int(n_predict))
    
    # calcuate input data sum in relation to n_prediction
    input_data_sum = 0
    while cur_date < end_date:
        cur_date += timedelta(days=1)
        input_data_sum += df_subset.loc[cur_date.strftime("%Y-%m-%d")]['volume_tests']
        
    percent_change = ((prediction_data_sum - input_data_sum) / input_data_sum) * 100

    return input_data, prediction_data, accuracy_score, avg_rmse, avg_mae, round(percent_change, 2)