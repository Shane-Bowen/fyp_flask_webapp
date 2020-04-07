# Import libraries
import numpy as np
from numpy import concatenate
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from tensorflow import keras

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

            # if min_commit in df
            if 'min_commit' in df.columns:
                input_features = np.append(input_features, [prediction_data[cur_date.strftime('%Y-%m-%d')], cur_date.day, cur_date.month, isWeekend, df.loc[last_date]['quality_too_poor'], df.loc[last_date]['number_busy'],
                                                              df.loc[last_date]['temporarily_unable_test'], df.loc[last_date]['outage_hrs'], df.loc[last_date]['number_test_types'], df.loc[last_date]['numbers_tested'], df.loc[last_date]['min_commit']])
            else:
                input_features = np.append(input_features, [prediction_data[cur_date.strftime('%Y-%m-%d')], cur_date.day, cur_date.month, isWeekend, df.loc[last_date]['quality_too_poor'], df.loc[last_date]['number_busy'],
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

def get_prediciton(company_id, date):

    # store all input data and prediction data
    input_data = {}
    prediction_data = {}

    # get model for that company
    model = keras.models.load_model(f"./models/model_{company_id}.h5")
    
    # load dataset
    df = read_csv(f'./reports/company_report_' + company_id + '.csv', header=0, index_col="time")
    df = df[['volume_tests', 'date', 'month', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]
    df = df.dropna(axis='columns')

    # get last date in dataframe
    cur_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")

    # if selected date is greater than last date in df, then prediction can be made
    if cur_date < datetime.strptime(date, '%Y-%m-%d'):
        # while loop until date selected
        while cur_date < datetime.strptime(date, '%Y-%m-%d'):

            # iterate cur_date
            cur_date += timedelta(days=1)

            input_features, scaler, input_data = get_input_features(df, cur_date.strftime('%Y-%m-%d'), company_id, input_data, prediction_data)
            prediction = model.predict(input_features)
                    
            inv_prediction = invert_scailing(input_features, prediction, scaler)
            prediction_data[cur_date.strftime('%Y-%m-%d')] = int(round(inv_prediction[0]))
    
    # else, no prediction can be made.
    else:
        # iterate cur_date
        date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)

        input_features, scaler, input_data = get_input_features(df, date.strftime('%Y-%m-%d'), company_id, input_data, prediction_data)

    return input_data, prediction_data