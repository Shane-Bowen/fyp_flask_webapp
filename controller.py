# Import libraries
import numpy as np
from numpy import concatenate
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv

#get previous inputs from date
def get_previous_inputs(date, company_id):

    # load dataset
    df = read_csv(f'./reports/company_report_{company_id}.csv', header=0, index_col="time")
    df = df[['volume_tests', 'company_id', 'company_type', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'number_test_types', 'numbers_tested']]

    # specify the number of days and features 
    n_days = 7
    n_features = df.shape[1]
    
    # integer encode direction
    encoder = LabelEncoder()
    df.iloc[:,[2]] = encoder.fit_transform(df.iloc[:,[2]])

    delta = timedelta(days=1)
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = target_date - timedelta(days=n_days)
    previous_inputs = np.array([])
    
    #df = df.loc[df['company_id'] == int(company_id)]
    
    # get last_date in stored in dataframe
    last_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")
    
    # if target_date is less than last_date then we can get previous inputs
    if target_date <= last_date + delta:    
        while start_date < target_date:
            cur_date = start_date.strftime("%Y-%m-%d")
            previous_inputs = np.append(previous_inputs, [df.loc[cur_date]])
            start_date += delta
    # else we will have to push back target and start date until inside dataframe range
    else:
        while target_date > last_date + delta:
            target_date -= timedelta(days=7)
            start_date -= timedelta(days=7)
        while start_date < target_date:
            cur_date = start_date.strftime("%Y-%m-%d")
            previous_inputs = np.append(previous_inputs, [df.loc[cur_date]])
            start_date += delta
        
    previous_inputs = previous_inputs.reshape((n_days, n_features))
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    #normalize previous_input features
    previous_inputs = scaler.fit_transform(previous_inputs)
    previous_inputs = previous_inputs.reshape((1, n_days, n_features))
    
    return previous_inputs, scaler

#invert scale prediction
def invert_scailing(previous_inputs, prediction, scaler):
    
    # specify the number of days and features 
    n_days = 7
    n_features = previous_inputs.shape[2]
    
    previous_inputs = previous_inputs.reshape((1, n_days*n_features))
    
    # invert scaling for forecast
    inv_prediction = concatenate((prediction, previous_inputs[:, -(n_features-1):][0:1]), axis=1)
    inv_prediction = scaler.inverse_transform(inv_prediction)
    inv_prediction = inv_prediction[:,0]
    
    return inv_prediction