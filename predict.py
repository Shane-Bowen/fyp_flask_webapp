# Import libraries
import numpy as np
from numpy import concatenate
from math import sqrt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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

#invert scale prediction
def invert_scailing(input_features, prediction, scaler):
    
    # specify the number of days and features 
    n_days = 7
    n_features = input_features.shape[2]
    
    input_features = input_features.reshape((n_days, n_features))
    pred_arr = np.empty((1, n_features))
        
    for i in range(0, prediction.shape[1]):
        # invert scaling for forecast
        pred_col = prediction[:, i].reshape(len(prediction[:, i]), 1)
        inv_prediction = concatenate((pred_col, input_features[:, -(n_features-1):][0:1]), axis=1)
        inv_prediction = scaler.inverse_transform(inv_prediction)
        inv_prediction = inv_prediction[:, 0]
        inv_prediction = inv_prediction.reshape(len(inv_prediction), 1)
        pred_arr = np.append(pred_arr, inv_prediction, axis=1)
    
    pred_arr = pred_arr[:, n_features:]
    
    return pred_arr

def history_accuracy(df, n_days, n_features, scaler):
    
    # set values
    values = df.values

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
    
    # reshape input to be 3D [samples, timesteps, features]
    test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
    
    # make prediction
    prediction = model.predict(test_X)
    
    # reshape input to be 2D
    test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
                
    # get inverted prediction in array
    pred_arr = np.empty((test_X.shape[0], 1))
    for i in range(0, prediction.shape[1]):
        prediction_col = prediction[:, i].reshape(len(prediction[:, i]), 1)
        inv_prediction = concatenate((prediction_col, test_X[:, -(n_features-1):]), axis=1)
        inv_prediction = scaler.inverse_transform(inv_prediction)
        inv_prediction = inv_prediction[:, 0]
        inv_prediction = inv_prediction.reshape(len(inv_prediction), 1)
        pred_arr = np.append(pred_arr, inv_prediction, axis=1)
        
    # slice prediction we need
    pred_arr = pred_arr[:,1:]
    
    # calculate RMSE
    for i in range(0, pred_arr.shape[1]):
        pred = pred_arr[:, i]
        rmse = sqrt(mean_squared_error(test_y[:, i], pred_arr[:, i]))
        print('t+{} RMSE: {:.3f}'.format(i+1, rmse))
        
    # calculate accuracy score based on expected and predicted
    accuracy_scores = []    
    for i in range(pred_arr.shape[0]):
        for j in range(pred_arr.shape[1]):
            if pred_arr[i][j] > test_y[i][j]:
                score = test_y[i][j] / pred_arr[i][j] * 100
                if np.isnan(score) == False:
                    accuracy_scores.append(score)
            else:
                score = pred_arr[i][j] / test_y[i][j] * 100
                if np.isnan(score) == False:
                    accuracy_scores.append(score)
            print('Row=%d Column=%d, Expected=%.2f, Predicted=%.2f, Accuracy=%.2f%%' % (i+1, j+1, test_y[i][j], pred_arr[i][j], score ))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    
    return round(np.mean(accuracy_scores), 2)

def get_input_predict_data(df, n_days, n_features, scaler):
    # get last elements from df
    input_features = df.tail(n_days).to_numpy()
    
    #normalize previous_input features
    input_features = scaler.fit_transform(input_features)
    input_features = input_features.reshape((1, n_days, n_features))
    
    prediction = model.predict(input_features)
    inv_prediction = invert_scailing(input_features, prediction, scaler)
    
    cur_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")
    end_date = cur_date + timedelta(days=7)
    
    i = 0
    #iterate cur_date
    while cur_date < end_date:
        cur_date += timedelta(days=1)
        prediction_data[cur_date.strftime('%Y-%m-%d')] = int(round(inv_prediction[0][i]))
        i += 1
        
    cur_date = datetime.strptime(df.tail(1).index.item(), "%Y-%m-%d")
    end_date = cur_date
    
    cur_date -= timedelta(days=7)
    
    while cur_date < end_date:
        cur_date += timedelta(days=1)
        input_data[cur_date.strftime("%Y-%m-%d")] = df.loc[cur_date.strftime("%Y-%m-%d")]['volume_tests']
    
    return input_data, prediction_data
    
def get_prediction(company_id, n_predict)

# store variables        
n_days = 7

# store all input data and prediction data
input_data = {}
prediction_data = {}

# get model for that company
model = keras.models.load_model(f"./models/model_{company_id}_n_{n_predict}.h5")

# load dataset
df = read_csv(f'./reports/company_report_' + company_id + '.csv', header=0, index_col="time")
df = df[['volume_tests', 'date', 'month', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]
df = df.dropna(axis='columns')

# specify the number of days, features
n_features = df.shape[1]

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))

score = history_accuracy(df, n_days, n_features, scaler)
input_data, prediction_data = get_input_predict_data(df, n_days, n_features, scaler)

#print(score)
#print(input_data)
#print(prediction_data)