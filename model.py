# Import libraries
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
def get_previous_inputs(date, df, n_days, n_features):
    
    delta = timedelta(days=1)
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = target_date - timedelta(days=n_days)
    previous_inputs = np.array([])
    
    while start_date < target_date:
        cur_date = start_date.strftime("%Y-%m-%d")
        previous_inputs = np.append(previous_inputs, [df.loc[cur_date]])
        start_date += delta
        
    previous_inputs = previous_inputs.reshape((n_days, n_features))
    
    #normalize previous_input features
    previous_inputs = scaler.fit_transform(previous_inputs)
    previous_inputs = previous_inputs.reshape((1, n_days, n_features))
    
    return previous_inputs

# load dataset
df = read_csv('./reports/company_report_2.csv', header=0, index_col="time")
df = df[['volume_tests', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'number_test_types', 'numbers_tested']]
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
model.save('./models/model.h5')

# =============================================================================
# # plot actual vs prediction
# num_days = 240
# pyplot.plot(list(inv_y[:num_days]), label='actual')
# pyplot.plot(inv_yhat[:num_days], label='prediction')
# pyplot.legend()
# pyplot.show()
# =============================================================================

# =============================================================================
# # date predicted
# date = '2019-12-31'
# 
# # get previous_inputs
# previous_inputs = get_previous_inputs(date, df, n_days, n_features)
# 
# # make a prediction
# yhat = model.predict(previous_inputs[0:1])
# X = X.reshape((X.shape[0], n_days*n_features))
#  
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, X[:, -(n_features-1):][0:1]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# print("Prediction", int(round(inv_yhat[0])))
# print("Actual", df.loc[date]['volume_tests'])
# 
# # Save the model
# model.save('model.h5')
# =============================================================================

# =============================================================================
# if not os.path.isdir('models'):
#     os.mkdir('models') 
# =============================================================================