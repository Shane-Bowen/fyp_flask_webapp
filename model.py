# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras

# convert series to supervised learning & normalize input variables
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
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
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_data():
    # load dataset
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, """reports/company_report_sorted.csv""")
    df = pd.read_csv(filename, header=0)
    
    # fill nan with 0
    df['min_commit'] = df['min_commit'].fillna(0)
    
    # remove outliers for number_busy
    df = df[df['number_busy'] < 2600]
    
    # remove outliers for quality_too_poor
    df = df[df['quality_too_poor'] < 450]
    
    # remove outliers for temporarily_unable_test
    df = df[df['temporarily_unable_test'] < 1000]
    
    # remove outliers for followup_tests
    df = df[df['followup_tests'] < 1200]
    
    return df

def train_models(df, company_list, predict_list):
    for company in company_list:
        for n_predict in predict_list:
            
            df_subset = df[df['company_id'] == company]
            df_subset = df_subset[['volume_tests', 'date', 'month', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]
            #df_subset = df_subset[['volume_tests', 'date', 'month', 'year', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit', 'has_min_commit', 'is_testing']]

            # get datframe values
            values = df_subset.values
                
            # specify the number of n_input, features
            n_input = 7
            n_features = df_subset.shape[1]
            
            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)
            
            # frame as supervised learning
            reframed = series_to_supervised(scaled, n_input, n_predict)
            
            # split into train and test sets
            values = reframed.values
            
            # train and test size
            train_size = int(len(values) * 0.80)
            test_size = len(values) - train_size
            train, test = values[:train_size,:], values[-test_size:,:]
            
            # split into input and outputs
            n_obs = n_predict * n_features
            n_predict_obs = n_predict * n_features
                        
            # split into inputs and output (cross-validation)
            X, y = values[:, :n_obs], values[:, -n_predict_obs::n_features]
            test_X, test_y = test[:, :n_obs], test[:, -n_predict_obs::n_features]
            
            # reshape input to be 3D [samples, timesteps, features]
            X = X.reshape((X.shape[0], n_predict, n_features))
            test_X = test_X.reshape((test_X.shape[0], n_predict, n_features))
            
            # design LSTM Model
            model = Sequential()
            model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(Dense(n_predict, kernel_initializer='lecun_uniform', activation='relu'))
            optimizer = Adam(lr=0.001, decay=1e-6)
            model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
            
            # num split for cross-validation
            num_splits = 10
            
            # k-fold cross validation
            kf = KFold(n_splits=num_splits)
            kf.get_n_splits(X)
            KFold(n_splits=num_splits, random_state=None, shuffle=False)
            
            # begin cross-validation procedure
            for train_index, test_index in kf.split(X):
                
                # split x and y into train and test
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
        
                # fit Model
                model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2, shuffle=False)
            
                # print test score and accuracy
                loss, score = model.evaluate(test_X, test_y)
                print('Test loss:', loss)
                print('Test score:', score)
            
            # make a prediction
            yhat = model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], n_predict*n_features))
               
            # initialize empty pred_arr and actual_arr
            pred_arr = np.empty((test_X.shape[0], 1))
            actual_arr = np.empty((test_X.shape[0], 1))
            
            # invert scaling for forecast
            for i in range(0, yhat.shape[1]):
                yhat_col = yhat[:, i].reshape(len(yhat[:, i]), 1)
                inv_yhat = np.concatenate((yhat_col, test_X[:, -(n_features-1):]), axis=1)
                inv_yhat = scaler.inverse_transform(inv_yhat)
                inv_yhat = inv_yhat[:, 0]
                inv_yhat = inv_yhat.reshape(len(inv_yhat), 1)
                pred_arr = np.append(pred_arr, inv_yhat, axis=1)
            pred_arr = pred_arr[:,1:]
            
            # invert scaling for actual
            for i in range(0, test_y.shape[1]):
                test_y_col = test_y[:, i].reshape(len(test_y[:, i]), 1)
                inv_y = np.concatenate((test_y_col, test_X[:, -(n_features-1):]), axis=1)
                inv_y = scaler.inverse_transform(inv_y)
                inv_y = inv_y[:,0]
                inv_y = inv_y.reshape(len(inv_y), 1)
                actual_arr = np.append(actual_arr, inv_y, axis=1)
            actual_arr = actual_arr[:,1:]
            
            # calculate RMSE
            for i in range(0, actual_arr.shape[1]):
                rmse = sqrt(mean_squared_error(actual_arr[:, i], pred_arr[:, i]))
                print('t+{} RMSE: {:.3f}'.format(i+1, rmse))
                          
            # plot actual vs prediction
            plt.plot(list(inv_y), label='actual')
            plt.plot(list(inv_yhat), label='prediction')
            plt.legend()
            plt.title("company_id = %s, n_predict = %s" % (company, n_predict))
            plt.show()
    
            # Save the model
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, """models/model_{0}_n_{1}.h5""".format(company, n_predict))
            model.save(filename)
            
            
def evaluate_models(df, company_list, predict_list):
    
    prediction = pd.DataFrame([], columns = ['company_id', 'accuracy'])

    for company in company_list:
        for n_predict in predict_list:
            df_subset = df[df['company_id'] == company]
            df_subset = df_subset[['volume_tests', 'date', 'month', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit']]
            #df_subset = df_subset[['volume_tests', 'date', 'month', 'year', 'is_weekend', 'quality_too_poor', 'number_busy', 'temporarily_unable_test', 'outage_hrs', 'number_test_types', 'numbers_tested', 'min_commit', 'has_min_commit', 'is_testing']]

            # get datframe values
            values = df_subset.values
                
            # specify the number of days, features
            n_input = 7
            n_features = df_subset.shape[1]
            
            #slice first n_predict (no prediction made)
            df_subset = df_subset[n_predict:]
            
            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)
            
            # frame as supervised learning
            reframed = series_to_supervised(scaled, n_input, n_predict)
            
            # split into train and test sets
            values = reframed.values
            
            # train and test size
            train_size = int(len(values) * 0.80)
            test = values[train_size:,:]
            
            # split into input and outputs
            n_obs = n_predict * n_features
            n_predict_obs = n_predict * n_features

            test_X, test_y = test[:, :n_obs], test[:, -n_predict_obs::n_features]
            test_X = test_X.reshape((test_X.shape[0], n_predict, n_features))
            
            # load saved model
            model = keras.models.load_model(f"./models/model_{company}_n_{n_predict}.h5")
            
            # make a prediction
            yhat = model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], n_predict*n_features))
               
            # initialize empty pred_arr and actual_arr
            pred_arr = np.empty((test_X.shape[0], 1))
            actual_arr = np.empty((test_X.shape[0], 1))
            
            # invert scaling for forecast
            for i in range(0, yhat.shape[1]):
                yhat_col = yhat[:, i].reshape(len(yhat[:, i]), 1)
                inv_yhat = np.concatenate((yhat_col, test_X[:, -(n_features-1):]), axis=1)
                inv_yhat = scaler.inverse_transform(inv_yhat)
                inv_yhat = inv_yhat[:, 0]
                inv_yhat = inv_yhat.reshape(len(inv_yhat), 1)
                pred_arr = np.append(pred_arr, inv_yhat, axis=1)
            pred_arr = pred_arr[:,1:]
            
            # invert scaling for actual
            for i in range(0, test_y.shape[1]):
                test_y_col = test_y[:, i].reshape(len(test_y[:, i]), 1)
                inv_y = np.concatenate((test_y_col, test_X[:, -(n_features-1):]), axis=1)
                inv_y = scaler.inverse_transform(inv_y)
                inv_y = inv_y[:,0]
                inv_y = inv_y.reshape(len(inv_y), 1)
                actual_arr = np.append(actual_arr, inv_y, axis=1)
            actual_arr = actual_arr[:,1:]
                          
            # plot actual vs prediction
            plt.plot(list(inv_y), label='actual')
            plt.plot(list(inv_yhat), label='prediction')
            plt.legend()
            plt.title("company_id = %s, n_predict = %s" % (company, n_predict))
            plt.show()
            
# =============================================================================
#             test_X = test_X.reshape((test_X.shape[0], n_input, n_features))
# 
#             inv_yhat = pred_arr[:, 0].reshape(len(pred_arr), 1)
#             inv_y = actual_arr[:, 0].reshape(len(actual_arr), 1)
#             
#             # get avg. accuracy score, compare expected and predicted
#             accuracy_scores = []
#             for i in range(len(inv_y)):
#                 if inv_y[i] == inv_yhat[i]:
#                     score = float(100)
#                     accuracy_scores.append(score)
#                 elif inv_y[i] > inv_yhat[i]:
#                     score = inv_yhat[i] / inv_y[i] * 100
#                     accuracy_scores.append(score[0])
#                 else:
#                     score = inv_y[i] / inv_yhat[i] * 100
#                     accuracy_scores.append(score[0])
#             print("Avg. Accuracy %.2f%% (+/- %.2f%%)" % (np.mean(accuracy_scores), np.std(accuracy_scores)))
#             model.evaluate(test_X, test_y)
#             
#             prediction = prediction.append({'company_id': str(company), 'accuracy': str(np.round(np.mean(accuracy_scores), 2))}, ignore_index=True)
# =============================================================================
            
            test_X = test_X.reshape((test_X.shape[0], n_input, n_features))

            inv_yhat = pred_arr[:, 0].reshape(len(pred_arr), 1)
            inv_y = actual_arr[:, 0].reshape(len(actual_arr), 1)
            
            expected_total = 0
            predicted_total = 0
            
            for i in range(len(inv_y)):
                expected_total += inv_y[i][0]
                predicted_total += inv_yhat[i][0]
            print(expected_total)
            print(predicted_total)
            
            
            if expected_total == predicted_total:
                score = float(100)
            elif expected_total > predicted_total:
                score = predicted_total / expected_total * 100
            else:
                score = expected_total / predicted_total * 100
                
            print("Avg. Accuracy %.2f%% " % (score))
            prediction = prediction.append({'company_id': str(company), 'accuracy': str(np.round(score, 2))}, ignore_index=True)
                        
    prediction.to_csv("predictions.csv", index=False)
    
def get_companies_ids(df):
    
    # initialize array
    company_list = []
    
    # store pandas series with company_id and count number of times vol. tests > 0
    count = df[df['volume_tests'] > 0].groupby('company_id')['volume_tests'].count()
    
    # iterate the index and value in series, and append id to array if greater than 30
    for index, value in count.items():
        if value > 30:
            company_list.append(index)
            
    # sort list
    company_list.sort()
    
    # return list
    return company_list

if __name__ == "__main__":
    
    #company_list = [2, 9, 49, 93, 130]
    predict_list = [7]

    df = prepare_data()
    company_list = get_companies_ids(df)
    evaluate_models(df, company_list, predict_list)
    #train_models(df, company_list, predict_list)