# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, flash
from controller import get_previous_inputs, invert_scailing
from tensorflow import keras

app = Flask(__name__)
app.debug = True
app.secret_key = 'development key'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    
    model = keras.models.load_model(f"./models/model_{request.form['company_id']}.h5")
            
    final_features, scaler, volume_tests, dates = get_previous_inputs(request.form['date'], request.form['company_id'])
    prediction = model.predict(final_features)
            
    inv_prediction = invert_scailing(final_features, prediction, scaler)
    #volume_tests.append(int(round(inv_prediction[0])))
    
    return render_template('index.html', prediction_text='Volume Tests should be {}'.format(int(round(inv_prediction[0]))), input_data=volume_tests, prediction_data=[int(round(inv_prediction[0]))], dates=dates)

if __name__ == "__main__":
    app.run(debug=True)