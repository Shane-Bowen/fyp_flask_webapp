# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, flash
from controller import get_previous_inputs, invert_scailing
from tensorflow import keras

app = Flask(__name__)
app.debug = True
app.secret_key = 'development key'
model = keras.models.load_model('./models/model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]co
            
    final_features, scaler = get_previous_inputs(request.form['date'])
    prediction = model.predict(final_features)
            
    inv_prediction = invert_scailing(final_features, prediction, scaler)
    
    return render_template('index.html', prediction_text='Volume Tests should be {}'.format(int(round(inv_prediction[0]))))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)