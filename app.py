# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, flash
from controller import get_prediciton

app = Flask(__name__)
app.debug = True
app.secret_key = 'development key'

@app.route('/')
def home():


    input_data, prediction_data, accuracy_score = get_prediciton('2', '7')

    return render_template('index.html', input_data=input_data, prediction_data=prediction_data, accuracy_score=accuracy_score)
    #return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_data, prediction_data, accuracy_score = get_prediciton(request.form['company_id'], request.form['predict_days'])
    #last_date = list(prediction_data.keys())[-1]
        
    return render_template('index.html', input_data=input_data, prediction_data=prediction_data, accuracy_score=accuracy_score)

if __name__ == "__main__":
    app.run(debug=True)