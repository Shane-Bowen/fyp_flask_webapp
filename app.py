# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, flash
from controller import get_input_features, invert_scailing, get_prediciton

app = Flask(__name__)
app.debug = True
app.secret_key = 'development key'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    input_data, prediction_data = get_prediciton(request.form['company_id'], request.form['date'])
        
    return render_template('index.html', input_data=input_data, prediction_data=prediction_data, date_selected=request.form['date'])

if __name__ == "__main__":
    app.run(debug=True)