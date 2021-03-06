# Import libraries
import numpy as np
import gc
from flask import Flask, request, jsonify, render_template, flash, url_for, redirect, session
from controller import get_prediciton
from passlib.hash import sha256_crypt
from functools import wraps
from MySQLdb import escape_string
from dbconnect import connection

app = Flask(__name__)
app.debug = True
app.secret_key = 'development key'

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect(url_for('login_page'))

    return wrap

@app.route('/')
@login_required
def home():
    input_data, prediction_data, accuracy_score, avg_rmse, avg_mae, percent_change = get_prediciton('2', '7')
    return render_template('index.html', input_data=input_data, prediction_data=prediction_data, accuracy_score=accuracy_score, avg_rmse=avg_rmse, avg_mae=avg_mae, percent_change=percent_change)

@app.route('/predict',methods=['POST'])
@login_required
def predict():
    input_data, prediction_data, accuracy_score, avg_rmse, avg_mae, percent_change = get_prediciton(request.form['company_id'], request.form['predict_days'])
    #last_date = list(prediction_data.keys())[-1]
    return render_template('index.html', input_data=input_data, prediction_data=prediction_data, accuracy_score=accuracy_score, avg_rmse=avg_rmse, avg_mae=avg_mae, percent_change=percent_change)

@app.route("/logout/")
@login_required
def logout():
    session.clear()
    gc.collect()
    return redirect(url_for("home"))

@app.route('/login/', methods=["GET","POST"])
def login_page():
    error = ''
    try:
        c = connection()[0]
        if request.method == "POST":
            sql = "SELECT * FROM users WHERE email = %s"
            val = (escape_string(request.form['email']), )
            data = c.execute(sql, val)
            data = c.fetchone()[2]

            if sha256_crypt.verify(request.form['password'], data):
                session['logged_in'] = True
                session['email'] = request.form['email']
                
                return redirect(url_for("home"))

            else:
                error = "Invalid credentials, Try again"
        gc.collect()

        return render_template("login.html", error=error)

    except Exception:
        error = "Invalid Credentials, Try again"
        return render_template("login.html", error=error) 

@app.route('/register/', methods=["GET","POST"])
def register_page():
    error = ''
    try:
        if request.method == "POST":
            email = request.form['email']
            password = sha256_crypt.encrypt((str(request.form['password'])))
            c, conn = connection()

            sql = "SELECT * FROM users WHERE email = %s"
            val = (escape_string(email), )
            x = c.execute(sql, val)
            
            if int(x) > 0 or request.form['password'] != request.form['confirm']:
                error = "Invalid Credentials, Try again"
                return render_template("register.html", error=error)

            else:
                sql = "INSERT INTO users (email, password) VALUES (%s, %s)"
                val = (escape_string(email), escape_string(password))
                c.execute(sql, val)

                conn.commit()
                c.close()
                conn.close()

                gc.collect() #garbage collection

                session['logged_in'] = True
                session['email'] = email

                return redirect(url_for('home'))
        
        return render_template("register.html", error=error)

    except Exception as e:
        return(str(e))

if __name__ == "__main__":
    app.run(debug=True)