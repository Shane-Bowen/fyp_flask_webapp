## ML-Flask-Webapp
This is a Flask web app with machine learning, with the goal of allowing users to predict volume of tests using the web app.

### Prerequisites
You must have Scikit Learn, Pandas, Numpy, Keras (for Machine Learning Model) and Flask (for API) installed.

### Project Structure
This project has six major parts :
1. models - This folder contains code for our Machine Learning model to predict volume tests based on training data in 'company_report_2.csv' file.
2. app.py - This contains Flask APIs that receives input from user, computes the predicted value based on our model and returns it.
3. controller.py - This contains any logic that we need for our web app to work.
4. templates - This folder contains the HTML template to allow user to enter inputs and display the predicted volume tests
5. scripts - This folder contains the scripts used to aggregate the data and export it to csv files.
6. reports - This is where our csv files are stored, this is used as our training data in our model.

### Running the project
1. Ensure that you are in the parent project directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model in the file ./models/model.h5

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

Enter a valid date and hit predict.

If everything goes well, you should  be able to see the predcited volume tests on the HTML pages
