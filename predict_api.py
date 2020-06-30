# Import libraries
from flask import Flask, request
from controller import get_prediciton
from flask_restx import Resource, Api, fields

app = Flask(__name__)
app.debug = True
app.secret_key = 'development key'

api = Api(app)

model = api.model('Model', {
    "company_id": fields.Integer(required = True),
    "predict_days": fields.Integer(required = True)
})

@api.route('/predict_api')
class Prediction(Resource):

    @api.expect(model)
    def post(self):

        input_data, prediction_data, accuracy_score, avg_rmse, avg_mae, percent_change = get_prediciton(request.json['company_id'], request.json['predict_days'])
        return {
            'input_data' : input_data,
            'prediction_data' : prediction_data,
            'accuracy_score' : accuracy_score,
            'avg_rmse' : avg_rmse,
            'avg_mae' : avg_mae,
            'percent_change' : percent_change
         }

if __name__ == "__main__":
    app.run(debug=True)