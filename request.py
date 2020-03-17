import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'day':2, 'is_weekend':0, 'season':3, 'number_test_types':4, 'numbers_tested':4000})

print(r.json())