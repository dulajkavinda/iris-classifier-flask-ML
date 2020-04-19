# app.py
import os
from flask import Flask, request, jsonify
app = Flask(__name__)
from model.Train import train_model
from sklearn.externals import joblib

if not os.path.isfile('iris-model.model'):
    train_model()

model = joblib.load('iris-model.model')


@app.route('/predict/', methods=['POST'])
def respond():

        sepal_length = request.form.get("sepal_length", type=float)
        sepal_width = request.form.get("sepal_width",  type=float)
        petal_length = request.form.get("petal_length",  type=float)
        petal_width = request.form.get("petal_width",  type=float)

        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        if prediction == 0:
            predicted_class = 'Iris-setosa'
        elif prediction == 1:
            predicted_class = 'Iris-versicolor'
        else:
            predicted_class = 'Iris-virginica'

        return jsonify({
            'Prediction': predicted_class
        })

@app.route('/post/', methods=['POST'])
def post_something():
    param = request.form.get('name')
    print(param)
    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        return jsonify({
            "Message": "Welcome {name} to our awesome platform!!",
            # Add this option to distinct the POST request
            "METHOD" : "POST"
        })
    else:
        return jsonify({
            "ERROR": "no name found, please send a name."
        })

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)