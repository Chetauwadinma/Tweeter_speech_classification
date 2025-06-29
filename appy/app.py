
# import the requiredlibraries
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# create a Flask application
app = Flask(__name__)
# load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
# Load the model vectorizer
vectorizer = pickle.load(open('cv.pkl', 'rb'))
# define a route for the home page


@app.route('/')
def home():
    return render_template('index.html')
# define a route for the prediction


@app.route('/predict', methods=['POST'])
def predict():
    # get the user input from the form
    user_input = request.form['text']
    # vectorize the input text
    input_vector = vectorizer.transform([user_input])
    # make a prediction
    prediction = model.predict(input_vector)
    # return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})


# run the app
if __name__ == "__main__":
    app.run(debug=True)