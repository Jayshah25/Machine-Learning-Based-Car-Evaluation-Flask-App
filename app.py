from flask import Flask, render_template, request
import pickle
import numpy as np

# intialize the app
app = Flask(__name__)

# load the model
model = pickle.load(open("model.pkl",'rb'))

@app.route('/')
@app.route('/home')
def home():
    """
    Renders the basic Front end from "index.html" file
    """
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    """
    Render the results on HTML GUI
    """
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    class_list = ['unacc', 'acc', 'vgood', 'good']
    predicted_class = class_list[output]
    return render_template('index.html', prediction_text='The Predicted Class is {}'.format(predicted_class))

@app.route('/about')
def about():
    """
    Renders the about page from "about.html"
    """
    return render_template("about.html")


if __name__=="__main__":
    app.run(debug=True)
