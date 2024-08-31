import pickle
from flask import Flask, render_template, request,url_for, app,jsonify
import numpy as np
import pandas as pd


app = Flask(__name__) #starting point of app

## Load the model
model = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('sacling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['Post'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))  
    output = model.predict(new_data) 
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['post'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediciton_text="The House Price prediciton is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)