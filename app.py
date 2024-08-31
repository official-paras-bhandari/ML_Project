import pickle
from flask import Flask, render_template, request, redirect, url_for, app,jsonify
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
           
if __name__=="__main__":
    app.run(debug=True)