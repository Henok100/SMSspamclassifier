# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:48:11 2020

@author: Henok Gashaw
"""
from flask import Flask,render_template,url_for,request
import pandas as pd 
import joblib

classifier = joblib.load('NB_classifierModel.pkl')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    		message = request.form['message']
    		data = [message]
    		my_prediction = classifier.predict(data)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)