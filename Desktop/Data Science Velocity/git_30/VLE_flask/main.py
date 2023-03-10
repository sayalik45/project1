import numpy as np 
import pandas as pd 
import pickle 
import json 
import flask 
from flask import Flask, render_template, request
from utils import vle_prediction

app = Flask(__name__) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data',methods = ['POST','GET'])
def get_data():

    data = request.form
    class_obj = vle_prediction(data)

    result = class_obj.liq_vapor_benzene_prediction()

    return render_template('index.html',prediction = result)


if __name__ == "__main__":
    app.run(host = '0.0.0.0',port = 5000,debug = True)
