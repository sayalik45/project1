import flask
from flask import Flask,render_template,request
import numpy as np
import pickle
import json

with open('artifacts/vle_model.pkl','rb') as file:
    model = pickle.load(file)

with open('artifacts/project_data.json') as file:
    project_data = json.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data',methods = ['POST'])
def get_data():
    data = request.form

    ## take all inputs from frontend
    Temperature = data['html_Temperature']
    liq_phase_comp_benzene = data['html_liq_phase_comp_benzene']
    liq_phase_comp_cyclohexane = data['html_liq_phase_comp_cyclohexane']
    vapor_phase_comp_cyclohexane = data['html_vapor_phase_comp_cyclohexane']
    Accentric_factor_benzene = data['html_Accentric_factor_benzene']
    Accentric_factor_cyclohexane = data['html_Accentric_factor_cyclohexane']


    user_data = np.zeros(len(project_data["columns"]))
    user_data[0] = Temperature
    user_data[1] = liq_phase_comp_benzene
    user_data[2] = liq_phase_comp_cyclohexane
    user_data[3] = vapor_phase_comp_cyclohexane
    user_data[4] = Accentric_factor_benzene
    user_data[5] = Accentric_factor_cyclohexane

    pred = model.predict([user_data])
    return render_template('index.html',prediction = pred)

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port = 5000,debug = True)