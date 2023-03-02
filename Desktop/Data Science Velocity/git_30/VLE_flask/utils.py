import pandas as pd 
import numpy as np 
import pickle 
import json 

class vle_prediction():
    def __init__(self,data):
        self.data = data

    def loading_files(self):
        with open('artifacts/vle_model.pkl','rb') as file :
            self.model = pickle.load(file)

        with open('artifacts/project_data.json','r') as file :
            self.project_data = json.load(file)

    def liq_vapor_benzene_prediction(self):
        self.loading_files()

        Temperature = self.data['html_Temperature']
        liq_phase_comp_benzene = self.data['html_liq_phase_comp_benzene']
        liq_phase_comp_cyclohexane = self.data['html_liq_phase_comp_cyclohexane']
        vapor_phase_comp_cyclohexane = self.data['html_vapor_phase_comp_cyclohexane']
        Accentric_factor_benzene = self.data['html_Accentric_factor_benzene']
        Accentric_factor_cyclohexane = self.data['html_Accentric_factor_cyclohexane']

        user_data = np.zeros(len(self.project_data['columns']))

        user_data[0] = Temperature 
        user_data[1] = liq_phase_comp_benzene
        user_data[2] = liq_phase_comp_cyclohexane
        user_data[3] = vapor_phase_comp_cyclohexane
        user_data[4] = Accentric_factor_benzene 
        user_data[5] = Accentric_factor_cyclohexane 

        result = self.model.predict([user_data])
        
        return result