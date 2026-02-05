from flask import Flask,request,jsonify,render_template;
import numpy as np;
import pandas as pd; 
from sklearn.preprocessing import StandardScaler;
import pickle;

application=Flask(__name__)
app=application;
ridge_model=pickle.load(open('models/ridgecv_model.pkl','rb'))
scalar=pickle.load(open('models/scalar_model.pkl','rb'))
@app.route("/")
def index():
    return render_template('index.html')
@app.route("/predictdata",methods=['GET','POST'])
def predict_data():
    if request.method=='POST':
        Tempareture=float(request.form.get('Tempareture'))
        RH=float(request.form.get('RH'))
        WS=float(request.form.get('WS'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        scaled_data=scalar.transform([[Tempareture,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        pred_value=ridge_model.predict(scaled_data)
        return render_template('predict_page.html',results=pred_value[0])
    else:
        return render_template('predict_page.html')

if __name__=="__main__":
    app.run(host="0.0.0.0");


