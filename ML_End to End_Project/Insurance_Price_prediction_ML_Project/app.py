#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
import pickle

app= Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])
def result():
    sex=int(request.form['sex'])
    age=int(request.form['age'])
    bmi=float(request.form['bmi'])
    children = int(request.form['children'])
    region = int(request.form['region'])
    smoker = int(request.form['smoker'])

    x=np.array([sex,age,bmi,children,region,somker]).reshape(1,-1)

    scaler_path=os.path.join('D:\Insurance_ML_Project','models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('D:\Insurance_ML_Project','models/mlr.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for insurance orice prediction
    print(Y_pred)

if __name__=="__main__":
    app.run(debug=True,port=7384)

