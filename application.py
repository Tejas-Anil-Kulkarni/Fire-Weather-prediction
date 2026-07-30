import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template

application = Flask(__name__)
app = application

# Load the trained model and scaler
with open("models/ridge.pkl", "rb") as f:
    ridge = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict_data():
    if request.method == "POST":
        temp = float(request.form.get('temp'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        fire_class = 0 if request.form.get('classes') == 'not_fire' else 1
        region = 0 if request.form.get('region') == 'Bejaia' else 1

        scaled_data = scaler.transform([[temp,rh,ws,rain,ffmc,dmc,isi,fire_class,region]])
        predicted_data = ridge.predict(scaled_data)

        return render_template("home.html", result = predicted_data[0])
    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
