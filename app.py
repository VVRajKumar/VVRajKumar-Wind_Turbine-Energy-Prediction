import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import requests
import os

app = Flask(__name__)
model = joblib.load('power_prediction.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/windapi', methods=['POST'])
def windapi():
    city = request.form.get('city')
    apikey = "735e0309a1dd27399215b1687cc50ab3"
    url = f"http://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+apikey
    resp = requests.get(url)
    resp = resp.json()
    
    temp = str(round(resp["main"]["temp"] - 273.15, 2)) + " °C"
    humid = str(resp["main"]["humidity"]) + " %"
    pressure = str(resp["main"]["pressure"]) + " mmHG"
    speed = str(resp["wind"]["speed"]) + " m/s"

    return render_template('predict.html', temp=temp, humid=humid, pressure=pressure, speed=speed)

@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        # Extract form data and convert to float
        theoretical_power = float(request.form['theoretical_power'])
        wind_speed = float(request.form['wind_speed'])
        wind_direction = float(request.form['wind_direction'])

        # Create DataFrame from the extracted data
        input_data = pd.DataFrame([{
            'Theoretical_Power_Curve(KWh)': theoretical_power,
            'WindSpeed(m/s)': wind_speed,
            'Wind_Direction': wind_direction
        }])

        # Make prediction using the model
        prediction = model.predict(input_data)
        output = prediction[0]

        return render_template('predict.html', prediction_text=f'The energy predicted is {output:.2f} KWh')

    except KeyError as ke:
        error_message = f"KeyError: Missing required form field: {ke}"
        return render_template('predict.html', prediction_text=error_message)

    except ValueError as ve:
        error_message = f"ValueError: {ve}"
        return render_template('predict.html', prediction_text=error_message)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('predict.html', prediction_text=error_message)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
