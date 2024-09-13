### BACKEND SERVER ###

## Python Libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

## File management
import os
from werkzeug.utils import secure_filename

## Server configuration
server = Flask(__name__)
CORS(server)

## Loading the model
dt = joblib.load('\Modelo\dt1_V5.joblib')
scaler = joblib.load('\Modelo\dt1_scaler_V5.joblib')

## Extracting age relation with spending
# Cargar el dataset que contiene los porcentajes por edad
porcentajes = pd.read_csv('\Modelo\porcentaje_gastos_edad.csv')


def calcular_gastos_por_edad_y_presupuesto(edad, presupuesto):
    # Find the row corresponding to the age
    edad = float(edad)
    presupuesto = float(presupuesto)
    fila_edad = porcentajes[porcentajes['Age'] == edad]
    
     # Check if the age exists in the dataset
    if fila_edad.empty:
        return f"No se encontraron datos para la edad {edad}."

    # Extract the percentages for that age
    room_service_pct = fila_edad['RoomService%'].values[0]
    food_court_pct = fila_edad['FoodCourt%'].values[0]
    shopping_mall_pct = fila_edad['ShoppingMall%'].values[0]
    spa_pct = fila_edad['Spa%'].values[0]
    vr_deck_pct = fila_edad['VRDeck%'].values[0]

    # Calculate the spending on each attribute according to the given budget
    room_service_gasto = presupuesto * room_service_pct
    food_court_gasto = presupuesto * food_court_pct
    shopping_mall_gasto = presupuesto * shopping_mall_pct
    spa_gasto = presupuesto * spa_pct
    vr_deck_gasto = presupuesto * vr_deck_pct

    # Format the results to two decimal places
    return [
        float(f"{room_service_gasto:.2f}"),
        float(f"{food_court_gasto:.2f}"),
        float(f"{shopping_mall_gasto:.2f}"),
        float(f"{spa_gasto:.2f}"),
        float(f"{vr_deck_gasto:.2f}")
    ]


## Defining a route to send JSON data
@server.route('/predict', methods=['POST'])
def predictJSON():
    # Process the input data
    data = request.json
    print(data)

    age = data['Age']
    budget = data['Budget']

    # Calculate the spending by service according to age and budget
    gastos = calcular_gastos_por_edad_y_presupuesto(age, budget)

    # Extract the calculated spending for each service
    room_service_gasto = (gastos[0])
    food_court_gasto = float(gastos[1])
    shopping_mall_gasto = float(gastos[2])
    spa_gasto = float(gastos[3])
    vr_deck_gasto = float(gastos[4])

    inputData = np.array([
        room_service_gasto,
        food_court_gasto,
        shopping_mall_gasto,
        spa_gasto,
        vr_deck_gasto,
        data['Age'],
        data['VIP_True'],
        data['CryoSleep_True'],
        data['HomePlanet_Mars'],
        data['Destination_PSO J318.5-22'],
        data['Destination_TRAPPIST-1e']
    ])

    print(inputData)

    inputData = inputData.reshape(1, -1)
    inputData_scaled = scaler.transform(inputData)

    prediction = dt.predict(inputData_scaled)
    return jsonify({'Prediction': str(prediction[0])})

if __name__ == '__main__':
    # Start the server
    server.run(debug=False, host='0.0.0.0', port=8080)