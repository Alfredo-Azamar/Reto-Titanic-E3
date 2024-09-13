# Python Libraries

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# File management
import os
from werkzeug.utils import secure_filename

server = Flask(__name__)
CORS(server)

# Loading the model
dt = joblib.load('dt1.joblib')
scaler = joblib.load('scaler.joblib')

# Creating the Flask
server = Flask(__name__)

# Defining a route to send JSON data
@server.route('/predict', methods=['POST'])
def predictJSON():
    # Procesar los datos de entrada
    data = request.json
    print(data)
    inputData = np.array([
        data['RoomService'],
        data['FoodCourt'],
        data['ShoppingMall'],
        data['Spa'],
        data['VRDeck'],
        data['Age'],
        data['VIP_True'],
        data['CryoSleep_True'],
        data['HomePlanet_Mars'],
        data['Destination_PSO J318.5-22'],
        data['Destination_TRAPPIST-1e']
    ])

    # OG
    # inputData = inputData.reshape(1, -1)
    # prediction = dt.predict(inputData)
    # return jsonify({'Prediction': str(prediction[0])})

    inputData = inputData.reshape(1, -1)
    inputData_scaled = scaler.transform(inputData)

    prediction = dt.predict(inputData_scaled)
    return jsonify({'Prediction': str(prediction[0])})

if __name__ == '__main__':
    # Iniciar la aplicación
    server.run(debug=False, host='0.0.0.0', port=8080)