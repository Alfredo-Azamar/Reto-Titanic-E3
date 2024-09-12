# Python Libraries

from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# File management
import os
from werkzeug.utils import secure_filename

# Loading the model
dt = joblib.load('dt1.joblib')

# Creating the Flask appp0'´ñ{-´ñp 5c}
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

    #Sosa's code
    # Realizar la predicción
    # result = dt.predict([inputData.reshape(-1, 1)])

    # Enviar la respuesta
    # return jsonify({'Prediction': str(result[0])})
    scaler = MinMaxScaler()
    inputData = scaler.transform(inputData.reshape(1, -1))
    prediction = dt.predict(inputData)
    return jsonify({'Prediction': str(prediction[0])})

if __name__ == '__main__':
    # Iniciar la aplicación
    server.run(debug=False, host='0.0.0.0', port=8080)