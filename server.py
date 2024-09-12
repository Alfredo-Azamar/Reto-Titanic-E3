# Python Libraries

from flask import Flask, request, jsonify
import numpy as np
import joblib

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
        data['HomePlanet'],
        data['CryoSleep'],
        data['Destination'],
        data['Age'],
        data['VIP'],
        data['RoomService'],
        data['FoodCourt'],
        data['ShoppingMall'],
        data['Spa'],
        data['VRDeck'],
        data['Transported'],
    ])

    # Realizar la predicción
    result = dt.predict([inputData.reshape(-1, 1)])

    # Enviar la respuesta
    return jsonify({'Prediction': str(result[0])})

if __name__ == '__main__':
    # Iniciar la aplicación
    server.run(debug=False, host='0.0.0.0', port=8080)