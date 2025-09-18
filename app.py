from flask import Flask, request, jsonify
from flask_cors import CORS # Import the CORS extension
import joblib
import numpy as np

# Iniciar la aplicación Flask
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes in your app

# Cargar el modelo KNN guardado
knn_model = joblib.load('ModeloCN.joblib')

@app.route('/')
def home():
    return "Servidor del modelo KNN en ejecución."

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del cuerpo de la solicitud (Request Body)
    data = request.json
    
    # Se espera que el JSON contenga una lista de 784 píxeles
    pixels = np.array(data['pixels']).reshape(1, -1)
    
    # Realizar la predicción
    prediction = knn_model.predict(pixels)
    
    # Devolver la predicción en formato JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
