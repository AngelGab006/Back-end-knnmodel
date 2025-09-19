# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar el modelo
try:
    knn_classifier = joblib.load('ModeloCN.joblib')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    knn_classifier = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if knn_classifier is None:
        return jsonify({'error': 'Modelo no disponible'}), 500

    try:
        data = request.get_json(force=True)
        pixels = np.array(data['pixels']).reshape(1, -1)

        # Realizar la predicción
        prediction = knn_classifier.predict(pixels)[0]
        
        # Obtener las probabilidades de cada clase
        probabilities = knn_classifier.predict_proba(pixels)[0]
        
        # Crear un array con las certezas
        certainty = np.round(probabilities, 4).tolist()

        # Reestructurar la matriz de píxeles a 28x28 para el frontend
        image_matrix = np.round(np.array(data['pixels']).reshape(28, 28) * 255).astype(int).tolist()

        return jsonify({
            'prediction': int(prediction),
            'certainty': certainty,
            'image_matrix': image_matrix
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)