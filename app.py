from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained model
model = joblib.load("C:/Users/noefo/Downloads/Race_pred_model5.joblib")

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        dff3['RIDAGEYR'], dff3['RACE'], dff3['SMOKER'],
        df3['EDUC'], df3['COUPLE'], df3['FAT'],
        df3['HTN'], df3['TOTAL_ACCULTURATION_SCORE_v2']
    ]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

# Render HTML Template
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)