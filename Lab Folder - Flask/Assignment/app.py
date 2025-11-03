from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
LE_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(LE_PATH, 'rb') as f:
    le = pickle.load(f)

FEATURE_LABELS = [
    ('sepal_length', 'Sepal length (cm)'),
    ('sepal_width', 'Sepal width (cm)'),
    ('petal_length', 'Petal length (cm)'),
    ('petal_width', 'Petal width (cm)'),
]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', feature_labels=FEATURE_LABELS)


@app.route('/predict', methods=['POST'])
def predict():
    # Read inputs
    try:
        values = []
        for key, _ in FEATURE_LABELS:
            v = request.form.get(key, None)
            if v is None or v.strip() == '':
                return render_template('result.html', error=f'Missing input for {key}')
            values.append(float(v))
    except ValueError:
        return render_template('result.html', error='Please enter numeric values for all fields.')

    arr = np.array(values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    pred = model.predict(arr_scaled)
    try:
        pred_label = le.inverse_transform(pred)[0]
    except Exception:
        pred_label = str(pred[0])

    probability = None
    if hasattr(model, 'predict_proba'):
        probability = float(model.predict_proba(arr_scaled).max())

    return render_template('result.html', prediction=pred_label, probability=probability, values=values, feature_labels=FEATURE_LABELS)


if __name__ == '__main__':
    # For local testing only. Use a production server in deployment.
    app.run(host='127.0.0.1', port=5000, debug=True)
