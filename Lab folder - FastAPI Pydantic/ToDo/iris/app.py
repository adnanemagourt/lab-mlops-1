from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(title="Iris Species Predictor")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load artifacts
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LE_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

model = None
scaler = None
label_encoder = None

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

try:
    label_encoder = joblib.load(LE_PATH)
except Exception:
    label_encoder = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the iris input form."""
    return templates.TemplateResponse("iris_index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request,
                       sepal_length: float = Form(...),
                       sepal_width: float = Form(...),
                       petal_length: float = Form(...),
                       petal_width: float = Form(...)):
    if model is None:
        return templates.TemplateResponse("iris_result.html", {"request": request, "error": "Model not available on server."})

    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    pred_idx = model.predict(X)[0]
    try:
        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform([int(pred_idx)])[0]
        else:
            pred_label = str(pred_idx)
    except Exception:
        pred_label = str(pred_idx)

    return templates.TemplateResponse("iris_result.html", {"request": request, "prediction": pred_label})


@app.post("/api/predict")
async def api_predict(body: dict):
    # expects JSON with keys: sepal_length, sepal_width, petal_length, petal_width
    try:
        vals = [float(body.get(k)) for k in ("sepal_length", "sepal_width", "petal_length", "petal_width")]
    except Exception:
        return JSONResponse({"error": "Invalid input"}, status_code=400)

    if model is None:
        return JSONResponse({"error": "Model not available"}, status_code=500)

    X = np.array([vals], dtype=float)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    pred_idx = model.predict(X)[0]
    try:
        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform([int(pred_idx)])[0]
        else:
            pred_label = str(pred_idx)
    except Exception:
        pred_label = str(pred_idx)

    return JSONResponse({"prediction": pred_label})
