from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn import preprocessing
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_CSV = os.path.join(BASE_DIR, "..", "furniture.csv")

app = FastAPI(title="Furniture Price Predictor")

# serve templates from ./templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


class PredictRequest(BaseModel):
    category: str
    sellable_online: str
    other_colors: str
    depth: float
    height: float
    width: float


def load_model_and_encoders():
    # load model
    model = joblib.load(MODEL_PATH)

    # rebuild label encoders using the original CSV so we encode inputs the same way
    df_path = os.path.join(BASE_DIR, "furniture.csv")
    if not os.path.exists(df_path):
        # try parent path (notebook location)
        df_path = os.path.join(BASE_DIR, "..", "furniture.csv")
    df_path = os.path.normpath(df_path)

    encoders = {}
    try:
        df = pd.read_csv(df_path, names=['item_id','name','category','old_price','sellable_online',
                                         'link','other_colors','short_description','designer','depth',
                                         'height','width','price'], skiprows=1, header=None)
        cat_cols = ['category', 'sellable_online', 'other_colors']
        le = preprocessing.LabelEncoder()
        mapping_dict = {}
        for c in cat_cols:
            # replace ? with NaN and fill with mode just like the notebook
            df[c] = df[c].replace("?", pd.NA)
            df[c] = df[c].fillna(df[c].mode().iloc[0])
            enc = preprocessing.LabelEncoder()
            enc.fit(df[c].astype(str))
            encoders[c] = enc
            mapping_dict[c] = dict(zip(enc.classes_, enc.transform(enc.classes_)))
    except Exception:
        # if CSV not present or reading fails, create passthrough encoders that expect numeric labels
        for c in ['category', 'sellable_online', 'other_colors']:
            encoders[c] = None

    return model, encoders


model, encoders = load_model_and_encoders()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # provide choices for categorical fields if encoders were built
    choices = {}
    for k, enc in encoders.items():
        if enc is not None:
            choices[k] = list(map(str, enc.classes_))
        else:
            choices[k] = []
    return templates.TemplateResponse("index.html", {"request": request, "choices": choices})


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request,
                       category: str = Form(...),
                       sellable_online: str = Form(...),
                       other_colors: str = Form(...),
                       depth: float = Form(...),
                       height: float = Form(...),
                       width: float = Form(...)):
    # prepare features in the order used in the notebook
    features = [category, sellable_online, other_colors, depth, height, width]

    X = []
    # encode categorical if encoder available, otherwise assume numeric
    for i, col in enumerate(['category', 'sellable_online', 'other_colors']):
        enc = encoders.get(col)
        val = features[i]
        if enc is not None:
            try:
                enc_val = int(enc.transform([str(val)])[0])
            except Exception:
                # if unseen label, append as 0
                enc_val = 0
        else:
            enc_val = float(val)
        X.append(enc_val)

    # append numeric values
    X.extend([float(depth), float(height), float(width)])

    pred = model.predict([X])[0]
    pred = float(pred)

    return templates.TemplateResponse("result.html", {"request": request, "prediction": round(pred, 2)})


@app.post("/api/predict")
async def api_predict(body: PredictRequest):
    # do the same preprocessing
    inputs = [body.category, body.sellable_online, body.other_colors, body.depth, body.height, body.width]
    X = []
    for i, col in enumerate(['category', 'sellable_online', 'other_colors']):
        enc = encoders.get(col)
        val = inputs[i]
        if enc is not None:
            try:
                enc_val = int(enc.transform([str(val)])[0])
            except Exception:
                enc_val = 0
        else:
            enc_val = float(val)
        X.append(enc_val)
    X.extend([float(body.depth), float(body.height), float(body.width)])
    pred = model.predict([X])[0]
    return JSONResponse({"prediction": float(pred)})
