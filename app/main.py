import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

art = joblib.load("models/top10_stack_model.joblib")

class HouseFeatures(BaseModel):
    """The 10 features the compact model needs. Field types/values match train.csv."""
    OverallQual: int
    GrLivArea: int
    ExterQual: str
    GarageCars: int
    KitchenQual: str
    TotalBsmtSF: int
    YearBuilt: int
    FirstFlrSF: int
    GarageFinish: str
    FullBath: int

def predict_single_house(raw_input, art):
    """Predict one house price from a raw dict."""
    raw_df = pd.DataFrame([raw_input])
    raw_df = raw_df.drop(columns=art["drop_columns"], errors="ignore")
    raw_df = raw_df.fillna(art["cat_na_fills"])

    for col, mapping in art["ordinal_mappings"].items():
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].map(mapping)

    raw_df = pd.get_dummies(
        raw_df,
        columns=[c for c in art['nominal_columns'] if c in raw_df.columns],
        dtype=int
        )

    raw_df = raw_df.fillna(art["train_medians"])

    raw_df = raw_df.reindex(
        columns=art["feature_columns"],
        fill_value=0)

    raw_df[art["scale_features"]] = art["scaler"].transform(
        raw_df[art["scale_features"]]
    )

    return float(art["model"].predict(raw_df)[0])


app  = FastAPI()
@app.post("/predict")
def predict(input_data: HouseFeatures):
    print("Received input data:", input_data.dict())
    return predict_single_house(input_data.dict(),art)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
