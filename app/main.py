import pandas as pd
from pydantic import BaseModel
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app import models

app = FastAPI()

# HTML templates folder (index.html lives here)
templates = Jinja2Templates(directory="templates")

# load the trained model + preprocessing info (scaler, mappings, columns...)
art = joblib.load("models/top10_stack_model.joblib")

class HouseFeatures(BaseModel):
    """The 10 features the compact model needs. Field types/values match train.csv."""
    overall_qual: int
    gr_liv_area: int
    exter_qual: str
    garage_cars: int
    kitchen_qual: str
    total_bsmt_sf: int
    year_built: int
    first_flr_sf: int
    garage_finish: str
    full_bathrooms: int

def predict_single_house(raw_input, art):

    """Predict one house price from a raw dict."""
    # put the input in a one-row DataFrame
    raw_df = pd.DataFrame([raw_input])

    # drop columns the model was not trained on
    raw_df = raw_df.drop(columns=art["drop_columns"], errors="ignore")

    # fill missing categorical values (e.g. "None")
    raw_df = raw_df.fillna(art["cat_na_fills"])

    # convert quality ratings to numbers (e.g. Gd -> 4)
    for col, mapping in art["ordinal_mappings"].items():
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].map(mapping)

    # one-hot encode the nominal categories
    raw_df = pd.get_dummies(
        raw_df,
        columns=[c for c in art['nominal_columns'] if c in raw_df.columns],
        dtype=int
        )

    # fill any remaining missing numbers with the training medians
    raw_df = raw_df.fillna(art["train_medians"])

    # keep exactly the columns the model expects, in the same order
    raw_df = raw_df.reindex(
        columns=art["feature_columns"],
        fill_value=0)

    # scale the features the same way as in training
    raw_df[art["scale_features"]] = art["scaler"].transform(
        raw_df[art["scale_features"]]
    )

    return float(art["model"].predict(raw_df)[0])


# predict a price, save the request + result to the DB, return the price
@app.post("/predict")
def predict(input_data: HouseFeatures):
    predicted_price = predict_single_house(input_data.dict(), art)

    data = input_data.dict()
    data["predicted_price"] = predicted_price

    models.save_data_to_db(data)

    return {'predicted_price': predicted_price}

# serve the web form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")
