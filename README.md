[README.md](https://github.com/user-attachments/files/30136549/README.md)
# House Price Predictor

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-almadhoun%2Fhouse--price--app-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/almadhoun/house-price-app)

An end-to-end machine-learning project for the [Kaggle *House Prices – Advanced Regression Techniques*](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset (Ames, Iowa housing data).

It covers the full lifecycle: exploratory analysis and feature engineering in notebooks, hyperparameter tuning with Optuna, a stacked ensemble model, and a **FastAPI** web service with a Bootstrap UI that serves live predictions and logs every request to a PostgreSQL database.

---

## What it does

- **Trains** a stacked ensemble regressor (CatBoost + XGBoost + scikit-learn gradient boosting / bagging) to predict house sale prices.
- **Tunes** hyperparameters with Optuna and freezes the best configuration to `models/best_params.txt`.
- **Serves** a compact 10-feature version of the model through a REST API and a web form.
- **Persists** each prediction (inputs + result) to a PostgreSQL table.

---

## Project structure

```
unlimited_involution_task/
├── app/
│   ├── main.py            # FastAPI app: GET / (web form) + POST /predict
│   └── models.py          # SQLAlchemy engine + save prediction to Postgres
├── templates/
│   └── index.html         # Bootstrap web form (calls POST /predict)
├── notebooks/
│   ├── house-prices-end-to-end-regression-pipeline.ipynb   # full pipeline
│   └── house-prices-top10-features-model.ipynb             # compact 10-feature model
├── models/
│   ├── final_stack_model.joblib     # full-feature stacking model
│   ├── top10_stack_model.joblib     # compact 10-feature model (served by the API)
│   └── best_params.txt              # frozen Optuna hyperparameters
├── data/
│   ├── train.csv                    # Kaggle training data
│   └── test.csv                     # Kaggle test data
├── outputs/
│   └── submission.csv               # Kaggle submission (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                             # DATABASE_URL (git-ignored)
```

---

## The model

Two artifacts are produced by the notebooks:

| Artifact                     | Features         | Purpose                                        | Hold-out RMSE | Hold-out R² |
| ---------------------------- | ---------------- | ---------------------------------------------- | ------------- | ------------ |
| `final_stack_model.joblib` | full feature set | Kaggle submission (`outputs/submission.csv`) | ≈ 28,253     | ≈ 0.913     |
| `top10_stack_model.joblib` | top 10 features  | Served by the web API                          | ≈ 32,066     | ≈ 0.888     |

Each artifact is a bundle that stores not just the fitted model but everything needed to predict from **raw** inputs: the scaler, ordinal mappings, one-hot columns, training medians, and the expected feature order. The API's `predict_single_house()` replays this exact preprocessing on incoming requests.

The web app serves the **top-10 model** so the form stays short. The 10 features (the most important ones by the model's own feature importances) are:

| Field              | Type | Description                       | Allowed / range                        |
| ------------------ | ---- | --------------------------------- | -------------------------------------- |
| `overall_qual`   | int  | Overall material & finish quality | 1–10                                  |
| `gr_liv_area`    | int  | Above-grade living area (sq ft)   | ≥ 0                                   |
| `exter_qual`     | str  | Exterior material quality         | `Ex`, `Gd`, `TA`, `Fa`, `Po` |
| `garage_cars`    | int  | Garage capacity (cars)            | 1–5                                   |
| `kitchen_qual`   | str  | Kitchen quality                   | `Ex`, `Gd`, `TA`, `Fa`, `Po` |
| `total_bsmt_sf`  | int  | Total basement area (sq ft)       | ≥ 0                                   |
| `year_built`     | int  | Original construction year        | e.g. 1800–2026                        |
| `first_flr_sf`   | int  | First floor area (sq ft)          | ≥ 0                                   |
| `garage_finish`  | str  | Interior garage finish            | `Fin`, `RFn`, `Unf`, `NA`      |
| `full_bathrooms` | int  | Full bathrooms above grade        | 1–5                                   |

> **Note:** the `predictions` table has `CHECK` constraints that reject `0` for `garage_cars` and `full_bathrooms`, so use `≥ 1` for those two fields.

---

## Getting started

### Prerequisites

- Python 3.13
- A PostgreSQL database (the project was built against [Neon](https://neon.tech)). A local Postgres works too.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the database

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME
```

The API expects a `predictions` table to already exist. Create it once:

```sql
CREATE TABLE predictions (
    id              SERIAL PRIMARY KEY,
    overall_qual    INTEGER NOT NULL,
    gr_liv_area     INTEGER NOT NULL,
    exter_qual      TEXT    NOT NULL,
    garage_cars     INTEGER NOT NULL CHECK (garage_cars > 0),
    kitchen_qual    TEXT    NOT NULL,
    total_bsmt_sf   INTEGER NOT NULL,
    year_built      INTEGER NOT NULL,
    first_flr_sf    INTEGER NOT NULL,
    garage_finish   TEXT    NOT NULL,
    full_bathrooms  INTEGER NOT NULL CHECK (full_bathrooms > 0),
    predicted_price DOUBLE PRECISION NOT NULL
);
```

> `/predict` writes to the database before returning, so a reachable `DATABASE_URL` and this table are required for the endpoint to succeed.

### 3. Run the API

Run from the **project root** (model and template paths are relative to the working directory):

```bash
uvicorn app.main:app --reload
```

Then open **http://localhost:8000** for the web form, or **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Run with Docker

The image bundles the API, templates, and the top-10 model. `DATABASE_URL` is injected at runtime from `.env` (never baked into the image), so create your `.env` (see [Configure the database](#2-configure-the-database)) before running.

### Option A — pull the pre-built image from Docker Hub

The image is published at [`almadhoun/house-price-app`](https://hub.docker.com/r/almadhoun/house-price-app):

```bash
docker pull almadhoun/house-price-app
docker run -p 8000:8000 --env-file .env almadhoun/house-price-app
```

### Option B — build it locally

```bash
docker compose up --build
```

Either way, the service is available at **http://localhost:8000**.

---

## API reference

### `GET /`

Serves the HTML prediction form.

### `POST /predict`

Predicts a sale price, stores the request + result, and returns the price.

**Request body**

```json
{
  "overall_qual": 7,
  "gr_liv_area": 1710,
  "exter_qual": "Gd",
  "garage_cars": 2,
  "kitchen_qual": "Gd",
  "total_bsmt_sf": 856,
  "year_built": 2003,
  "first_flr_sf": 856,
  "garage_finish": "RFn",
  "full_bathrooms": 2
}
```

**Response**

```json
{ "predicted_price": 197824.41 }
```

---

## Reproducing the model

The notebooks in `notebooks/` regenerate the artifacts. Run them from inside the `notebooks/` folder (paths are relative to it, e.g. `../data/`, `../models/`).

1. **`house-prices-end-to-end-regression-pipeline.ipynb`** — the full pipeline: EDA, missing-value handling, ordinal + one-hot encoding, scaling, 5-fold cross-validation, Optuna tuning, the final stacked ensemble → saves `final_stack_model.joblib` and `outputs/submission.csv`.
2. **`house-prices-top10-features-model.ipynb`** — retrains the same stack on the 10 most important features and saves `top10_stack_model.joblib`, the artifact the API serves.

> Optuna hyperparameters are already frozen in `models/best_params.txt`, so you can skip the (slow) tuning cells and reuse them directly.

---

## Tech stack

- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, pandas, NumPy
- **API / web:** FastAPI, Uvicorn, Jinja2, Bootstrap 5
- **Database:** PostgreSQL (Neon) via SQLAlchemy
- **Packaging:** Docker, Docker Compose

---

## Data

Place the Kaggle competition files in `data/` as `train.csv` and `test.csv`. They are available from the [competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
