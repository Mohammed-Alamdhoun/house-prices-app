import os
import dotenv
from sqlalchemy import text
from sqlalchemy import create_engine

# read DATABASE_URL from the .env file
dotenv.load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(SQLALCHEMY_DATABASE_URL)

# insert one prediction (features + result) into the predictions table
def save_data_to_db(data):

    query = text("""
    INSERT INTO predictions (
        overall_qual,
        gr_liv_area,
        exter_qual,
        garage_cars,
        kitchen_qual,
        total_bsmt_sf,
        year_built,
        first_flr_sf,
        garage_finish,
        full_bathrooms,
        predicted_price
    )
    VALUES (
        :overall_qual,
        :gr_liv_area,
        :exter_qual,
        :garage_cars,
        :kitchen_qual,
        :total_bsmt_sf,
        :year_built,
        :first_flr_sf,
        :garage_finish,
        :full_bathrooms,
        :predicted_price
    )
    """)

    with engine.connect() as conn:
        conn.execute(query, data)
        conn.commit()

    return {
        "message": "Data saved successfully"
    }