from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
from functools import lru_cache
import uvicorn
import logging
import traceback
import os
import json
import ee
from dotenv import load_dotenv

# ----------------- Load .env for local dev ------------------
load_dotenv()

# ----------------- Logging Setup ------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------- Middleware ------------------

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            body = await request.body()
            logging.info(f"➡ Request: {request.method} {request.url.path} | Body: {body.decode('utf-8') or 'No Body'}")
            response = await call_next(request)

            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            logging.info(f"⬅ Response: {response.status_code} | Body: {response_body.decode('utf-8')}")
            return Response(content=response_body, status_code=response.status_code,
                            headers=dict(response.headers), media_type=response.media_type)

        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"❌ Error in middleware: {str(e)}\n{tb}")
            raise e

# ----------------- Earth Engine Init ------------------

PROJECT_ID = "my-flood-556"
SERVICE_ACCOUNT = "gee-service@my-flood-556.iam.gserviceaccount.com"

def init_earth_engine():
    try:
        credentials = ee.ServiceAccountCredentials(
            "gee-service@my-flood-556.iam.gserviceaccount.com",
            "service-account.json"  # ← actual path to the JSON file
        )
        ee.Initialize(credentials, project=PROJECT_ID)
        logging.info("✅ Earth Engine initialized with service account.")
    except Exception as e:
        logging.error(f"❌ Failed to initialize Earth Engine: {e}")
        raise


# ----------------- App Init ------------------

app = FastAPI()
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------- Models ------------------

class Location(BaseModel):
    latitude: float
    longitude: float

# ----------------- Geometry Utils ------------------

@lru_cache()
def get_karnataka_geometry():
    logging.info("Fetching Karnataka state geometry...")
    states = ee.FeatureCollection("FAO/GAUL/2015/level1")
    filtered = states.filter(
        ee.Filter.And(
            ee.Filter.eq('ADM0_NAME', 'India'),
            ee.Filter.eq('ADM1_NAME', 'Karnataka')
        )
    )
    return filtered.geometry()

def is_within_karnataka(lat, lon):
    point = ee.Geometry.Point(lon, lat)
    result = get_karnataka_geometry().contains(point).getInfo()
    logging.info(f"Point ({lat}, {lon}) inside Karnataka? {result}")
    return result

# ----------------- Flood Analysis ------------------

def get_annual_flood_counts(region):
    logging.info("Computing annual flood counts...")
    def count_year(year):
        start = ee.Date(f"{year}-06-01")
        end = ee.Date(f"{year}-10-01")
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(region) \
            .filterDate(start, end) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
            .select('VV') \
            .sort('system:time_start')

        timestamps = s1.aggregate_array('system:time_start').getInfo()
        dates = [datetime.utcfromtimestamp(t / 1000) for t in timestamps]

        flood_events = 0
        for i in range(len(dates) - 1):
            try:
                date_before = dates[i]
                date_after = date_before + timedelta(days=12)

                img_before = s1.filterDate(date_before.strftime('%Y-%m-%d'),
                                           (date_before + timedelta(days=1)).strftime('%Y-%m-%d')).mean()
                img_after = s1.filterDate(date_after.strftime('%Y-%m-%d'),
                                          (date_after + timedelta(days=1)).strftime('%Y-%m-%d')).mean()

                diff = img_before.subtract(img_after)
                flood_mask = diff.lt(-1.5)
                flooded_pixels = flood_mask.reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=region, scale=250, bestEffort=True
                ).get('VV')
                flooded_count = ee.Number(flooded_pixels).getInfo()
                logging.info(f"{year} - Flood check {date_before.strftime('%Y-%m-%d')} => {flooded_count}")
                if flooded_count and flooded_count > 0:
                    flood_events += 1
            except Exception as e:
                logging.warning(f"⚠ Flood check failed for {date_before}: {e}")
        return flood_events

    return [count_year(y) for y in range(2020, 2023)]  # Example: 2020–2022

# ----------------- Indicators ------------------

def get_landcover_label(code):
    labels = {
        range(0, 11): "Tree cover",
        range(11, 21): "Shrubland",
        range(21, 31): "Grassland",
        range(31, 41): "Cropland",
        range(41, 51): "Built-up",
        range(51, 61): "Bare / sparse vegetation",
        range(61, 71): "Snow and ice",
        range(71, 81): "Permanent water bodies",
        range(81, 91): "Herbaceous wetland",
        range(91, 96): "Mangroves",
        range(96, 101): "Moss and lichen"
    }
    for k in labels:
        if code in k:
            return labels[k]
    return "Unknown"

def get_indicators(region):
    logging.info("Calculating environmental indicators...")
    ndwi_img = ee.ImageCollection("MODIS/006/MOD09GA") \
        .select(['sur_refl_b04', 'sur_refl_b02']) \
        .filterDate('2022-06-01', '2022-10-01') \
        .mean()
    ndwi = ndwi_img.normalizedDifference(['sur_refl_b04', 'sur_refl_b02'])
    ndwi_val = ndwi.reduceRegion(ee.Reducer.mean(), region, 500, bestEffort=True).get('nd').getInfo()

    dem = ee.Image('USGS/SRTMGL1_003')
    dem_val = dem.reduceRegion(ee.Reducer.mean(), region, 500, bestEffort=True).get('elevation').getInfo()

    landcover = ee.Image("ESA/WorldCover/v100/2020")
    land_val = landcover.reduceRegion(ee.Reducer.mode(), region, 100, bestEffort=True).get('Map').getInfo()
    land_val_int = int(land_val or 0)

    rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate('2022-06-01', '2022-10-01') \
        .filterBounds(region).sum()
    rain_val = rainfall.reduceRegion(ee.Reducer.mean(), region, 5000, bestEffort=True).get('precipitation').getInfo()

    logging.info(f"NDWI: {ndwi_val}, Rainfall: {rain_val}, Elevation: {dem_val}, Landcover: {land_val_int}")

    return {
        "ndwi": round(ndwi_val or 0, 4),
        "rainfall": round(rain_val or 0, 2),
        "elevation": round(dem_val or 0, 2),
        "landcover": {
            "code": land_val_int,
            "label": get_landcover_label(land_val_int)
        }
    }

# ----------------- Risk Classification ------------------

def classify_risk(flood_count, indicators):
    score = 0
    score += 3 if flood_count >= 15 else 2 if flood_count >= 7 else 1 if flood_count >= 2 else 0
    if indicators["ndwi"] > 0.3: score += 2
    if indicators["rainfall"] > 500: score += 2
    if indicators["elevation"] < 100: score += 2
    if indicators["landcover"]["code"] in [80, 90]: score += 1

    risk = "High" if score >= 10 else "Moderate" if score >= 5 else "Low"
    logging.info(f"Classified risk score: {score} => {risk}")
    return risk

# ----------------- Cached Prediction ------------------

@lru_cache(maxsize=256)
def cached_flood_prediction(lat: float, lon: float):
    logging.info(f"Predicting flood risk for lat={lat}, lon={lon}")
    if not is_within_karnataka(lat, lon):
        raise HTTPException(status_code=400, detail="Location is outside Karnataka")

    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(500)

    flood_counts = get_annual_flood_counts(region)
    total_flood_events = sum(flood_counts)
    indicators = get_indicators(region)
    risk_level = classify_risk(total_flood_events, indicators)

    return {
        "state": "Karnataka",
        "risk_level": risk_level,
        "flood_history": flood_counts,
        "total_flood_events": total_flood_events,
        "indicators": indicators
    }

# ----------------- API Endpoint ------------------

@app.post("/predict_flood")
def predict_flood(location: Location):
    try:
        logging.info(f"Received prediction request: {location}")
        result = cached_flood_prediction(location.latitude, location.longitude)
        return result
    except HTTPException as he:
        logging.warning(f"Client error: {he.detail}")
        raise he
    except Exception as e:
        logging.error(f"Server error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def read_root():
    return {"message": "FastAPI app is running 🚀"}

# ----------------- Run ------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
