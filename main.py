from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas
import tensorflow as tf
import pdfplumber
import requests
import io
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

model = tf.keras.models.load_model('models/model2-90.h5')  


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

soil_map = {0: 'Black', 1: 'Clayey', 2: 'Loamy', 3: 'Red', 4: 'Sandy'}
crop_map = {0: 'Barley', 1: 'Cotton', 2: 'Ground Nuts', 3: 'Maize', 4: 'Millets', 5: 'Oil seeds', 6: 'Paddy', 7: 'Pulses', 8: 'Sugarcane', 9: 'Tobacco', 10: 'Wheat'}
fertilizer_map = {0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}



def extract_soil_data(pdf_content):
    pdf_file = io.BytesIO(pdf_content)
    table = None
    with pdfplumber.open(pdf_file) as file:
        table = file.pages[0].extract_tables()[0]
        print(f"table : {table} tables : {len(table)} tablel[] : {table[0][:]}")
    headers = table[0][:]
    print(f"headers : {headers}")
    
    soil_type_index = headers.index('Soil Type')
    n_index,p_index,k_index = 0,0,0
    for i,h in enumerate(headers):
        if h.lower().startswith("nitrogen") or h.lower() == "n":
            n_index = i
        elif h.lower().startswith("phosphorus") or h.lower() == "p":
            p_index = i
        elif h.lower().startswith("potassium") or  h.lower() =="k":
            k_index = i
    soil_data = table[1][:]
    print(f"soil data:{soil_data} n index: {n_index} p index : {p_index} k index : {k_index}")

    soil_type = soil_data[soil_type_index].strip()
    nitrogen = float(soil_data[n_index]) 
    phosphorous = float(soil_data[p_index]) 
    potassium = float(soil_data[k_index]) 
    print(f"ty : {soil_type} n : {nitrogen} p :{phosphorous} k : {potassium}")
    return {
        "soil_type": soil_type,
        "nitrogen": nitrogen,
        "phosphorous": phosphorous,
        "potassium": potassium,
    }

def get_weather(location):
    api_url = f"http://api.weatherapi.com/v1/forecast.json?key=6bac00c67d4144e5ad2180607240809&q={location[0]},{location[1]}&days=1&aqi=no&alerts=no"
    print(api_url) 
    response = requests.get(api_url)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching weather data")
    weather_data = response.json()
    current_weather = weather_data.get('current', {})
    temperature = current_weather.get('temp_c', None)
    humidity = current_weather.get('humidity', None)
    precipitation = current_weather.get('precip_mm', None)
    data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Moisture': precipitation
    }
    print(data)
    return data

def preprocess_data(data):
    soil_encoder = LabelEncoder()
    data["Soil Type"] = soil_encoder.fit_transform(data["Soil Type"])
    print(f"soil : {data}")

    crop_encoder = LabelEncoder()
    data["Crop Type"] = crop_encoder.fit_transform(data["Crop Type"])
    print(f"crop : {data}")

    scaler = MinMaxScaler()
    numeric_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    print(f"full : {data}")

    return data

@app.get("/")
def home():
    return {"Message" : "As you can see I am not Dead"}

@app.post("/ferti-predict/")
async def predict_fertilizer(file: UploadFile = File(...),
                             lat: float = Query(..., description="Latitude of the location"), 
                            lon: float = Query(..., description="Longitude of the location"),
                            crop_type: str = Query(..., description="Type of the crop")
    ):
    print(f"lattitude : {lat} \nlongitude : {lon} \ncroptype : {crop_type}")
    pdf_content = await file.read()
    # print(f"Read file : {pdf_content}")
    soil_data = extract_soil_data(pdf_content)
    soil_type = soil_data["soil_type"]
    n = soil_data["nitrogen"]
    p = soil_data["phosphorous"]
    k = soil_data["potassium"]

    weather_data = get_weather([lat, lon])
    t = weather_data['Temperature']
    h = weather_data['Humidity']
    m = weather_data['Moisture']

    data = {
        "Temperature": [t],
        "Humidity": [h],
        "Moisture": [m],
        "Soil Type": [soil_type],
        "Crop Type": [crop_type],
        "Nitrogen": [n],
        "Potassium": [k],
        "Phosphorous": [p],
    }
    print(f"data : {data}")
    df = pandas.DataFrame(data)
    print(f"df : {df}") 

    processed_data = preprocess_data(df)
    print(f"process data : {processed_data}")

    pred = model.predict(processed_data)
    print(f"pred : {pred}")

    max_index = np.argmax(pred[0])

    return JSONResponse(content={"Response": "Function completed", "Recommended_Fertilizer": fertilizer_map[max_index]})


