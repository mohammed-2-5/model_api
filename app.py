# app.py
"""
FastAPI service
• Parkinson – numeric & drawing
• Wilson's disease – numeric (uses saved StandardScaler)
• Liver disease – numeric (handles invisible spaces in feature names)
• Colorectal cancer – numeric (with categorical encoding)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # hide TF info banners

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import joblib
import io

from PIL import Image
from google.cloud import storage
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

def clean_column(col):
    """Remove all types of whitespace and invisible characters from start/end of feature name."""
    return col.replace('\xa0', '').strip()

# ── Google Cloud Storage settings ──────────────────────────────────────
BUCKET_NAME = "parkinson_models"
FILES = {
    # Parkinson artefacts
    "model.pkl":          "model.pkl",
    "scaler.pkl":         "scaler.pkl",
    "drawings.keras":     "drawings.keras",
    # Wilson artefacts
    "wilson_model.pkl":   "wilson_model.pkl",
    "wilson_scaler.pkl":  "wilson_scaler.pkl",
    # Liver artefacts
    "liver_model.pkl":    "liver_model.pkl",
    "liver_scaler.pkl":   "liver_scaler.pkl",
    # Colorectal cancer artefacts
    "colorectal_model.pkl": "colorectal_model.pkl",
}

def download_models_from_gcs() -> None:
    """Download all artefacts once at startup."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    os.makedirs("models", exist_ok=True)

    for filename, blob_name in FILES.items():
        local_path = f"models/{filename}"
        if not os.path.exists(local_path):
            print(f"🔽  Downloading {filename} …")
            bucket.blob(blob_name).download_to_filename(local_path)

# ── FastAPI lifespan: load artefacts into memory ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pd_model, pd_scaler, cnn, wilson_model, wilson_scaler, liver_model, liver_scaler, colorectal_model

    download_models_from_gcs()

    # Parkinson
    pd_model   = joblib.load("models/model.pkl")
    pd_scaler  = joblib.load("models/scaler.pkl")
    cnn        = load_model("models/drawings.keras")

    # Wilson
    wilson_model  = joblib.load("models/wilson_model.pkl")
    wilson_scaler = joblib.load("models/wilson_scaler.pkl")
   
    # Liver
    liver_model  = joblib.load("models/liver_model.pkl")
    liver_scaler = joblib.load("models/liver_scaler.pkl")
    # Clean feature names in scaler immediately after loading!
    liver_scaler.feature_names_in_ = [clean_column(f) for f in liver_scaler.feature_names_in_]

    # Colorectal cancer
    colorectal_model = joblib.load("models/colorectal_model.pkl")

    yield

app = FastAPI(lifespan=lifespan)

# ── tiny redirect so visiting "/" opens Swagger UI ─────────────────────
@app.get("/", include_in_schema=False)
def _root():
    return RedirectResponse(url="/docs", status_code=308)

# ═══════════════════════════════════════════════════════════════════════
#                P  A  R  K  I  N  S  O  N   E N D P O I N T S
# ═══════════════════════════════════════════════════════════════════════

class PDInput(BaseModel):
    UPDRS: float
    FunctionalAssessment: float
    MoCA: float
    Tremor: int
    Rigidity: int
    Bradykinesia: int
    Age: int
    AlcoholConsumption: float
    BMI: float
    SleepQuality: float
    DietQuality: float
    CholesterolTriglycerides: float

@app.post("/predict", tags=["Parkinson – numeric"])
async def predict_parkinson(data: PDInput):
    df = pd.DataFrame([data.dict()])
    df = df.reindex(columns=pd_scaler.feature_names_in_, fill_value=0)
    scaled = pd_scaler.transform(df)
    pred   = int(pd_model.predict(scaled)[0])
    msg    = (
        "The person has Parkinson disease"
        if pred == 1
        else "The person does not have Parkinson disease"
    )
    return {"prediction_class": "parkinson" if pred else "healthy",
            "prediction_value": pred,
            "result": msg}

@app.post("/predict_image", tags=["Parkinson – drawing"])
async def predict_parkinson_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())) \
                 .convert("RGB") \
                 .resize((64, 64))
    arr   = np.expand_dims(np.array(image), axis=0)
    value = float(cnn.predict(arr)[0][0])
    label = "healthy" if value < 0.5 else "parkinson"
    return {"prediction_class": label,
            "prediction_value": value}

# ═══════════════════════════════════════════════════════════════════════
#                W  I  L  S  O  N ’ S   D  I  S  E  A  S  E
# ═══════════════════════════════════════════════════════════════════════

class WilsonInput(BaseModel):
    Age: int
    ATB7B_Gene_Mutation: int = Field(..., alias="ATB7B Gene Mutation")
    Kayser_Fleischer_Rings: int = Field(..., alias="Kayser-Fleischer Rings")
    Copper_in_Blood_Serum: float = Field(..., alias="Copper in Blood Serum")
    Copper_in_Urine: float = Field(..., alias="Copper in Urine")
    Neurological_Symptoms_Score: float = Field(..., alias="Neurological Symptoms Score")
    Ceruloplasmin_Level: float = Field(..., alias="Ceruloplasmin Level")
    AST: float
    ALT: float
    Family_History: int = Field(..., alias="Family History")
    Gamma_Glutamyl_Transferase: float = Field(..., alias="Gamma-Glutamyl Transferase (GGT)")
    Total_Bilirubin: float

    class Config:
        validate_by_name = True    # renamed key in Pydantic v2
        extra = "allow"            # accept the other 11 features the model saw

@app.post("/predict_wilson", tags=["Wilson disease – numeric"])
async def predict_wilson(data: WilsonInput):
    df = pd.DataFrame([data.dict(by_alias=True)])
    df = df.reindex(columns=wilson_scaler.feature_names_in_, fill_value=0)
    scaled = wilson_scaler.transform(df)
    prob   = float(wilson_model.predict(scaled)[0])
    has_disease = int(round(prob))
    return {
        "prediction_class": "wilson" if has_disease else "healthy",
        "prediction_value": prob,
        "result": (
            "The person has Wilson's disease."
            if has_disease
            else "The person does not have Wilson's disease."
        ),
    }

# ═══════════════════════════════════════════════════════════════════════
#                    L i v e r    E N D P O I N T S
# ═══════════════════════════════════════════════════════════════════════
class LiverInput(BaseModel):
    Total_Bilirubin: float = Field(..., alias="Total Bilirubin")
    Direct_Bilirubin: float = Field(..., alias="Direct Bilirubin")
    Alkphos_Alkaline_Phosphatase: float = Field(..., alias="Alkphos Alkaline Phosphotase")
    Sgpt_Alamine_Aminotransferase: float = Field(..., alias="Sgpt Alamine Aminotransferase")
    Sgot_Aspartate_Aminotransferase: float = Field(..., alias="Sgot Aspartate Aminotransferase")
    ALB_Albumin: float = Field(..., alias="ALB Albumin")
    AG_Ratio_Albumin_and_Globulin_Ratio: float = Field(..., alias="A/G Ratio Albumin and Globulin Ratio")
    Total_Protiens: float = Field(..., alias="Total Protiens")

    class Config:
        # For Pydantic v1
        allow_population_by_field_name = True
        # For Pydantic v2
        populate_by_name = True
        extra = "allow"

@app.post("/predict_liver")
async def predict_liver(data: dict):
    import pandas as pd

    # Convert incoming dict to DataFrame
    input_df = pd.DataFrame([data])

    # Align columns exactly to what the model expects
    input_df = input_df.reindex(columns=liver_model.feature_names_in_, fill_value=0)

    # Predict directly (no scaling!)
    pred = liver_model.predict(input_df)[0]

    msg = "The person has liver disease" if pred == 1 else "The person does not have liver disease"
    return {
        "prediction_class": "liver" if pred else "healthy",
        "prediction_value": int(pred),
        "result": msg
    }

# ═══════════════════════════════════════════════════════════════════════
#              C O L O R E C T A L   C A N C E R   E N D P O I N T S
# ═══════════════════════════════════════════════════════════════════════

class ColorectalInput(BaseModel):
    Age: int = Field(..., description="Patient age in years", example=65)
    Gender: str = Field(..., description="Patient gender", example="Male", enum=["Male", "Female"])
    BMI: float = Field(..., description="Body Mass Index", example=28.5)
    Lifestyle: str = Field(..., description="Lifestyle category", example="Smoker", 
                          enum=["Sedentary", "Active", "Moderate", "Smoker", "Non-smoker", "Athlete"])
    Family_History_CRC: str = Field(..., alias="Family_History_CRC", 
                                   description="Family history of colorectal cancer", 
                                   example="Yes", enum=["Yes", "No"])
    Pre_existing_Conditions: str = Field(..., alias="Pre-existing Conditions",
                                        description="Pre-existing medical conditions",
                                        example="Diabetes",
                                        enum=["None", "Diabetes", "Hypertension", "Heart Disease", "Obesity", "IBD", "Polyps"])
    Carbohydrates_g: float = Field(..., alias="Carbohydrates (g)", 
                                  description="Daily carbohydrate intake in grams", example=300.0)
    Proteins_g: float = Field(..., alias="Proteins (g)", 
                             description="Daily protein intake in grams", example=80.0)
    Fats_g: float = Field(..., alias="Fats (g)", 
                         description="Daily fat intake in grams", example=50.0)
    Vitamin_A_IU: float = Field(..., alias="Vitamin A (IU)", 
                               description="Daily Vitamin A intake in International Units", example=3000.0)
    Vitamin_C_mg: float = Field(..., alias="Vitamin C (mg)", 
                               description="Daily Vitamin C intake in milligrams", example=60.0)
    Iron_mg: float = Field(..., alias="Iron (mg)", 
                          description="Daily iron intake in milligrams", example=12.0)

    class Config:
        populate_by_name = True
        extra = "allow"
        schema_extra = {
            "examples": [
                {
                    "title": "High Risk Patient",
                    "description": "Elderly male smoker with multiple risk factors",
                    "value": {
                        "Age": 76,
                        "Gender": "Male",
                        "BMI": 31.4,
                        "Lifestyle": "Smoker",
                        "Family_History_CRC": "Yes",
                        "Pre-existing Conditions": "Obesity",
                        "Carbohydrates (g)": 398.0,
                        "Proteins (g)": 56.0,
                        "Fats (g)": 61.0,
                        "Vitamin A (IU)": 2500.0,
                        "Vitamin C (mg)": 38.0,
                        "Iron (mg)": 9.5
                    }
                },
                {
                    "title": "Low Risk Patient",
                    "description": "Young active female with healthy lifestyle",
                    "value": {
                        "Age": 32,
                        "Gender": "Female",
                        "BMI": 22.1,
                        "Lifestyle": "Active",
                        "Family_History_CRC": "No",
                        "Pre-existing Conditions": "None",
                        "Carbohydrates (g)": 220.0,
                        "Proteins (g)": 95.0,
                        "Fats (g)": 35.0,
                        "Vitamin A (IU)": 5200.0,
                        "Vitamin C (mg)": 125.0,
                        "Iron (mg)": 18.0
                    }
                },
                {
                    "title": "Medium Risk Patient",
                    "description": "Middle-aged person with some risk factors",
                    "value": {
                        "Age": 55,
                        "Gender": "Male",
                        "BMI": 26.8,
                        "Lifestyle": "Sedentary",
                        "Family_History_CRC": "No",
                        "Pre-existing Conditions": "Diabetes",
                        "Carbohydrates (g)": 280.0,
                        "Proteins (g)": 70.0,
                        "Fats (g)": 45.0,
                        "Vitamin A (IU)": 3800.0,
                        "Vitamin C (mg)": 75.0,
                        "Iron (mg)": 14.0
                    }
                }
            ]
        }

@app.post("/predict_colorectal", tags=["Colorectal Cancer – numeric"])
async def predict_colorectal(data: ColorectalInput):
    """
    Predict colorectal cancer risk based on demographic, lifestyle, and nutritional factors.
    Returns: healthy or at_risk
    """
    # Convert input to dictionary and then to DataFrame
    input_dict = data.dict(by_alias=True)
    input_df = pd.DataFrame([input_dict])
    
    # Get expected columns from model
    if hasattr(colorectal_model, 'feature_names_in_'):
        expected_columns = colorectal_model.feature_names_in_.tolist()
    else:
        expected_columns = [
            'Age', 'Gender', 'BMI', 'Lifestyle', 'Family_History_CRC',
            'Pre-existing Conditions', 'Carbohydrates (g)', 'Proteins (g)',
            'Fats (g)', 'Vitamin A (IU)', 'Vitamin C (mg)', 'Iron (mg)'
        ]
    
    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Handle categorical encoding
    # Gender
    if 'Gender' in input_df.columns:
        gender_map = {'Female': 0, 'Male': 1}
        input_df['Gender'] = input_df['Gender'].map(gender_map).fillna(0)
    
    # Lifestyle
    if 'Lifestyle' in input_df.columns:
        lifestyle_map = {
            'Sedentary': 0, 'Active': 1, 'Moderate': 2, 
            'Smoker': 3, 'Non-smoker': 1, 'Athlete': 4
        }
        input_df['Lifestyle'] = input_df['Lifestyle'].map(lifestyle_map).fillna(0)
    
    # Family History
    if 'Family_History_CRC' in input_df.columns:
        family_map = {'No': 0, 'Yes': 1}
        input_df['Family_History_CRC'] = input_df['Family_History_CRC'].map(family_map).fillna(0)
    
    # Pre-existing Conditions
    if 'Pre-existing Conditions' in input_df.columns:
        condition_map = {
            'None': 0, 'Diabetes': 1, 'Hypertension': 2, 
            'Heart Disease': 3, 'Obesity': 4, 'IBD': 5, 'Polyps': 6
        }
        input_df['Pre-existing Conditions'] = input_df['Pre-existing Conditions'].map(condition_map).fillna(0)
    
    # Convert all to numeric
    for col in expected_columns:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
    
    # Reorder columns to match model expectations
    input_df = input_df[expected_columns]
    
    # Make prediction
    try:
        pred = int(colorectal_model.predict(input_df)[0])
        
        if pred == 1:
            result_class = "at_risk"
            message = "The person is at risk for colorectal cancer"
        else:
            result_class = "healthy"
            message = "The person is not at risk for colorectal cancer"
        
        return {
            "prediction": result_class,
            "message": message,
            "risk_score": pred
        }
        
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "prediction": "unknown",
            "message": "Unable to make prediction due to data processing error"
        }

# ═══════════════════════════════════════════════════════════════════════