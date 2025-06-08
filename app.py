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
#                W  I  L  S  O  N ' S   D  I  S  E  A  S  E
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

def _prepare_wilson_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Wilson's disease data from Excel for prediction:
    • Normalizes column names
    • Adds missing columns with default 0
    • Scales features using the pre-trained scaler
    Returns a cleaned DataFrame ready for model.predict()
    """
    # Map Excel columns to model's expected feature names
    column_map = {
        "Age": "Age",
        "ATB7B Gene Mutation": "ATB7B Gene Mutation",
        "Kayser-Fleischer Rings": "Kayser-Fleischer Rings",
        "Copper in Blood Serum": "Copper in Blood Serum",
        "Copper in Urine": "Copper in Urine",
        "Neurological Symptoms Score": "Neurological Symptoms Score",
        "Ceruloplasmin Level": "Ceruloplasmin Level",
        "AST": "AST",
        "ALT": "ALT",
        "Family History": "Family History",
        "Gamma-Glutamyl Transferase (GGT)": "Gamma-Glutamyl Transferase (GGT)",
        "Total_Bilirubin": "Total Bilirubin"
    }

    # Rename columns according to mapping
    df = raw_df.copy()
    df.columns = [col.strip() for col in df.columns]  # Remove any whitespace
    df = df.rename(columns=column_map)

    # Get expected columns from scaler
    expected_cols = wilson_scaler.feature_names_in_.tolist()

    # Create DataFrame with all expected columns, filled with 0s
    result_df = pd.DataFrame(0, index=df.index, columns=expected_cols)

    # Fill in values from input data where we have them
    for excel_col, model_col in column_map.items():
        if excel_col in raw_df.columns and model_col in expected_cols:
            result_df[model_col] = pd.to_numeric(df[model_col], errors='coerce').fillna(0)

    # Set default values for missing columns that might be important
    if 'Sex' not in result_df.columns:
        result_df['Sex'] = 0  # Default to 0 (can be adjusted based on your encoding)
    if 'Region' not in result_df.columns:
        result_df['Region'] = 0  # Default region
    if 'Socioeconomic Status' not in result_df.columns:
        result_df['Socioeconomic Status'] = 0  # Middle status
    if 'Alcohol Use' not in result_df.columns:
        result_df['Alcohol Use'] = 0  # No alcohol use
    if 'BMI' not in result_df.columns:
        result_df['BMI'] = 22  # Normal BMI
    if 'Psychiatric Symptoms' not in result_df.columns:
        result_df['Psychiatric Symptoms'] = 0  # No symptoms
    if 'Cognitive Function Score' not in result_df.columns:
        result_df['Cognitive Function Score'] = 0  # Normal cognitive function
    if 'Free Copper in Blood Serum' not in result_df.columns:
        result_df['Free Copper in Blood Serum'] = result_df['Copper in Blood Serum'] * 0.1  # Estimate from total copper
    if 'Alkaline Phosphatase (ALP)' not in result_df.columns:
        result_df['Alkaline Phosphatase (ALP)'] = 0
    if 'Prothrombin Time / INR' not in result_df.columns:
        result_df['Prothrombin Time / INR'] = 1  # Normal INR
    if 'Albumin' not in result_df.columns:
        result_df['Albumin'] = 4  # Normal albumin level

    # Scale the features
    scaled_data = wilson_scaler.transform(result_df)
    return pd.DataFrame(scaled_data, columns=expected_cols)

@app.post("/predict_wilson_excel",
          tags=["Wilson disease – batch"],
          summary="Upload an Excel sheet and get Wilson's disease predictions for every row")
async def predict_wilson_excel(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(status_code=415, detail="Please upload an .xls or .xlsx file")

    try:
        binary = await file.read()
        df = pd.read_excel(io.BytesIO(binary))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read Excel file: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded sheet contains no rows")

    # Print expected vs actual columns for debugging
    print("Expected columns:", wilson_scaler.feature_names_in_.tolist())
    print("Received columns:", df.columns.tolist())

    # Prepare data and predict
    try:
        clean = _prepare_wilson_df(df)
        # For Sequential model, get raw predictions (probabilities)
        probs = wilson_model.predict(clean)
        # Convert probabilities to binary predictions
        preds = (probs > 0.5).astype(int)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing data: {str(e)}\nExpected columns: {wilson_scaler.feature_names_in_.tolist()}"
        )

    result = [
        {
            "row": int(i) + 2,  # +2 for Excel row numbers (header = 1)
            "prediction_class": "wilson" if p else "healthy",
            "prediction_value": float(prob),  # probability score
            "result": "The person has Wilson's disease" if p else "The person does not have Wilson's disease"
        }
        for i, (p, prob) in enumerate(zip(preds, probs))
    ]
    return {"rows": len(result), "predictions": result}

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

@app.post("/predict_liver", tags=["Liver disease – numeric"])
async def predict_liver(data: LiverInput):
    df = pd.DataFrame([data.dict(by_alias=True)])
    df = df.reindex(columns=liver_model.feature_names_in_, fill_value=0)
    
    # Predict directly (no scaling!)
    pred = liver_model.predict(df)[0]

    msg = "The person has liver disease" if pred == 1 else "The person does not have liver disease"
    return {
        "prediction_class": "liver" if pred else "healthy",
        "prediction_value": int(pred),
        "result": msg
    }

def _prepare_liver_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare liver disease data from Excel for prediction:
    • Normalizes column names
    • Adds missing columns with default 0
    • Handles invisible spaces in feature names
    Returns a cleaned DataFrame ready for model.predict()
    """
    # Expected columns (from model if available)
    expected = (liver_model.feature_names_in_.tolist()
               if hasattr(liver_model, "feature_names_in_")
               else [
                   "Total Bilirubin", "Direct Bilirubin",
                   "Alkphos Alkaline Phosphotase",
                   "Sgpt Alamine Aminotransferase",
                   "Sgot Aspartate Aminotransferase",
                   "ALB Albumin",
                   "A/G Ratio Albumin and Globulin Ratio",
                   "Total Protiens"
               ])

    # Accept both raw names and aliases
    alias_map = {
        "Total_Bilirubin": "Total Bilirubin",
        "Direct_Bilirubin": "Direct Bilirubin",
        "Alkphos_Alkaline_Phosphatase": "Alkphos Alkaline Phosphotase",
        "Sgpt_Alamine_Aminotransferase": "Sgpt Alamine Aminotransferase",
        "Sgot_Aspartate_Aminotransferase": "Sgot Aspartate Aminotransferase",
        "ALB_Albumin": "ALB Albumin",
        "AG_Ratio": "A/G Ratio Albumin and Globulin Ratio",
        "Total_Proteins": "Total Protiens"
    }
    raw_df = raw_df.rename(columns=alias_map)

    # Clean column names (remove invisible spaces)
    raw_df.columns = [clean_column(col) for col in raw_df.columns]

    # Ensure all columns exist
    for col in expected:
        if col not in raw_df.columns:
            raw_df[col] = 0

    # Convert all to numeric
    for col in expected:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0)

    # Reorder columns to match model expectations
    return raw_df[expected]

@app.post("/predict_liver_excel",
          tags=["Liver disease – batch"],
          summary="Upload an Excel sheet and get liver disease predictions for every row")
async def predict_liver_excel(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(status_code=415, detail="Please upload an .xls or .xlsx file")

    try:
        binary = await file.read()
        df = pd.read_excel(io.BytesIO(binary))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read Excel file: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded sheet contains no rows")

    # Prepare data and predict
    clean = _prepare_liver_df(df)
    preds = liver_model.predict(clean)

    result = [
        {
            "row": int(i) + 2,  # +2 for Excel row numbers (header = 1)
            "prediction_class": "liver" if p else "healthy",
            "prediction_value": int(p),
            "result": "The person has liver disease" if p else "The person does not have liver disease"
        }
        for i, p in enumerate(preds)
    ]
    return {"rows": len(result), "predictions": result}

# ──────────────────────────────────────────────────────────────────────
#  C O L O R E C T A L   C A N C E R – batch (Excel upload)
# ──────────────────────────────────────────────────────────────────────
from fastapi import UploadFile, File, HTTPException
import io

# helper reused by both single-row & batch endpoints
def _prepare_colorectal_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    • Normalises column names / aliases
    • Adds any missing columns with default 0
    • Encodes categoricals numerically
    • Reorders columns to exactly `expected_columns`
    Returns a cleaned DataFrame ready for model.predict()
    """
    # 1️⃣ expected columns (pull from model if present)
    expected = (colorectal_model.feature_names_in_.tolist()
                if hasattr(colorectal_model, "feature_names_in_")
                else [
                    "Age","Gender","BMI","Lifestyle","Family_History_CRC",
                    "Pre-existing Conditions","Carbohydrates (g)","Proteins (g)",
                    "Fats (g)","Vitamin A (IU)","Vitamin C (mg)","Iron (mg)"
                ])

    # 2️⃣ Accept both "nice" aliases and raw names
    alias_map = {
        "Family History CRC": "Family_History_CRC",
        "Pre-existing Conditions": "Pre-existing Conditions",
        # add more if your spreadsheet uses other spellings
    }
    raw_df = raw_df.rename(columns=alias_map)

    # 3️⃣ Ensure all columns exist
    for c in expected:
        if c not in raw_df.columns:
            raw_df[c] = 0

    # 4️⃣ Encode categoricals
    gender_map      = {"Female": 0, "Male": 1}
    lifestyle_map   = {
        "Sedentary": 0, "Active": 1, "Moderate": 2,
        "Smoker": 3, "Non-smoker": 1, "Athlete": 4
    }
    family_map      = {"No": 0, "Yes": 1}
    condition_map   = {
        "None": 0, "Diabetes": 1, "Hypertension": 2,
        "Heart Disease": 3, "Obesity": 4, "IBD": 5, "Polyps": 6
    }

    raw_df["Gender"]                  = raw_df["Gender"].map(gender_map).fillna(0)
    raw_df["Lifestyle"]               = raw_df["Lifestyle"].map(lifestyle_map).fillna(0)
    raw_df["Family_History_CRC"]      = raw_df["Family_History_CRC"].map(family_map).fillna(0)
    raw_df["Pre-existing Conditions"] = raw_df["Pre-existing Conditions"].map(condition_map).fillna(0)

    # 5️⃣ Numeric coercion & column order
    for c in expected:
        raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce").fillna(0)
    return raw_df[expected]

@app.post("/predict_colorectal_excel",
          tags=["Colorectal Cancer – batch"],
          summary="Upload an Excel sheet and get risk predictions for every row")
async def predict_colorectal_excel(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(status_code=415, detail="Please upload an .xls or .xlsx file")

    # read entire workbook into memory (guard size in real deployments)
    try:
        binary = await file.read()
        df     = pd.read_excel(io.BytesIO(binary))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read Excel file: {exc}")

    # empty sheet guard
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded sheet contains no rows")

    # data wrangling → model.predict
    clean  = _prepare_colorectal_df(df)
    preds  = colorectal_model.predict(clean)           # shape (n,)
    result = [
        {
            "row": int(i) + 2,                         # +2 to match human Excel row numbers (header = row-1)
            "prediction_class": "at_risk" if p else "healthy",
            "prediction_value": int(p)
        }
        for i, p in enumerate(preds)
    ]
    return {"rows": len(result), "predictions": result}

# ═══════════════════════════════════════════════════════════════════════