# Disease Prediction API

A FastAPI-based service that provides machine learning predictions for multiple diseases:
- Parkinson's Disease (numeric features & drawing analysis)
- Wilson's Disease
- Liver Disease
- Colorectal Cancer

## 🚀 Features

- **Multiple Disease Predictions**: Single endpoint for each disease type
- **Batch Processing**: Excel file upload support for bulk predictions
- **Image Analysis**: Spiral drawing analysis for Parkinson's disease
- **Input Validation**: Comprehensive validation of input data
- **Swagger Documentation**: Interactive API documentation at `/docs`

## 🛠️ Technologies

- FastAPI
- TensorFlow
- scikit-learn
- Pandas
- Docker
- Google Cloud Platform (Cloud Run & Storage)

## 📋 API Endpoints

### Parkinson's Disease

#### 1. Numeric Prediction
- **Endpoint**: `POST /predict`
- **Input Example**:
```json
{
  "UPDRS": 67.83,
  "FunctionalAssessment": 2.13,
  "MoCA": 29.92,
  "Tremor": 1,
  "Rigidity": 0,
  "Bradykinesia": 0,
  "Age": 70,
  "AlcoholConsumption": 2.24,
  "BMI": 15.36,
  "SleepQuality": 9.93,
  "DietQuality": 6.49,
  "CholesterolTriglycerides": 395.66
}
```

#### 2. Drawing Analysis
- **Endpoint**: `POST /predict_image`
- **Input**: Image file (spiral drawing)

### Wilson's Disease

#### 1. Single Prediction
- **Endpoint**: `POST /predict_wilson`
- **Input Example**:
```json
{
  "Age": 35,
  "ATB7B Gene Mutation": 1,
  "Kayser-Fleischer Rings": 1,
  "Copper in Blood Serum": 150.5,
  "Copper in Urine": 180.2,
  "Neurological Symptoms Score": 7.5,
  "Ceruloplasmin Level": 15.3,
  "AST": 45.2,
  "ALT": 38.7,
  "Family History": 1,
  "Gamma-Glutamyl Transferase (GGT)": 55.8,
  "Total_Bilirubin": 1.2
}
```

#### 2. Batch Prediction
- **Endpoint**: `POST /predict_wilson_excel`
- **Input**: Excel file with multiple patient records

### Liver Disease

#### 1. Single Prediction
- **Endpoint**: `POST /predict_liver`
- **Input Example**:
```json
{
  "Total Bilirubin": 1.5,
  "Direct Bilirubin": 0.8,
  "Alkphos Alkaline Phosphotase": 200.0,
  "Sgpt Alamine Aminotransferase": 45.0,
  "Sgot Aspartate Aminotransferase": 40.0,
  "ALB Albumin": 4.2,
  "A/G Ratio Albumin and Globulin Ratio": 1.8,
  "Total Protiens": 7.5
}
```

#### 2. Batch Prediction
- **Endpoint**: `POST /predict_liver_excel`
- **Input**: Excel file with multiple patient records

### Colorectal Cancer

#### 1. Single Prediction
- **Endpoint**: `POST /predict_colorectal`
- **Input Example**:
```json
{
  "Age": 65,
  "Gender": "Male",
  "BMI": 28.5,
  "Lifestyle": "Smoker",
  "Family_History_CRC": "Yes",
  "Pre-existing Conditions": "Diabetes",
  "Carbohydrates (g)": 300.0,
  "Proteins (g)": 80.0,
  "Fats (g)": 50.0,
  "Vitamin A (IU)": 3000.0,
  "Vitamin C (mg)": 60.0,
  "Iron (mg)": 12.0
}
```

#### 2. Batch Prediction
- **Endpoint**: `POST /predict_colorectal_excel`
- **Input**: Excel file with multiple patient records

## 🚀 Deployment

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd parkinson_model_api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app:app --reload
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t parkinson-model-api .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 parkinson-model-api
```

### Cloud Run Deployment

1. Tag the image:
```bash
docker tag parkinson-model-api gcr.io/[PROJECT-ID]/my-fastapi-app
```

2. Push to Google Container Registry:
```bash
docker push gcr.io/[PROJECT-ID]/my-fastapi-app
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy my-fastapi-app \
  --image gcr.io/[PROJECT-ID]/my-fastapi-app \
  --platform managed \
  --region [REGION] \
  --allow-unauthenticated
```

## 📝 Input Data Validation

The API includes comprehensive input validation:
- Checks for all-zero inputs
- Validates feature names
- Provides clear error messages for incorrect or missing features
- Supports multiple naming conventions through alias mapping

## 🔒 Security

- Environment variables for sensitive configurations
- Google Cloud Storage for model storage
- Input validation to prevent malicious data

## 📊 Excel File Format

For batch predictions, prepare Excel files with appropriate column names as shown in the single prediction examples. The API supports both exact column names and common aliases.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.