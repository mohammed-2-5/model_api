# Medical Disease Prediction API 🏥🤖

A comprehensive FastAPI service for predicting multiple diseases using machine learning models. The API supports predictions for Parkinson's disease, Wilson's disease, Liver disease, and Colorectal cancer.

## 🚀 Features

- **Parkinson's Disease**: Prediction from clinical data and spiral drawing analysis
- **Wilson's Disease**: Clinical data-based prediction with biochemical markers
- **Liver Disease**: Hepatic function assessment
- **Colorectal Cancer**: Risk assessment based on lifestyle and nutritional factors
- **Cloud Integration**: Models automatically downloaded from Google Cloud Storage
- **Interactive Documentation**: Built-in Swagger UI

## 📋 Prerequisites

- Python 3.8 or higher
- Google Cloud Storage access (for model downloading)
- pip package manager

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mohammed-2-5/model_api.git
cd model_api
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Google Cloud Storage

Ensure you have proper Google Cloud credentials configured:

```bash
# Option 1: Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"

# Option 2: Use gcloud CLI
gcloud auth application-default login
```

### 5. Run the Application

```bash
# Development server with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📖 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 API Endpoints

### Parkinson's Disease

#### 1. Numeric Prediction
- **Endpoint**: `POST /predict`
- **Description**: Predict Parkinson's disease from clinical data

**Example Request:**
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
- **Description**: Analyze spiral drawings for Parkinson's detection
- **Input**: Image file (PNG, JPG, etc.)

### Wilson's Disease

- **Endpoint**: `POST /predict_wilson`
- **Description**: Predict Wilson's disease from clinical markers

**Example Request:**
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

### Liver Disease

- **Endpoint**: `POST /predict_liver`
- **Description**: Assess liver disease risk

**Example Request:**
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

### Colorectal Cancer

- **Endpoint**: `POST /predict_colorectal`
- **Description**: Assess colorectal cancer risk

**Example Request:**
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

## 🧪 Quick Test Commands

### Using curl

```bash
# Test Parkinson's prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
```

## 🏗️ Project Structure

```
model_api/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── vercel.json          # Vercel deployment config
├── vercel_app.py        # Vercel entry point
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## 🚀 Deployment Options

### Local Development
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build image
docker build -t medical-api .

# Run container
docker run -p 8000:8000 medical-api
```

### Vercel Serverless
The project includes Vercel configuration for serverless deployment:
```bash
vercel --prod
```

## 🔧 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud service account key | Yes |
| `CUDA_VISIBLE_DEVICES` | GPU configuration (set to -1 for CPU) | No |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow logging level | No |

## 📊 Model Information

- **Models**: Automatically downloaded from Google Cloud Storage
- **Storage**: Models are cached locally in `models/` directory (excluded from git)
- **Types**: Support for scikit-learn, TensorFlow/Keras models
- **Size**: Models range from 1KB to 9MB

## 🔄 Development Workflow

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/mohammed-2-5/model_api.git
   cd model_api
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Run Development Server**:
   ```bash
   uvicorn app:app --reload
   ```

3. **Test the API**:
   - Visit http://localhost:8000/docs for interactive testing
   - Use curl or Python requests for programmatic testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues:

1. Check the [Issues](https://github.com/mohammed-2-5/model_api/issues) page
2. Create a new issue with detailed description
3. Include error logs and environment information

## 🔄 API Response Format

All endpoints return JSON responses with consistent structure:

```json
{
  "prediction_class": "healthy|disease_name",
  "prediction_value": 0.85,
  "result": "Human-readable result message"
}
```

---

**Made with ❤️ by Mohammed** | [GitHub](https://github.com/mohammed-2-5/model_api)