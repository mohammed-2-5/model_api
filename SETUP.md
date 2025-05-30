# Quick Setup Guide 🚀

## Installation Commands

```bash
# 1. Clone the repository
git clone https://github.com/mohammed-2-5/model_api.git
cd model_api

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure Google Cloud (choose one option)
# Option A: Using service account key
set GOOGLE_APPLICATION_CREDENTIALS=path\to\your\service-account-key.json

# Option B: Using gcloud CLI
gcloud auth application-default login

# 5. Run the application
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Access Points

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

## Quick Test

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"UPDRS": 67.83, "FunctionalAssessment": 2.13, "MoCA": 29.92, "Tremor": 1, "Rigidity": 0, "Bradykinesia": 0, "Age": 70, "AlcoholConsumption": 2.24, "BMI": 15.36, "SleepQuality": 9.93, "DietQuality": 6.49, "CholesterolTriglycerides": 395.66}'
```

## Important Notes

- **Models**: Downloaded automatically from Google Cloud Storage
- **Heavy files**: Excluded via .gitignore (models/, .venv/, etc.)
- **Environment**: Virtual environment is required
- **Credentials**: Google Cloud credentials are needed for model access 