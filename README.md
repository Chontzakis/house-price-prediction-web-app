# House Price Prediction System

## Overview
A machine learning system for predicting house prices with a FastAPI backend, a Streamlit frontend, and Docker deployment support.

The backend loads a trained model from `models/` and exposes a `/predict` endpoint. The user interface is served by Streamlit in `frontend/app.py`.

## Features
- FastAPI prediction API
- Streamlit interactive frontend
- Pre-trained house price model loaded from `models/`
- Docker and Docker Compose support
- MLflow experiment tracking environment present in `mlruns/` (should be created by the user)

## Tech Stack
- Python 3.11
- FastAPI
- Streamlit
- Scikit-learn
- XGBoost / LightGBM
- MLflow
- Docker

## Project Layout
- `Dockerfile` — container image definition
- `docker-compose.yml` — multi-service Docker setup for API + frontend
- `src/api/main.py` — FastAPI application
- `frontend/app.py` — Streamlit dashboard
- `models/model.pkl` — saved model loaded by the API
- `requirements.txt` — Python dependencies

## Local Setup (Python)

1. Create a virtual environment

```bash
python3 -m venv .venv
```

2. Activate it

```bash
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Start the API

```bash
python3 -m uvicorn src.api.main:app --reload
```

5. Start the frontend (from a new terminal window)

```bash
streamlit run frontend/app.py    
```

6. If the end user want to change something in the training procedure then that can happen inside the 

> Note: the frontend is currently configured to call the API at `http://api:8000/predict`, which works directly in Docker Compose. For direct local execution, update the API URL in `frontend/app.py` to `http://localhost:8501` or run via Docker Compose.

> Note: If the end user want to change something in the training procedure then that can happen inside the models/train.py. The command that executes this procedure is 

```bash
python3 -m src.models.train
```

## Docker Compose (Recommended)

Use Docker Compose to start both services together with the correct internal networking:

```bash
docker compose up --build
```

Then open:
- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`

## Docker Only

Build the container:

```bash
docker build -t house-price-system .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 house-price-system
```

## Notes
- `src/api/main.py` loads the model from `models/model.pkl` at startup.
- The frontend currently posts requests to the internal service name `api`, so Docker Compose is the simplest way to run both services together.
- If you want to use local mode without Docker, update the API URL in `frontend/app.py` to the local host address.
