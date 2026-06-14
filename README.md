# AI Music Genre Analyzer

This repository contains a bachelor project focused on automatic music genre classification from audio recordings.

The project compares classical machine learning methods and neural network models on the GTZAN dataset. It includes trained models, a FastAPI inference backend, and a React web application that allows users to upload an audio file and view genre predictions.

## Repository Structure

```text
frontend/   React web application
backend/    FastAPI inference backend
training/   Training and evaluation scripts
```

## Trained Models

The trained model files are included so the application can be tested without retraining:

```text
backend/gtzan_ast_best.pt
backend/artifacts/cnn_fbank_best.pt
backend/artifacts/rf_mfcc.joblib
backend/artifacts/knn_mfcc.joblib
```

These files are stored through Git LFS because the AST checkpoint is larger than the normal GitHub file limit.

The GTZAN dataset itself is not included. The repository contains only the splits, training/evaluation code, trained artifacts, backend, and frontend.

Before committing or cloning the repository with model files, install Git LFS:

```bash
git lfs install
```

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MODEL_DEVICE=cpu uvicorn app:app --host 127.0.0.1 --port 8001
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

## Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev -- --host 127.0.0.1 --port 5173
```

Default frontend URL:

```text
http://127.0.0.1:5173/
```

## Notes

- The frontend profile and history functionality is prototype-only and uses client-side storage.
- Production deployment would require server-side authentication, a database, and secure handling of uploaded files.
- The GTZAN dataset has known limitations; reported metrics should be interpreted in the context of the experimental setup.
