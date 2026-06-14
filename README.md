# AI Music Genre Analyzer

Web application for automatic music genre recognition from audio recordings.

The project combines several trained classifiers and exposes them through a FastAPI backend and a React frontend. A user can upload an audio file, run the analysis, see the predicted genre, compare outputs of individual models, and view basic model metrics.

## Implemented Models

The repository contains trained model artifacts, so the application can be tested without retraining:

| Model | File |
| --- | --- |
| AST | `backend/gtzan_ast_best.pt` |
| CNN | `backend/artifacts/cnn_fbank_best.pt` |
| Random Forest | `backend/artifacts/rf_mfcc.joblib` |
| KNN | `backend/artifacts/knn_mfcc.joblib` |

The large model files are stored with Git LFS. The GTZAN dataset itself is not included in the repository.

## Requirements

Install these tools before running the project:

- Python 3.10 or newer
- Node.js 18 or newer
- npm
- Git LFS
- ffmpeg, recommended for decoding MP3/M4A/OGG and other audio formats

On Arch Linux, for example:

```bash
sudo pacman -S python nodejs npm git-lfs ffmpeg
```

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv nodejs npm git-lfs ffmpeg
```

## Quick Start

Clone the repository:

```bash
git clone https://github.com/kznkutsubyo/music-content-analysis-nn.git
cd music-content-analysis-nn
```

Download the Git LFS model files:

```bash
git lfs install
git lfs pull
```

Install backend and frontend dependencies:

```bash
./scripts/setup.sh
```

Start the backend and frontend together:

```bash
./scripts/run_app.sh
```

Open the web application:

```text
http://127.0.0.1:5173/
```

To stop the application, press `Ctrl+C` in the terminal where `run_app.sh` is running.

## Manual Run

Use this option if you prefer to run backend and frontend in separate terminals.

Terminal 1, backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-runtime.txt
MODEL_DEVICE=cpu uvicorn app:app --host 127.0.0.1 --port 8001
```

Backend health check:

```bash
curl http://127.0.0.1:8001/health
```

Terminal 2, frontend:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend URL:

```text
http://127.0.0.1:5173/
```

## Configuration

The run script uses CPU by default:

```bash
./scripts/run_app.sh
```

If CUDA is available and the installed PyTorch build supports it, the backend can be started with:

```bash
MODEL_DEVICE=cuda ./scripts/run_app.sh
```

Default ports:

| Service | URL |
| --- | --- |
| Backend API | `http://127.0.0.1:8001` |
| Frontend | `http://127.0.0.1:5173` |

## Repository Structure

```text
backend/    FastAPI backend, inference code, trained models, metrics
frontend/   React web application
training/   Training and evaluation scripts
scripts/    Setup and launch scripts
```

## Supported Audio Formats

The upload form accepts common audio formats such as:

```text
WAV, MP3, FLAC, OGG, AU, AIFF, M4A, AAC
```

For best compatibility with compressed formats, install `ffmpeg`.

## Troubleshooting

If model files are missing after cloning, run:

```bash
git lfs pull
```

If the backend cannot decode an audio file, install `ffmpeg` and try again.

If a port is already in use, choose another port:

```bash
BACKEND_PORT=8010 FRONTEND_PORT=5174 ./scripts/run_app.sh
```

If dependency installation fails, make sure that the active Python version is 3.10 or newer and that `python3-venv` is installed on Debian/Ubuntu systems.

## Notes

- The frontend profile and history functionality is a prototype and uses browser local storage.
- The GTZAN dataset has known limitations, so reported metrics should be interpreted in the context of the experimental setup.
- Production deployment would require server-side authentication, database storage, and stricter upload security.
