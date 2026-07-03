# Prescription API Backend

FastAPI backend for the future Kotlin Android app.

## Features

- Register
- Login
- Bearer-token auth
- Prescription image extraction
- Department classification
- CSV workspace start
- Append image extraction result to workspace CSV
- Download workspace CSV

## Run Locally

macOS/Linux:

```bash
cd api_backend
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Windows CMD:

```bat
cd api_backend
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Windows PowerShell:

```powershell
cd api_backend
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Or double-click:

```text
RUN_API.bat
```

Local API docs:

```text
http://127.0.0.1:8001/docs
```

## Render Deploy

This repository includes a Render Blueprint service named `prescription-api`.

Render build command:

```text
pip install -r api_backend/requirements.txt
```

Render start command:

```text
uvicorn api_backend.main:app --host 0.0.0.0 --port $PORT
```

Set these Render environment variables:

```text
GEMINI_API_KEY=your_api_key
GEMINI_MODEL_NAME=gemini-2.5-flash
DATABASE_URL=postgres_connection_string
```

`DATABASE_URL` should be shared with the Streamlit dashboard service so website and
mobile app users use the same persistent account database. Without `DATABASE_URL`,
the API falls back to local SQLite for development.

## Environment

Create `.env` in the repo root or `api_backend/`:

```text
GEMINI_API_KEY=your_api_key
GEMINI_MODEL_NAME=gemma-4-31b-it
```

## Main Endpoints

```text
POST /api/auth/register
POST /api/auth/login
GET  /api/me
POST /api/extract
POST /api/workspaces
POST /api/workspaces/{workspace_id}/images
GET  /api/workspaces/{workspace_id}/download
GET  /api/workspaces
```
