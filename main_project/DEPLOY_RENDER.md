# Render Deployment

This project is ready for Render deployment through `render.yaml`.

## Steps

1. Go to Render dashboard.
2. Click `New +`.
3. Choose `Blueprint`.
4. Connect the GitHub repository:

```text
https://github.com/taimuralam0181/handwritting-to-text-machine-readable-dataset
```

5. Render will detect `render.yaml`.
6. Set the required secret:

```text
GEMINI_API_KEY=your_api_key_here
```

7. Deploy.

## Manual Web Service Settings

If you use `New Web Service` instead of Blueprint:

Build command:

```bash
pip install -r main_project/requirements-render.txt
```

Start command:

```bash
streamlit run main_project/app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --browser.gatherUsageStats false
```

Environment variables:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemma-4-31b-it
```

## Notes

- Free Render services may sleep when inactive.
- First request after sleep can be slow.
- Runtime `users.db` and generated CSV files are stored on the Render instance and may not be permanent on free hosting.
- For permanent user/database storage, use a paid persistent disk or move auth/workspace data to PostgreSQL.

