# Smart Prescription Analysis System

Streamlit-based prescription analysis application that extracts medicine names from prescription images and classifies each detected medicine into `Cardiology`, `Ophthalmology`, or `Unknown`.

The system supports single-image and bulk-image workflows, Gemini-based text extraction, alias and fuzzy correction, dataset-driven classification, manual review, export, caching, and an evaluation dashboard.

## Features

- Single and bulk prescription image processing
- Gemini/Gemma-based prescription text extraction
- Image preprocessing with optional auto-crop workflow
- Medicine name cleaning, alias matching, and fuzzy matching
- Department classification using:
  - dataset matching
  - keyword rules
  - local trained classifier
  - SVM classifier
  - KNN classifier
  - Gemini fallback review
- Manual review and correction workflow
- CSV, Excel, PDF, and bulk ZIP export
- Cached repeat analysis for faster reruns
- Evaluation dashboard with accuracy, macro F1, unknown rate, and confusion matrix

## Project Workflow

1. Upload one or more prescription images.
2. Preprocess the image and optionally auto-crop the medicine-heavy region.
3. Send the prepared image to Gemini for text extraction.
4. Parse medicine lines from the extracted text.
5. Clean and resolve medicine names using alias and fuzzy matching.
6. Classify each medicine into a department.
7. Review and correct the final output.
8. Export the structured result.

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- OpenCV
- Pandas
- scikit-learn
- ReportLab

## Requirements

Main dependencies are listed in [requirements.txt](./requirements.txt):

- `streamlit`
- `easyocr`
- `opencv-python`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `pillow`
- `google-generativeai`
- `python-dotenv`
- `reportlab`
- `openpyxl`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemma-4-31b-it
```

4. Run the app:

```bash
streamlit run app.py
```

## Datasets

The application uses seed and support datasets from the `dataset/` directory, including:

- `medicine_department_seed.csv`
- `real_cardiology_medicine_department_12000.csv`
- `ophthalmology_1000_plus_medicines.csv`
- `medicine_name_aliases.csv`

These are used for name normalization, alias correction, and department classification.

## Evaluation Dashboard

The built-in evaluation dashboard can be used with a labeled CSV file to measure:

- Accuracy
- Precision
- Recall
- Macro F1 score
- Unknown rate
- Confusion matrix

This is useful for thesis, report, and research-style validation.

## Repository Structure

```text
app.py
prescription_services.py
extract_app.py
train_model.py
train_department_model.py
dataset/
requirements.txt
```

## Notes

- The app expects a valid Gemini API key for extraction and fallback review.
- The local dataset and manual corrections improve future classification quality.
- Cached repeat uploads can return much faster when the same image and mode are used again.
