# Main Project

This folder contains the runnable Streamlit app and the files it needs.

## Run

```powershell
cd "main_project"
pip install -r requirements.txt
streamlit run app.py
```

On Windows, you can also double-click `Launch_App.vbs` after installing requirements. This opens the app like a desktop launcher without keeping a command window visible.

To create a desktop shortcut with the app logo:

```powershell
powershell -ExecutionPolicy Bypass -File Create_Desktop_Shortcut.ps1
```

Launcher and branding files:

```text
Launch_App.vbs
RUN_APP.bat
Create_Desktop_Shortcut.ps1
assets/app_logo.png
assets/app_icon.ico
```

Login features:

- Register account
- Login account
- Forgot Password tab for resetting a local account password by registered email

## CSV Workspace Mode

Use `CSV Workspace` when a user wants to process images one by one and keep saving every extraction into the same CSV.

Flow:

1. Select `CSV Workspace`.
2. Enter a CSV file name, for example `prescription_workspace.csv`.
3. Click `Start Workspace`.
4. Upload one prescription image.
5. Click `Extract and Save to Workspace CSV`.
6. Upload the next image and repeat.
7. Download the growing CSV from the right-side workspace panel.

The workspace CSV is saved in:

```text
output/csv_collections/
```

## Included

- `app.py`, `auth.py`
- `ui/`
- `models/`
- required `dataset/*.csv` files
- `users.db`
- `.env`
- `output/csv_collections/` and `output/analysis_cache/`
- `training_dataset/` for the image-based Gemma/PaliGemma dataset evidence
- `assets/` for the app logo and Windows shortcut icon

Large test folders, zip files, and virtual environments were not copied.

## Training Dataset For Demonstration

The image-based training/demo dataset is inside:

```text
training_dataset/
```

Prepared split files:

```text
training_dataset/gemma_line_crop_dataset/train.jsonl
training_dataset/gemma_line_crop_dataset/validation.jsonl
training_dataset/gemma_line_crop_dataset/test.jsonl
training_dataset/gemma_line_crop_dataset/summary.json
```

Available cropped line images and metadata:

```text
training_dataset/medicine_line_dataset_final/images/
training_dataset/medicine_line_dataset_final/metadata.csv
training_dataset/medicine_line_dataset_final/metadata_with_ocr.csv
```

Custom department image datasets:

```text
training_dataset/cardiology_custom_20260625_200149/
training_dataset/ophthalmology_custom_20260625_202639/
```

Summary:

- Total prepared examples: 2456
- Train: 1964
- Validation: 246
- Test: 246
- Custom cardiology/ophthalmology folders are included for showing the original image dataset used to build department-specific line-crop data.

Note for presentation: plain Gemma is text-only, so image-based fine-tuning needs PaliGemma or another vision-language model. This project has the dataset prepared and the runnable OCR/classification app included.

## Trained Custom Image Department Classifier

The custom cardiology and ophthalmology line-crop images were used to train a lightweight image-based department classifier.

Training command:

```powershell
python train_custom_image_department_model.py --max-iter 300
```

Training data:

- Cardiology labeled line-crop images: 5007
- Ophthalmology labeled line-crop images: 5571
- Total labeled line-crop images used for training: 10578
- Train examples: 8462
- Validation examples: 1058
- Test examples: 1058

Result:

- Validation accuracy: 97.07%
- Test accuracy: 97.07%

Saved files:

```text
models/custom_image_department_classifier/custom_image_department_classifier.joblib
models/custom_image_department_classifier/metrics.json
models/custom_image_department_classifier/validation_report.txt
models/custom_image_department_classifier/test_report.txt
models/custom_image_department_classifier/prediction_preview.csv
```
