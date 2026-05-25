from __future__ import annotations

import io
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

try:
    import google.generativeai as genai
except ImportError:
    genai = None

BASE_DIR = Path(__file__).resolve().parent
DEPARTMENT_MODEL_PATH = BASE_DIR / "models" / "medicine_department_classifier.pkl"
DEPARTMENT_DATASET_PATH = BASE_DIR / "dataset" / "medicine_department_seed.csv"

EXTRACTION_PROMPT = """Extract medical information from this prescription image.

Use EXACT format with these section headers:

PATIENT INFORMATION:
Name: [Full patient name]
Age: [Age in years]
Gender: [Male/Female]
Date: [Prescription date]

DOCTOR INFORMATION:
Doctor Name: [Full doctor name with title]
Qualifications: [All degrees]
Hospital/Clinic: [Hospital name]
Registration Number: [Medical council registration]
Contact: [Phone number]

MEDICAL NOTES:
Diagnosis: [Medical diagnosis]
Symptoms: [Patient symptoms]
Tests Ordered: [Lab tests]

PRESCRIBED MEDICATIONS:
• [Medication name] - [dosage pattern like 1-0-1] for [X] days
• [Next medication] - [dosage pattern] for [X] days

FOLLOW-UP DATE: [Date or Not specified]

Use "Not specified" for any missing information. List ALL medications found. No extra text or explanations."""


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and genai is not None:
    genai.configure(api_key=GEMINI_API_KEY)

_department_seed_df: pd.DataFrame | None = None
_department_model: Any = None
_department_model_loaded = False


def _load_department_seed_data() -> pd.DataFrame:
    global _department_seed_df
    if _department_seed_df is not None:
        return _department_seed_df

    if not DEPARTMENT_DATASET_PATH.exists():
        _department_seed_df = pd.DataFrame(columns=["medicine_name", "department", "normalized_name"])
        return _department_seed_df

    seed_df = pd.read_csv(DEPARTMENT_DATASET_PATH)
    seed_df["normalized_name"] = seed_df["medicine_name"].apply(normalize_medication_name)
    _department_seed_df = seed_df
    return _department_seed_df


def _load_department_model() -> Any:
    global _department_model, _department_model_loaded
    if _department_model_loaded:
        return _department_model

    _department_model_loaded = True
    if not DEPARTMENT_MODEL_PATH.exists():
        _department_model = None
        return _department_model

    try:
        _department_model = joblib.load(DEPARTMENT_MODEL_PATH)
    except Exception:
        _department_model = None
    return _department_model


def normalize_medication_name(name: str) -> str:
    normalized = (name or "").lower().strip()
    normalized = re.sub(r"^[a-z]+\.\s*", "", normalized)
    normalized = re.sub(r"\b(tab|tablet|cap|capsule|inj|injection|syp|syrup|drop|drops)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def clean_medication_name(raw_name: str) -> str:
    name = (raw_name or "").strip()
    name = re.sub(r"^\d+\.\s*", "", name)
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = re.sub(r"\s+(?:before|after)\s+\w+.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+\d+(?:\.\d+)?\s*[\+\-]\s*\d+(?:\.\d+)?\s*[\+\-]\s*\d+(?:\.\d+)?(?:.*)?$", "", name)
    name = re.sub(r"\s*-\s*\d+(?:[.-]\d+)?-\d+(?:[.-]\d+)?-\d+(?:[.-]\d+)?(?:.*)?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" -.")
    return name


def is_valid_medication_candidate(raw_line: str, cleaned_name: str) -> bool:
    line = (raw_line or "").strip()
    name = (cleaned_name or "").strip()
    if not name:
        return False

    upper_line = line.upper()
    blocked_prefixes = (
        "PATIENT INFORMATION",
        "DOCTOR INFORMATION",
        "MEDICAL NOTES",
        "PRESCRIBED MEDICATIONS",
        "FOLLOW-UP DATE",
        "NAME:",
        "AGE:",
        "GENDER:",
        "DATE:",
        "DOCTOR NAME:",
        "QUALIFICATIONS:",
        "HOSPITAL/CLINIC:",
        "REGISTRATION NUMBER:",
        "CONTACT:",
        "DIAGNOSIS:",
        "SYMPTOMS:",
        "TESTS ORDERED:",
        "FOLLOW-UP",
        "EXACT FORMAT",
        "SECTION HEADERS",
        "NO EXTRA TEXT",
        "TASK:",
        "OUTPUT FORMAT",
        "SPECIFIC REQUIREMENTS",
        "CHECK AGAINST",
        "CHECK MEDICATIONS",
        "RX",
    )
    if upper_line.startswith(blocked_prefixes):
        return False

    if ":" in line and not re.match(r"^\d+\.\s*", line):
        return False

    if len(name) < 3:
        return False

    medication_hint = re.search(
        r"\b(tab|tablet|cap|capsule|inj|injection|syp|syrup|drop|drops|ointment|gel|spray)\b",
        line,
        re.IGNORECASE,
    )
    strength_hint = re.search(r"\b\d+(?:\.\d+)?\s*(mg|mcg|g|ml|iu|%)\b", line, re.IGNORECASE)
    if medication_hint or strength_hint:
        return True

    known_name = normalize_medication_name(name)
    if known_name:
        seed_df = _load_department_seed_data()
        if not seed_df.empty:
            if any(
                seed_name and (seed_name in known_name or known_name in seed_name)
                for seed_name in seed_df["normalized_name"].tolist()
            ):
                return True

    return False


def parse_prescription_to_columns(extracted_text: str) -> dict[str, str]:
    structured_data = {
        "filename": "",
        "doctor_name": "",
        "doctor_qualifications": "",
        "doctor_designation": "",
        "hospital_clinic": "",
        "doctor_reg_no": "",
        "doctor_contact": "",
        "patient_name": "",
        "patient_age": "",
        "patient_gender": "",
        "prescription_date": "",
        "diagnosis": "",
        "symptoms": "",
        "tests_ordered": "",
        "follow_up_date": "",
    }

    patient_match = re.search(r"PATIENT INFORMATION:(.*?)(?=DOCTOR INFORMATION:|$)", extracted_text, re.IGNORECASE | re.DOTALL)
    if patient_match:
        patient_text = patient_match.group(1)
        for label, key in [
            ("Name", "patient_name"),
            ("Age", "patient_age"),
            ("Gender", "patient_gender"),
            ("Date", "prescription_date"),
        ]:
            match = re.search(rf"{label}:\s*(.+?)(?:\n|$)", patient_text, re.IGNORECASE)
            if match:
                structured_data[key] = match.group(1).strip()

    doctor_match = re.search(r"DOCTOR INFORMATION:(.*?)(?=MEDICAL NOTES:|$)", extracted_text, re.IGNORECASE | re.DOTALL)
    if doctor_match:
        doctor_text = doctor_match.group(1)
        for label, key in [
            ("Doctor Name", "doctor_name"),
            ("Qualifications", "doctor_qualifications"),
            ("Hospital/Clinic", "hospital_clinic"),
            ("Registration Number", "doctor_reg_no"),
            ("Contact", "doctor_contact"),
        ]:
            match = re.search(rf"{label}:\s*(.+?)(?:\n|$)", doctor_text, re.IGNORECASE)
            if match:
                structured_data[key] = match.group(1).strip()

    medical_match = re.search(r"MEDICAL NOTES:(.*?)(?=PRESCRIBED MEDICATIONS:|$)", extracted_text, re.IGNORECASE | re.DOTALL)
    if medical_match:
        medical_text = medical_match.group(1)
        for label, key in [
            ("Diagnosis", "diagnosis"),
            ("Symptoms", "symptoms"),
            ("Tests Ordered", "tests_ordered"),
        ]:
            match = re.search(rf"{label}:\s*(.+?)(?:\n|$)", medical_text, re.IGNORECASE)
            if match:
                structured_data[key] = match.group(1).strip()

    followup_match = re.search(r"FOLLOW-UP DATE:\s*(.+?)(?:\n|$)", extracted_text, re.IGNORECASE)
    if followup_match:
        structured_data["follow_up_date"] = followup_match.group(1).strip()

    return structured_data


def extract_medications_list(extracted_text: str) -> list[dict[str, str]]:
    medications: list[dict[str, str]] = []
    med_section_match = re.search(r"PRESCRIBED MEDICATIONS:(.*?)(?=FOLLOW-UP DATE:|$)", extracted_text, re.IGNORECASE | re.DOTALL)
    med_section = med_section_match.group(1) if med_section_match else ""

    if med_section_match:
        lines = med_section.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("PRESCRIBED MEDICATIONS"):
                continue

            line = re.sub(r"^[•â€¢\-*]\s*", "", line)
            pattern1 = r"^(.+?)\s*[-]\s*(\d+(?:[.-]\d+)?[-]\d+(?:[.-]\d+)?[-]\d+(?:[.-]\d+)?)(?:\s*for\s*(.+?)\s*days?)?$"
            match = re.match(pattern1, line, re.IGNORECASE)
            if match:
                med_name = clean_medication_name(match.group(1))
                dosage = match.group(2)
                duration = match.group(3) if match.group(3) else ""
                if not is_valid_medication_candidate(line, med_name):
                    continue
                parts = dosage.split("-")
                medications.append({
                    "medication_name": med_name,
                    "morning": parts[0] if len(parts) > 0 else "0",
                    "noon": parts[1] if len(parts) > 1 else "0",
                    "night": parts[2] if len(parts) > 2 else "0",
                    "duration_days": duration if duration and duration != "Not specified" else "",
                })
                continue

            pattern2 = r"^(.+?)\s*[-]\s*Not specified(?:\s*for\s*Not specified\s*days?)?$"
            match = re.match(pattern2, line, re.IGNORECASE)
            if match:
                med_name = clean_medication_name(match.group(1))
                if not is_valid_medication_candidate(line, med_name):
                    continue
                medications.append({
                    "medication_name": med_name,
                    "morning": "0",
                    "noon": "0",
                    "night": "0",
                    "duration_days": "",
                })
                continue

            if line and len(line) > 3 and not line.startswith("Not specified"):
                med_name = clean_medication_name(line)
                if not is_valid_medication_candidate(line, med_name):
                    continue
                medications.append({
                    "medication_name": med_name,
                    "morning": "0",
                    "noon": "0",
                    "night": "0",
                    "duration_days": "",
                })

    if not medications:
        alt_pattern = r"[•â€¢\-]\s*(.+?)\s*-\s*(\d+(?:[.-]\d+)?[-]\d+(?:[.-]\d+)?[-]\d+(?:[.-]\d+)?)"
        matches = re.findall(alt_pattern, med_section if med_section_match else extracted_text)
        for match in matches:
            med_name = clean_medication_name(match[0])
            if not is_valid_medication_candidate(match[0], med_name):
                continue
            parts = match[1].split("-")
            medications.append({
                "medication_name": med_name,
                "morning": parts[0] if len(parts) > 0 else "0",
                "noon": parts[1] if len(parts) > 1 else "0",
                "night": parts[2] if len(parts) > 2 else "0",
                "duration_days": "",
            })

    return medications


def infer_department_from_seed(medication_name: str) -> str:
    normalized_name = normalize_medication_name(medication_name)
    if not normalized_name:
        return "Unknown"

    seed_df = _load_department_seed_data()
    if seed_df.empty:
        return "Unknown"

    exact_matches = seed_df[seed_df["normalized_name"] == normalized_name]
    if not exact_matches.empty:
        return str(exact_matches.iloc[0]["department"])

    substring_matches = seed_df[
        seed_df["normalized_name"].apply(lambda item: bool(item) and (item in normalized_name or normalized_name in item))
    ]
    if not substring_matches.empty:
        best_match = substring_matches.iloc[substring_matches["normalized_name"].str.len().argmax()]
        return str(best_match["department"])

    if any(keyword in normalized_name for keyword in ["eye", "ophthalmic", "ocular", "intravitreal", "tear gel"]):
        return "Ophthalmology"

    return "Unknown"


def classify_department(medication_name: str) -> str:
    model = _load_department_model()
    if model is not None:
        try:
            predicted_department = str(model.predict([medication_name])[0])
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([medication_name])[0]
                if float(max(probabilities)) < 0.55:
                    return infer_department_from_seed(medication_name)
            return predicted_department
        except Exception:
            pass
    return infer_department_from_seed(medication_name)


def classify_medications_by_department(medications: list[dict[str, str]]) -> list[dict[str, str]]:
    classified_medications: list[dict[str, str]] = []
    for medication in medications:
        item = medication.copy()
        item["department"] = classify_department(item.get("medication_name", ""))
        classified_medications.append(item)
    return classified_medications


def extract_prescription_text_from_image(image_bytes: bytes, image_mime_type: str = "image/jpeg", model_name: str = "gemma-4-31b-it") -> str:
    if genai is None:
        raise RuntimeError("google-generativeai is not installed in this environment.")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    model = genai.GenerativeModel(model_name)
    response = model.generate_content([
        EXTRACTION_PROMPT,
        {"mime_type": image_mime_type or "image/jpeg", "data": image_bytes},
    ])
    return getattr(response, "text", "") or ""


def build_export_dataframe(medications: list[dict[str, str]]) -> pd.DataFrame:
    export_rows = []
    for med in medications:
        medication_name = (med.get("medication_name", "") or "").strip()
        if medication_name:
            export_rows.append({
                "medicine_name": medication_name,
                "department": med.get("department", "Unknown"),
            })
    return pd.DataFrame(export_rows, columns=["medicine_name", "department"]).fillna("")


def build_spreadsheet_export(df: pd.DataFrame, timestamp: str | None = None) -> dict[str, Any]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    spreadsheet_filename = f"prescription_{timestamp}.xlsx"
    spreadsheet_label = "Excel"
    spreadsheet_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Medicines")
        spreadsheet_data = excel_buffer.getvalue()
    except ModuleNotFoundError:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        spreadsheet_data = csv_buffer.getvalue().encode("utf-8")
        spreadsheet_filename = f"prescription_{timestamp}.csv"
        spreadsheet_label = "CSV"
        spreadsheet_mime = "text/csv"

    return {
        "filename": spreadsheet_filename,
        "label": spreadsheet_label,
        "mime": spreadsheet_mime,
        "data": spreadsheet_data,
    }


def build_pdf_export(medications: list[dict[str, str]], timestamp: str | None = None) -> dict[str, Any]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"prescription_{timestamp}.pdf"

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#667eea"),
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName="Helvetica-Bold",
    )

    story.append(Paragraph("Medicine Department Classification", title_style))
    story.append(Spacer(1, 20))

    grouped_rows: dict[str, list[str]] = {}
    for med in medications:
        medication_name = (med.get("medication_name", "") or "").strip()
        if medication_name:
            grouped_rows.setdefault(med.get("department", "Unknown"), []).append(medication_name)

    if grouped_rows:
        for department, names in grouped_rows.items():
            story.append(Paragraph(escape(department), styles["Heading2"]))
            story.append(Spacer(1, 8))
            for index, medication_name in enumerate(names, start=1):
                story.append(Paragraph(f"{index}. {escape(medication_name)}", styles["Normal"]))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No medicine names found.", styles["Normal"]))

    doc.build(story)
    return {"filename": pdf_filename, "mime": "application/pdf", "data": pdf_buffer.getvalue()}
try:
    import google.generativeai as genai
except ImportError:
    genai = None
