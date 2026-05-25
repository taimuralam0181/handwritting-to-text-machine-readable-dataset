import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
import io
import base64
import json
import re
import hashlib
import warnings
import time
from pathlib import Path
from difflib import SequenceMatcher, get_close_matches
import zipfile

import joblib
from xml.sax.saxutils import escape

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = Warning

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
except Exception:
    TfidfVectorizer = None
    KNeighborsClassifier = None
    Pipeline = None
    LinearSVC = None

BASE_DIR = Path(__file__).resolve().parent
DEPARTMENT_MODEL_PATH = BASE_DIR / "models" / "medicine_department_classifier.pkl"
DEPARTMENT_DATASET_PATH = BASE_DIR / "dataset" / "medicine_department_seed.csv"
CARDIOLOGY_SUPPLEMENT_PATH = BASE_DIR / "dataset" / "real_cardiology_medicine_department_12000.csv"
OPHTHALMOLOGY_SUPPLEMENT_PATH = BASE_DIR / "dataset" / "ophthalmology_1000_plus_medicines.csv"
ALIAS_DATASET_PATH = BASE_DIR / "dataset" / "medicine_name_aliases.csv"
CSV_COLLECTIONS_DIR = BASE_DIR / "output" / "csv_collections"
ANALYSIS_CACHE_DIR = BASE_DIR / "output" / "analysis_cache"

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PREFERRED_GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemma-4-31b-it").strip() or "gemma-4-31b-it"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'structured_data' not in st.session_state:
    st.session_state.structured_data = None
if 'medications_list' not in st.session_state:
    st.session_state.medications_list = []
if 'export_timestamp' not in st.session_state:
    st.session_state.export_timestamp = None
if 'bulk_results' not in st.session_state:
    st.session_state.bulk_results = []
if 'bulk_processing_complete' not in st.session_state:
    st.session_state.bulk_processing_complete = False
if 'bulk_export_timestamp' not in st.session_state:
    st.session_state.bulk_export_timestamp = None
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'current_image_hash' not in st.session_state:
    st.session_state.current_image_hash = None
if 'bulk_current_hashes' not in st.session_state:
    st.session_state.bulk_current_hashes = []
if 'review_save_message' not in st.session_state:
    st.session_state.review_save_message = None
if 'current_uploaded_image' not in st.session_state:
    st.session_state.current_uploaded_image = None
if 'current_cropped_image' not in st.session_state:
    st.session_state.current_cropped_image = None
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}
if 'export_cache' not in st.session_state:
    st.session_state.export_cache = {}
if 'cache_epoch' not in st.session_state:
    st.session_state.cache_epoch = 0
if 'last_analysis_from_cache' not in st.session_state:
    st.session_state.last_analysis_from_cache = False
if 'single_pure_medicine_name_only' not in st.session_state:
    st.session_state.single_pure_medicine_name_only = True
if 'bulk_pure_medicine_name_only' not in st.session_state:
    st.session_state.bulk_pure_medicine_name_only = True
if 'single_visible_columns' not in st.session_state:
    st.session_state.single_visible_columns = [
        'classification_status',
        'confidence',
        'name_source',
        'department_source',
        'resolution_explanation',
    ]
if 'bulk_visible_columns' not in st.session_state:
    st.session_state.bulk_visible_columns = [
        'classification_status',
    ]
if 'evaluation_report' not in st.session_state:
    st.session_state.evaluation_report = None
if 'evaluation_source_shape' not in st.session_state:
    st.session_state.evaluation_source_shape = None

# Page configuration
st.set_page_config(
    page_title="Smart Prescription Analysis System",
    page_icon="📋",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(21, 147, 127, 0.10), transparent 22%),
            radial-gradient(circle at top left, rgba(30, 64, 175, 0.08), transparent 20%),
            linear-gradient(180deg, #f6fbff 0%, #fbfdfd 100%);
        font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    }
    .main-header {
        padding: 1.75rem 1.9rem;
        background: linear-gradient(135deg, #123a6f 0%, #0f766e 100%);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 24px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.12);
    }
    .main-header h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.15rem;
        letter-spacing: -0.03em;
        font-weight: 800;
    }
    .main-header p {
        margin: 0;
        opacity: 0.92;
        font-size: 1rem;
    }
    .dashboard-note {
        color: #335c67;
        font-size: 0.95rem;
        margin: 0.3rem 0 1rem 0.1rem;
    }
    .section-chip {
        display: inline-block;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        background: #e6f4f1;
        color: #0f766e;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .section-heading {
        margin: 0 0 0.3rem 0;
        color: #123a6f;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .section-copy {
        margin: 0 0 1rem 0;
        color: #58727d;
        font-size: 0.95rem;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.35rem 0;
    }
    .stat-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(18, 58, 111, 0.08);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }
    .stat-label {
        color: #58727d;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }
    .stat-value {
        color: #123a6f;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.05;
    }
    .image-preview-card {
        border-radius: 20px;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(18, 58, 111, 0.08);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        padding: 0.9rem;
        margin-bottom: 0.85rem;
    }
    .image-preview-title {
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #0f766e;
        margin-bottom: 0.45rem;
    }
    .image-preview-sub {
        color: #6b7f86;
        font-size: 0.9rem;
        margin-bottom: 0.65rem;
    }
    .stat-sub {
        color: #6b7f86;
        font-size: 0.86rem;
        margin-top: 0.3rem;
    }
    .panel-shell {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(18, 58, 111, 0.08);
        border-radius: 24px;
        padding: 1.15rem 1.15rem 1.25rem 1.15rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    .sidebar-card {
        padding: 0.85rem 0.95rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(18, 58, 111, 0.08);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.8rem;
    }
    .sidebar-kicker {
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #0f766e;
        margin-bottom: 0.35rem;
    }
    .sidebar-value {
        font-size: 1.05rem;
        font-weight: 800;
        color: #123a6f;
        line-height: 1.2;
    }
    .sidebar-sub {
        color: #6b7f86;
        font-size: 0.86rem;
        margin-top: 0.2rem;
    }
    .sidebar-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.6rem;
        margin: 0.4rem 0 0.8rem 0;
    }
    .sidebar-tile {
        padding: 0.7rem 0.75rem;
        border-radius: 14px;
        background: #f8fcfb;
        border: 1px solid rgba(15, 118, 110, 0.12);
    }
    .sidebar-tile-label {
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #58727d;
        font-weight: 700;
    }
    .sidebar-tile-value {
        margin-top: 0.18rem;
        font-size: 1rem;
        font-weight: 800;
        color: #123a6f;
    }
    .success-message {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fbff 0%, #eff7f7 100%);
        border-right: 1px solid rgba(18, 58, 111, 0.08);
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 14px;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(230, 244, 241, 0.85);
        border-radius: 12px 12px 0 0;
        padding: 0.65rem 1rem;
        font-weight: 700;
    }
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
    }
    @media (max-width: 900px) {
        .stat-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>📋 Smart Prescription Analysis System</h1><p>AI-Powered Medical Text Extraction</p></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-note">Upload prescription images, review corrected medicine names, and export clean department-wise outputs from one focused dashboard.</div>',
    unsafe_allow_html=True,
)

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

Use "Not specified" for any missing information.
Do not add comments, brackets, explanations, conversions, checks, confirmations, repeated sections, or any text outside the exact template.
Under PRESCRIBED MEDICATIONS, each line must contain only:
• [Medication name] - [dosage pattern] for [X] days
List ALL medications found. No extra text or explanations."""

MEDICATION_ONLY_PROMPT = """Read this prescription image and extract only the prescribed medicines.

Return only medicine lines. No headings. No explanations. No extra text.
Each line must follow this format exactly:
[Medication name] - [dosage pattern like 1-0-1] for [X] days

If duration is missing, use:
[Medication name] - [dosage pattern] for Not specified days

If dosage pattern is missing, use:
[Medication name] - Not specified for Not specified days

List all medicines found, one per line."""

FAST_MODEL_CANDIDATES = [PREFERRED_GEMINI_MODEL]

HIGH_ACCURACY_MODEL_CANDIDATES = [PREFERRED_GEMINI_MODEL]

CARDIOLOGY_KEYWORDS = (
    'amlodipine', 'olmesartan', 'telmisartan', 'losartan', 'valsartan',
    'bisoprolol', 'metoprolol', 'atenolol', 'carvedilol', 'nebivolol',
    'clopidogrel', 'aspirin', 'atorvastatin', 'rosuvastatin', 'pitavastatin',
    'ezetimibe', 'fenofibrate', 'warfarin', 'rivaroxaban', 'apixaban',
    'dabigatran', 'furosemide', 'torsemide', 'spironolactone', 'eplerenone',
    'trimetazidine', 'selexipag', 'ambrisentan', 'nifedipine', 'enalapril',
    'ramipril', 'perindopril', 'chlorthalidone', 'moxonidine', 'ivabradine',
    'sacubitril', 'digoxin', 'isosorbide', 'nitroglycerin', 'ticagrelor',
    'prasugrel', 'diltiazem', 'verapamil', 'atorva', 'rosuva', 'cardivas',
    'olmevest', 'olmezest', 'olmesat', 'bisocor', 'bisotab', 'telma',
    'napa extra cardiac'
)

OPHTHALMOLOGY_KEYWORDS = (
    'eye', 'ophthalmic', 'ocular', 'intravitreal', 'tear', 'drop', 'drops',
    'ointment', 'gel', 'solution', 'timolol', 'latanoprost', 'travoprost',
    'brimonidine', 'moxifloxacin', 'olopatadine', 'carboxymethylcellulose',
    'gatifloxacin', 'nepafenac', 'ketorolac', 'dorzolamide', 'bimatoprost',
    'prednisolone acetate', 'cyclopentolate', 'tropicamide', 'lubricant eye',
    'artificial tear'
)


def is_high_accuracy_mode():
    return st.session_state.get("accuracy_mode", "High Accuracy") == "High Accuracy"


def get_active_model_candidates():
    return HIGH_ACCURACY_MODEL_CANDIDATES if is_high_accuracy_mode() else FAST_MODEL_CANDIDATES

def parse_prescription_to_columns(extracted_text):
    """
    Parse extracted text into structured prescription fields
    """
    structured_data = {
        'filename': '',
        'doctor_name': '',
        'doctor_qualifications': '',
        'doctor_designation': '',
        'hospital_clinic': '',
        'doctor_reg_no': '',
        'doctor_contact': '',
        'patient_name': '',
        'patient_age': '',
        'patient_gender': '',
        'prescription_date': '',
        'diagnosis': '',
        'symptoms': '',
        'tests_ordered': '',
        'follow_up_date': ''
    }
    
    # Extract PATIENT INFORMATION
    patient_match = re.search(r'PATIENT INFORMATION:(.*?)(?=DOCTOR INFORMATION:|$)', extracted_text, re.IGNORECASE | re.DOTALL)
    if patient_match:
        patient_text = patient_match.group(1)
        
        name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', patient_text, re.IGNORECASE)
        if name_match:
            structured_data['patient_name'] = name_match.group(1).strip()
        
        age_match = re.search(r'Age:\s*(.+?)(?:\n|$)', patient_text, re.IGNORECASE)
        if age_match:
            structured_data['patient_age'] = age_match.group(1).strip()
        
        gender_match = re.search(r'Gender:\s*(.+?)(?:\n|$)', patient_text, re.IGNORECASE)
        if gender_match:
            structured_data['patient_gender'] = gender_match.group(1).strip()
        
        date_match = re.search(r'Date:\s*(.+?)(?:\n|$)', patient_text, re.IGNORECASE)
        if date_match:
            structured_data['prescription_date'] = date_match.group(1).strip()
    
    # Extract DOCTOR INFORMATION
    doctor_match = re.search(r'DOCTOR INFORMATION:(.*?)(?=MEDICAL NOTES:|$)', extracted_text, re.IGNORECASE | re.DOTALL)
    if doctor_match:
        doctor_text = doctor_match.group(1)
        
        doc_name_match = re.search(r'Doctor Name:\s*(.+?)(?:\n|$)', doctor_text, re.IGNORECASE)
        if doc_name_match:
            structured_data['doctor_name'] = doc_name_match.group(1).strip()
        
        qual_match = re.search(r'Qualifications:\s*(.+?)(?:\n|$)', doctor_text, re.IGNORECASE)
        if qual_match:
            structured_data['doctor_qualifications'] = qual_match.group(1).strip()
        
        hosp_match = re.search(r'Hospital/Clinic:\s*(.+?)(?:\n|$)', doctor_text, re.IGNORECASE)
        if hosp_match:
            structured_data['hospital_clinic'] = hosp_match.group(1).strip()
        
        reg_match = re.search(r'Registration Number:\s*(.+?)(?:\n|$)', doctor_text, re.IGNORECASE)
        if reg_match:
            structured_data['doctor_reg_no'] = reg_match.group(1).strip()
        
        contact_match = re.search(r'Contact:\s*(.+?)(?:\n|$)', doctor_text, re.IGNORECASE)
        if contact_match:
            structured_data['doctor_contact'] = contact_match.group(1).strip()
    
    # Extract MEDICAL NOTES
    medical_match = re.search(r'MEDICAL NOTES:(.*?)(?=PRESCRIBED MEDICATIONS:|$)', extracted_text, re.IGNORECASE | re.DOTALL)
    if medical_match:
        medical_text = medical_match.group(1)
        
        diag_match = re.search(r'Diagnosis:\s*(.+?)(?:\n|$)', medical_text, re.IGNORECASE)
        if diag_match:
            structured_data['diagnosis'] = diag_match.group(1).strip()
        
        symp_match = re.search(r'Symptoms:\s*(.+?)(?:\n|$)', medical_text, re.IGNORECASE)
        if symp_match:
            structured_data['symptoms'] = symp_match.group(1).strip()
        
        tests_match = re.search(r'Tests Ordered:\s*(.+?)(?:\n|$)', medical_text, re.IGNORECASE)
        if tests_match:
            structured_data['tests_ordered'] = tests_match.group(1).strip()
    
    # Extract FOLLOW-UP DATE
    followup_match = re.search(r'FOLLOW-UP DATE:\s*(.+?)(?:\n|$)', extracted_text, re.IGNORECASE)
    if followup_match:
        structured_data['follow_up_date'] = followup_match.group(1).strip()
    
    return structured_data


def clean_medication_name(raw_name):
    name = (raw_name or '').strip()
    name = re.sub(r'^[*•\-]\s*Input\s+text:\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^[*•\-]\s*Input:\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^Input\s+text:\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^Input:\s*', '', name, flags=re.IGNORECASE)
    name = name.strip().strip('"').strip("'")
    name = re.sub(r'^\d+\.\s*', '', name)
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s+(?:before|after)\s+\w+.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+\d+(?:\.\d+)?\s*[\+\-]\s*\d+(?:\.\d+)?\s*[\+\-]\s*\d+(?:\.\d+)?(?:.*)?$', '', name)
    name = re.sub(r'\s*-\s*\d+(?:[.-]\d+)?-\d+(?:[.-]\d+)?-\d+(?:[.-]\d+)?(?:.*)?$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip(' -.')
    name = name.strip().strip('"').strip("'")
    return name


def get_pure_medicine_name(raw_name):
    """Return a stripped-down medicine name without dosage or form words."""
    original_name = clean_medication_name(raw_name)
    if not original_name:
        return ""

    prefix = ""
    prefix_pattern = r'^\s*(?:tab\.?|tablet|cap\.?|capsule|inj\.?|injection|syp\.?|syrup|drop\.?|drops|eye\s+drop\.?|eye\s+drops)\s+'
    prefix_match = re.match(prefix_pattern, original_name, flags=re.IGNORECASE)
    if prefix_match:
        prefix = prefix_match.group(0).strip()
        prefix = re.sub(r'\s+', ' ', prefix)
        prefix = prefix.rstrip('.')
        if prefix.lower() in ('tab', 'tablet'):
            prefix = 'Tab.'
        elif prefix.lower() in ('cap', 'capsule'):
            prefix = 'Cap.'
        elif prefix.lower() in ('inj', 'injection'):
            prefix = 'Inj.'
        elif prefix.lower() in ('syp', 'syrup'):
            prefix = 'Syp.'
        elif prefix.lower() in ('drop', 'drops', 'eye drop', 'eye drops'):
            prefix = 'Drops'
        else:
            prefix = prefix.title()

    name = original_name
    name = re.sub(
        r'^\s*(?:tab\.?|tablet|cap\.?|capsule|inj\.?|injection|syp\.?|syrup|drop\.?|drops|eye\s+drop\.?|eye\s+drops)\s+',
        ' ',
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(
        r'\b(tablet|cap|capsule|inj|injection|syp|syrup|drop|drops|eye|ophthalmic|ointment|gel|solution|suspension|cream|lotion|spray|ear|nasal)\b',
        ' ',
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r'\b\d+(?:\.\d+)?\b', ' ', name)
    name = re.sub(r'\b(mg|mcg|g|ml|iu|%)\b', ' ', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip()
    if prefix:
        return f"{prefix} {name}".strip()
    return name.title()


def prepare_output_dataframe(rows_df, strict_medicine_only=False, selected_columns=None, required_columns=None):
    if rows_df is None:
        return pd.DataFrame()

    output_df = rows_df.copy().fillna('')
    if strict_medicine_only and 'medicine_name' in output_df.columns:
        output_df['medicine_name'] = output_df['medicine_name'].apply(get_pure_medicine_name)
        output_df = output_df[output_df['medicine_name'].astype(str).str.strip() != '']

    required_columns = [column for column in (required_columns or []) if column in output_df.columns]
    if selected_columns:
        visible_columns = required_columns + [
            column for column in selected_columns
            if column in output_df.columns and column not in required_columns
        ]
        if visible_columns:
            output_df = output_df[visible_columns]
    elif required_columns:
        output_df = output_df[required_columns + [
            column for column in output_df.columns if column not in required_columns
        ]]

    return output_df.reset_index(drop=True)

def extract_medications_list(extracted_text):
    """
    Extract ALL medications from PRESCRIBED MEDICATIONS section - Enhanced to capture everything
    """
    medications = []

    def is_valid_medication_candidate(raw_line, cleaned_name):
        line = (raw_line or '').strip()
        name = (cleaned_name or '').strip()
        if not name:
            return False

        upper_line = line.upper()
        blocked_prefixes = (
            'PATIENT INFORMATION',
            'DOCTOR INFORMATION',
            'MEDICAL NOTES',
            'PRESCRIBED MEDICATIONS',
            'FOLLOW-UP DATE',
            'NAME:',
            'AGE:',
            'GENDER:',
            'DATE:',
            'DOCTOR NAME:',
            'QUALIFICATIONS:',
            'HOSPITAL/CLINIC:',
            'REGISTRATION NUMBER:',
            'CONTACT:',
            'DIAGNOSIS:',
            'SYMPTOMS:',
            'TESTS ORDERED:',
            'FOLLOW-UP',
            'EXACT FORMAT',
            'SECTION HEADERS',
            'NO EXTRA TEXT',
            'TASK:',
            'OUTPUT FORMAT',
            'SPECIFIC REQUIREMENTS',
            'CHECK AGAINST',
            'CHECK MEDICATIONS',
            'RX',
        )
        if upper_line.startswith(blocked_prefixes):
            return False

        if ':' in line and not re.match(r'^\d+\.\s*', line):
            return False

        if len(name) < 3:
            return False

        medication_hint = re.search(
            r'\b(tab|tablet|cap|capsule|inj|injection|syp|syrup|drop|drops|ointment|gel|spray)\b',
            line,
            re.IGNORECASE
        )
        strength_hint = re.search(r'\b\d+(?:\.\d+)?\s*(mg|mcg|g|ml|iu|%)\b', line, re.IGNORECASE)
        if medication_hint or strength_hint:
            return True

        known_name = normalize_medication_name(name)
        if known_name:
            seed_df = load_department_seed_data()
            if not seed_df.empty:
                if any(
                    seed_name and (seed_name in known_name or known_name in seed_name)
                    for seed_name in seed_df['normalized_name'].tolist()
                ):
                    return True

        return False

    def append_medication(raw_line, raw_name, dosage_parts=None, duration=''):
        med_name = clean_medication_name(raw_name)
        if not is_valid_medication_candidate(raw_line, med_name):
            return

        dosage_parts = dosage_parts or ['0', '0', '0']
        medications.append({
            'medication_name': med_name,
            'morning': dosage_parts[0] if len(dosage_parts) > 0 else '0',
            'noon': dosage_parts[1] if len(dosage_parts) > 1 else '0',
            'night': dosage_parts[2] if len(dosage_parts) > 2 else '0',
            'duration_days': duration if duration and duration != 'Not specified' else ''
        })
    
    # Find PRESCRIBED MEDICATIONS section
    med_section_match = re.search(r'PRESCRIBED MEDICATIONS:(.*?)(?=FOLLOW-UP DATE:|$)', extracted_text, re.IGNORECASE | re.DOTALL)
    
    if med_section_match:
        med_section = med_section_match.group(1)
        
        # Split into individual lines
        lines = med_section.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('PRESCRIBED MEDICATIONS'):
                continue
            
            # Remove bullet points
            line = re.sub(r'^[•\-*]\s*', '', line)
            
            # Pattern 1: With dosage pattern like "1-0-1"
            # Matches: "Cap. Denver 200 - 1-0-1 for Not specified days"
            pattern1 = r'^(.+?)\s*[-]\s*(\d+(?:[.-]\d+)?[-]\d+(?:[.-]\d+)?[-]\d+(?:[.-]\d+)?)(?:\s*for\s*(.+?)\s*days?)?$'
            match = re.match(pattern1, line, re.IGNORECASE)
            
            if match:
                dosage = match.group(2)
                duration = match.group(3) if match.group(3) else ''
                append_medication(line, match.group(1), dosage.split('-'), duration)
                continue
            
            # Pattern 2: Without dosage pattern (inhalers, nebulization)
            # Matches: "Sulprex HFA Inhaler - Not specified for Not specified days"
            pattern2 = r'^(.+?)\s*[-]\s*Not specified(?:\s*for\s*Not specified\s*days?)?$'
            match = re.match(pattern2, line, re.IGNORECASE)
            
            if match:
                append_medication(line, match.group(1))
                continue
            
            # Pattern 3: Any other medication line
            if line and len(line) > 3 and not line.startswith('Not specified'):
                append_medication(line, line)
    
    # If no medications found with bullet points, try alternative extraction
    if not medications:
        alt_pattern = r'[•\-]\s*(.+?)\s*-\s*(\d+(?:[.-]\d+)?[-]\d+(?:[.-]\d+)?[-]\d+(?:[.-]\d+)?)'
        matches = re.findall(alt_pattern, med_section if med_section_match else extracted_text)
        
        for match in matches:
            append_medication(match[0], match[0], match[1].split('-'))

    if not medications:
        for raw_line in extracted_text.split('\n'):
            line = raw_line.strip()
            if not line:
                continue

            line = re.sub(r'^[•\-*]\s*', '', line)
            line = re.sub(r'^\d+\.\s*', '', line)
            if not line:
                continue

            dosage_match = re.search(
                r'(\d+(?:\.\d+)?)\s*[\+\-]\s*(\d+(?:\.\d+)?)\s*[\+\-]\s*(\d+(?:\.\d+)?)',
                line
            )
            duration_match = re.search(r'for\s+(\d+)\s+days?', line, re.IGNORECASE)
            dosage_parts = None
            if dosage_match:
                dosage_parts = [dosage_match.group(1), dosage_match.group(2), dosage_match.group(3)]

            append_medication(
                line,
                line,
                dosage_parts,
                duration_match.group(1) if duration_match else ''
            )
    
    return medications


def normalize_medication_name(name):
    """Normalize medication text before classification."""
    normalized = (name or "").lower().strip()
    normalized = re.sub(r'^[a-z]+\.\s*', '', normalized)
    normalized = re.sub(r'\b(tab|tablet|cap|capsule|inj|injection|syp|syrup|drop|drops)\b', ' ', normalized)
    normalized = re.sub(r'[^a-z0-9]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def extract_base_medication_name(name):
    """Drop strengths/forms so OCR matching can focus on the core medicine name."""
    normalized = normalize_medication_name(name)
    if not normalized:
        return ""

    base_name = re.sub(r'\b\d+(?:\.\d+)?\s*(mg|mcg|g|ml|iu|%)\b', ' ', normalized)
    base_name = re.sub(r'\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\b', ' ', base_name)
    base_name = re.sub(r'\s+', ' ', base_name).strip()
    return base_name


def compute_bytes_hash(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest() if data_bytes else ""


@st.cache_data
def load_department_seed_data():
    """Load curated medicine labels for rule-based fallback."""
    frames = []
    for path in (
        DEPARTMENT_DATASET_PATH,
        CARDIOLOGY_SUPPLEMENT_PATH,
        OPHTHALMOLOGY_SUPPLEMENT_PATH,
    ):
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                continue

    if not frames:
        return pd.DataFrame(columns=['medicine_name', 'department'])

    seed_df = pd.concat(frames, ignore_index=True)
    if 'medicine_name' not in seed_df.columns or 'department' not in seed_df.columns:
        return pd.DataFrame(columns=['medicine_name', 'department'])

    seed_df = seed_df[['medicine_name', 'department']].copy()
    seed_df['medicine_name'] = seed_df['medicine_name'].astype(str).str.strip()
    seed_df['department'] = seed_df['department'].astype(str).str.strip()
    seed_df = seed_df[(seed_df['medicine_name'] != '') & (seed_df['department'] != '')]
    seed_df = seed_df.drop_duplicates(subset=['medicine_name', 'department'], keep='last')
    seed_df['normalized_name'] = seed_df['medicine_name'].apply(normalize_medication_name)
    seed_df['base_name'] = seed_df['medicine_name'].apply(extract_base_medication_name)
    return seed_df


@st.cache_data
def load_alias_seed_data():
    if not ALIAS_DATASET_PATH.exists():
        return pd.DataFrame(columns=['ocr_name', 'corrected_name', 'department', 'normalized_ocr_name'])

    alias_df = pd.read_csv(ALIAS_DATASET_PATH)
    if alias_df.empty:
        return pd.DataFrame(columns=['ocr_name', 'corrected_name', 'department', 'normalized_ocr_name'])
    alias_df['ocr_name'] = alias_df['ocr_name'].astype(str).str.strip()
    alias_df['corrected_name'] = alias_df['corrected_name'].astype(str).str.strip()
    alias_df['department'] = alias_df['department'].astype(str).str.strip()
    alias_df['normalized_ocr_name'] = alias_df['ocr_name'].apply(normalize_medication_name)
    alias_df['base_ocr_name'] = alias_df['ocr_name'].apply(extract_base_medication_name)
    return alias_df


def find_best_alias_match(medication_name):
    """Return the closest OCR alias match for noisy brand/generic names."""
    normalized_name = normalize_medication_name(medication_name)
    base_name = extract_base_medication_name(medication_name)
    if not normalized_name:
        return None, 0.0

    alias_df = load_alias_seed_data()
    if alias_df.empty:
        return None, 0.0

    best_corrected_name = None
    best_similarity = 0.0

    normalized_candidates = [item for item in alias_df['normalized_ocr_name'].dropna().tolist() if item]
    close_matches = get_close_matches(normalized_name, normalized_candidates, n=1, cutoff=0.78)
    if close_matches:
        best_normalized = close_matches[0]
        similarity = SequenceMatcher(None, normalized_name, best_normalized).ratio()
        best_rows = alias_df[alias_df['normalized_ocr_name'] == best_normalized]
        if not best_rows.empty:
            best_corrected_name = str(best_rows.iloc[0]['corrected_name'])
            best_similarity = similarity

    if base_name:
        base_candidates = [item for item in alias_df['base_ocr_name'].dropna().tolist() if item]
        base_matches = get_close_matches(base_name, base_candidates, n=1, cutoff=0.72)
        if base_matches:
            best_base = base_matches[0]
            similarity = SequenceMatcher(None, base_name, best_base).ratio()
            if similarity > best_similarity:
                best_rows = alias_df[alias_df['base_ocr_name'] == best_base]
                if not best_rows.empty:
                    best_corrected_name = str(best_rows.iloc[0]['corrected_name'])
                    best_similarity = similarity

    if best_corrected_name:
        return best_corrected_name, best_similarity

    return None, 0.0


@st.cache_resource
def load_department_model():
    """Load the locally trained department classifier if available."""
    if not DEPARTMENT_MODEL_PATH.exists():
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            return joblib.load(DEPARTMENT_MODEL_PATH)
    except Exception:
        return None


@st.cache_resource
def load_knn_department_model():
    """Build a simple KNN text classifier from the curated seed dataset."""
    if Pipeline is None or TfidfVectorizer is None or KNeighborsClassifier is None:
        return None

    seed_df = load_department_seed_data()
    if seed_df.empty or 'medicine_name' not in seed_df.columns or 'department' not in seed_df.columns:
        return None

    training_df = seed_df[['medicine_name', 'department']].copy()
    training_df['medicine_name'] = training_df['medicine_name'].astype(str).str.strip()
    training_df['department'] = training_df['department'].astype(str).str.strip()
    training_df = training_df[
        (training_df['medicine_name'] != '') &
        (training_df['department'].isin(['Cardiology', 'Ophthalmology', 'Unknown']))
    ].drop_duplicates()

    if training_df.empty or training_df['department'].nunique() < 2:
        return None

    neighbor_count = max(1, min(5, len(training_df)))
    knn_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ('knn', KNeighborsClassifier(n_neighbors=neighbor_count, weights='distance')),
    ])
    knn_pipeline.fit(training_df['medicine_name'], training_df['department'])
    return knn_pipeline


@st.cache_resource
def load_svm_department_model():
    """Build a linear SVM text classifier from the curated seed dataset."""
    if Pipeline is None or TfidfVectorizer is None or LinearSVC is None:
        return None

    seed_df = load_department_seed_data()
    if seed_df.empty or 'medicine_name' not in seed_df.columns or 'department' not in seed_df.columns:
        return None

    training_df = seed_df[['medicine_name', 'department']].copy()
    training_df['medicine_name'] = training_df['medicine_name'].astype(str).str.strip()
    training_df['department'] = training_df['department'].astype(str).str.strip()
    training_df = training_df[
        (training_df['medicine_name'] != '') &
        (training_df['department'].isin(['Cardiology', 'Ophthalmology', 'Unknown']))
    ].drop_duplicates()

    if training_df.empty or training_df['department'].nunique() < 2:
        return None

    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ('svm', LinearSVC(class_weight='balanced')),
    ])
    svm_pipeline.fit(training_df['medicine_name'], training_df['department'])
    return svm_pipeline


def estimate_classifier_confidence(model, medication_name):
    if model is None:
        return 0.0

    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([medication_name])[0]
            return float(max(probabilities))

        if hasattr(model, 'decision_function'):
            scores = np.asarray(model.decision_function([medication_name]))
            scores = scores.ravel()
            if scores.size == 0:
                return 0.0
            if scores.size == 1:
                return float(1.0 / (1.0 + np.exp(-abs(scores[0]))))
            shifted = scores - np.max(scores)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / np.sum(exp_scores)
            return float(np.max(probabilities))
    except Exception:
        return 0.0

    return 0.75


def predict_with_classifier(model, medication_name, keyword_department, keyword_confidence, source_label):
    if model is None:
        return None

    try:
        predicted_department = model.predict([medication_name])[0]
        confidence = estimate_classifier_confidence(model, medication_name)

        if predicted_department == "Cardiology" and not is_cardiology_like_medication_name(medication_name):
            if keyword_department == "Ophthalmology" and keyword_confidence >= 0.74:
                return {
                    'department': 'Ophthalmology',
                    'department_source': 'keyword_override',
                    'confidence': keyword_confidence,
                }
            if confidence < 0.70 and is_high_accuracy_mode():
                gemini_department = infer_department_with_gemini(medication_name)
                if gemini_department != 'Unknown':
                    return {
                        'department': gemini_department,
                        'department_source': 'gemini_fallback',
                        'confidence': 0.55,
                    }

        if predicted_department == "Ophthalmology" and not is_eye_like_medication_name(medication_name):
            if confidence < 0.90 and is_high_accuracy_mode():
                gemini_department = infer_department_with_gemini(medication_name)
                return {
                    'department': gemini_department,
                    'department_source': 'gemini_fallback' if gemini_department != 'Unknown' else 'unknown',
                    'confidence': 0.55 if gemini_department != 'Unknown' else 0.20,
                }
            return {'department': 'Unknown', 'department_source': 'model_rejected', 'confidence': 0.20}

        if confidence < 0.55:
            if keyword_department != "Unknown" and keyword_confidence >= 0.74:
                return {
                    'department': keyword_department,
                    'department_source': 'keyword_rules',
                    'confidence': keyword_confidence,
                }
            if not is_high_accuracy_mode():
                return {'department': 'Unknown', 'department_source': 'low_confidence', 'confidence': confidence}
            gemini_department = infer_department_with_gemini(medication_name)
            return {
                'department': gemini_department,
                'department_source': 'gemini_fallback' if gemini_department != 'Unknown' else 'unknown',
                'confidence': 0.55 if gemini_department != 'Unknown' else confidence,
            }

        return {
            'department': predicted_department,
            'department_source': source_label,
            'confidence': confidence,
        }
    except Exception:
        return None


def infer_department_from_seed(medication_name):
    """Fallback rule-based department inference from the curated seed list."""
    normalized_name = normalize_medication_name(medication_name)
    base_name = extract_base_medication_name(medication_name)
    if not normalized_name:
        return "Unknown"

    seed_df = load_department_seed_data()
    if seed_df.empty:
        return "Unknown"

    exact_matches = seed_df[seed_df['normalized_name'] == normalized_name]
    if not exact_matches.empty:
        return exact_matches.iloc[0]['department']

    if base_name:
        base_matches = seed_df[seed_df['base_name'] == base_name]
        if not base_matches.empty:
            return base_matches.iloc[0]['department']

    substring_matches = seed_df[
        seed_df['normalized_name'].apply(
            lambda item: item and (item in normalized_name or normalized_name in item)
        )
    ]
    if not substring_matches.empty:
        best_match = substring_matches.iloc[substring_matches['normalized_name'].str.len().argmax()]
        return best_match['department']

    if base_name:
        base_substring_matches = seed_df[
            seed_df['base_name'].apply(
                lambda item: item and (item in base_name or base_name in item)
            )
        ]
        if not base_substring_matches.empty:
            best_match = base_substring_matches.iloc[base_substring_matches['base_name'].str.len().argmax()]
            return best_match['department']

    keyword_department, keyword_confidence = infer_department_from_keywords(medication_name)
    if keyword_confidence >= 0.90:
        return keyword_department

    return "Unknown"


@st.cache_data(show_spinner=False)
def correct_medicine_name_with_gemini(medication_name):
    """Try to clean OCR-distorted medicine names using Gemini."""
    if not is_high_accuracy_mode() or not GEMINI_API_KEY or genai is None:
        return medication_name

    prompt = f"""Read this prescription medicine name and correct only obvious OCR spelling mistakes.

Rules:
- Return only the corrected medicine name.
- Keep strength/dose if present.
- Do not add explanation.
- If you are not confident, return the original text unchanged.

Medicine text:
{medication_name}
"""
    for model_name in get_active_model_candidates():
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            corrected = (getattr(response, 'text', '') or '').strip().splitlines()[0].strip()
            if corrected:
                return corrected
        except Exception:
            continue

    return medication_name


def find_best_seed_match(medication_name):
    """Find the nearest known medicine name from the local dataset."""
    normalized_name = normalize_medication_name(medication_name)
    base_name = extract_base_medication_name(medication_name)
    if not normalized_name:
        return None, 0.0

    seed_df = load_department_seed_data()
    if seed_df.empty:
        return None, 0.0

    normalized_candidates = [item for item in seed_df['normalized_name'].dropna().tolist() if item]
    close_matches = get_close_matches(normalized_name, normalized_candidates, n=1, cutoff=0.82)
    best_match_name = None
    best_similarity = 0.0

    if close_matches:
        best_normalized = close_matches[0]
        similarity = SequenceMatcher(None, normalized_name, best_normalized).ratio()
        best_rows = seed_df[seed_df['normalized_name'] == best_normalized]
        if not best_rows.empty:
            best_match_name = best_rows.iloc[0]['medicine_name']
            best_similarity = similarity

    if base_name:
        base_candidates = [item for item in seed_df['base_name'].dropna().tolist() if item]
        base_close_matches = get_close_matches(base_name, base_candidates, n=1, cutoff=0.74)
        if base_close_matches:
            best_base = base_close_matches[0]
            base_similarity = SequenceMatcher(None, base_name, best_base).ratio()
            if base_similarity > best_similarity:
                best_rows = seed_df[seed_df['base_name'] == best_base]
                if not best_rows.empty:
                    best_match_name = best_rows.iloc[0]['medicine_name']
                    best_similarity = base_similarity

    if best_match_name:
        return best_match_name, best_similarity

    return None, 0.0


def resolve_medicine_name_for_classification(medication_name):
    """Normalize OCR-noisy medicine names before department classification."""
    if is_obviously_invalid_medication_name(medication_name):
        return medication_name

    alias_df = load_alias_seed_data()
    normalized_name = normalize_medication_name(medication_name)
    if normalized_name and not alias_df.empty:
        alias_match = alias_df[alias_df['normalized_ocr_name'] == normalized_name]
        if not alias_match.empty:
            return str(alias_match.iloc[0]['corrected_name'])

    alias_best_match, alias_similarity = find_best_alias_match(medication_name)
    if alias_best_match and alias_similarity >= 0.82:
        return str(alias_best_match)

    best_match, similarity = find_best_seed_match(medication_name)
    if best_match and similarity >= 0.84:
        return str(best_match)

    corrected_name = correct_medicine_name_with_gemini(medication_name)
    corrected_match, corrected_similarity = find_best_seed_match(corrected_name)
    if corrected_match and corrected_similarity >= 0.84:
        return str(corrected_match)

    return corrected_name or medication_name


def get_name_resolution_source(original_name, resolved_name):
    if (original_name or '').strip() == (resolved_name or '').strip():
        alias_df = load_alias_seed_data()
        normalized_name = normalize_medication_name(original_name)
        if normalized_name and not alias_df.empty and any(alias_df['normalized_ocr_name'] == normalized_name):
            return "alias"
        return "original"

    alias_df = load_alias_seed_data()
    normalized_name = normalize_medication_name(original_name)
    if normalized_name and not alias_df.empty and any(alias_df['normalized_ocr_name'] == normalized_name):
        return "alias"

    best_match, similarity = find_best_seed_match(original_name)
    if best_match and best_match == resolved_name and similarity >= 0.88:
        return "fuzzy_match"

    return "gemini_correction"


def refine_unknown_medications_with_image(medications, target_bytes, target_mime_type):
    """Ask Gemini to re-check only unknown medicines against the original image."""
    if not is_high_accuracy_mode() or not medications or not GEMINI_API_KEY or genai is None:
        return medications

    refined_medications = []

    for medication in medications:
        item = medication.copy()
        if item.get('department') != 'Unknown':
            refined_medications.append(item)
            continue

        candidate_name = item.get('medication_name', '') or item.get('original_medication_name', '')
        if not candidate_name or is_obviously_invalid_medication_name(candidate_name):
            refined_medications.append(item)
            continue

        prompt = f"""Look at this prescription image and review one uncertain medicine name.

Candidate medicine text:
{candidate_name}

Allowed departments:
Cardiology
Ophthalmology
Unknown

Rules:
- Correct only obvious OCR mistakes if you are confident.
- If you are not confident about the name, keep the original candidate name.
- Return Unknown unless the medicine is clearly Cardiology or clearly Ophthalmology.
- Do not guess.

Return valid JSON only with this shape:
{{"corrected_name":"...", "department":"Cardiology|Ophthalmology|Unknown"}}"""

        reviewed = False
        for model_name in get_active_model_candidates():
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([
                    prompt,
                    {"mime_type": target_mime_type, "data": target_bytes}
                ])
                raw_text = (getattr(response, 'text', '') or '').strip()
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if not json_match:
                    continue

                payload = json.loads(json_match.group(0))
                corrected_name = str(payload.get('corrected_name', '') or candidate_name).strip()
                reviewed_department = str(payload.get('department', 'Unknown') or 'Unknown').strip()
                if reviewed_department not in ('Cardiology', 'Ophthalmology', 'Unknown'):
                    reviewed_department = 'Unknown'

                resolved_name = resolve_medicine_name_for_classification(corrected_name)
                final_department = reviewed_department
                if final_department == 'Unknown':
                    final_department = classify_department(resolved_name)

                item['original_medication_name'] = item.get('original_medication_name', candidate_name)
                item['medication_name'] = resolved_name
                item['department'] = final_department
                reviewed = True
                break
            except Exception:
                continue

        refined_medications.append(item if reviewed else item)

    return refined_medications


@st.cache_data(show_spinner=False)
def infer_department_with_gemini(medication_name):
    """Use Gemini only as a backup when local matching cannot classify a medicine."""
    if not GEMINI_API_KEY or genai is None:
        return "Unknown"

    normalized_name = normalize_medication_name(medication_name)
    eye_keywords = OPHTHALMOLOGY_KEYWORDS

    prompt = f"""Classify this medicine into exactly one department.

Allowed labels only:
Cardiology
Ophthalmology
Unknown

Medicine: {medication_name}

Rules:
- Return Unknown unless the medicine is clearly and specifically used for Cardiology or clearly and specifically used for Ophthalmology.
- Do not guess.
- If the medicine belongs to Gastroenterology, General Medicine, Neurology, Psychiatry, Pulmonology, Endocrinology, Infectious Disease, Pain, or any other department, return Unknown.
- If the medicine name is unclear, incomplete, or you are not fully certain, return Unknown.

Return only one label."""

    for model_name in get_active_model_candidates():
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw_text = (getattr(response, 'text', '') or '').strip()
            label = raw_text.splitlines()[0].strip().lower()

            if label == 'ophthalmology':
                if any(keyword in normalized_name for keyword in eye_keywords):
                    return "Ophthalmology"
                return "Unknown"
            if label == 'cardiology':
                return "Cardiology"
            if label == 'unknown':
                return "Unknown"

            if 'ophthalmology' in label:
                if any(keyword in normalized_name for keyword in eye_keywords):
                    return "Ophthalmology"
                return "Unknown"
            if 'cardiology' in label:
                return "Cardiology"
            if 'unknown' in label:
                return "Unknown"
        except Exception:
            continue

    return "Unknown"


def is_eye_like_medication_name(medication_name):
    normalized_name = normalize_medication_name(medication_name)
    return any(keyword in normalized_name for keyword in OPHTHALMOLOGY_KEYWORDS)


def is_cardiology_like_medication_name(medication_name):
    normalized_name = normalize_medication_name(medication_name)
    return any(keyword in normalized_name for keyword in CARDIOLOGY_KEYWORDS)


def infer_department_from_keywords(medication_name):
    """Use department-specific medicine keywords before falling back to the model."""
    normalized_name = normalize_medication_name(medication_name)
    base_name = extract_base_medication_name(medication_name)
    haystacks = [item for item in (normalized_name, base_name) if item]
    if not haystacks:
        return "Unknown", 0.0

    cardiology_hits = sum(
        1 for keyword in CARDIOLOGY_KEYWORDS
        if any(keyword in haystack for haystack in haystacks)
    )
    ophthalmology_hits = sum(
        1 for keyword in OPHTHALMOLOGY_KEYWORDS
        if any(keyword in haystack for haystack in haystacks)
    )

    if ophthalmology_hits and not cardiology_hits:
        return "Ophthalmology", min(0.96, 0.82 + ophthalmology_hits * 0.06)
    if cardiology_hits and not ophthalmology_hits:
        return "Cardiology", min(0.96, 0.82 + cardiology_hits * 0.04)
    if cardiology_hits and ophthalmology_hits:
        if ophthalmology_hits >= cardiology_hits + 1:
            return "Ophthalmology", 0.74
        if cardiology_hits >= ophthalmology_hits + 1:
            return "Cardiology", 0.74

    return "Unknown", 0.0


def is_obviously_invalid_medication_name(medication_name):
    lowered = (medication_name or '').strip().lower()
    blocked_fragments = (
        'not clearly marked',
        'illegible',
        "let's",
        're-examine',
        'it says',
        'month',
        'unclear',
        'comment',
        'explanation',
        'conversion',
        'check',
        'confirm',
    )
    return any(fragment in lowered for fragment in blocked_fragments)


def classify_department(medication_name):
    """Predict department using the trained model with a rule-based fallback."""
    return classify_department_details(medication_name)['department']


def classify_department_details(medication_name):
    """Predict department and expose source/confidence for review."""
    if is_obviously_invalid_medication_name(medication_name):
        return {'department': 'Unknown', 'department_source': 'invalid_text', 'confidence': 0.0}

    fallback_department = infer_department_from_seed(medication_name)
    if fallback_department != "Unknown":
        return {'department': fallback_department, 'department_source': 'dataset', 'confidence': 0.99}

    keyword_department, keyword_confidence = infer_department_from_keywords(medication_name)
    if keyword_department != "Unknown" and keyword_confidence >= 0.86:
        return {
            'department': keyword_department,
            'department_source': 'keyword_rules',
            'confidence': keyword_confidence,
        }

    model = load_department_model()
    model_result = predict_with_classifier(model, medication_name, keyword_department, keyword_confidence, 'local_model')
    if model_result is not None:
        return model_result

    svm_model = load_svm_department_model()
    svm_result = predict_with_classifier(svm_model, medication_name, keyword_department, keyword_confidence, 'svm_model')
    if svm_result is not None:
        return svm_result

    knn_model = load_knn_department_model()
    knn_result = predict_with_classifier(knn_model, medication_name, keyword_department, keyword_confidence, 'knn_model')
    if knn_result is not None:
        return knn_result

    if not is_high_accuracy_mode():
        return {'department': 'Unknown', 'department_source': 'no_local_match', 'confidence': 0.20}

    gemini_department = infer_department_with_gemini(medication_name)
    return {
        'department': gemini_department,
        'department_source': 'gemini_fallback' if gemini_department != 'Unknown' else 'unknown',
        'confidence': 0.55 if gemini_department != 'Unknown' else 0.20,
    }


def classify_medications_by_department(medications):
    """Append department labels to each extracted medication."""
    classified_medications = []

    for medication in medications:
        item = medication.copy()
        original_name = item.get('medication_name', '')
        resolved_name = resolve_medicine_name_for_classification(original_name)
        item['original_medication_name'] = original_name
        item['medication_name'] = resolved_name
        item['name_resolution_source'] = get_name_resolution_source(original_name, resolved_name)
        department_details = classify_department_details(resolved_name)
        item['department'] = department_details['department']
        item['department_source'] = department_details['department_source']
        item['confidence'] = department_details['confidence']
        classified_medications.append(item)

    return classified_medications


def build_resolution_explanation(medication):
    department = str(medication.get('department', 'Unknown') or 'Unknown').strip()
    name_source = str(medication.get('name_resolution_source', 'original') or 'original').strip()
    department_source = str(medication.get('department_source', 'unknown') or 'unknown').strip()
    confidence = float(medication.get('confidence', 0.0) or 0.0)
    confidence_text = f"{int(round(confidence * 100))}%"

    source_labels = {
        'dataset': 'dataset match',
        'keyword_rules': 'keyword rules',
        'keyword_override': 'keyword override',
        'local_model': 'local model',
        'svm_model': 'svm model',
        'knn_model': 'knn model',
        'gemini_fallback': 'AI fallback',
        'unknown': 'uncertain match',
        'low_confidence': 'low confidence',
        'model_rejected': 'model rejected',
        'invalid_text': 'invalid text',
        'manual_review': 'manual review',
        'original': 'original extraction',
    }

    name_source_label = source_labels.get(name_source, name_source.replace('_', ' '))
    department_source_label = source_labels.get(department_source, department_source.replace('_', ' '))
    return f"{department} via {department_source_label}; name resolved from {name_source_label}; confidence {confidence_text}"


def build_clean_medication_rows(medications):
    """Conservative cleaner: remove obvious wrappers but avoid dropping valid medicine lines."""
    clean_rows = []
    seen_rows = set()

    blocked_fragments = (
        'patient information',
        'doctor information',
        'medical notes',
        'follow-up',
        'diagnosis',
        'symptoms',
        'tests ordered',
        'exact format',
        'section headers',
        'no extra text',
        'task:',
        'output format',
        'specific requirements',
        'check against',
        'check medications',
    )

    for med in medications or []:
        medication_name = clean_medication_name((med.get('medication_name', '') or '').strip())
        department = (med.get('department', 'Unknown') or 'Unknown').strip()
        if not medication_name:
            continue

        lowered = medication_name.lower()
        if any(fragment in lowered for fragment in blocked_fragments):
            continue
        if ':' in medication_name and not re.match(r'^\d+\.\s*', medication_name):
            continue
        if len(medication_name) > 80:
            continue

        normalized_name = normalize_medication_name(medication_name)
        row_key = (normalized_name, department)
        if not normalized_name or row_key in seen_rows:
            continue
        seen_rows.add(row_key)

        clean_rows.append({
            'medicine_name': medication_name,
            'department': department,
            'classification_status': 'Unknown' if department == 'Unknown' else 'Known',
            'confidence': round(float(med.get('confidence', 0.0) or 0.0), 2),
            'name_source': med.get('name_resolution_source', 'original'),
            'department_source': med.get('department_source', 'unknown'),
            'resolution_explanation': build_resolution_explanation(med),
        })

    if clean_rows:
        return clean_rows

    fallback_rows = []
    fallback_seen = set()
    for med in medications or []:
        medication_name = clean_medication_name((med.get('medication_name', '') or '').strip())
        department = (med.get('department', 'Unknown') or 'Unknown').strip()
        normalized_name = normalize_medication_name(medication_name)
        row_key = (normalized_name, department)
        if not medication_name or not normalized_name or row_key in fallback_seen:
            continue
        fallback_seen.add(row_key)
        fallback_rows.append({
            'medicine_name': medication_name,
            'department': department,
            'classification_status': 'Unknown' if department == 'Unknown' else 'Known',
            'confidence': round(float(med.get('confidence', 0.0) or 0.0), 2),
            'name_source': med.get('name_resolution_source', 'original'),
            'department_source': med.get('department_source', 'unknown'),
            'resolution_explanation': build_resolution_explanation(med),
        })

    return fallback_rows


def build_review_rows(medications):
    review_rows = []
    for med in medications or []:
        medication_name = (med.get('medication_name', '') or '').strip()
        if not medication_name:
            continue
        if is_obviously_invalid_medication_name(medication_name):
            continue
        review_rows.append({
            'original_name': (med.get('original_medication_name', medication_name) or '').strip(),
            'corrected_name': medication_name,
            'department': (med.get('department', 'Unknown') or 'Unknown').strip(),
            'classification_status': 'Unknown' if str(med.get('department', 'Unknown') or 'Unknown').strip() == 'Unknown' else 'Known',
            'name_source': med.get('name_resolution_source', 'original'),
            'department_source': med.get('department_source', 'unknown'),
            'confidence': round(float(med.get('confidence', 0.0) or 0.0), 2),
            'resolution_explanation': build_resolution_explanation(med),
        })
    return review_rows


def build_department_summary_df(rows_df, image_count=1):
    if rows_df is None or rows_df.empty:
        return pd.DataFrame(columns=['metric', 'value'])

    summary_rows = [
        {'metric': 'Images', 'value': int(image_count)},
        {'metric': 'Total Medicines', 'value': int(len(rows_df))},
    ]
    for department in ['Cardiology', 'Ophthalmology', 'Unknown']:
        summary_rows.append({
            'metric': f'{department} Count',
            'value': int((rows_df['department'] == department).sum()),
        })
    return pd.DataFrame(summary_rows)


def normalize_department_label(value):
    normalized = str(value or '').strip().lower()
    if normalized.startswith('cardio'):
        return 'Cardiology'
    if normalized.startswith('ophthal'):
        return 'Ophthalmology'
    if normalized == 'unknown':
        return 'Unknown'
    return str(value or '').strip().title() if str(value or '').strip() else 'Unknown'


def build_evaluation_report(eval_df, medicine_col='medicine_name', truth_col='true_department', predicted_col=None):
    if eval_df is None or eval_df.empty:
        raise ValueError("Evaluation CSV is empty.")
    if medicine_col not in eval_df.columns:
        raise ValueError(f"Missing medicine column: {medicine_col}")
    if truth_col not in eval_df.columns:
        raise ValueError(f"Missing ground truth column: {truth_col}")
    if predicted_col and predicted_col not in eval_df.columns:
        raise ValueError(f"Missing predicted column: {predicted_col}")

    working_df = eval_df.copy().fillna('')
    y_true = working_df[truth_col].apply(normalize_department_label)

    if predicted_col:
        y_pred = working_df[predicted_col].apply(normalize_department_label)
    else:
        y_pred = working_df[medicine_col].astype(str).apply(
            lambda name: classify_department_details(name).get('department', 'Unknown')
        )

    labels = ['Cardiology', 'Ophthalmology', 'Unknown']
    y_true_arr = y_true.astype(str).to_numpy()
    y_pred_arr = y_pred.astype(str).to_numpy()
    total = len(working_df)
    accuracy = float((y_true_arr == y_pred_arr).mean()) if total else 0.0
    unknown_rate = float((y_pred_arr == 'Unknown').mean()) if total else 0.0
    true_unknown_rate = float((y_true_arr == 'Unknown').mean()) if total else 0.0

    per_class_rows = []
    for label in labels:
        tp = int(((y_true_arr == label) & (y_pred_arr == label)).sum())
        fp = int(((y_true_arr != label) & (y_pred_arr == label)).sum())
        fn = int(((y_true_arr == label) & (y_pred_arr != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = int((y_true_arr == label).sum())
        per_class_rows.append({
            'class': label,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'support': support,
        })

    macro_precision = float(np.mean([row['precision'] for row in per_class_rows])) if per_class_rows else 0.0
    macro_recall = float(np.mean([row['recall'] for row in per_class_rows])) if per_class_rows else 0.0
    macro_f1 = float(np.mean([row['f1_score'] for row in per_class_rows])) if per_class_rows else 0.0
    confusion_matrix = pd.crosstab(
        pd.Categorical(y_true, categories=labels),
        pd.Categorical(y_pred, categories=labels),
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    return {
        'total': total,
        'accuracy': round(accuracy, 4),
        'unknown_rate': round(unknown_rate, 4),
        'true_unknown_rate': round(true_unknown_rate, 4),
        'macro_precision': round(macro_precision, 4),
        'macro_recall': round(macro_recall, 4),
        'macro_f1': round(macro_f1, 4),
        'per_class_df': pd.DataFrame(per_class_rows),
        'confusion_matrix_df': confusion_matrix,
    }


def save_alias_corrections(review_df):
    if review_df is None or review_df.empty:
        return {'saved_aliases': 0, 'saved_dataset_rows': 0}

    alias_rows = []
    dataset_rows = []
    for _, row in review_df.iterrows():
        original_name = str(row.get('original_name', '') or '').strip()
        corrected_name = str(row.get('corrected_name', '') or '').strip()
        department = str(row.get('department', 'Unknown') or 'Unknown').strip()
        if not corrected_name:
            continue

        if original_name and corrected_name and normalize_medication_name(original_name) != normalize_medication_name(corrected_name):
            alias_rows.append({
                'ocr_name': original_name,
                'corrected_name': corrected_name,
                'department': department,
            })

        if department in ('Cardiology', 'Ophthalmology'):
            dataset_rows.append({
                'medicine_name': corrected_name,
                'department': department,
                'source_group': 'Manual_Review',
            })

    saved_aliases = 0
    if alias_rows:
        alias_df = pd.DataFrame(alias_rows).drop_duplicates(subset=['ocr_name', 'corrected_name', 'department'])
        if ALIAS_DATASET_PATH.exists():
            existing_alias_df = pd.read_csv(ALIAS_DATASET_PATH)
        else:
            existing_alias_df = pd.DataFrame(columns=['ocr_name', 'corrected_name', 'department'])
        merged_alias_df = pd.concat([existing_alias_df, alias_df], ignore_index=True)
        merged_alias_df = merged_alias_df.drop_duplicates(subset=['ocr_name', 'corrected_name', 'department'], keep='last')
        merged_alias_df.to_csv(ALIAS_DATASET_PATH, index=False)
        load_alias_seed_data.clear()
        saved_aliases = int(len(alias_df))

    saved_dataset_rows = 0
    if dataset_rows:
        merge_result = merge_uploaded_department_dataset(pd.DataFrame(dataset_rows))
        saved_dataset_rows = int(merge_result['added_rows'])

    if saved_aliases or saved_dataset_rows:
        invalidate_all_caches()

    return {'saved_aliases': saved_aliases, 'saved_dataset_rows': saved_dataset_rows}


def apply_review_corrections_to_medications(medications, review_df):
    """Apply manual review edits back to the in-session medications list."""
    if review_df is None or review_df.empty:
        return medications

    updated_medications = []
    review_rows = review_df.to_dict('records')
    review_index = 0

    for med in medications or []:
        item = med.copy()
        medication_name = (item.get('medication_name', '') or '').strip()
        if not medication_name or is_obviously_invalid_medication_name(medication_name):
            updated_medications.append(item)
            continue

        if review_index >= len(review_rows):
            updated_medications.append(item)
            continue

        review_row = review_rows[review_index]
        review_index += 1

        corrected_name = str(review_row.get('corrected_name', medication_name) or medication_name).strip()
        department = str(review_row.get('department', item.get('department', 'Unknown')) or 'Unknown').strip()

        if corrected_name:
            item['original_medication_name'] = str(
                review_row.get('original_name', item.get('original_medication_name', medication_name)) or medication_name
            ).strip()
            item['medication_name'] = corrected_name
            item['name_resolution_source'] = 'manual_review'

        item['department'] = department if department in ('Cardiology', 'Ophthalmology', 'Unknown') else 'Unknown'
        item['department_source'] = 'manual_review'
        item['confidence'] = 1.0
        updated_medications.append(item)

    return updated_medications


def normalize_history_hashes(image_hash):
    if isinstance(image_hash, (list, tuple, set)):
        return [str(item).strip() for item in image_hash if str(item).strip()]

    text = str(image_hash or '').strip()
    if not text:
        return []

    if '|' in text:
        return [item.strip() for item in text.split('|') if item.strip()]

    return [text]


def history_contains_image_hash(image_hash):
    target_hash = str(image_hash or '').strip()
    if not target_hash:
        return False

    for entry in st.session_state.processing_history:
        if target_hash in normalize_history_hashes(entry.get('image_hashes', entry.get('image_hash', ''))):
            return True

    return False


def add_processing_history_entry(mode, filename, image_hash, rows_df):
    rows_df = rows_df if rows_df is not None else pd.DataFrame(columns=['medicine_name', 'department'])
    history = st.session_state.processing_history
    history_hashes = normalize_history_hashes(image_hash)
    history.insert(0, {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': mode,
        'file': filename,
        'image_hash': '|'.join(history_hashes),
        'image_hashes': history_hashes,
        'rows': int(len(rows_df)),
        'cardiology': int((rows_df['department'] == 'Cardiology').sum()) if 'department' in rows_df else 0,
        'ophthalmology': int((rows_df['department'] == 'Ophthalmology').sum()) if 'department' in rows_df else 0,
        'unknown': int((rows_df['department'] == 'Unknown').sum()) if 'department' in rows_df else 0,
    })
    st.session_state.processing_history = history[:20]


def build_bulk_zip_bytes(bulk_df, bulk_data, bulk_filename):
    zip_buffer = io.BytesIO()
    summary_df = build_department_summary_df(bulk_df, image_count=bulk_df['filename'].nunique() if 'filename' in bulk_df else 1)
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(bulk_filename, bulk_data)
        zip_file.writestr('bulk_summary.csv', summary_df.to_csv(index=False))
        for filename, group_df in bulk_df.groupby('filename'):
            safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(filename))
            zip_file.writestr(f'per_image/{safe_name}.csv', group_df.to_csv(index=False))
    return zip_buffer.getvalue()


def safe_dataframe(dataframe, **kwargs):
    kwargs.pop('hide_index', None)
    st.dataframe(dataframe, **kwargs)


def slugify_cache_part(value):
    return re.sub(r'[^a-z0-9._-]+', '_', str(value).strip().lower()) or "default"


def encode_cache_blob(data):
    if data is None:
        return ""
    return base64.b64encode(data).decode("ascii")


def decode_cache_blob(data):
    if not data:
        return b""
    return base64.b64decode(data.encode("ascii"))


def build_export_signature(dataframe, extra_parts=None):
    normalized_df = dataframe.fillna('').copy()
    payload_parts = [f"epoch_{st.session_state.get('cache_epoch', 0)}", normalized_df.to_csv(index=False)]
    if extra_parts:
        payload_parts.extend(str(part) for part in extra_parts if part is not None)
    signature_source = "\n".join(payload_parts)
    return hashlib.sha256(signature_source.encode('utf-8')).hexdigest()


def build_export_cache_key(export_kind, signature):
    return "__".join([
        slugify_cache_part(export_kind),
        str(signature or "").strip(),
    ])


def get_export_cache_path(cache_key):
    ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_CACHE_DIR / f"{cache_key}.json"


def load_cached_export(cache_key):
    cached = st.session_state.export_cache.get(cache_key)
    if cached is not None:
        return cached

    cache_path = get_export_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        st.session_state.export_cache[cache_key] = cached
        return cached
    except Exception:
        return None


def store_cached_export(cache_key, payload):
    st.session_state.export_cache[cache_key] = payload
    cache_path = get_export_cache_path(cache_key)
    try:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_analysis_cache_key(image_hash, extraction_mode, accuracy_mode):
    return "__".join([
        str(image_hash or "").strip(),
        slugify_cache_part(extraction_mode),
        slugify_cache_part(accuracy_mode),
    ])


def get_analysis_cache_path(cache_key):
    ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_CACHE_DIR / f"{cache_key}.json"


def load_cached_analysis(cache_key):
    cached = st.session_state.analysis_cache.get(cache_key)
    if cached is not None:
        return cached

    cache_path = get_analysis_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        st.session_state.analysis_cache[cache_key] = cached
        return cached
    except Exception:
        return None


def store_cached_analysis(cache_key, payload):
    st.session_state.analysis_cache[cache_key] = payload
    cache_path = get_analysis_cache_path(cache_key)
    try:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass


def clear_analysis_cache():
    st.session_state.analysis_cache = {}
    st.session_state.export_cache = {}
    if ANALYSIS_CACHE_DIR.exists():
        for cache_file in ANALYSIS_CACHE_DIR.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception:
                pass


def invalidate_all_caches():
    st.session_state.cache_epoch = int(st.session_state.get('cache_epoch', 0)) + 1
    clear_analysis_cache()


def run_cached_extraction(full_image_bytes, full_image_mime_type, extraction_mode, image_hash, image=None):
    accuracy_mode = st.session_state.get("accuracy_mode", "High Accuracy")
    cache_key = build_analysis_cache_key(image_hash, extraction_mode, accuracy_mode)
    cached_result = load_cached_analysis(cache_key)

    if cached_result is not None:
        selected_result = {
            'extracted_text': cached_result.get('extracted_text', ''),
            'medications': cached_result.get('medications', []),
            'model_used': cached_result.get('model_used', 'cache'),
        }
        cached_image = image if image is not None else Image.open(io.BytesIO(full_image_bytes)).convert('RGB')
        cropped_image = auto_crop_medication_region(cached_image) if extraction_mode in ("Use auto crop", "Use both and keep better result") else None
        return selected_result, cropped_image, True

    cached_image = image if image is not None else Image.open(io.BytesIO(full_image_bytes)).convert('RGB')
    selected_result, cropped_image = run_extraction_with_mode(
        cached_image,
        full_image_bytes,
        full_image_mime_type,
        extraction_mode,
    )
    store_cached_analysis(cache_key, {
        'image_hash': image_hash,
        'extraction_mode': extraction_mode,
        'accuracy_mode': accuracy_mode,
        'extracted_text': selected_result.get('extracted_text', ''),
        'medications': selected_result.get('medications', []),
        'model_used': selected_result.get('model_used', ''),
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    return selected_result, cropped_image, False


def build_single_export_bundle(df):
    spreadsheet_filename, spreadsheet_label, spreadsheet_mime, spreadsheet_data = build_spreadsheet_export_file(df, "export")

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
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#667eea'),
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName='Helvetica-Bold'
    )

    story.append(Paragraph("Medicine Department Classification", title_style))
    story.append(Spacer(1, 20))

    export_rows = df.to_dict('records')
    if export_rows:
        grouped_rows = {}
        for row in export_rows:
            grouped_rows.setdefault(row['department'], []).append(row['medicine_name'])

        for department, names in grouped_rows.items():
            story.append(Paragraph(escape(department), styles['Heading2']))
            story.append(Spacer(1, 8))
            for index, medication_name in enumerate(names, start=1):
                story.append(Paragraph(f"{index}. {escape(medication_name)}", styles['Normal']))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No medicine names found.", styles['Normal']))

    doc.build(story)
    pdf_data = pdf_buffer.getvalue()

    return {
        'spreadsheet_filename': spreadsheet_filename,
        'spreadsheet_label': spreadsheet_label,
        'spreadsheet_mime': spreadsheet_mime,
        'spreadsheet_data': spreadsheet_data,
        'pdf_data': pdf_data,
    }


def build_bulk_export_bundle(bulk_df):
    bulk_filename, bulk_label, bulk_mime, bulk_data = build_spreadsheet_export_file(
        bulk_df,
        "bulk_export"
    )
    bulk_zip_data = build_bulk_zip_bytes(bulk_df, bulk_data, bulk_filename)
    return {
        'bulk_filename': bulk_filename,
        'bulk_label': bulk_label,
        'bulk_mime': bulk_mime,
        'bulk_data': bulk_data,
        'bulk_zip_data': bulk_zip_data,
    }


def render_dashboard_section(chip, heading, copy):
    st.markdown(f"**{chip.upper()}**")
    st.subheader(heading)
    st.caption(copy)


def render_stat_cards(cards):
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.metric(card["label"], card["value"])
            st.caption(card["subtext"])


def merge_uploaded_department_dataset(uploaded_dataset_df):
    """Merge a user-uploaded medicine dataset into the local seed dataset."""
    required_columns = {'medicine_name', 'department'}
    normalized_columns = {column.strip().lower(): column for column in uploaded_dataset_df.columns}
    if 'department' not in normalized_columns and 'department_name' in normalized_columns:
        normalized_columns['department'] = normalized_columns['department_name']
    if not required_columns.issubset(normalized_columns.keys()):
        missing_columns = sorted(required_columns - set(normalized_columns.keys()))
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    renamed_df = uploaded_dataset_df.rename(columns={
        normalized_columns['medicine_name']: 'medicine_name',
        normalized_columns['department']: 'department',
    }).copy()

    renamed_df['medicine_name'] = renamed_df['medicine_name'].astype(str).str.strip()
    renamed_df['department'] = renamed_df['department'].astype(str).str.strip()
    renamed_df = renamed_df[(renamed_df['medicine_name'] != '') & (renamed_df['department'] != '')]
    renamed_df['source_group'] = renamed_df.get('source_group', 'User_Upload')
    renamed_df = renamed_df.drop_duplicates(subset=['medicine_name', 'department'])

    if renamed_df.empty:
        raise ValueError("Uploaded CSV has no valid medicine_name and department rows.")

    if DEPARTMENT_DATASET_PATH.exists():
        existing_df = pd.read_csv(DEPARTMENT_DATASET_PATH)
    else:
        existing_df = pd.DataFrame(columns=['medicine_name', 'department', 'source_group'])

    existing_df['medicine_name'] = existing_df['medicine_name'].astype(str).str.strip()
    existing_df['department'] = existing_df['department'].astype(str).str.strip()
    if 'source_group' not in existing_df.columns:
        existing_df['source_group'] = 'Seed'
    merged_df = pd.concat([
        existing_df[['medicine_name', 'department', 'source_group']],
        renamed_df[['medicine_name', 'department', 'source_group']]
    ], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['medicine_name', 'department'], keep='last')
    merged_df = merged_df.sort_values(by=['department', 'medicine_name']).reset_index(drop=True)
    merged_df.to_csv(DEPARTMENT_DATASET_PATH, index=False)

    load_department_seed_data.clear()
    load_department_model.clear()
    load_svm_department_model.clear()
    load_knn_department_model.clear()
    invalidate_all_caches()
    return {
        'added_rows': int(len(renamed_df)),
        'total_rows': int(len(merged_df)),
        'labels': sorted(merged_df['department'].astype(str).str.strip().unique().tolist()),
        'report_text': 'Dataset updated. Rule-based matching refreshed. No local retraining required.',
    }


def render_dataset_upload_section(widget_prefix="main"):
    st.subheader("Dataset Upload")
    st.caption("Upload a CSV with `medicine_name` and `department` columns to extend the classifier dataset.")

    department_dataset_file = st.file_uploader(
        "Upload department dataset CSV",
        type=['csv'],
        key=f"department_dataset_uploader_{widget_prefix}"
    )

    if department_dataset_file is not None:
        try:
            preview_df = pd.read_csv(department_dataset_file)
            st.caption(f"Rows found: {len(preview_df)}")
            st.dataframe(preview_df.head(5), use_container_width=True)
            department_dataset_file.seek(0)

            if st.button("Add To Dataset", use_container_width=True, key=f"add_dataset_button_{widget_prefix}"):
                try:
                    upload_df = pd.read_csv(department_dataset_file)
                    result = merge_uploaded_department_dataset(upload_df)
                    st.success(
                        f"Added {result['added_rows']} row(s). "
                        f"Dataset now has {result['total_rows']} row(s). "
                        f"Labels: {', '.join(result['labels'])}"
                    )
                    st.text(result['report_text'])
                except Exception as retrain_error:
                    st.error(f"Dataset update failed: {retrain_error}")
        except Exception as dataset_error:
            st.error(f"Dataset upload failed: {dataset_error}")


def get_collection_csv_options():
    CSV_COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.name for path in CSV_COLLECTIONS_DIR.glob("*.csv"))


def append_rows_to_collection_csv(rows_df, selected_csv_name="", new_csv_name=""):
    CSV_COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    target_name = (new_csv_name or "").strip() or (selected_csv_name or "").strip()
    if not target_name:
        raise ValueError("Select an existing CSV or provide a new CSV file name.")
    if not target_name.lower().endswith(".csv"):
        target_name = f"{target_name}.csv"

    target_path = CSV_COLLECTIONS_DIR / target_name
    rows_df = rows_df.copy()
    required_columns = ['source_image', 'medicine_name', 'department']
    for column in required_columns:
        if column not in rows_df.columns:
            rows_df[column] = ''
    rows_df = rows_df[required_columns].fillna('')

    if target_path.exists():
        existing_df = pd.read_csv(target_path)
        for column in required_columns:
            if column not in existing_df.columns:
                existing_df[column] = ''
        combined_df = pd.concat([existing_df[required_columns], rows_df], ignore_index=True)
    else:
        combined_df = rows_df

    combined_df.to_csv(target_path, index=False)
    return {
        'path': str(target_path),
        'filename': target_name,
        'rows_written': int(len(rows_df)),
        'total_rows': int(len(combined_df)),
    }


def resolve_collection_csv_path(selected_csv_name="", new_csv_name=""):
    target_name = (new_csv_name or "").strip() or (selected_csv_name or "").strip()
    if not target_name:
        return None
    if not target_name.lower().endswith(".csv"):
        target_name = f"{target_name}.csv"
    return CSV_COLLECTIONS_DIR / target_name


def read_collection_csv_bytes(target_path):
    target_path = Path(target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Collection CSV not found: {target_path}")
    return target_path.read_bytes()


def clear_collection_csv(target_path):
    target_path = Path(target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Collection CSV not found: {target_path}")
    target_path.unlink()


def maybe_auto_append_collection_csv(rows_df, widget_prefix, run_token):
    auto_append_enabled = st.session_state.get(f"collection_csv_auto_{widget_prefix}", False)
    if not auto_append_enabled or rows_df is None or len(rows_df) == 0:
        return None

    selected_csv = st.session_state.get(f"collection_csv_select_{widget_prefix}", "")
    new_csv_name = st.session_state.get(f"collection_csv_new_{widget_prefix}", "")
    target_path = resolve_collection_csv_path(selected_csv, new_csv_name)
    if target_path is None:
        return {
            'status': 'missing_target',
            'message': "Auto append is on, but no target CSV is selected.",
        }

    last_token_key = f"collection_csv_last_auto_append_{widget_prefix}"
    if st.session_state.get(last_token_key) == run_token:
        return None

    result = append_rows_to_collection_csv(rows_df, selected_csv, new_csv_name)
    st.session_state[last_token_key] = run_token
    return {
        'status': 'appended',
        'message': (
            f"Auto-added {result['rows_written']} row(s) to {result['filename']}. "
            f"Total rows now: {result['total_rows']}"
        ),
        'path': result['path'],
    }


def render_collection_csv_section(rows_df, widget_prefix):
    st.subheader("CSV Collection")
    st.caption("Ekta CSV file choose koro. Chaile notun file create koro, chaile existing file e add koro.")

    existing_files = [""] + get_collection_csv_options()
    mode_key = f"collection_csv_mode_{widget_prefix}"
    select_key = f"collection_csv_select_{widget_prefix}"
    new_key = f"collection_csv_new_{widget_prefix}"
    auto_key = f"collection_csv_auto_{widget_prefix}"

    st.caption("Step 1: File choose korun")
    collection_mode = st.radio(
        "CSV file option",
        options=["Use existing CSV", "Create new CSV"],
        horizontal=True,
        key=mode_key,
        label_visibility="collapsed",
    )

    selected_csv = ""
    new_csv_name = ""
    if collection_mode == "Use existing CSV":
        selected_csv = st.selectbox(
            "Choose CSV file",
            options=existing_files,
            format_func=lambda value: value if value else "Select a CSV file",
            key=select_key,
        )
    else:
        new_csv_name = st.text_input(
            "New CSV file name",
            placeholder="example: my_department_collection.csv",
            key=new_key,
        )

    target_path = resolve_collection_csv_path(selected_csv, new_csv_name)
    target_name = target_path.name if target_path is not None else "No file selected"
    st.caption(f"Current target file: {target_name}")

    st.caption("Step 2: Auto save on/off")
    st.toggle(
        "Auto append future processed images to this CSV",
        key=auto_key,
        help="Switch on korle next processed image/result automatically ei selected CSV file e joma hobe.",
    )

    if st.button("Append To Selected CSV", use_container_width=True, key=f"collection_csv_append_{widget_prefix}"):
        try:
            result = append_rows_to_collection_csv(rows_df, selected_csv, new_csv_name)
            st.success(
                f"Added {result['rows_written']} row(s) to {result['filename']}. "
                f"Total rows now: {result['total_rows']}"
            )
            st.caption(f"Saved at: {result['path']}")
        except Exception as collection_error:
            st.error(f"CSV append failed: {collection_error}")

    st.caption(f"Collection folder: {CSV_COLLECTIONS_DIR}")

    if target_path is not None and target_path.exists():
        try:
            csv_bytes = read_collection_csv_bytes(target_path)
            st.download_button(
                label="Download Current Collection CSV",
                data=csv_bytes,
                file_name=target_path.name,
                mime="text/csv",
                use_container_width=True,
                key=f"collection_csv_download_{widget_prefix}",
            )
        except Exception as download_error:
            st.error(f"Collection download failed: {download_error}")

    if st.button("Clear Selected Collection", use_container_width=True, key=f"collection_csv_clear_{widget_prefix}"):
            try:
                clear_collection_csv(target_path)
                st.success(f"Cleared collection file: {target_path.name}")
                st.rerun()
            except Exception as clear_error:
                st.error(f"Collection clear failed: {clear_error}")


def choose_better_extraction_result(result_a, result_b):
    """Pick the extraction result with the stronger medication list."""
    def score(item):
        medications = item.get('medications', [])
        known_departments = sum(1 for med in medications if med.get('department', 'Unknown') != 'Unknown')
        total_name_length = sum(len((med.get('medication_name', '') or '').strip()) for med in medications)
        return (
            len(medications),
            known_departments,
            total_name_length,
            len(item.get('extracted_text', '') or ''),
        )

    return result_a if score(result_a) >= score(result_b) else result_b


def generate_with_model_fallback(prompt, target_bytes, target_mime_type):
    last_error = None

    for model_name in get_active_model_candidates():
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([
                    prompt,
                    {"mime_type": target_mime_type, "data": target_bytes}
                ])
                response_text = getattr(response, 'text', '') or ''
                if response_text.strip():
                    return response_text, model_name
            except Exception as model_error:
                last_error = model_error
                error_text = str(model_error).lower()
                # Retry transient server-side failures.
                if "500" in error_text or "internal error" in error_text or "unavailable" in error_text:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break

    if last_error is not None:
        raise RuntimeError(
            "Model extraction failed. The API returned a temporary server error. "
            f"Details: {last_error}"
        )
    raise RuntimeError("Model extraction failed: no response text returned.")


def run_extraction_once(target_bytes, target_mime_type):
    extracted, model_used = generate_with_model_fallback(EXTRACTION_PROMPT, target_bytes, target_mime_type)
    meds = classify_medications_by_department(extract_medications_list(extracted))

    if not meds:
        medication_only_text, fallback_model_used = generate_with_model_fallback(
            MEDICATION_ONLY_PROMPT,
            target_bytes,
            target_mime_type,
        )
        fallback_meds = classify_medications_by_department(extract_medications_list(medication_only_text))
        if fallback_meds:
            extracted = medication_only_text
            meds = fallback_meds
            model_used = fallback_model_used

    if any((med.get('department') == 'Unknown') for med in meds):
        meds = refine_unknown_medications_with_image(meds, target_bytes, target_mime_type)

    return {
        'extracted_text': extracted,
        'medications': meds,
        'model_used': model_used,
    }


def run_extraction_with_mode(image, full_image_bytes, full_image_mime_type, extraction_mode):
    cropped_image = None
    if extraction_mode in ("Use auto crop", "Use both and keep better result"):
        cropped_image = auto_crop_medication_region(image)

    if extraction_mode == "Use auto crop" and cropped_image is not None:
        cropped_buffer = io.BytesIO()
        cropped_image.save(cropped_buffer, format="PNG")
        selected_result = run_extraction_once(cropped_buffer.getvalue(), "image/png")
    elif extraction_mode == "Use both and keep better result" and cropped_image is not None:
        cropped_buffer = io.BytesIO()
        cropped_image.save(cropped_buffer, format="PNG")
        full_result = run_extraction_once(full_image_bytes, full_image_mime_type)
        cropped_result = run_extraction_once(cropped_buffer.getvalue(), "image/png")
        selected_result = choose_better_extraction_result(full_result, cropped_result)
    else:
        selected_result = run_extraction_once(full_image_bytes, full_image_mime_type)

    return selected_result, cropped_image


def build_spreadsheet_export_file(df, timestamp):
    spreadsheet_filename = f"prescription_{timestamp}.xlsx"
    spreadsheet_label = "Excel"
    spreadsheet_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    export_df = df.copy().fillna('')
    if not export_df.empty and 'classification_status' not in export_df.columns:
        export_df['classification_status'] = export_df.get('department', '').astype(str).apply(
            lambda value: 'Unknown' if value.strip() == 'Unknown' else 'Known'
        )

    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Medicines')
        spreadsheet_data = excel_buffer.getvalue()
    except ModuleNotFoundError:
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        spreadsheet_data = csv_buffer.getvalue().encode('utf-8')
        spreadsheet_filename = f"prescription_{timestamp}.csv"
        spreadsheet_label = "CSV"
        spreadsheet_mime = "text/csv"

    return spreadsheet_filename, spreadsheet_label, spreadsheet_mime, spreadsheet_data


def extract_from_uploaded_file(uploaded_file, extraction_mode):
    if not GEMINI_API_KEY:
        raise RuntimeError("System configuration error. Please contact support.")
    if genai is None:
        raise RuntimeError("google-generativeai package is not available in this environment.")

    full_image_bytes = uploaded_file.getvalue()
    if not full_image_bytes:
        raise RuntimeError("Uploaded image is empty or could not be read.")

    image = Image.open(io.BytesIO(full_image_bytes)).convert('RGB')
    full_image_mime_type = uploaded_file.type or "image/jpeg"
    selected_result, cropped_image = run_extraction_with_mode(
        image,
        full_image_bytes,
        full_image_mime_type,
        extraction_mode,
    )
    return image, selected_result, cropped_image


def auto_crop_medication_region(image):
    """Heuristically crop the lower text-dense medication region from a prescription image."""
    rgb_image = image.convert('RGB')
    image_array = np.array(rgb_image)
    gray = np.mean(image_array, axis=2)

    # Highlight darker text strokes against a mostly white page.
    text_mask = gray < 210
    row_density = text_mask.mean(axis=1)
    col_density = text_mask.mean(axis=0)

    height, width = gray.shape
    search_start = int(height * 0.30)
    active_rows = np.where(row_density[search_start:] > 0.015)[0]

    if active_rows.size == 0:
        top = int(height * 0.40)
        bottom = height
    else:
        top = search_start + int(active_rows[0])
        bottom = search_start + int(active_rows[-1]) + 1

    active_cols = np.where(col_density > 0.01)[0]
    if active_cols.size == 0:
        left = 0
        right = width
    else:
        left = int(active_cols[0])
        right = int(active_cols[-1]) + 1

    row_padding = max(20, int(height * 0.03))
    col_padding = max(20, int(width * 0.03))
    top = max(0, top - row_padding)
    bottom = min(height, bottom + row_padding)
    left = max(0, left - col_padding)
    right = min(width, right + col_padding)

    # Keep the crop focused on the lower portion where medicines usually appear.
    top = max(top, int(height * 0.20))

    if bottom - top < int(height * 0.20):
        top = int(height * 0.35)
        bottom = height

    return rgb_image.crop((left, top, right, bottom))

# Sidebar
with st.sidebar:
    st.header("ℹ️ System Information")
    st.caption("Focused dashboard for prescription extraction, review, and export.")
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-kicker">Live Overview</div>
            <div class="sidebar-grid">
                <div class="sidebar-tile">
                    <div class="sidebar-tile-label">Alias Rules</div>
                    <div class="sidebar-tile-value">{len(load_alias_seed_data())}</div>
                </div>
                <div class="sidebar-tile">
                    <div class="sidebar-tile-label">Recent Runs</div>
                    <div class="sidebar-tile-value">{len(st.session_state.processing_history)}</div>
                </div>
            </div>
            <div class="sidebar-sub">Best for handwritten or noisy prescriptions when correction history matters.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-kicker">What This App Does</div>
            <div class="sidebar-sub" style="margin-top:0;">
                Extracts text, resolves medicine names, classifies departments, and exports clean outputs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-kicker">Supported</div>
            <div class="sidebar-sub" style="margin-top:0;">
                JPG, JPEG, PNG and handwritten or printed prescriptions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")

    if 'total_processed' not in st.session_state:
        st.session_state.total_processed = 0

    st.metric("Total Processed", st.session_state.total_processed)
    st.caption("Recent history keeps duplicate detection and review tracking usable.")
    if st.session_state.processing_history:
        st.caption("Recent History")
        history_df = pd.DataFrame(st.session_state.processing_history)[['time', 'mode', 'file', 'rows', 'unknown']].head(5)
        safe_dataframe(history_df, use_container_width=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.extracted_text = None
        st.session_state.processing_complete = False
        st.session_state.uploaded_file_name = None
        st.session_state.structured_data = None
        st.session_state.medications_list = []
        st.session_state.bulk_results = []
        st.session_state.bulk_processing_complete = False
        st.session_state.bulk_export_timestamp = None
        st.session_state.current_image_hash = None
        st.session_state.bulk_current_hashes = []
        st.session_state.current_uploaded_image = None
        st.session_state.current_cropped_image = None
        st.session_state.last_analysis_from_cache = False
        st.rerun()

    if st.button("🧹 Clear Cache", use_container_width=True):
        invalidate_all_caches()
        st.session_state.last_analysis_from_cache = False
        st.success("Analysis cache cleared.")
    
    st.markdown("---")
    st.caption("© 2025 Smart Prescription System")

# Main layout
workflow_mode = st.radio(
    "Workflow Mode",
    options=["Single Image", "Bulk Images"],
    horizontal=True,
)
st.radio(
    "Processing Mode",
    options=["High Accuracy", "Fast"],
    horizontal=True,
    key="accuracy_mode",
    help="High Accuracy mode e extra correction/review cholbe, tai time beshi nite pare.",
)

alias_count = len(load_alias_seed_data())
history_count = len(st.session_state.processing_history)
current_medicine_count = len(build_clean_medication_rows(st.session_state.medications_list)) if st.session_state.medications_list else 0
current_confidences = [
    float(row.get('confidence', 0.0) or 0.0)
    for row in build_clean_medication_rows(st.session_state.medications_list)
    if float(row.get('confidence', 0.0) or 0.0) > 0
]
average_confidence = round(sum(current_confidences) / len(current_confidences), 2) if current_confidences else 0.0
render_stat_cards([
    {
        'label': 'Workflow',
        'value': workflow_mode,
        'subtext': 'Choose one image or a batch of images',
    },
    {
        'label': 'Processing Mode',
        'value': st.session_state.get("accuracy_mode", "High Accuracy"),
        'subtext': 'Accuracy mode adds correction and fallback review',
    },
    {
        'label': 'Current Medicines',
        'value': current_medicine_count,
        'subtext': 'Rows ready for correction and export',
    },
    {
        'label': 'Avg Confidence',
        'value': f"{int(average_confidence * 100)}%",
        'subtext': f'{alias_count} alias rule(s) and {history_count} session(s) tracked',
    },
])

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    if workflow_mode == "Single Image":
        render_dashboard_section(
            "Input",
            "Upload Prescription",
            "Select the extraction method, preview the uploaded prescription, and run the analysis pipeline.",
        )
        with st.expander("Upload Department Dataset", expanded=False):
            render_dataset_upload_section("main")

        uploaded_file = st.file_uploader(
            "Select an image file",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed",
            key="single_image_uploader",
        )

        if uploaded_file:
            uploaded_file_bytes = uploaded_file.getvalue()
            uploaded_file_hash = compute_bytes_hash(uploaded_file_bytes)
            duplicate_single = history_contains_image_hash(uploaded_file_hash)
            if duplicate_single:
                st.warning("Ei image ta age process kora hoyeche. Chaile abar process korte paro.")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Uploaded Prescription")
            st.caption(f"File: {uploaded_file.name} | Size: {uploaded_file.size/1024:.1f} KB")

            extraction_mode = st.radio(
                "Extraction mode",
                options=[
                    "Use full image",
                    "Use auto crop",
                    "Use both and keep better result",
                ],
                index=0,
                key="single_extraction_mode",
                help="Auto crop focuses on the lower medicine-heavy area. The combined mode runs both and keeps the stronger medicine extraction.",
            )

            if extraction_mode in ("Use auto crop", "Use both and keep better result"):
                cropped_preview = auto_crop_medication_region(image)
                st.image(cropped_preview, use_column_width=True, caption="Auto-cropped medicine region")

            if st.button("🔍 Extract & Analyze", type="primary", use_container_width=True, key="single_extract_button"):
                progress_bar = st.progress(0, text="Starting... 0%")
                try:
                    progress_bar.progress(10, text="Preparing image... 10%")
                    selected_result, cropped_image, used_cache = run_cached_extraction(
                        uploaded_file_bytes,
                        uploaded_file.type or "image/jpeg",
                        extraction_mode,
                        uploaded_file_hash,
                        image=image,
                    )
                    st.session_state.last_analysis_from_cache = used_cache
                    progress_bar.progress(75, text="Using cached result... 75%" if used_cache else "Extracting medicines... 75%")
                    if used_cache:
                        st.info("Cached result loaded for the same image.")
                    extracted_text = selected_result['extracted_text']

                    if extracted_text:
                        st.session_state.extracted_text = extracted_text
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.processing_complete = True
                        st.session_state.bulk_results = []
                        st.session_state.bulk_processing_complete = False
                        st.session_state.bulk_export_timestamp = None
                        st.session_state.total_processed += 1
                        st.session_state.export_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.session_state.current_image_hash = uploaded_file_hash
                        structured = parse_prescription_to_columns(extracted_text)
                        structured['filename'] = uploaded_file.name
                        st.session_state.structured_data = structured
                        st.session_state.medications_list = selected_result['medications']
                        st.session_state.current_uploaded_image = image
                        st.session_state.current_cropped_image = cropped_image
                        progress_bar.progress(100, text="Completed 100%")
                        st.rerun()
                    else:
                        progress_bar.progress(100, text="Completed 100%")
                        st.warning("No text could be extracted.")
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Error processing prescription: {str(e)}")
    else:
        render_dashboard_section(
            "Batch Input",
            "Bulk Prescription Upload",
            "Queue multiple prescription images, process them together, and build one combined export-ready output.",
        )
        with st.expander("Upload Department Dataset", expanded=False):
            render_dataset_upload_section("bulk")

        bulk_files = st.file_uploader(
            "Select multiple image files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="bulk_image_uploader",
        )

        bulk_extraction_mode = st.radio(
            "Bulk extraction mode",
            options=[
                "Use full image",
                "Use auto crop",
                "Use both and keep better result",
            ],
            index=0,
            key="bulk_extraction_mode",
            help="Bulk mode processes each image with the selected extraction strategy.",
        )

        if bulk_files:
            bulk_hashes = [compute_bytes_hash(item.getvalue()) for item in bulk_files]
            duplicate_bulk = sum(1 for item_hash in bulk_hashes if history_contains_image_hash(item_hash))
            if duplicate_bulk:
                st.warning(f"{duplicate_bulk} ta image ageo process kora chilo.")
            st.caption(f"Selected files: {len(bulk_files)}")
            for bulk_file in bulk_files[:5]:
                st.write(f"- {bulk_file.name}")
            if len(bulk_files) > 5:
                st.caption(f"And {len(bulk_files) - 5} more file(s)")

            if st.button("🔍 Process All Images", type="primary", use_container_width=True, key="bulk_process_button"):
                progress_bar = st.progress(0, text="Starting bulk processing... 0%")
                try:
                    if not GEMINI_API_KEY:
                        st.error("System configuration error. Please contact support.")
                        st.stop()

                    combined_rows = []
                    processed_items = []
                    total_files = len(bulk_files)

                    for index, bulk_file in enumerate(bulk_files, start=1):
                        percent_complete = int(((index - 1) / total_files) * 100)
                        progress_bar.progress(
                            percent_complete,
                            text=f"Processing {index}/{total_files}: {bulk_file.name}... {percent_complete}%"
                        )
                        bulk_bytes = bulk_file.getvalue()
                        bulk_hash = compute_bytes_hash(bulk_bytes)
                        bulk_image = Image.open(io.BytesIO(bulk_bytes)).convert('RGB')
                        selected_result, _, used_cache = run_cached_extraction(
                            bulk_bytes,
                            bulk_file.type or "image/jpeg",
                            bulk_extraction_mode,
                            bulk_hash,
                            image=bulk_image,
                        )
                        st.session_state.last_analysis_from_cache = used_cache
                        if used_cache:
                            st.caption(f"Cached: {bulk_file.name}")
                        cleaned_rows = build_clean_medication_rows(selected_result['medications'])
                        processed_items.append({
                            'filename': bulk_file.name,
                            'row_count': len(cleaned_rows),
                        })
                        for row in cleaned_rows:
                            combined_rows.append({
                                'filename': bulk_file.name,
                                'medicine_name': row['medicine_name'],
                                'department': row['department'],
                            })

                    st.session_state.bulk_results = combined_rows
                    st.session_state.bulk_processing_complete = True
                    st.session_state.extracted_text = None
                    st.session_state.processing_complete = False
                    st.session_state.uploaded_file_name = None
                    st.session_state.structured_data = None
                    st.session_state.medications_list = []
                    st.session_state.bulk_export_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.session_state.bulk_current_hashes = bulk_hashes
                    st.session_state.current_uploaded_image = None
                    st.session_state.current_cropped_image = None
                    st.session_state.total_processed += len(bulk_files)
                    progress_bar.progress(100, text="Completed 100%")
                    st.success(f"Processed {len(bulk_files)} image(s)")
                    if processed_items:
                        summary_df = pd.DataFrame(processed_items)
                        st.dataframe(summary_df, use_container_width=True)
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Bulk processing error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    if workflow_mode == "Single Image":
        render_dashboard_section(
            "Output",
            "Extraction Workspace",
            "Review medicine rows, confirm department labels, and prepare the final export set from this session.",
        )
    
    if workflow_mode == "Single Image" and st.session_state.processing_complete and st.session_state.extracted_text:
        
        st.markdown('<div class="success-message">✓ Extraction complete</div>', unsafe_allow_html=True)
        if st.session_state.last_analysis_from_cache:
            st.success("Loaded from cache")
        if st.session_state.review_save_message:
            st.success(st.session_state.review_save_message)
            st.session_state.review_save_message = None
        
        medication_preview_rows = build_clean_medication_rows(st.session_state.medications_list)
        single_output_base_df = pd.DataFrame(medication_preview_rows).fillna('')
        single_required_columns = ['medicine_name', 'department']
        single_available_columns = [
            column for column in (
                list(single_output_base_df.columns)
                if not single_output_base_df.empty
                else [
                    'medicine_name',
                    'department',
                    'classification_status',
                    'confidence',
                    'name_source',
                    'department_source',
                    'resolution_explanation',
                ]
            )
            if column not in single_required_columns
        ]
        st.markdown("#### Output Options")
        option_col_1, option_col_2 = st.columns([0.8, 1.2], gap="large")
        with option_col_1:
            st.checkbox(
                "Pure medicine name only",
                key="single_pure_medicine_name_only",
                help="Removes dosage and form words so the medicine column keeps only the core name.",
            )
        with option_col_2:
            st.multiselect(
                "Visible columns",
                options=single_available_columns,
                default=[column for column in st.session_state.single_visible_columns if column in single_available_columns],
                key="single_visible_columns",
                help="You can hide any optional column from the result table and CSV export.",
            )
        single_display_df = prepare_output_dataframe(
            single_output_base_df,
            strict_medicine_only=st.session_state.single_pure_medicine_name_only,
            selected_columns=st.session_state.single_visible_columns,
            required_columns=single_required_columns,
        )
        single_core_df = prepare_output_dataframe(
            single_output_base_df,
            strict_medicine_only=st.session_state.single_pure_medicine_name_only,
            required_columns=single_required_columns,
        )
        review_rows = build_review_rows(st.session_state.medications_list)
        review_df = pd.DataFrame(review_rows)
        if medication_preview_rows and st.session_state.current_image_hash:
            add_processing_history_entry(
                "Single",
                st.session_state.uploaded_file_name or "",
                st.session_state.current_image_hash,
                pd.DataFrame(medication_preview_rows),
            )
            st.session_state.current_image_hash = None

        result_tab, review_tab, analytics_tab = st.tabs(["Medicine Output", "Review", "Summary"])

        with result_tab:
            left_col, right_col = st.columns([1.05, 1.25], gap="large")
            with left_col:
                if st.session_state.current_uploaded_image is not None:
                    st.markdown(
                        """
                        <div class="image-preview-card">
                            <div class="image-preview-title">Prescription Preview</div>
                            <div class="image-preview-sub">Original uploaded image for visual verification.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(st.session_state.current_uploaded_image, use_column_width=True)
                else:
                    st.info("No prescription preview available.")

            with right_col:
                if medication_preview_rows:
                    st.dataframe(single_display_df, use_container_width=True)
                else:
                    st.info("No medicine names found.")
                with st.expander("Parsed Rows", expanded=False):
                    if medication_preview_rows:
                        st.dataframe(single_display_df, use_container_width=True)
                    else:
                        st.info("No parsed medicine data available.")

        edited_review_df = review_df.copy()
        with review_tab:
            review_left, review_right = st.columns([1.15, 0.85], gap="large")
            with review_left:
                if not review_df.empty:
                    st.caption("Original, corrected, source, confidence dekho. Chaile corrected name ar department edit korte parba.")
                    if hasattr(st, "data_editor"):
                        edited_review_df = st.data_editor(
                            review_df,
                            use_container_width=True,
                            num_rows="dynamic",
                            key="single_review_editor",
                            column_config={
                                "original_name": st.column_config.TextColumn(disabled=True),
                                "name_source": st.column_config.TextColumn(disabled=True),
                                "department_source": st.column_config.TextColumn(disabled=True),
                                "confidence": st.column_config.NumberColumn(disabled=True, format="%.2f"),
                                "resolution_explanation": st.column_config.TextColumn(disabled=True),
                                "department": st.column_config.SelectboxColumn(options=["Cardiology", "Ophthalmology", "Unknown"]),
                            },
                        )
                    else:
                        st.warning("Manual review editor ei Streamlit version e available na. Table only dekhano hocche.")
                        safe_dataframe(review_df, use_container_width=True)
                    unknown_only_df = edited_review_df[edited_review_df['department'] == 'Unknown']
                    if not unknown_only_df.empty:
                        st.caption("Unknown only")
                        safe_dataframe(unknown_only_df, use_container_width=True)
                    if st.button("Save Review Corrections", use_container_width=True, key="save_review_corrections"):
                        save_result = save_alias_corrections(edited_review_df)
                        st.session_state.medications_list = apply_review_corrections_to_medications(
                            st.session_state.medications_list,
                            edited_review_df,
                        )
                        st.session_state.review_save_message = (
                            f"Saved aliases: {save_result['saved_aliases']} | "
                            f"Dataset rows added: {save_result['saved_dataset_rows']}"
                        )
                        st.rerun()
                else:
                    st.info("No review rows available for this extraction.")
            with review_right:
                st.markdown(
                    """
                    <div class="image-preview-card">
                        <div class="image-preview-title">Review Notes</div>
                        <div class="image-preview-sub">Edit only the medicines that need correction. Unknown rows deserve the most attention.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.session_state.current_uploaded_image is not None:
                    st.markdown(
                        """
                        <div class="image-preview-card">
                            <div class="image-preview-title">Prescription Reference</div>
                            <div class="image-preview-sub">Use this image while correcting names and departments.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(st.session_state.current_uploaded_image, use_column_width=True)
                if st.session_state.current_cropped_image is not None:
                    st.markdown(
                        """
                        <div class="image-preview-card">
                            <div class="image-preview-title">Cropped Region</div>
                            <div class="image-preview-sub">Auto-cropped medicine-heavy area for quick checking.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(st.session_state.current_cropped_image, use_column_width=True)

        with analytics_tab:
            if medication_preview_rows:
                confidence_df = single_output_base_df.copy()
                if 'confidence' in confidence_df.columns and not confidence_df.empty:
                    avg_conf = round(float(confidence_df['confidence'].fillna(0).astype(float).mean()), 2)
                    high_conf = int((confidence_df['confidence'].fillna(0).astype(float) >= 0.80).sum())
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Average Confidence", f"{int(avg_conf * 100)}%")
                    with metric_col2:
                        st.metric("High Confidence Rows", high_conf)
                summary_df = build_department_summary_df(confidence_df, image_count=1)
                safe_dataframe(summary_df, use_container_width=True)
                st.caption("Confidence and explanation are carried through the review pipeline for traceability.")
            else:
                st.info("No department summary available yet.")

        if medication_preview_rows:
            st.markdown("---")
            st.subheader("Dataset Ready Output")
            st.caption("Current single-image result is already in dataset format: `medicine_name, department`.")

            dataset_ready_df = single_core_df.copy()
            st.dataframe(dataset_ready_df, use_container_width=True)
            collection_ready_df = dataset_ready_df.copy()
            collection_ready_df['source_image'] = st.session_state.uploaded_file_name or ''
            auto_append_result = maybe_auto_append_collection_csv(
                collection_ready_df[['source_image', 'medicine_name', 'department']],
                "single_result",
                st.session_state.export_timestamp or st.session_state.uploaded_file_name or "",
            )
            if auto_append_result:
                if auto_append_result['status'] == 'appended':
                    st.success(auto_append_result['message'])
                    st.caption(f"Saved at: {auto_append_result['path']}")
                elif auto_append_result['status'] == 'missing_target':
                    st.warning(auto_append_result['message'])
            render_collection_csv_section(collection_ready_df[['source_image', 'medicine_name', 'department']], "single_result")

            if st.button("Add Current Result To Dataset", use_container_width=True, key="add_current_result_to_dataset"):
                try:
                    current_result_df = dataset_ready_df.copy()
                    current_result_df["source_group"] = "Single_Image_Result"
                    result = merge_uploaded_department_dataset(current_result_df)
                    st.success(
                        f"Added {result['added_rows']} row(s) from current image. "
                        f"Dataset now has {result['total_rows']} row(s). "
                        f"Labels: {', '.join(result['labels'])}"
                    )
                    st.text(result['report_text'])
                except Exception as current_dataset_error:
                    st.error(f"Failed to add current result to dataset: {current_dataset_error}")
        
        # Metrics
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Characters", len(st.session_state.extracted_text))
        with col_s2:
            st.metric("Words", len(st.session_state.extracted_text.split()))
        with col_s3:
            st.metric("Medications", len(st.session_state.medications_list))
        with col_s4:
            st.metric("Status", "Success")
        
        # Generate export files with medicine names only
        timestamp = st.session_state.export_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        export_rows = build_clean_medication_rows(st.session_state.medications_list)
        export_base_df = pd.DataFrame(export_rows).fillna('')
        df = prepare_output_dataframe(
            export_base_df,
            strict_medicine_only=st.session_state.single_pure_medicine_name_only,
            selected_columns=st.session_state.single_visible_columns,
            required_columns=single_required_columns,
        )
        if df.empty:
            df = export_base_df[['medicine_name', 'department']].copy() if not export_base_df.empty else pd.DataFrame(columns=['medicine_name', 'department'])

        export_signature = build_export_signature(
            df,
            extra_parts=[
                st.session_state.get("workflow_mode", ""),
                st.session_state.get("accuracy_mode", ""),
                st.session_state.get("extraction_mode", ""),
            ],
        )
        export_cache_key = build_export_cache_key("single_export", export_signature)
        cached_export = load_cached_export(export_cache_key)

        if cached_export is not None:
            spreadsheet_label = cached_export.get('spreadsheet_label', 'Excel')
            spreadsheet_mime = cached_export.get('spreadsheet_mime', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            spreadsheet_data = decode_cache_blob(cached_export.get('spreadsheet_data'))
            pdf_data = decode_cache_blob(cached_export.get('pdf_data'))
            st.success("Loaded cached export")
        else:
            export_bundle = build_single_export_bundle(df)
            spreadsheet_label = export_bundle['spreadsheet_label']
            spreadsheet_mime = export_bundle['spreadsheet_mime']
            spreadsheet_data = export_bundle['spreadsheet_data']
            pdf_data = export_bundle['pdf_data']
            store_cached_export(export_cache_key, {
                'spreadsheet_label': spreadsheet_label,
                'spreadsheet_mime': spreadsheet_mime,
                'spreadsheet_data': encode_cache_blob(spreadsheet_data),
                'pdf_data': encode_cache_blob(pdf_data),
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })

        spreadsheet_filename = f"prescription_{timestamp}.xlsx" if spreadsheet_label == "Excel" else f"prescription_{timestamp}.csv"
        pdf_filename = f"prescription_{timestamp}.pdf"
        
        # Export section
        st.markdown("---")
        render_dashboard_section(
            "Export",
            "Download Final Files",
            "Export the cleaned medicine rows as spreadsheet or PDF, or save generated files to the output folder.",
        )
        
        if spreadsheet_label == "CSV":
            st.warning("Excel export unavailable in this environment. Downloading CSV instead.")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.download_button(
                label=f"📊 Download {spreadsheet_label}",
                data=spreadsheet_data,
                file_name=spreadsheet_filename,
                mime=spreadsheet_mime,
                use_container_width=True,
                key="spreadsheet_download"
            )
            st.caption(f"✅ {len(export_rows)} medication(s)")
        
        with col_d2:
            st.download_button(
                label="📄 Download PDF",
                data=pdf_data,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
                key="pdf_download"
            )
        
        if st.button("💾 Save Files To Output Folder", use_container_width=True):
            os.makedirs('output', exist_ok=True)
            with open(f"output/{spreadsheet_filename}", 'wb') as f:
                f.write(spreadsheet_data)
            with open(f"output/{pdf_filename}", 'wb') as f:
                f.write(pdf_data)
            st.success("💾 Files saved to output folder")

        # Show export preview
        with st.expander("Preview Export Data", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.caption(f"Total medications captured: {len(export_rows)}")

        
    elif workflow_mode == "Single Image":
        st.info("👈 Upload a prescription image and click 'Extract & Analyze'")
        st.markdown("""
        **Instructions:**
        1. Upload a clear image of the prescription
        2. Choose an extraction mode and click 'Extract & Analyze'
        3. Review extracted information
        4. Download as spreadsheet or PDF
        
        **The spreadsheet and PDF files will contain medicine names with departments**
        
        **Supported images:** JPG, JPEG, PNG
        """)
    else:
        render_dashboard_section(
            "Batch Output",
            "Bulk Results",
            "Inspect combined medicine rows, review the department summary, and export the full multi-image dataset.",
        )
        if st.session_state.bulk_processing_complete and st.session_state.bulk_results:
            bulk_df = pd.DataFrame(
                st.session_state.bulk_results,
                columns=['filename', 'medicine_name', 'department']
            ).fillna('')
            bulk_required_columns = ['filename', 'medicine_name', 'department']
            bulk_available_columns = [
                column for column in (list(bulk_df.columns) + ['classification_status'])
                if column not in bulk_required_columns
            ]
            st.markdown("#### Output Options")
            bulk_option_col_1, bulk_option_col_2 = st.columns([0.8, 1.2], gap="large")
            with bulk_option_col_1:
                st.checkbox(
                    "Pure medicine name only",
                    key="bulk_pure_medicine_name_only",
                    help="Removes dosage and form words so the medicine column keeps only the core name.",
                )
            with bulk_option_col_2:
                st.multiselect(
                    "Visible columns",
                    options=bulk_available_columns,
                    default=[column for column in st.session_state.bulk_visible_columns if column in bulk_available_columns],
                    key="bulk_visible_columns",
                    help="Hide any optional column from the bulk result table and CSV export.",
                )
            bulk_display_df = prepare_output_dataframe(
                bulk_df,
                strict_medicine_only=st.session_state.bulk_pure_medicine_name_only,
                selected_columns=st.session_state.bulk_visible_columns,
                required_columns=bulk_required_columns,
            )
            bulk_export_columns = bulk_required_columns + [
                column for column in st.session_state.bulk_visible_columns
                if column not in bulk_required_columns
            ]
            bulk_export_df = prepare_output_dataframe(
                bulk_df,
                strict_medicine_only=st.session_state.bulk_pure_medicine_name_only,
                selected_columns=bulk_export_columns,
                required_columns=bulk_required_columns,
            )
            if st.session_state.bulk_current_hashes:
                add_processing_history_entry(
                    "Bulk",
                    f"{len(st.session_state.bulk_current_hashes)} image(s)",
                    st.session_state.bulk_current_hashes,
                    bulk_df[['medicine_name', 'department']].copy(),
                )
                st.session_state.bulk_current_hashes = []
            bulk_collection_df = bulk_df.rename(columns={'filename': 'source_image'})
            auto_append_result = maybe_auto_append_collection_csv(
                bulk_collection_df[['source_image', 'medicine_name', 'department']],
                "bulk_result",
                st.session_state.bulk_export_timestamp or "",
            )
            if auto_append_result:
                if auto_append_result['status'] == 'appended':
                    st.success(auto_append_result['message'])
                    st.caption(f"Saved at: {auto_append_result['path']}")
                elif auto_append_result['status'] == 'missing_target':
                    st.warning(auto_append_result['message'])
            render_collection_csv_section(bulk_collection_df[['source_image', 'medicine_name', 'department']], "bulk_result")

            bulk_timestamp = st.session_state.bulk_export_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
            bulk_export_signature = build_export_signature(
                bulk_export_df[['filename', 'medicine_name', 'department']].fillna('') if {'filename', 'medicine_name', 'department'}.issubset(bulk_export_df.columns) else bulk_df[['filename', 'medicine_name', 'department']].fillna(''),
                extra_parts=[
                    st.session_state.get("workflow_mode", ""),
                    st.session_state.get("accuracy_mode", ""),
                ],
            )
            bulk_export_cache_key = build_export_cache_key("bulk_export", bulk_export_signature)
            cached_bulk_export = load_cached_export(bulk_export_cache_key)
            if cached_bulk_export is not None:
                bulk_label = cached_bulk_export.get('bulk_label', 'Excel')
                bulk_mime = cached_bulk_export.get('bulk_mime', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                bulk_data = decode_cache_blob(cached_bulk_export.get('bulk_data'))
                bulk_zip_data = decode_cache_blob(cached_bulk_export.get('bulk_zip_data'))
                st.success("Loaded cached bulk export")
            else:
                bulk_export_bundle = build_bulk_export_bundle(bulk_export_df if not bulk_export_df.empty else bulk_df)
                bulk_label = bulk_export_bundle['bulk_label']
                bulk_mime = bulk_export_bundle['bulk_mime']
                bulk_data = bulk_export_bundle['bulk_data']
                bulk_zip_data = bulk_export_bundle['bulk_zip_data']
                store_cached_export(bulk_export_cache_key, {
                    'bulk_label': bulk_label,
                    'bulk_mime': bulk_mime,
                    'bulk_data': encode_cache_blob(bulk_data),
                    'bulk_zip_data': encode_cache_blob(bulk_zip_data),
                    'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })

            bulk_filename = f"bulk_{bulk_timestamp}.xlsx" if bulk_label == "Excel" else f"bulk_{bulk_timestamp}.csv"

            bulk_summary_df = build_department_summary_df(
                bulk_df[['medicine_name', 'department']].copy(),
                image_count=bulk_df['filename'].nunique(),
            )
            bulk_results_tab, bulk_summary_tab, bulk_export_tab = st.tabs(["Combined Rows", "Summary", "Export"])

            with bulk_results_tab:
                st.dataframe(bulk_display_df, use_container_width=True)

            with bulk_summary_tab:
                safe_dataframe(bulk_summary_df, use_container_width=True)
                st.caption("Combined rows from all uploaded images in dataset-ready format.")

            with bulk_export_tab:
                st.download_button(
                    label=f"📊 Download Bulk {bulk_label}",
                    data=bulk_data,
                    file_name=bulk_filename,
                    mime=bulk_mime,
                    use_container_width=True,
                    key="bulk_download_button",
                )
                st.download_button(
                    label="Download Bulk ZIP",
                    data=bulk_zip_data,
                    file_name=f"bulk_{bulk_timestamp}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="bulk_zip_download_button",
                )

                if st.button("Add Bulk Result To Dataset", use_container_width=True, key="add_bulk_result_to_dataset"):
                    try:
                        bulk_dataset_df = bulk_df[['medicine_name', 'department']].copy()
                        bulk_dataset_df['source_group'] = 'Bulk_Image_Result'
                        result = merge_uploaded_department_dataset(bulk_dataset_df)
                        st.success(
                            f"Added {result['added_rows']} row(s) from bulk result. "
                            f"Dataset now has {result['total_rows']} row(s). "
                            f"Labels: {', '.join(result['labels'])}"
                        )
                        st.text(result['report_text'])
                    except Exception as bulk_dataset_error:
                        st.error(f"Failed to add bulk result to dataset: {bulk_dataset_error}")
        else:
            st.info("👈 Upload multiple prescription images and click 'Process All Images'")
            st.markdown("""
            **Bulk mode does this:**
            1. Takes multiple prescription images
            2. Extracts medicine names from each image
            3. Classifies each medicine by department
            4. Combines all rows into one dataset-ready table
            5. Exports one spreadsheet for all uploaded files
            """)

    with st.expander("Evaluation Dashboard", expanded=False):
        st.caption("Upload a labeled CSV to measure classification quality with precision, recall, F1 score, confusion matrix, and Unknown rate.")
        evaluation_file = st.file_uploader(
            "Upload evaluation CSV",
            type=['csv'],
            key="evaluation_csv_uploader",
        )
        if evaluation_file is not None:
            try:
                evaluation_df = pd.read_csv(evaluation_file)
                evaluation_columns = list(evaluation_df.columns)

                medicine_defaults = ['medicine_name', 'corrected_name', 'name', 'medication_name']
                truth_defaults = ['true_department', 'department', 'label', 'ground_truth']
                medicine_default_index = next((evaluation_columns.index(col) for col in medicine_defaults if col in evaluation_columns), 0)
                truth_default_index = next((evaluation_columns.index(col) for col in truth_defaults if col in evaluation_columns), 0)

                with st.form("evaluation_dashboard_form"):
                    eval_left, eval_right = st.columns(2)
                    with eval_left:
                        medicine_col = st.selectbox(
                            "Medicine column",
                            options=evaluation_columns,
                            index=medicine_default_index if evaluation_columns else 0,
                        )
                        truth_col = st.selectbox(
                            "Ground truth column",
                            options=evaluation_columns,
                            index=truth_default_index if evaluation_columns else 0,
                        )
                    with eval_right:
                        predicted_options = ["(Predict from medicine column)"] + evaluation_columns
                        predicted_default_index = 0
                        if 'predicted_department' in evaluation_columns:
                            predicted_default_index = predicted_options.index('predicted_department')
                        predicted_choice = st.selectbox(
                            "Predicted column",
                            options=predicted_options,
                            index=predicted_default_index,
                            help="Leave as prediction mode if the file has no model output column.",
                        )
                        st.caption("If no predicted column is chosen, the app will generate predictions from medicine names.")

                    submitted = st.form_submit_button("Run Evaluation", use_container_width=True)

                if submitted:
                    predicted_col = None if predicted_choice == "(Predict from medicine column)" else predicted_choice
                    try:
                        st.session_state.evaluation_report = build_evaluation_report(
                            evaluation_df,
                            medicine_col=medicine_col,
                            truth_col=truth_col,
                            predicted_col=predicted_col,
                        )
                        st.session_state.evaluation_source_shape = evaluation_df.shape
                    except Exception as evaluation_error:
                        st.session_state.evaluation_report = None
                        st.error(f"Evaluation failed: {evaluation_error}")

                if st.session_state.evaluation_report:
                    report = st.session_state.evaluation_report
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Accuracy", f"{report['accuracy']:.2%}")
                    with metric_col2:
                        st.metric("Macro F1", f"{report['macro_f1']:.2%}")
                    with metric_col3:
                        st.metric("Unknown Rate", f"{report['unknown_rate']:.2%}")
                    with metric_col4:
                        st.metric("Samples", report['total'])

                    st.caption("Per-class performance")
                    st.dataframe(report['per_class_df'], use_container_width=True)

                    st.caption("Confusion matrix")
                    st.dataframe(report['confusion_matrix_df'], use_container_width=True)

                    st.caption(
                        f"True Unknown share: {report['true_unknown_rate']:.2%} | "
                        f"Evaluation source: {st.session_state.evaluation_source_shape[0]} rows"
                    )
            except Exception as evaluation_load_error:
                st.error(f"Could not load evaluation file: {evaluation_load_error}")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Smart Prescription Analysis System | Medicine Department Export | Spreadsheet and PDF"
    "</div>",
    unsafe_allow_html=True
)
