from __future__ import annotations

import hashlib
import hmac
import io
import csv
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, Field

try:
    import psycopg
    from psycopg import errors as pg_errors
    from psycopg.rows import dict_row
except ImportError:  # Local SQLite mode does not need psycopg installed.
    psycopg = None
    pg_errors = None
    dict_row = None


API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
MAIN_PROJECT_DIR = ROOT_DIR / "main_project"
DATASET_DIR = MAIN_PROJECT_DIR / "dataset"
STORAGE_DIR = API_DIR / "storage"
WORKSPACE_DIR = STORAGE_DIR / "workspaces"
DB_PATH = STORAGE_DIR / "api.db"

DEPARTMENT_DATASET_PATH = DATASET_DIR / "medicine_department_seed_balanced_10000.csv"
CUSTOM_DEPARTMENT_DATASET_PATH = DATASET_DIR / "medicine_department_from_custom_folders.csv"
ALIAS_DATASET_PATH = DATASET_DIR / "medicine_name_aliases.csv"

SESSION_DAYS = 14
load_dotenv(ROOT_DIR / ".env")
load_dotenv(API_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = [
    model_name
    for model_name in (
        GEMINI_MODEL_NAME,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    )
    if model_name
]

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Prescription Extraction API",
    version="1.0.0",
    description="Backend API for the Kotlin mobile app.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)


EXTRACTION_PROMPT = """Read this prescription image and extract only the prescribed medicines.

Return only medicine lines. No headings. No explanations. No extra text.
Each line must follow this format exactly:
[Medication name] - [dosage pattern like 1-0-1] for [X] days

If duration is missing, use:
[Medication name] - [dosage pattern] for Not specified days

If dosage pattern is missing, use:
[Medication name] - Not specified for Not specified days

List all medicines found, one per line."""

CARDIOLOGY_KEYWORDS = (
    "amlodipine", "olmesartan", "telmisartan", "losartan", "valsartan",
    "bisoprolol", "metoprolol", "atenolol", "carvedilol", "nebivolol",
    "clopidogrel", "aspirin", "atorvastatin", "rosuvastatin", "warfarin",
    "rivaroxaban", "apixaban", "dabigatran", "furosemide", "torsemide",
    "spironolactone", "diltiazem", "verapamil", "nifedipine", "ramipril",
    "perindopril", "digoxin", "isosorbide", "nitroglycerin", "heparin",
    "lisinopril", "ivabradine", "sacubitril", "ecosprin",
)

OPHTHALMOLOGY_KEYWORDS = (
    "eye", "ophthalmic", "ocular", "tear", "drop", "drops", "ointment",
    "gel", "solution", "timolol", "latanoprost", "travoprost",
    "brimonidine", "moxifloxacin", "olopatadine", "carboxymethylcellulose",
    "gatifloxacin", "nepafenac", "ketorolac", "dorzolamide", "bimatoprost",
    "prednisolone", "cyclopentolate", "tropicamide", "lubricant",
    "difluprednate", "fluconazole", "gentamicin", "flurbiprofen",
)


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    token: str
    user: dict


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceResponse(BaseModel):
    id: int
    name: str
    csv_filename: str
    row_count: int
    created_at: str


def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    ensure_storage()
    if USE_POSTGRES:
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is set but psycopg is not installed.")
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    return sqlite3.connect(DB_PATH)


def db_execute(connection, query: str, params: tuple = ()):
    if USE_POSTGRES:
        query = query.replace("?", "%s")
    return connection.execute(query, params)


def is_unique_violation(error: Exception) -> bool:
    if isinstance(error, sqlite3.IntegrityError):
        return True
    return bool(pg_errors is not None and isinstance(error, pg_errors.UniqueViolation))


def initialize_database() -> None:
    with get_connection() as connection:
        if USE_POSTGRES:
            db_execute(
                connection,
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            db_execute(
                connection,
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            db_execute(
                connection,
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    csv_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
        else:
            db_execute(
                connection,
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            db_execute(
                connection,
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """,
            )
            db_execute(
                connection,
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    csv_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """,
            )


@app.on_event("startup")
def on_startup() -> None:
    initialize_database()


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def validate_email_or_400(email: str) -> str:
    email = normalize_email(email)
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    return email


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt_hex, _ = str(stored_hash).split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    expected = hash_password(password, salt=bytes.fromhex(salt_hex))
    return hmac.compare_digest(expected, stored_hash)


def hash_token(token: str) -> str:
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now()
    expires_at = now + timedelta(days=SESSION_DAYS)
    with get_connection() as connection:
        db_execute(
            connection,
            """
            INSERT INTO sessions (token_hash, user_id, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                hash_token(token),
                int(user_id),
                expires_at.isoformat(timespec="seconds"),
                now.isoformat(timespec="seconds"),
            ),
        )
    return token


def get_current_user(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    with get_connection() as connection:
        if not USE_POSTGRES:
            connection.row_factory = sqlite3.Row
        row = db_execute(
            connection,
            """
            SELECT sessions.token_hash, sessions.expires_at,
                   users.id, users.full_name, users.email
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
            """,
            (hash_token(token),),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="Invalid session.")
        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
        except ValueError:
            expires_at = datetime.min
        if expires_at <= datetime.now():
            db_execute(connection, "DELETE FROM sessions WHERE token_hash = ?", (row["token_hash"],))
            raise HTTPException(status_code=401, detail="Session expired.")

    return {"id": row["id"], "full_name": row["full_name"], "email": row["email"]}


def normalize_medication_name(name: str) -> str:
    normalized = (name or "").lower().strip()
    normalized = re.sub(r"^[a-z]+\.\s*", "", normalized)
    normalized = re.sub(r"\b(tab|tablet|cap|capsule|inj|injection|syp|syrup|drop|drops)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def clean_medication_name(name: str) -> str:
    value = str(name or "").strip()
    value = re.sub(r"^\s*[-*•\d.)]+\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:120]


def extract_medications_list(extracted_text: str) -> list[dict]:
    medications: list[dict] = []
    for raw_line in str(extracted_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line and not re.match(r"^\d+\.\s*", line):
            continue
        name_part = re.split(r"\s+-\s+|\s+for\s+", line, maxsplit=1, flags=re.IGNORECASE)[0]
        name = clean_medication_name(name_part)
        if len(name) < 3:
            continue
        medications.append({"medication_name": name, "raw_line": line})
    return medications


def load_seed_rows() -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in (DEPARTMENT_DATASET_PATH, CUSTOM_DEPARTMENT_DATASET_PATH):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                if "medicine_name" not in (reader.fieldnames or []) or "department" not in (reader.fieldnames or []):
                    continue
                for row in reader:
                    medicine_name = str(row.get("medicine_name", "")).strip()
                    department = str(row.get("department", "")).strip()
                    normalized_name = normalize_medication_name(medicine_name)
                    key = (normalized_name, department)
                    if not medicine_name or not department or not normalized_name or key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "medicine_name": medicine_name,
                            "department": department,
                            "normalized_name": normalized_name,
                        }
                    )
        except Exception:
            continue
    return rows


def classify_department(medicine_name: str) -> str:
    normalized = normalize_medication_name(medicine_name)
    if not normalized:
        return "Unknown"

    seed_rows = load_seed_rows()
    for row in seed_rows:
        if row["normalized_name"] == normalized:
            return str(row["department"])
    for row in seed_rows:
        item = row["normalized_name"]
        if item and (item in normalized or normalized in item):
            return str(row["department"])

    if any(keyword in normalized for keyword in CARDIOLOGY_KEYWORDS):
        return "Cardiology"
    if any(keyword in normalized for keyword in OPHTHALMOLOGY_KEYWORDS):
        return "Ophthalmology"
    return "Unknown"


def build_clean_rows(medications: list[dict], source_image: str | None = None) -> list[dict]:
    rows = []
    seen = set()
    for item in medications:
        medicine_name = clean_medication_name(item.get("medication_name", ""))
        if not medicine_name:
            continue
        department = classify_department(medicine_name)
        key = (normalize_medication_name(medicine_name), department, source_image or "")
        if key in seen:
            continue
        seen.add(key)
        row = {"medicine_name": medicine_name, "department": department}
        if source_image is not None:
            row = {"source_image": source_image, **row}
        rows.append(row)
    return rows


def generate_with_gemini(prompt: str, image_bytes: bytes, mime_type: str) -> tuple[str, str]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    last_error: Exception | None = None
    tried_models: list[str] = []
    for model_name in dict.fromkeys(GEMINI_FALLBACK_MODELS):
        tried_models.append(model_name)
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, {"mime_type": mime_type, "data": image_bytes}])
                text = getattr(response, "text", "") or ""
                if text.strip():
                    return text, model_name
            except Exception as error:
                last_error = error
                if any(fragment in str(error).lower() for fragment in ("500", "internal", "unavailable")):
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break
    raise HTTPException(status_code=502, detail=f"Gemini extraction failed after trying {tried_models}: {last_error}")


def normalize_image_for_extraction(image_bytes: bytes) -> tuple[bytes, str, dict]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    original_size = image.size
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    max_side = 1600
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    normalized_bytes = output.getvalue()
    return normalized_bytes, "image/jpeg", {
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "processed_size": {"width": image.size[0], "height": image.size[1]},
        "processed_bytes": len(normalized_bytes),
    }


def extract_prescription(image_bytes: bytes, mime_type: str, filename: str | None = None) -> dict:
    normalized_bytes, normalized_mime_type, image_info = normalize_image_for_extraction(image_bytes)
    extracted_text, model_name = generate_with_gemini(EXTRACTION_PROMPT, normalized_bytes, normalized_mime_type)
    medications = extract_medications_list(extracted_text)
    rows = build_clean_rows(medications, source_image=filename)
    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "medicine_count": len(rows),
        "medicines": rows,
        "model": model_name,
        "image_info": image_info,
    }


def sanitize_csv_name(name: str) -> str:
    cleaned = Path(str(name or "").strip()).name
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", cleaned).strip("._")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Enter a valid workspace CSV name.")
    if not cleaned.lower().endswith(".csv"):
        cleaned = f"{cleaned}.csv"
    return cleaned


def workspace_csv_path(user_id: int, csv_filename: str) -> Path:
    user_dir = WORKSPACE_DIR / f"user_{int(user_id)}"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / sanitize_csv_name(csv_filename)


def append_rows_to_csv(path: Path, rows: list[dict]) -> int:
    columns = ["source_image", "medicine_name", "department"]
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: str(row.get(column, "") or "") for column in columns})
    return count_csv_rows(path)


def create_empty_workspace_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["source_image", "medicine_name", "department"])
        writer.writeheader()


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def get_workspace_or_404(workspace_id: int, user_id: int) -> dict:
    with get_connection() as connection:
        if not USE_POSTGRES:
            connection.row_factory = sqlite3.Row
        row = db_execute(
            connection,
            "SELECT id, user_id, name, csv_filename, created_at FROM workspaces WHERE id = ? AND user_id = ?",
            (int(workspace_id), int(user_id)),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return dict(row)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}


@app.post("/api/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest) -> AuthResponse:
    email = validate_email_or_400(payload.email)
    try:
        with get_connection() as connection:
            params = (
                payload.full_name.strip(),
                email,
                hash_password(payload.password),
                datetime.now().isoformat(timespec="seconds"),
            )
            if USE_POSTGRES:
                cursor = db_execute(
                    connection,
                    """
                    INSERT INTO users (full_name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    RETURNING id
                    """,
                    params,
                )
                user_id = int(cursor.fetchone()["id"])
            else:
                cursor = db_execute(
                    connection,
                    """
                    INSERT INTO users (full_name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    params,
                )
                user_id = int(cursor.lastrowid)
    except Exception as error:
        if is_unique_violation(error):
            raise HTTPException(status_code=409, detail="An account already exists for this email.")
        raise

    token = create_session(user_id)
    return AuthResponse(token=token, user={"id": user_id, "full_name": payload.full_name.strip(), "email": email})


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    email = validate_email_or_400(payload.email)
    with get_connection() as connection:
        if not USE_POSTGRES:
            connection.row_factory = sqlite3.Row
        user = db_execute(
            connection,
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_session(int(user["id"]))
    return AuthResponse(token=token, user={"id": user["id"], "full_name": user["full_name"], "email": user["email"]})


@app.get("/api/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return {"user": user}


@app.post("/api/extract")
async def extract_image(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> dict:
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    return extract_prescription(image_bytes, mime_type, file.filename)


@app.post("/api/workspaces", response_model=WorkspaceResponse)
def create_workspace(payload: WorkspaceCreateRequest, user: dict = Depends(get_current_user)) -> WorkspaceResponse:
    csv_filename = sanitize_csv_name(payload.name)
    csv_path = workspace_csv_path(user["id"], csv_filename)
    if not csv_path.exists():
        create_empty_workspace_csv(csv_path)
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection() as connection:
        params = (int(user["id"]), payload.name.strip(), csv_filename, now)
        if USE_POSTGRES:
            cursor = db_execute(
                connection,
                """
                INSERT INTO workspaces (user_id, name, csv_filename, created_at)
                VALUES (?, ?, ?, ?)
                RETURNING id
                """,
                params,
            )
            workspace_id = int(cursor.fetchone()["id"])
        else:
            cursor = db_execute(
                connection,
                """
                INSERT INTO workspaces (user_id, name, csv_filename, created_at)
                VALUES (?, ?, ?, ?)
                """,
                params,
            )
            workspace_id = int(cursor.lastrowid)
    return WorkspaceResponse(
        id=workspace_id,
        name=payload.name.strip(),
        csv_filename=csv_filename,
        row_count=0,
        created_at=now,
    )


@app.get("/api/workspaces")
def list_workspaces(user: dict = Depends(get_current_user)) -> dict:
    with get_connection() as connection:
        if not USE_POSTGRES:
            connection.row_factory = sqlite3.Row
        rows = db_execute(
            connection,
            "SELECT id, name, csv_filename, created_at FROM workspaces WHERE user_id = ? ORDER BY id DESC",
            (int(user["id"]),),
        ).fetchall()
    workspaces = []
    for row in rows:
        path = workspace_csv_path(user["id"], row["csv_filename"])
        workspaces.append({**dict(row), "row_count": count_csv_rows(path)})
    return {"workspaces": workspaces}


@app.post("/api/workspaces/{workspace_id}/images")
async def append_workspace_image(
    workspace_id: int,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> dict:
    workspace = get_workspace_or_404(workspace_id, user["id"])
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    result = extract_prescription(image_bytes, mime_type, file.filename)
    rows = result["medicines"]
    csv_path = workspace_csv_path(user["id"], workspace["csv_filename"])
    total_rows = append_rows_to_csv(csv_path, rows) if rows else count_csv_rows(csv_path)
    return {
        "workspace": {
            "id": workspace["id"],
            "name": workspace["name"],
            "csv_filename": workspace["csv_filename"],
            "total_rows": total_rows,
        },
        "extraction": result,
        "rows_appended": len(rows),
    }


@app.get("/api/workspaces/{workspace_id}/download")
def download_workspace_csv(workspace_id: int, user: dict = Depends(get_current_user)) -> FileResponse:
    workspace = get_workspace_or_404(workspace_id, user["id"])
    csv_path = workspace_csv_path(user["id"], workspace["csv_filename"])
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Workspace CSV not found.")
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=workspace["csv_filename"],
    )
