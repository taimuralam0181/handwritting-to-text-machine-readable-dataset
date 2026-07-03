import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

try:
    import psycopg
    from psycopg import errors as pg_errors
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    pg_errors = None
    dict_row = None


AUTH_DB_PATH = Path(__file__).resolve().parent / "users.db"
LOGIN_CSS_PATH = Path(__file__).resolve().parent / "ui" / "login.css"
SESSION_DAYS = 14
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL)


def apply_login_styles():
    if LOGIN_CSS_PATH.exists():
        st.markdown(f"<style>{LOGIN_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def get_connection():
    if USE_POSTGRES:
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is set but psycopg is not installed.")
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    return sqlite3.connect(AUTH_DB_PATH)


def db_execute(connection, query, params=()):
    if USE_POSTGRES:
        query = query.replace("?", "%s")
    return connection.execute(query, params)


def is_unique_violation(error):
    if isinstance(error, sqlite3.IntegrityError):
        return True
    return bool(pg_errors is not None and isinstance(error, pg_errors.UniqueViolation))


def initialize_auth_database():
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
                CREATE TABLE IF NOT EXISTS login_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
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
                CREATE TABLE IF NOT EXISTS login_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """,
            )


def normalize_email(email):
    return str(email or "").strip().lower()


def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    password_bytes = str(password).encode("utf-8")
    digest = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, 120_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    try:
        algorithm, salt_hex, digest_hex = str(stored_hash).split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    salt = bytes.fromhex(salt_hex)
    expected_hash = hash_password(password, salt=salt)
    return hmac.compare_digest(expected_hash, stored_hash)


def create_user(full_name, email, password):
    full_name = str(full_name or "").strip()
    email = normalize_email(email)
    password = str(password or "")

    if len(full_name) < 2:
        return False, "Enter your full name."
    if "@" not in email or "." not in email:
        return False, "Enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    try:
        with get_connection() as connection:
            db_execute(
                connection,
                """
                INSERT INTO users (full_name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (full_name, email, hash_password(password), datetime.now().isoformat(timespec="seconds")),
            )
        return True, "Registration successful. You can sign in now."
    except Exception as error:
        if is_unique_violation(error):
            return False, "An account already exists for this email."
        raise


def authenticate_user(email, password):
    email = normalize_email(email)
    with get_connection() as connection:
        if not USE_POSTGRES:
            connection.row_factory = sqlite3.Row
        user = db_execute(
            connection,
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if user is None or not verify_password(password, user["password_hash"]):
        return None

    return {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
    }


def reset_user_password(email, new_password):
    email = normalize_email(email)
    new_password = str(new_password or "")

    if "@" not in email or "." not in email:
        return False, "Enter a valid email address."
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."

    with get_connection() as connection:
        cursor = db_execute(
            connection,
            """
            UPDATE users
            SET password_hash = ?
            WHERE email = ?
            """,
            (hash_password(new_password), email),
        )
        if cursor.rowcount == 0:
            return False, "No account found for this email."

        db_execute(
            connection,
            """
            DELETE FROM login_sessions
            WHERE user_id = (SELECT id FROM users WHERE email = ?)
            """,
            (email,),
        )

    return True, "Password reset successful. You can sign in now."


def hash_session_token(token):
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def create_login_session(user_id):
    token = secrets.token_urlsafe(32)
    now = datetime.now()
    expires_at = now + timedelta(days=SESSION_DAYS)
    with get_connection() as connection:
        db_execute(
            connection,
            """
            INSERT INTO login_sessions (token_hash, user_id, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                hash_session_token(token),
                int(user_id),
                expires_at.isoformat(timespec="seconds"),
                now.isoformat(timespec="seconds"),
            ),
        )
    return token


def get_query_auth_token():
    try:
        value = st.query_params.get("auth_token", "")
    except Exception:
        return ""
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def set_query_auth_token(token):
    try:
        st.query_params["auth_token"] = token
    except Exception:
        pass


def clear_query_auth_token():
    try:
        if "auth_token" in st.query_params:
            del st.query_params["auth_token"]
    except Exception:
        pass


def authenticate_session_token(token):
    token = str(token or "").strip()
    if not token:
        return None

    with get_connection() as connection:
        if not USE_POSTGRES:
            connection.row_factory = sqlite3.Row
        row = db_execute(
            connection,
            """
            SELECT login_sessions.token_hash, login_sessions.expires_at,
                   users.id, users.full_name, users.email
            FROM login_sessions
            JOIN users ON users.id = login_sessions.user_id
            WHERE login_sessions.token_hash = ?
            """,
            (hash_session_token(token),),
        ).fetchone()

        if row is None:
            return None

        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
        except ValueError:
            expires_at = datetime.min

        if expires_at <= datetime.now():
            db_execute(connection, "DELETE FROM login_sessions WHERE token_hash = ?", (row["token_hash"],))
            return None

    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
    }


def delete_login_session(token):
    token = str(token or "").strip()
    if not token:
        return
    with get_connection() as connection:
        db_execute(connection, "DELETE FROM login_sessions WHERE token_hash = ?", (hash_session_token(token),))


def render_brand_panel():
    st.markdown(
        """
        <div class="auth-brand-panel">
            <h1 class="auth-title">Handwritten Image to Machine-Readable Text Recognition</h1>
            <p class="auth-copy">
                Secure access for prescription image upload, OCR extraction, medicine review,
                department classification, and dataset-ready export.
            </p>
            <div class="auth-feature-row">
                <span>Image OCR</span>
                <span>AI Correction</span>
                <span>Structured Dataset</span>
                <span>Export Ready</span>
            </div>
            <div class="scan-stage">
                <div class="ocr-grid"></div>
                <div class="document-sheet"></div>
                <div class="scan-line"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login_panel():
    st.markdown(
        """
        <div class="auth-form-panel">
            <div class="auth-form-heading">Secure Access</div>
            <div class="auth-form-sub">Sign in or create an account to continue to the dashboard.</div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, register_tab, reset_tab = st.tabs(["Login", "Register", "Forgot Password"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
        if submitted:
            user = authenticate_user(email, password)
            if user is None:
                st.error("Invalid email or password.")
            else:
                st.session_state.authenticated_user = user
                set_query_auth_token(create_login_session(user["id"]))
                st.rerun()

    with register_tab:
        with st.form("register_form", clear_on_submit=False):
            full_name = st.text_input("Full Name", key="register_full_name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(full_name, email, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with reset_tab:
        with st.form("reset_password_form", clear_on_submit=False):
            email = st.text_input("Registered Email", key="reset_email")
            new_password = st.text_input("New Password", type="password", key="reset_new_password")
            confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
            submitted = st.form_submit_button("Reset Password", use_container_width=True)
        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = reset_user_password(email, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.markdown("</div>", unsafe_allow_html=True)


def render_auth_gate():
    initialize_auth_database()
    user = st.session_state.get("authenticated_user")
    if user:
        return user

    token_user = authenticate_session_token(get_query_auth_token())
    if token_user:
        st.session_state.authenticated_user = token_user
        return token_user

    apply_login_styles()
    st.markdown(
        """
        <div class="auth-keywords">
            <span>OCR</span><span>AI</span><span>Text</span><span>Dataset</span><span>Recognition</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_col, right_col = st.columns([1.08, 0.92], gap="large")
    with left_col:
        render_brand_panel()
    with right_col:
        render_login_panel()
    return None


def render_logout_control(user):
    st.caption(f"Signed in as {user.get('full_name', 'User')}")
    st.caption(user.get("email", ""))
    if st.button("Logout", use_container_width=True):
        delete_login_session(get_query_auth_token())
        clear_query_auth_token()
        st.session_state.pop("authenticated_user", None)
        st.rerun()
