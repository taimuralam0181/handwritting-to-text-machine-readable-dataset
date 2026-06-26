import streamlit as st
from xml.sax.saxutils import escape


DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

.stApp {
    background:
        radial-gradient(circle at 12% 12%, rgba(20, 184, 166, 0.16), transparent 28%),
        radial-gradient(circle at 88% 0%, rgba(59, 130, 246, 0.14), transparent 26%),
        radial-gradient(circle at 80% 85%, rgba(245, 158, 11, 0.10), transparent 30%),
        #f4f7fb !important;
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    min-height: 100vh;
    color: #172033;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}
.block-container {
    padding: 1.35rem 1.5rem 1.6rem !important;
    max-width: 1280px !important;
}

.main-header {
    padding: 1.35rem 1.5rem;
    background: linear-gradient(135deg, #12355b 0%, #0f766e 58%, #14b8a6 100%);
    border: 1px solid rgba(15, 118, 110, 0.18);
    border-radius: 8px;
    color: white;
    margin-bottom: 0.8rem;
    box-shadow: 0 12px 30px rgba(18, 53, 91, 0.16);
}
.main-header h1 {
    margin: 0;
    font-size: 1.75rem;
    letter-spacing: 0;
    font-weight: 800;
    line-height: 1.25;
    color: #ffffff !important;
}
.main-header p {
    margin: 0;
    opacity: 0.88;
    font-size: 0.98rem;
}

.dashboard-note {
    color: #526173;
    font-size: 1rem;
    margin: 0.2rem 0 1.1rem 0.05rem;
}
.control-bar {
    background: #ffffff;
    border: 1px solid #dfe7f1;
    border-radius: 8px;
    padding: 0.95rem 1rem 0.7rem;
    margin: 0 0 1rem 0;
    box-shadow: 0 8px 22px rgba(23, 32, 51, 0.05);
}
.control-bar-title {
    color: #172033;
    font-size: 0.9rem;
    font-weight: 850;
    margin-bottom: 0.35rem;
}
.control-bar-copy {
    color: #66758a;
    font-size: 0.84rem;
    margin-bottom: 0.55rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #dfe7f1 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    box-shadow: 0 8px 22px rgba(23, 32, 51, 0.05);
}
.section-wrap { padding: 0.15rem 0 0.2rem 0; }
.section-chip {
    display: inline-block;
    padding: 0.28rem 0.58rem;
    border-radius: 6px;
    background: linear-gradient(135deg, #e7f6f4, #eff6ff);
    border: 1px solid #bde5df;
    color: #0f766e;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.section-heading {
    margin: 0 0 0.25rem 0;
    color: #172033;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: 0;
}
.section-copy {
    margin: 0 0 0.9rem 0;
    color: #5b6778;
    font-size: 0.95rem;
    line-height: 1.5;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.95rem;
    margin: 1rem 0 1.25rem 0;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #dfe7f1;
    border-left: 5px solid #0f766e;
    border-radius: 8px;
    padding: 1rem 1.05rem;
    box-shadow: 0 8px 22px rgba(23, 32, 51, 0.06);
    min-height: 122px;
}
.stat-card:nth-child(2) {
    border-left-color: #2563eb;
}
.stat-card:nth-child(3) {
    border-left-color: #f59e0b;
}
.stat-card:nth-child(4) {
    border-left-color: #7c3aed;
}
.stat-label {
    color: #66758a;
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}
.stat-value {
    color: #172033;
    font-size: 1.36rem;
    font-weight: 850;
    line-height: 1.12;
    overflow-wrap: anywhere;
}
.stat-sub {
    color: #66758a;
    font-size: 0.84rem;
    margin-top: 0.3rem;
    line-height: 1.45;
}

.image-preview-card {
    border-radius: 8px;
    background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
    border: 1px solid #dfe7f1;
    padding: 0.8rem;
    margin-bottom: 0.75rem;
}
.upload-helper,
.empty-state-card {
    background: linear-gradient(135deg, #f8fafc 0%, #ecfdf5 54%, #eff6ff 100%);
    border: 1px solid #dfe7f1;
    border-left: 5px solid #14b8a6;
    border-radius: 8px;
    padding: 0.95rem 1rem;
    margin: 0.75rem 0;
}
.upload-helper-title,
.empty-state-title {
    color: #172033;
    font-size: 0.98rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.upload-helper-copy,
.empty-state-copy {
    color: #66758a;
    font-size: 0.88rem;
    line-height: 1.45;
}
.file-meta-card {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: center;
    background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
    border: 1px solid #dfe7f1;
    border-left: 4px solid #0f766e;
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
    margin: 0.85rem 0 0.7rem;
}
.file-meta-name {
    color: #172033;
    font-size: 0.95rem;
    font-weight: 800;
    overflow-wrap: anywhere;
}
.file-meta-sub {
    color: #66758a;
    font-size: 0.82rem;
    margin-top: 0.16rem;
}
.status-badge-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.25rem 0 0.8rem;
}
.dept-badge {
    border-radius: 999px;
    padding: 0.3rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 800;
    border: 1px solid transparent;
}
.dept-cardiology {
    background: linear-gradient(135deg, #eef6ff, #dbeafe);
    color: #1d4ed8;
    border-color: #bfdbfe;
}
.dept-ophthalmology {
    background: linear-gradient(135deg, #ecfdf5, #ccfbf1);
    color: #047857;
    border-color: #bbf7d0;
}
.dept-unknown {
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    color: #c2410c;
    border-color: #fed7aa;
}
.image-preview-title {
    font-size: 0.76rem;
    font-weight: 850;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #0f766e;
    margin-bottom: 0.45rem;
}
.image-preview-sub {
    color: #66758a;
    font-size: 0.9rem;
    margin-bottom: 0.65rem;
}

.panel-shell {
    background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    border: 1px solid #dfe7f1;
    border-radius: 8px;
    padding: 1.15rem;
    box-shadow: 0 10px 28px rgba(23, 32, 51, 0.07);
    min-height: 100%;
}

.sidebar-card {
    padding: 0.85rem;
    border-radius: 8px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dfe7f1;
    box-shadow: 0 8px 20px rgba(23, 32, 51, 0.05);
    margin-bottom: 0.75rem;
}
.sidebar-kicker {
    font-size: 0.76rem;
    font-weight: 850;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #0f766e;
    margin-bottom: 0.35rem;
}
.sidebar-value {
    font-size: 1.05rem;
    font-weight: 800;
    color: #172033;
    line-height: 1.2;
}
.sidebar-sub {
    color: #66758a;
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
    padding: 0.65rem;
    border-radius: 8px;
    background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
    border: 1px solid #dfe7f1;
}
.sidebar-tile-label {
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #66758a;
    font-weight: 800;
}
.sidebar-tile-value {
    margin-top: 0.18rem;
    font-size: 1rem;
    font-weight: 800;
    color: #172033;
}
.sidebar-tile:nth-child(2) {
    background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%);
}
.sidebar-tile:nth-child(3) {
    background: linear-gradient(135deg, #fff7ed 0%, #fffbeb 100%);
}

.success-message {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%);
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    padding: 0.85rem 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    color: #166534;
    font-weight: 700;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef7ff 0%, #ecfdf5 100%);
    border-right: 1px solid #d8e2ee;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #172033;
}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dfe7f1;
    border-radius: 8px;
    padding: 0.75rem;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 8px;
    font-weight: 750;
    border: 0;
    color: white;
    background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 100%);
    box-shadow: 0 8px 18px rgba(15, 118, 110, 0.18);
    min-height: 2.55rem;
}
.stButton > button[kind="secondary"] {
    background: #e7edf5;
    color: #172033;
    box-shadow: none;
}
.stButton > button[kind="secondary"]:hover {
    background: #dce6f1;
    color: #172033;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    background: linear-gradient(135deg, #115e59 0%, #0284c7 100%);
    box-shadow: 0 10px 22px rgba(15, 118, 110, 0.24);
}
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
    background: #f8fafc;
    border: 1px solid #dfe7f1;
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 0.9rem;
    font-weight: 750;
    color: #526173;
}
.stTabs [aria-selected="true"] {
    background: #ffffff;
    color: #0f766e;
    border-bottom-color: #ffffff;
}
div[data-testid="stRadio"] { margin-bottom: 0.25rem; }
div[data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.55rem;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #dfe7f1;
    border-radius: 8px;
    padding: 0.42rem 0.75rem;
    color: #172033;
}
div[data-testid="stRadio"] label:hover { border-color: #0f766e; }
div[data-testid="stExpander"] details {
    background: #f8fafc;
    border: 1px solid #dfe7f1;
    border-radius: 8px;
}
[data-testid="stFileUploader"] section {
    background: linear-gradient(135deg, #f8fafc 0%, #ecfdf5 100%);
    border: 1px dashed #5eead4;
    border-radius: 8px;
    padding: 1.35rem;
    min-height: 112px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #0f766e;
    background: linear-gradient(135deg, #ecfdf5 0%, #eff6ff 100%);
}
[data-testid="stFileUploader"] button {
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #172033;
    font-weight: 750;
}
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #dfe7f1;
}
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #dfe7f1;
    box-shadow: 0 8px 18px rgba(23, 32, 51, 0.05);
}
.stDataFrame, [data-testid="stDataFrame"] {
    max-width: 100%;
}
.dashboard-footer {
    color: #7a8798;
    font-size: 0.78rem;
    text-align: center;
    padding: 0.65rem 0 0.2rem;
}
.stAlert {
    border-radius: 8px;
}
h1, h2, h3, h4, h5, h6,
p, label, span, div {
    letter-spacing: 0;
}
@media (max-width: 900px) {
    .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .main-header h1 { font-size: 1.25rem; }
    .file-meta-card { align-items: flex-start; flex-direction: column; }
}
@media (max-width: 640px) {
    .stat-grid { grid-template-columns: 1fr; }
    .sidebar-grid { grid-template-columns: 1fr; }
    .panel-shell { padding: 0.85rem; }
}
</style>
"""

THEME_OPTIONS = {
    "Colorful Day": {
        "app_bg": "radial-gradient(circle at 12% 12%, rgba(20, 184, 166, 0.16), transparent 28%), radial-gradient(circle at 88% 0%, rgba(59, 130, 246, 0.14), transparent 26%), radial-gradient(circle at 80% 85%, rgba(245, 158, 11, 0.10), transparent 30%), #f4f7fb",
        "sidebar_bg": "linear-gradient(180deg, #eef7ff 0%, #ecfdf5 100%)",
        "header_bg": "linear-gradient(135deg, #12355b 0%, #0f766e 58%, #14b8a6 100%)",
        "panel_bg": "linear-gradient(180deg, #ffffff 0%, #fbfdff 100%)",
        "soft_bg": "linear-gradient(135deg, #f8fafc 0%, #ecfdf5 54%, #eff6ff 100%)",
        "card_bg": "linear-gradient(180deg, #ffffff 0%, #f8fbff 100%)",
        "text": "#172033",
        "muted": "#66758a",
        "border": "#dfe7f1",
        "accent": "#0f766e",
        "accent_2": "#0ea5e9",
        "accent_3": "#f59e0b",
        "button_text": "#ffffff",
    },
    "Clinical Light": {
        "app_bg": "radial-gradient(circle at 15% 8%, rgba(14, 165, 233, 0.10), transparent 30%), #f7fafc",
        "sidebar_bg": "linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%)",
        "header_bg": "linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%)",
        "panel_bg": "linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)",
        "soft_bg": "linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%)",
        "card_bg": "linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)",
        "text": "#142033",
        "muted": "#5f6f84",
        "border": "#d8e2ee",
        "accent": "#2563eb",
        "accent_2": "#0891b2",
        "accent_3": "#7c3aed",
        "button_text": "#ffffff",
    },
    "Deep Night": {
        "app_bg": "radial-gradient(circle at 18% 8%, rgba(20, 184, 166, 0.18), transparent 30%), radial-gradient(circle at 85% 12%, rgba(59, 130, 246, 0.16), transparent 28%), #07111f",
        "sidebar_bg": "linear-gradient(180deg, #0b1628 0%, #0f1f35 100%)",
        "header_bg": "linear-gradient(135deg, #0f172a 0%, #0f766e 55%, #2563eb 100%)",
        "panel_bg": "linear-gradient(180deg, #101c2e 0%, #0b1628 100%)",
        "soft_bg": "linear-gradient(135deg, #111f33 0%, #102b31 55%, #111f33 100%)",
        "card_bg": "linear-gradient(180deg, #122036 0%, #0e1a2d 100%)",
        "text": "#eef6ff",
        "muted": "#a8b6c8",
        "border": "#26384f",
        "accent": "#2dd4bf",
        "accent_2": "#60a5fa",
        "accent_3": "#fbbf24",
        "button_text": "#06121f",
    },
    "Emerald Care": {
        "app_bg": "radial-gradient(circle at 10% 10%, rgba(16, 185, 129, 0.18), transparent 30%), radial-gradient(circle at 86% 18%, rgba(45, 212, 191, 0.14), transparent 28%), #f1fbf7",
        "sidebar_bg": "linear-gradient(180deg, #ecfdf5 0%, #dff8ee 100%)",
        "header_bg": "linear-gradient(135deg, #064e3b 0%, #059669 58%, #14b8a6 100%)",
        "panel_bg": "linear-gradient(180deg, #ffffff 0%, #f3fcf8 100%)",
        "soft_bg": "linear-gradient(135deg, #f0fdf4 0%, #ccfbf1 100%)",
        "card_bg": "linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%)",
        "text": "#12312a",
        "muted": "#58746b",
        "border": "#bde8d8",
        "accent": "#059669",
        "accent_2": "#0d9488",
        "accent_3": "#84cc16",
        "button_text": "#ffffff",
    },
    "Warm Thesis": {
        "app_bg": "radial-gradient(circle at 12% 12%, rgba(245, 158, 11, 0.16), transparent 30%), radial-gradient(circle at 82% 8%, rgba(20, 184, 166, 0.12), transparent 26%), #fff8ed",
        "sidebar_bg": "linear-gradient(180deg, #fff7ed 0%, #fef3c7 100%)",
        "header_bg": "linear-gradient(135deg, #7c2d12 0%, #d97706 54%, #0f766e 100%)",
        "panel_bg": "linear-gradient(180deg, #ffffff 0%, #fffaf0 100%)",
        "soft_bg": "linear-gradient(135deg, #fffbeb 0%, #fff7ed 55%, #ecfdf5 100%)",
        "card_bg": "linear-gradient(180deg, #ffffff 0%, #fff7ed 100%)",
        "text": "#2e2418",
        "muted": "#76685a",
        "border": "#f2d3a6",
        "accent": "#d97706",
        "accent_2": "#0f766e",
        "accent_3": "#dc2626",
        "button_text": "#ffffff",
    },
}


def build_theme_css(theme_name):
    theme = THEME_OPTIONS.get(theme_name, THEME_OPTIONS["Colorful Day"])
    return f"""
    <style>
    .stApp {{
        background: {theme["app_bg"]} !important;
        color: {theme["text"]} !important;
    }}
    [data-testid="stSidebar"] {{
        background: {theme["sidebar_bg"]} !important;
        border-right-color: {theme["border"]} !important;
    }}
    .main-header {{
        background: {theme["header_bg"]} !important;
    }}
    .panel-shell,
    .stat-card,
    .sidebar-card,
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {theme["panel_bg"]} !important;
        border-color: {theme["border"]} !important;
    }}
    .sidebar-tile,
    .image-preview-card,
    .upload-helper,
    .empty-state-card,
    .file-meta-card,
    div[data-testid="stExpander"] details,
    [data-testid="stFileUploader"] section {{
        background: {theme["soft_bg"]} !important;
        border-color: {theme["border"]} !important;
    }}
    .section-chip {{
        background: {theme["soft_bg"]} !important;
        border-color: {theme["border"]} !important;
        color: {theme["accent"]} !important;
    }}
    .dashboard-note,
    .section-copy,
    .stat-label,
    .stat-sub,
    .sidebar-sub,
    .sidebar-tile-label,
    .file-meta-sub,
    .upload-helper-copy,
    .empty-state-copy,
    .control-bar-copy,
    .dashboard-footer {{
        color: {theme["muted"]} !important;
    }}
    .section-heading,
    .stat-value,
    .sidebar-tile-value,
    .upload-helper-title,
    .empty-state-title,
    .file-meta-name,
    .control-bar-title,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    div[data-testid="stRadio"] label {{
        color: {theme["text"]} !important;
    }}
    .stat-card,
    .file-meta-card,
    .upload-helper,
    .empty-state-card {{
        border-left-color: {theme["accent"]} !important;
    }}
    .stat-card:nth-child(2) {{
        border-left-color: {theme["accent_2"]} !important;
    }}
    .stat-card:nth-child(3),
    .stat-card:nth-child(4) {{
        border-left-color: {theme["accent_3"]} !important;
    }}
    .sidebar-kicker,
    .image-preview-title {{
        color: {theme["accent"]} !important;
    }}
    .stButton > button,
    .stDownloadButton > button {{
        background: linear-gradient(135deg, {theme["accent"]} 0%, {theme["accent_2"]} 100%) !important;
        color: {theme["button_text"]} !important;
        box-shadow: 0 10px 22px color-mix(in srgb, {theme["accent"]} 24%, transparent) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: {theme["card_bg"]} !important;
        color: {theme["text"]} !important;
        border: 1px solid {theme["border"]} !important;
    }}
    div[data-testid="stRadio"] label,
    [data-testid="stFileUploader"] button {{
        background: {theme["card_bg"]} !important;
        border-color: {theme["border"]} !important;
        color: {theme["text"]} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {theme["card_bg"]} !important;
        border-color: {theme["border"]} !important;
        color: {theme["muted"]} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {theme["accent"]} !important;
    }}
    </style>
    """


def apply_dashboard_styles(theme_name="Colorful Day"):
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown(build_theme_css(theme_name), unsafe_allow_html=True)


def render_dashboard_section(chip, heading, copy):
    st.markdown(
        (
            '<div class="section-wrap">'
            f'<div class="section-chip">{escape(str(chip))}</div>'
            f'<h2 class="section-heading">{escape(str(heading))}</h2>'
            f'<p class="section-copy">{escape(str(copy))}</p>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_stat_cards(cards):
    html = ['<div class="stat-grid">']
    for card in cards:
        html.append(
            (
                '<div class="stat-card">'
                f'<div class="stat-label">{escape(str(card["label"]))}</div>'
                f'<div class="stat-value">{escape(str(card["value"]))}</div>'
                f'<div class="stat-sub">{escape(str(card["subtext"]))}</div>'
                '</div>'
            )
        )
    html.append('</div>')
    st.markdown(''.join(html), unsafe_allow_html=True)
