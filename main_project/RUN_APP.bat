@echo off
cd /d "%~dp0"
title Prescription Data Extraction Dashboard
streamlit run app.py --server.headless=false
