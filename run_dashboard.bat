@echo off
title CardioPredict AI Dashboard
echo ===================================================
echo Starting CardioPredict Heart Disease AI Dashboard
echo ===================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
