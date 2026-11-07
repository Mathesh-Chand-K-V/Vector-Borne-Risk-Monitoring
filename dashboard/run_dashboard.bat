@echo off
setlocal
cd /d "%~dp0"
echo Starting Streamlit dashboard. It will open in the browser.
python -m streamlit run app.py
if errorlevel 1 (
    echo Streamlit failed to start. Check Python and dependency installation.
    pause
)
