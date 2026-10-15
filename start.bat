@echo off
echo Starting NeuroForge...
start "NeuroForge API" powershell -NoExit -Command "uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
timeout /t 4
start "NeuroForge Site" powershell -NoExit -Command "python -m streamlit run src/ui/app.py"
echo Open http://localhost:8501 in browser
