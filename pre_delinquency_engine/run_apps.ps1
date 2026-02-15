# Start API in background, then dashboard in foreground.
# API: http://localhost:8000  |  Dashboard: http://localhost:8501
Set-Location $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
Start-Sleep -Seconds 2
streamlit run dashboard/app.py
