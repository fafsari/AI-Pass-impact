@echo off
rem Start the Ki-67 app. Double-click this file.
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv is not installed yet.
  echo Open PowerShell and run:
  echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  echo Then close PowerShell and double-click run_app.bat again.
  pause
  exit /b 1
)
echo Starting the app. The first start installs packages and takes a few minutes.
echo Your browser will open http://localhost:8501 - keep this window open while you use the app.
uv run streamlit run app.py
pause
