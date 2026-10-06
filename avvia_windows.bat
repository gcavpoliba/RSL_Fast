@echo off
REM Avvio EQL Site Response (Windows)
cd /d "%~dp0"
where py >nul 2>nul && (py -3 run_gui.py) || (python run_gui.py)
if errorlevel 1 pause
