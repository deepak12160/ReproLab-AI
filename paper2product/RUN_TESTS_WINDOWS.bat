@echo off
title Paper2Product AI - tests
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 run.py --test) else (python run.py --test)
echo.
pause
