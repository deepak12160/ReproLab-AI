@echo off
title Paper2Product AI
cd /d "%~dp0"
echo ============================================
echo   Paper2Product AI - starting (first run installs things, please wait)
echo ============================================
where py >nul 2>nul
if %errorlevel%==0 goto usepy
where python >nul 2>nul
if %errorlevel%==0 goto usepython
echo.
echo Python was not found on this computer.
echo 1. Open https://www.python.org/downloads/  and install Python 3.13
echo 2. On the first installer screen TICK "Add python.exe to PATH"
echo 3. Double-click this file again.
goto end
:usepy
py -3 run.py %*
goto end
:usepython
python run.py %*
:end
echo.
pause
