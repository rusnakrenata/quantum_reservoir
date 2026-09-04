@echo off
setlocal

set REPO=%~dp0
set REPO=%REPO:~0,-1%
set VENV=C:\venvs\qrc_venv

echo.
echo === QRC Environment Setup ===
echo Repo : %REPO%
echo Venv : %VENV%
echo.

REM Kill any Python/Jupyter processes that may be locking venv files
echo Stopping any running Python / Jupyter processes...
taskkill /F /IM python.exe    >nul 2>&1
taskkill /F /IM pythonw.exe   >nul 2>&1
taskkill /F /IM jupyter.exe   >nul 2>&1
taskkill /F /IM jupyter-lab.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Remove broken venv inside repo if present
if exist "%REPO%\venv" (
    echo Removing venv from repo folder...
    rmdir /s /q "%REPO%\venv"
)

REM Remove old external venv
if exist "%VENV%" (
    echo Removing old venv at %VENV%...
    rmdir /s /q "%VENV%"
    if exist "%VENV%" (
        echo ERROR: Could not delete %VENV% -- try running this script as Administrator.
        pause
        exit /b 1
    )
)

REM Create venv outside OneDrive
if not exist "C:\venvs" mkdir "C:\venvs"
REM Find a supported Python (3.13, 3.12, or 3.11 - in that order).
REM Python 3.14 is intentionally excluded: no numpy 2.2.6 wheel for it.
set PYEXE=
for %%V in (3.13 3.12 3.11) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1 && set PYEXE=py -%%V
    )
)
if not defined PYEXE (
    echo ERROR: No supported Python found.
    echo Install Python 3.11, 3.12, or 3.13 ^(64-bit^) from https://www.python.org/downloads/
    echo Note: Python 3.14 is NOT supported yet ^(packages have no 3.14 wheels^).
    pause
    exit /b 1
)

echo Creating virtual environment with: %PYEXE%
%PYEXE% -m venv "%VENV%"
if errorlevel 1 (
    echo ERROR: Could not create the virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo Installing packages...
"%VENV%\Scripts\pip.exe" install -r "%REPO%\requirements.txt"
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo Registering Jupyter kernel...
"%VENV%\Scripts\python.exe" -m ipykernel install --user --name qrc_venv --display-name "Python (QRC)"

echo.
echo === Done! ===
echo To start JupyterLab:
echo   "%VENV%\Scripts\jupyter.exe" lab
echo.
pause
