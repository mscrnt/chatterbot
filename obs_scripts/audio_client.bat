@echo off
REM chatterbot audio client launcher (Windows).
REM
REM Uses an isolated `.venv-win` so it never collides with the Linux/WSL
REM `.venv` the dashboard container creates. First run installs deps;
REM subsequent runs reuse the existing venv (fast).
REM
REM Usage:
REM   audio_client.bat                    -- starts capturing from default device
REM   audio_client.bat --list-devices     -- prints available devices
REM   audio_client.bat --device "Mic"     -- pick a device by name fragment
REM   audio_client.bat --url http://...   -- override dashboard URL
REM
REM Pass any audio_client.py flags through; they're forwarded as-is.

setlocal

REM Move to the repo root so relative paths work regardless of where
REM the bat is launched from.
cd /d "%~dp0\.."

set VENV_DIR=.venv-win
set REQS=numpy sounddevice

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [audio_client] first run — bootstrapping %VENV_DIR%
    REM Prefer Python 3.11 if available (matches what OBS likes too),
    REM but fall back to whatever `py` resolves.
    where py >nul 2>nul
    if errorlevel 1 (
        echo [audio_client] error: 'py' launcher not found. Install Python from python.org.
        exit /b 1
    )
    py -3.11 -m venv "%VENV_DIR%" 2>nul
    if errorlevel 1 (
        echo [audio_client] python 3.11 not available, trying default Python...
        py -m venv "%VENV_DIR%"
        if errorlevel 1 (
            echo [audio_client] error: failed to create venv.
            exit /b 1
        )
    )
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >nul
    "%VENV_DIR%\Scripts\python.exe" -m pip install %REQS%
    if errorlevel 1 (
        echo [audio_client] error: failed to install deps.
        exit /b 1
    )
)

"%VENV_DIR%\Scripts\python.exe" obs_scripts\audio_client.py %*

endlocal
