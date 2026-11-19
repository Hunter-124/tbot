@echo off
rem TBOT Console launcher (Windows). Double-click it, or run from a terminal:
rem
rem   launch_console.bat          open the console for the gun on the Ethernet cable (169.254.20.20 / tbot.local)
rem   launch_console.bat dev      no gun: start a stand-in gun on this computer (synthetic camera, simulated
rem                               servo) and open the console on it
rem
rem Needs Python 3.10+ from python.org ("Add to PATH"). Connecting to the gun needs no packages; "dev" needs the
rem software's own dependencies (pip install -e .). The console window is an Edge app window; closing it is enough.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
    echo Python was not found. Install Python 3.10+ from python.org with "Add to PATH" ticked.
    pause
    exit /b 1
)

if /i "%~1"=="dev" (
    echo Starting the stand-in gun on http://127.0.0.1:8765/ in its own window ...
    start "TBOT dev gun (close to stop)" %PY% -m tbot.console.server --dev --host 127.0.0.1 --port 8765
    %PY% desktop\tbot_console.pyw --pi 127.0.0.1:8765
) else (
    %PY% desktop\tbot_console.pyw %*
)
if errorlevel 1 pause
endlocal
