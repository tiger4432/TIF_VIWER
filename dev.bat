@echo off
REM Runs the viewer from source with jurigged (live code reload):
REM saved or pulled changes to functions/methods are applied to the running program.
REM git pull only with DEV_PULL=1 (set DEV_PULL=1 before dev.bat): sites that must not pull use a
REM release bundle (run.bat) instead.
REM   dev.bat              (pull once if DEV_PULL=1,) then run
REM   dev.bat watch [sec]  with DEV_PULL=1 also pull every sec seconds (default 30) in a minimized window
REM Changes to __init__ / new widgets and config.json still need a restart (or Reload Declarations).
setlocal
cd /d "%~dp0"
set "VENV=.venv-dev"

if "%DEV_PULL%"=="1" (git pull --ff-only || echo WARNING: git pull failed - running the local code.)

set "REQS=requirements.txt requirements-dev.txt"
if "%WITH_PATCHCORE%"=="1" set "REQS=%REQS% requirements-patchcore.txt"
call tools\setup_venv.bat %VENV% %REQS% || goto :error
call %VENV%\Scripts\activate.bat || goto :error

if /i not "%~1"=="watch" goto :run
if not "%DEV_PULL%"=="1" goto :run
set "SEC=%~2"
if "%SEC%"=="" set "SEC=30"
start "TIF_Viewer auto-pull" /min cmd /c "for /l %%i in (0,0,1) do (git pull --ff-only -q & timeout /t %SEC% /nobreak >nul)"
echo Auto-pull every %SEC% s in the minimized "TIF_Viewer auto-pull" window - close it to stop.

:run
python -m jurigged -v tif_view_GL.py
exit /b %errorlevel%

:error
echo.
echo dev.bat FAILED
exit /b 1
