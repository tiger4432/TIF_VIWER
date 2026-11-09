@echo off
REM Updates this folder to the latest GitHub release, without git (tools\update.py): downloads the
REM release zip, checks its sha256 and unpacks it here. config.json, products.json, deploy.json and
REM plugins\ are kept. Close the viewer first.
REM   update.bat            latest release
REM   update.bat 1.1.2      that version
REM   update.bat --check    only show the installed and the latest version
setlocal
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" tools\update.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo update.bat FAILED
exit /b 1
