@echo off
REM Builds dist\TIF_Viewer\TIF_Viewer.exe with PyInstaller (Windows x64, Python 3.10 - 3.13).
REM The build uses its own virtual environment (.venv-build), separate from a .venv an IDE may
REM keep open; see tools\setup_venv.bat for how Python is chosen.
setlocal
cd /d "%~dp0"
set "VENV=.venv-build"

call tools\setup_venv.bat %VENV% requirements.txt requirements-build.txt || goto :error
call %VENV%\Scripts\activate.bat || goto :error

pyinstaller --noconfirm --clean TIF_Viewer.spec || goto :error

REM Editable deployment files live next to the exe
copy /Y config.json dist\TIF_Viewer\ >nul || goto :error
copy /Y products.json dist\TIF_Viewer\ >nul || goto :error
copy /Y deploy.json dist\TIF_Viewer\ >nul || goto :error

echo.
echo Build OK: dist\TIF_Viewer\TIF_Viewer.exe
exit /b 0

:error
echo.
echo Build FAILED
exit /b 1
