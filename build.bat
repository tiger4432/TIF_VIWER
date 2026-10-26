@echo off
REM Builds dist\TIF_Viewer\TIF_Viewer.exe with PyInstaller (Windows x64, Python 3.10).
setlocal
cd /d "%~dp0"

if not exist .venv (
    py -3.10 -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat || goto :error

python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt -r requirements-build.txt || goto :error

pyinstaller --noconfirm --clean TIF_Viewer.spec || goto :error

REM Editable deployment files live next to the exe
copy /Y config.json dist\TIF_Viewer\ >nul || goto :error
copy /Y products.json dist\TIF_Viewer\ >nul || goto :error

echo.
echo Build OK: dist\TIF_Viewer\TIF_Viewer.exe
exit /b 0

:error
echo.
echo Build FAILED
exit /b 1
