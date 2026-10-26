@echo off
REM Builds dist\TIF_Viewer\TIF_Viewer.exe with PyInstaller (Windows x64, Python 3.10 - 3.13).
REM Python used to create .venv, in this order:
REM   1. %PYTHON% if set           e.g.  set PYTHON=C:\Python310\python.exe
REM   2. the py launcher           py -3.10
REM   3. python on PATH            e.g. an activated conda environment
setlocal
cd /d "%~dp0"

if exist .venv\Scripts\python.exe goto :venv_ready

set "PYEXE="
if defined PYTHON set PYEXE="%PYTHON%"
if defined PYEXE goto :check_python

where py >nul 2>nul
if errorlevel 1 goto :try_python
py -3.10 -c "import sys" >nul 2>nul
if errorlevel 1 goto :try_python
set "PYEXE=py -3.10"
goto :check_python

:try_python
where python >nul 2>nul
if errorlevel 1 goto :no_python
set "PYEXE=python"

:check_python
%PYEXE% -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) and sys.maxsize > 2**32 else 1)"
if errorlevel 1 goto :bad_python
echo Using Python: %PYEXE%
%PYEXE% -c "import sys; print(sys.version)"
%PYEXE% -m venv .venv || goto :error

:venv_ready
call .venv\Scripts\activate.bat || goto :error

python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt -r requirements-build.txt || goto :error

REM Qt must load in this Python before PyInstaller can collect it
python -c "import PySide6.QtCore" || goto :qt_broken

pyinstaller --noconfirm --clean TIF_Viewer.spec || goto :error

REM Editable deployment files live next to the exe
copy /Y config.json dist\TIF_Viewer\ >nul || goto :error
copy /Y products.json dist\TIF_Viewer\ >nul || goto :error
copy /Y deploy.json dist\TIF_Viewer\ >nul || goto :error

echo.
echo Build OK: dist\TIF_Viewer\TIF_Viewer.exe
exit /b 0

:no_python
echo Python not found. Install Python 3.10 x64, activate a conda environment, or set PYTHON=path\to\python.exe
goto :error

:bad_python
echo %PYEXE% is not a 64-bit Python 3.10 - 3.13.
echo Conda: conda create -n tifviewer -c conda-forge python=3.10  then  conda activate tifviewer  and run build.bat again.
goto :error

:qt_broken
echo.
echo PySide6 / Qt cannot be loaded with this Python - "DLL load failed ... procedure not found".
echo Cause: Anaconda base Python ships an older MSVC runtime - msvcp140.dll - that Qt 6 cannot use.
echo Fix, either:
echo   1. Install Python 3.10 x64 from python.org, open a new console without conda activated.
echo   2. conda create -n tifviewer -c conda-forge python=3.10   then   conda activate tifviewer
echo Then delete the .venv folder - rmdir /s /q .venv - and run build.bat again.
goto :error

:error
echo.
echo Build FAILED
exit /b 1
