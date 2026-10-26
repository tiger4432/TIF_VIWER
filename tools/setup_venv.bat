@echo off
REM Prepares a virtual environment for build.bat / dev.bat (run from the repository root).
REM Usage: call tools\setup_venv.bat <venv dir> <requirements file> [more requirements files...]
REM Python used for the venv, in this order:
REM   1. %PYTHON% if set           e.g.  set PYTHON=C:\Python310\python.exe
REM   2. the py launcher           py -3.10
REM   3. python on PATH            e.g. an activated conda environment
REM The venv is recreated when it was made with a different Python. Exit code 1 on failure.
setlocal
set "VENV=%~1"
shift

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
%PYEXE% -c "import sys; print(sys.version); print(sys.base_prefix)"

if not exist %VENV%\pyvenv.cfg goto :make_venv
REM Reuse the venv only if it was created from this Python installation
%PYEXE% -c "import os, sys; cfg = {k.strip(): v.strip() for k, v in (l.split('=', 1) for l in open(r'%VENV%\pyvenv.cfg') if '=' in l)}; base = os.path.dirname(getattr(sys, '_base_executable', sys.executable)); same = os.path.normcase(os.path.abspath(cfg.get('home', ''))) == os.path.normcase(os.path.abspath(base)); sys.exit(0 if same else 1)"
if not errorlevel 1 goto :venv_ready
echo %VENV% was created with another Python - recreating it.
rmdir /s /q %VENV% 2>nul
if exist %VENV% goto :venv_locked

:make_venv
%PYEXE% -m venv %VENV% || goto :fail

:venv_ready
call %VENV%\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip || goto :fail

:install_loop
if "%~1"=="" goto :installed
python -m pip install -r "%~1" || goto :fail
shift
goto :install_loop

:installed
REM Qt must load in this Python (PyInstaller collects it; the viewer needs it)
python -c "import PySide6.QtCore" || goto :qt_broken
exit /b 0

:no_python
echo Python not found. Install Python 3.10 x64, activate a conda environment, or set PYTHON=path\to\python.exe
goto :fail

:bad_python
echo %PYEXE% is not a 64-bit Python 3.10 - 3.13.
echo Conda: conda create -n tifviewer -c conda-forge python=3.10  then  conda activate tifviewer  and run again.
goto :fail

:venv_locked
echo.
echo %VENV% is in use and cannot be removed: a python.exe from it is still running
echo or an editor uses it as interpreter. Close those, or end them in PowerShell with
echo   Get-Process python* ^| Where-Object { $_.Path -like '*%VENV%*' } ^| Stop-Process
echo then run again.
goto :fail

:qt_broken
echo.
echo PySide6 / Qt cannot be loaded with this Python - "DLL load failed ... procedure not found".
echo Cause: this Python installation - e.g. Anaconda base - ships an older MSVC runtime, msvcp140.dll, that Qt 6 cannot use.
echo Fix, either:
echo   1. Install Python 3.10 x64 from python.org and run from a console without conda activated.
echo   2. conda create -n tifviewer -c conda-forge python=3.10   then   conda activate tifviewer   and run again.
echo %VENV% is recreated automatically for the new Python.
goto :fail

:fail
exit /b 1
