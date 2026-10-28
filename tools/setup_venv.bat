@echo off
REM Prepares a virtual environment for build.bat / dev.bat (run from the repository root).
REM Usage: call tools\setup_venv.bat <venv dir> <requirements file> [more requirements files...]
REM Python used for the venv, in this order (conda does not need to be activated, e.g. in a VS Code
REM terminal that does not load conda):
REM   1. %PYTHON% if set               e.g.  set PYTHON=C:\Python310\python.exe
REM   2. python.local in the repository root: one line, the path of python.exe (not in git)
REM   3. an activated conda environment (not base)
REM   4. the conda environment "tifviewer" (or %CONDA_ENV%) in the usual conda install folders
REM   5. the py launcher               py -3.10
REM   6. python on PATH
REM The venv is recreated when it was made with another Python (installation or version) or its
REM pip is broken. Exit code 1 on failure.
setlocal
set "VENV=%~1"
shift

set "PYEXE="
if defined PYTHON set PYEXE="%PYTHON%"
if defined PYEXE goto :check_python

set "PYLOCAL="
if exist python.local set /p PYLOCAL=<python.local
if defined PYLOCAL set "PYLOCAL=%PYLOCAL:"=%"
if defined PYLOCAL set PYEXE="%PYLOCAL%"
if defined PYEXE goto :check_python

if not defined CONDA_PREFIX goto :conda_envs
if /i "%CONDA_DEFAULT_ENV%"=="base" goto :conda_envs
if exist "%CONDA_PREFIX%\python.exe" set PYEXE="%CONDA_PREFIX%\python.exe"
if defined PYEXE goto :check_python

:conda_envs
if not defined CONDA_ENV set "CONDA_ENV=tifviewer"
set CONDA_DIRS="%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3"
set CONDA_DIRS=%CONDA_DIRS% "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniforge3"
set CONDA_DIRS=%CONDA_DIRS% "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniforge3" "%USERPROFILE%\.conda"
for %%D in (%CONDA_DIRS%) do if not defined PYEXE if exist "%%~D\envs\%CONDA_ENV%\python.exe" set PYEXE="%%~D\envs\%CONDA_ENV%\python.exe"
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
REM Reuse the venv only if it was created from this Python installation AND version (an upgraded
REM Python at the same path leaves an old pip in the venv: "pkgutil has no attribute ImpImporter")
%PYEXE% -c "import os, sys; cfg = {k.strip(): v.strip() for k, v in (l.split('=', 1) for l in open(r'%VENV%\pyvenv.cfg') if '=' in l)}; base = os.path.dirname(getattr(sys, '_base_executable', sys.executable)); same = os.path.normcase(os.path.abspath(cfg.get('home', ''))) == os.path.normcase(os.path.abspath(base)); ver = (cfg.get('version_info') or cfg.get('version') or '').split('.')[:2]; same = same and ver == [str(sys.version_info[0]), str(sys.version_info[1])]; sys.exit(0 if same else 1)"
if not errorlevel 1 goto :venv_ready
echo %VENV% was created with another Python - recreating it.
rmdir /s /q %VENV% 2>nul
if exist %VENV% goto :venv_locked

:make_venv
%PYEXE% -m venv %VENV% || goto :fail

:venv_ready
call %VENV%\Scripts\activate.bat || goto :fail
REM pip bundled with this Python first (repairs an outdated pip), then the latest
python -m ensurepip --upgrade >nul 2>nul
REM pip health without network (an old pip fails here with "pkgutil ... ImpImporter")
python -m pip list >nul 2>nul
if not errorlevel 1 goto :pip_ok
if defined VENV_RETRIED goto :fail
echo.
echo pip in %VENV% does not work with this Python - recreating %VENV%.
set "VENV_RETRIED=1"
call %VENV%\Scripts\deactivate.bat >nul 2>nul
rmdir /s /q %VENV% 2>nul
if exist %VENV% goto :venv_locked
goto :make_venv

:pip_ok
python -m pip install --upgrade pip || echo WARNING: pip could not be upgraded (offline?) - continuing with the installed pip.

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
echo Python not found. Install Python 3.10 x64, or create the conda environment "tifviewer",
echo or write the path of python.exe into python.local in the repository root, e.g.
echo   C:\Users\me\miniconda3\envs\tifviewer\python.exe
goto :fail

:bad_python
echo %PYEXE% is not a 64-bit Python 3.10 - 3.13.
echo Conda: conda create -n tifviewer -c conda-forge python=3.10   (found automatically, no activation needed)
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
echo   2. conda create -n tifviewer -c conda-forge python=3.10   (found automatically, no activation needed)
echo %VENV% is recreated automatically for the new Python.
goto :fail

:fail
exit /b 1
