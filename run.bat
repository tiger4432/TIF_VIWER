@echo off
REM Runs the viewer from an unpacked release (no git). First run: creates .venv and installs
REM requirements.txt. The site's own files (config.json, products.json, deploy.json, plugins\) are
REM copied from defaults\ only when missing, so unpacking a new release over this folder keeps them.
setlocal
cd /d "%~dp0"
call :site_files || goto :error
set "REQS=requirements.txt"
if "%WITH_PATCHCORE%"=="1" set "REQS=%REQS% requirements-patchcore.txt"
call tools\setup_venv.bat .venv %REQS% || goto :error
call .venv\Scripts\activate.bat || goto :error
python tif_view_GL.py
exit /b %errorlevel%

:site_files
for %%F in (config.json products.json deploy.json) do (
    if not exist "%%F" if exist "defaults\%%F" copy /Y "defaults\%%F" "%%F" >nul
)
if not exist "plugins\results_api.py" if exist "defaults\plugins" xcopy /E /I /Y /Q "defaults\plugins" "plugins" >nul
exit /b 0

:error
echo.
echo run.bat FAILED
exit /b 1
