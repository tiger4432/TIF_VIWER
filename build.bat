@echo off
REM Builds dist\TIF_Viewer\TIF_Viewer.exe with PyInstaller (Windows x64, Python 3.10 - 3.13).
REM The build uses the development environment of dev.bat (.venv-dev: the packages the viewer runs
REM with from source) plus PyInstaller, so the exe gets exactly those. set BUILD_VENV=<dir> for another
REM one; see tools\setup_venv.bat for how Python is chosen.
setlocal
cd /d "%~dp0"
set "VENV=.venv-dev"
if not "%BUILD_VENV%"=="" set "VENV=%BUILD_VENV%"

REM set WITH_PATCHCORE=1 before build.bat to include the optional PatchCore scorer (onnxruntime)
set "REQS=requirements.txt requirements-dev.txt requirements-build.txt"
if "%WITH_PATCHCORE%"=="1" set "REQS=%REQS% requirements-patchcore.txt"
call tools\setup_venv.bat %VENV% %REQS% || goto :error
call %VENV%\Scripts\activate.bat || goto :error

pyinstaller --noconfirm --clean TIF_Viewer.spec || goto :error

REM A release bundle keeps the site's files in defaults\: use them when missing here
for %%F in (config.json products.json deploy.json) do if not exist "%%F" if exist "defaults\%%F" copy /Y "defaults\%%F" "%%F" >nul
if not exist plugins if exist defaults\plugins xcopy /E /I /Y /Q defaults\plugins plugins >nul

REM Editable deployment files live next to the exe
copy /Y config.json dist\TIF_Viewer\ >nul || goto :error
copy /Y products.json dist\TIF_Viewer\ >nul || goto :error
copy /Y deploy.json dist\TIF_Viewer\ >nul || goto :error
REM Results API plugin (edited per site, loaded at run time)
if exist plugins xcopy /E /I /Y /Q plugins dist\TIF_Viewer\plugins >nul
REM PatchCore models (used only when PatchCore is selected and onnxruntime is in the build)
if exist models xcopy /E /I /Y /Q models dist\TIF_Viewer\models >nul

echo.
echo Build OK: dist\TIF_Viewer\TIF_Viewer.exe
exit /b 0

:error
echo.
echo Build FAILED
exit /b 1
