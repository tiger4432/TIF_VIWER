@echo off
REM Updates this git checkout from release bundles, without network access (no git pull):
REM   bundle_update.bat                       the newest bundle in releases\ (put the downloaded bundles there)
REM   bundle_update.bat D:\down\TIF_Viewer-1.1.6.bundle   that bundle
REM The bundles come with each GitHub release: TIF_Viewer-<ver>.bundle (whole history, always works) or
REM TIF_Viewer-<ver>-since-<prev>.bundle (small; needs release <prev> in this checkout already).
REM dev.bat and run.bat do the same at start (dev.bat watch: also while running).
REM Local changes (config.json, plugins\results_api.py, ...) are put aside and put back after the update
REM (git --autostash); if they collide with the update, git says so and keeps them in "git stash list".
setlocal
cd /d "%~dp0"
set "PY=python"
if exist ".venv-dev\Scripts\python.exe" set "PY=.venv-dev\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" tools\bundle_update.py %*
exit /b %errorlevel%
