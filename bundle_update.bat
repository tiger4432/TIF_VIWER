@echo off
REM Updates this git checkout from a git bundle file, without network access (no git pull):
REM   bundle_update.bat D:\down\TIF_Viewer-1.1.6.bundle
REM The bundle comes with each GitHub release: TIF_Viewer-<ver>.bundle (whole history, always works) or
REM TIF_Viewer-<ver>-since-<prev>.bundle (small; needs release <prev> in this checkout already).
REM Local changes (config.json, plugins\results_api.py, ...) are put aside and put back after the update
REM (git --autostash); if they collide with the update, git says so and keeps them in "git stash list".
setlocal
cd /d "%~dp0"
if "%~1"=="" goto :nofile
echo No such file: "%~1"
exit /b 1

:usage
if not exist "%~1" goto :nofile
git bundle verify "%~1" || goto :error
git fetch "%~1" main:refs/remotes/bundle/main || goto :error
echo.
echo Before:
git log --oneline -1 HEAD
git merge --ff-only --autostash refs/remotes/bundle/main || goto :error
echo Now:
git log --oneline -1 HEAD
echo Start the viewer with dev.bat or run.bat.
exit /b 0

:nofile
echo No such file: "%~1"
exit /b 1

:usage
echo Usage: bundle_update.bat path\TIF_Viewer-^<version^>.bundle
exit /b 1

:error
echo.
echo bundle_update.bat FAILED. "git status" shows local changes or commits in the way.
exit /b 1
