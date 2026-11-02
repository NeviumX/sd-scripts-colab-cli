@echo off
rem Create a Colab runtime and prepare sd-scripts for training.
rem To use a non-default WSL distro, set SD_COLAB_WSL_DISTRO, for example to Ubuntu-24.04.
setlocal
chcp 65001 >nul
set "DISTRO_OPT="
if defined SD_COLAB_WSL_DISTRO set "DISTRO_OPT=-d %SD_COLAB_WSL_DISTRO%"
title sd-scripts on Colab
wsl.exe %DISTRO_OPT% --cd "%~dp0." --exec bash -l ./up.sh %*
echo.
pause
