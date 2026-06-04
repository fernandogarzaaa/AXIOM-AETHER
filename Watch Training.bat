@echo off
REM Double-click to watch Axiom train. Opens a live dashboard in your browser.
REM Closing this window stops the dashboard only — training keeps running.
title Axiom Training Dashboard
cd /d "%~dp0"
start "" http://127.0.0.1:7070
python scripts\dashboard.py
