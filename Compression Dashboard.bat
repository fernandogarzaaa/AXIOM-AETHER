@echo off
REM Double-click to open the Axiom compression control dashboard.
REM Set compression Off/Low/Medium/High and watch savings live.
REM Closing this window stops the dashboard only — the proxy keeps running.
title Axiom Compression Dashboard
cd /d "%~dp0"
start "" http://127.0.0.1:7071
python scripts\compress_dashboard.py
