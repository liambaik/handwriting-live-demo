@echo off
chcp 65001 >nul
rem Windows 실행: 추가 옵션은 그대로 main.py 에 전달됩니다.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [오류] 가상환경이 없습니다. 먼저 setup.bat 을 실행하세요.
  exit /b 1
)
.venv\Scripts\python main.py %*
