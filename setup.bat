@echo off
chcp 65001 >nul
rem 최초 1회 (Windows): 이 폴더 안에 전용 가상환경(.venv)을 만들고 패키지를 설치합니다.
rem Python 3.9 이상 필요. 설치할 때 "Add python.exe to PATH" 를 체크하세요.
cd /d "%~dp0"
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>nul
if errorlevel 1 (
  echo [오류] Python 3.9 이상을 찾지 못했습니다. https://www.python.org/downloads/ 에서 설치하세요.
  echo        설치할 때 "Add python.exe to PATH" 를 체크한 뒤 이 창을 다시 여세요.
  exit /b 1
)
python -m venv .venv || goto :err
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || goto :err
echo.
echo 설치 완료.
echo   카메라 번호 확인 : run.bat --list-cameras
echo   손글씨 판별 화면 : run_check.bat --device 번호
exit /b 0
:err
echo [오류] 설치에 실패했습니다. 위의 오류 메시지를 확인하세요.
exit /b 1
