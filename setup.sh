#!/usr/bin/env bash
# 최초 1회: 이 폴더 안에 전용 가상환경(.venv)을 만들고 패키지를 설치합니다 (macOS / Linux).
# Python 3.9 이상이 필요합니다. 설치된 것 중 가장 새 버전을 자동으로 고릅니다.
# 다른 Python 을 쓰려면: PYTHON=/경로/python3 ./setup.sh
set -e
cd "$(dirname "$0")"

PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      PY=$c; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "[오류] Python 3.9 이상을 찾지 못했습니다."
  echo "  macOS: https://www.python.org/downloads/ 에서 설치하거나  brew install python@3.12"
  echo "  설치 후 터미널을 다시 열고 ./setup.sh 를 다시 실행하세요."
  exit 1
fi
echo "사용할 Python: $PY ($("$PY" -V 2>&1))"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo
echo "설치 완료."
echo "  카메라 번호 확인 : ./run.sh --list-cameras"
echo "  손글씨 판별 화면 : ./run_check.sh --device 번호"
