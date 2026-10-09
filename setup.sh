#!/usr/bin/env bash
# 최초 1회: 이 폴더 안에 전용 가상환경(.venv)을 만들고 패키지를 설치합니다.
# torch 는 Python 3.10 이상이 필요합니다. 다른 Python 을 쓰려면: PYTHON=/경로/python3 ./setup.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 이상이 필요합니다: " + sys.version'
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "설치 완료. 카메라 번호 확인: ./run.sh --list-cameras / 실행: ./run.sh --device 번호"
