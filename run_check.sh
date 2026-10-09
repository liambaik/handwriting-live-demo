#!/usr/bin/env bash
# 손글씨 판별 화면: 손글씨면 1, 아니면 아무것도 표시 안 함. 예) ./run_check.sh  (기본 iPhone = --device 0)
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[오류] 가상환경(.venv)이 없습니다. 먼저 ./setup.sh 를 실행하세요."
  exit 1
fi
exec .venv/bin/python check.py "$@"
