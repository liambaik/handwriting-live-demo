#!/usr/bin/env bash
# 실행: 추가 옵션은 그대로 main.py 에 전달됩니다. 예) ./run.sh --device 1
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[오류] 가상환경(.venv)이 없습니다. 먼저 ./setup.sh 를 실행하세요."
  exit 1
fi
exec .venv/bin/python main.py "$@"
