#!/usr/bin/env bash
# 학습 사진 모으기 (iPhone): h=손글씨, p=인쇄 글자, u=취소, q=종료
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[오류] 가상환경(.venv)이 없습니다. 먼저 ./setup.sh 를 실행하세요."
  exit 1
fi
exec .venv/bin/python collect.py "$@"
