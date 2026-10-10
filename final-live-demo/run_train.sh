#!/usr/bin/env bash
# 모은 사진으로 가중치 다시 학습 → models/model_parameters_iphone.json
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[오류] 가상환경(.venv)이 없습니다. 먼저 ./setup.sh 를 실행하세요."
  exit 1
fi
exec .venv/bin/python train_weights.py "$@"
