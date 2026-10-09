"""학습용 사진 모으기: iPhone 카메라로 손글씨/인쇄 글자를 비추고 저장합니다.

저장 방법 (셋 중 아무거나)
  - 화면 아래 버튼 클릭: 왼쪽 [HANDWRITTEN] / 오른쪽 [PRINTED]
  - 숫자 키: 1 = 손글씨, 2 = 인쇄 글자   (한/영 상태와 상관없이 동작)
  - 영문 키: h = 손글씨, p = 인쇄 글자   (한글 입력 상태여도 동작하도록 처리)
그 밖의 키
  u : 마지막 저장 취소      r : 화면 90° 회전 (휴대폰을 세워 글씨가 옆으로 보일 때)
  q / ESC : 종료

팁
  - 손글씨·인쇄 각각 최소 10장, 가능하면 30장 이상. 글자 내용·펜 색·크기·거리·각도·조명을 바꿔 가며.
  - 박스 테두리가 초록(글씨 확인 통과)일 때 저장해야 학습에 쓰입니다.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

from check import find_writing
from hw_demo.camera import Camera, load_rotation, save_rotation
from hw_demo.keys import read_key

DATA = Path(__file__).resolve().parent / "data" / "iphone"
WINDOW = "Collect training photos"
BAR = 70   # 아래 버튼 높이 (px)


def count(cls: str) -> int:
    return len(list((DATA / cls).glob("*.png")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="손글씨/인쇄 글자 학습 사진 모으기")
    ap.add_argument("--device", type=int, default=0, help="카메라 번호 (이 Mac: 0 iPhone, 1 내장 웹캠)")
    ap.add_argument("--roi", type=float, nargs=2, default=[0.6, 0.45], metavar=("W", "H"))
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None,
                    help="화면 회전 각도 (생략하면 지난번에 r 로 맞춘 각도)")
    args = ap.parse_args(argv)
    for c in ("handwritten", "printed"):
        (DATA / c).mkdir(parents=True, exist_ok=True)

    cam = Camera(args.device, rotate=load_rotation() if args.rotate is None else args.rotate)
    try:
        cam.open()
    except RuntimeError as e:
        print(f"[오류] {e}", file=sys.stderr)
        return 1
    print(f"저장 위치: {DATA}\n  버튼 클릭 또는 1/h = 손글씨, 2/p = 인쇄, u = 취소, r = 회전, q = 종료", flush=True)

    state = {"click": None, "width": 1}
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        # 아래 버튼 영역 클릭: 왼쪽 절반 = 손글씨, 오른쪽 절반 = 인쇄
        if event == cv2.EVENT_LBUTTONDOWN and y >= param["height"] - BAR:
            state["click"] = "handwritten" if x < state["width"] // 2 else "printed"

    last, flash = None, (None, 0.0)
    try:
        while True:
            frame = cam.read()
            if frame is None:
                if read_key(1) in ("q", "esc"):
                    break
                continue
            H, W = frame.shape[:2]
            w, h = int(W * args.roi[0]), int(H * args.roi[1])
            x, y = (W - w) // 2, (H - h) // 2
            roi = frame[y:y + h, x:x + w]
            crop, reason = find_writing(roi)
            ok = crop is not None

            view = cv2.copyMakeBorder(frame, 0, BAR, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0) if ok else (0, 0, 255), 3)
            n_h, n_p = count("handwritten"), count("printed")
            cv2.rectangle(view, (0, 0), (W, 44), (0, 0, 0), -1)
            cv2.putText(view, f"handwritten {n_h}   printed {n_p}   rotate {cam.rotate}" + ("" if ok else "   (no writing found)"),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            # 버튼 2개
            for i, (label, color) in enumerate((("HANDWRITTEN  [1 / h]", (60, 120, 60)), ("PRINTED  [2 / p]", (120, 80, 40)))):
                x0, x1 = i * W // 2 + 6, (i + 1) * W // 2 - 6
                cv2.rectangle(view, (x0, H + 8), (x1, H + BAR - 8), color, -1)
                cv2.putText(view, label, (x0 + 20, H + BAR // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            # 방금 저장했다는 표시 (0.6초)
            msg, t = flash
            if msg and (datetime.now().timestamp() - t) < 0.6:
                cv2.putText(view, msg, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            state["width"] = W
            cv2.setMouseCallback(WINDOW, on_mouse, {"height": view.shape[0]})
            cv2.imshow(WINDOW, view)

            key = read_key(1)
            target = state.pop("click", None)
            state["click"] = None
            if key in ("q", "esc"):
                break
            if key in ("h", "1"):
                target = "handwritten"
            elif key in ("p", "2"):
                target = "printed"
            elif key == "u" and last is not None and last.exists():
                last.unlink()
                print(f"취소: {last.name}", flush=True)
                flash, last = ("UNDO", datetime.now().timestamp()), None
            elif key == "r":
                save_rotation(cam.turn())
                print(f"화면 회전: {cam.rotate}°", flush=True)
            if target:
                path = DATA / target / f"{target}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
                cv2.imwrite(str(path), roi)
                last = path
                flash = (f"SAVED: {target.upper()}", datetime.now().timestamp())
                print(f"저장: {target} ({'글씨 확인 통과' if ok else '주의: ' + reason}) → {path.name}", flush=True)
    finally:
        cam.close()
        cv2.destroyAllWindows()
    print(f"모은 사진: 손글씨 {count('handwritten')}장, 인쇄 {count('printed')}장. 학습: ./run_train.sh", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
