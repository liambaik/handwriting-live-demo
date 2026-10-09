"""실시간 손글씨 인식 데모.

카메라 화면 가운데 가이드 박스에 종이 손글씨를 비추면, 글자를 한 글자씩 잘라 CNN 으로 읽고 화면에 띄웁니다.
읽을 수 있는 글자: 숫자 0~9, 대문자 A~Z (36자). 소문자·하이픈·한글은 읽지 못합니다.

예)
  python main.py                     # 카메라 0번
  python main.py --device 1          # 다른 카메라 (iPhone 연속성 카메라 등)
  python main.py --list-cameras      # 연결된 카메라 번호 확인
  python main.py --image 사진.jpg    # 사진 파일 한 장 읽기 (카메라 없이)

단축키
  q / ESC  종료             space  화면 멈춤/재개 (멈춘 화면을 천천히 볼 때)
  s        현재 결과 저장    d      디버그 보기 (잉크 마스크, 모델에 들어가는 글자 이미지)
  p        글씨 색 판단 바꾸기: auto → dark(흰 종이에 어두운 글씨) → light(어두운 바탕에 밝은 글씨)
  [ / ]    민감도 낮춤/높임 (흐린 글씨가 안 잡히면 ], 잡음이 많이 잡히면 [)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2

from hw_demo.camera import Camera, list_cameras
from hw_demo.overlay import draw
from hw_demo.recognizer import DEFAULT_MODEL, Recognizer
from hw_demo.segment import SegmentConfig, find_glyphs

WINDOW = "Handwriting Live Demo"
POLARITIES = ["auto", "dark", "light"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="카메라로 비춘 손글씨(숫자·대문자)를 실시간으로 읽는 데모")
    ap.add_argument("--device", type=int, default=0, help="카메라 번호 (--list-cameras 로 확인)")
    ap.add_argument("--list-cameras", action="store_true", help="연결된 카메라 번호를 보여주고 종료")
    ap.add_argument("--image", help="카메라 대신 사진 파일 한 장을 읽음")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--roi", type=float, nargs=2, default=[0.7, 0.5], metavar=("W", "H"),
                    help="가이드 박스 크기 (화면 대비 가로, 세로 비율)")
    ap.add_argument("--polarity", choices=POLARITIES, default="auto")
    ap.add_argument("--sensitivity", type=float, default=12.0, help="잉크 판단 기준 (작을수록 흐린 글씨도 잡음)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="확신도가 이보다 낮으면 ? 로 표시")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--output", default="captures", help="s 키로 저장할 폴더")
    ap.add_argument("--no-window", action="store_true", help="--image 와 함께: 창 없이 결과만 출력")
    return ap.parse_args(argv)


def roi_box(shape, ratio) -> tuple:
    """화면 가운데 가이드 박스 (x, y, w, h)."""
    H, W = shape[:2]
    w, h = int(W * ratio[0]), int(H * ratio[1])
    return (W - w) // 2, (H - h) // 2, w, h


def read_frame(frame, roi, cfg, rec):
    """한 프레임 처리: 가이드 박스 안 글자 찾기 → 읽기 → 줄별 문자열."""
    x, y, w, h = roi
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    glyphs, mask = find_glyphs(gray, cfg)
    reads = rec.read([g.image for g in glyphs])
    lines = {}
    for g, r in zip(glyphs, reads):
        lines.setdefault(g.line, []).append(r.char if r.conf >= MIN_CONF else "?")
    text = " / ".join("".join(v) for _, v in sorted(lines.items()))
    return glyphs, reads, text, mask


def save(out_dir: Path, frame, view, glyphs, reads, text, stable, cfg) -> Path:
    d = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "frame.png"), frame)
    cv2.imwrite(str(d / "result.png"), view)
    for i, g in enumerate(glyphs):
        cv2.imwrite(str(d / f"glyph_{i:02d}_{reads[i].char}.png"), g.image)
    info = {"text": text, "stable_text": stable, "polarity": cfg.polarity, "sensitivity": cfg.sensitivity,
            "glyphs": [{"box_in_roi": list(g.box), "line": g.line, "char": r.char, "conf": round(r.conf, 3),
                        "second": r.alt, "second_conf": round(r.alt_conf, 3)} for g, r in zip(glyphs, reads)]}
    (d / "result.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def main(argv=None) -> int:
    global MIN_CONF
    args = parse_args(argv)
    MIN_CONF = args.min_conf

    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("열 수 있는 카메라가 없습니다. 카메라 권한(시스템 설정 > 개인정보 보호 및 보안 > 카메라)을 확인하세요.")
        for i, res, mean in cams:
            note = "  ← 화면이 거의 검음 (렌즈가 가려졌거나 iPhone 화면이 꺼짐?)" if mean < 10 else ""
            print(f"  --device {i}: {res}, 평균 밝기 {mean:.0f}{note}")
        return 0

    rec = Recognizer(args.model)
    cfg = SegmentConfig(polarity=args.polarity, sensitivity=args.sensitivity)
    out_dir = Path(args.output)
    print(f"모델: {Path(args.model).name} ({len(rec.chars)}글자, 장치 {rec.device})")

    # ---- 사진 한 장 모드
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[오류] 사진을 읽을 수 없습니다: {args.image}", file=sys.stderr)
            return 1
        roi = (0, 0, frame.shape[1], frame.shape[0])   # 사진은 전체를 읽음
        glyphs, reads, text, mask = read_frame(frame, roi, cfg, rec)
        print(f"읽은 글자: {text or '(없음)'}")
        for g, r in zip(glyphs, reads):
            print(f"  줄{g.line + 1} {r.char} (확신도 {r.conf:.2f}, 2순위 {r.alt} {r.alt_conf:.2f})")
        view = draw(frame, roi, glyphs, reads, text, "", "image mode", args.min_conf, False)
        if args.no_window:
            return 0
        cv2.imshow(WINDOW, view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    # ---- 실시간 카메라 모드
    cam = Camera(args.device, args.width, args.height)
    try:
        cam.open()
    except RuntimeError as e:
        print(f"[오류] {e}", file=sys.stderr)
        return 1
    print(f"카메라 #{args.device} 시작. 가이드 박스 안에 숫자·대문자를 또박또박, 글자 사이를 띄워서 비춰 주세요.")

    history = deque(maxlen=10)    # 최근 10프레임 결과 → 가장 많이 나온 결과를 '안정된 결과'로 표시
    frozen, debug, frame, misses = False, False, None, 0
    fps, t_prev = 0.0, time.time()
    try:
        while True:
            if not frozen or frame is None:
                f = cam.read()
                if f is None:
                    misses += 1
                    if misses >= 30:
                        print("[오류] 카메라 영상이 끊겼습니다.", file=sys.stderr)
                        return 1
                    continue
                frame, misses = f, 0
            roi = roi_box(frame.shape, args.roi)
            glyphs, reads, text, mask = read_frame(frame, roi, cfg, rec)
            if not frozen:
                history.append(text)
            common = Counter(t for t in history if t).most_common(1)
            stable = common[0][0] if common and common[0][1] >= 5 else ""

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            info = f"FPS {fps:4.1f} | polarity {cfg.polarity} | sensitivity {cfg.sensitivity:.0f} | glyphs {len(glyphs)}"
            view = draw(frame, roi, glyphs, reads, text, stable, info, args.min_conf, frozen,
                        mask if debug else None)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                frozen = not frozen
            elif key == ord("d"):
                debug = not debug
            elif key == ord("p"):
                cfg.polarity = POLARITIES[(POLARITIES.index(cfg.polarity) + 1) % len(POLARITIES)]
                print(f"글씨 색 판단: {cfg.polarity}")
            elif key == ord("["):
                cfg.sensitivity = min(60, cfg.sensitivity + 2)   # 기준을 올림 → 덜 민감
                print(f"민감도 기준: {cfg.sensitivity:.0f}")
            elif key == ord("]"):
                cfg.sensitivity = max(2, cfg.sensitivity - 2)    # 기준을 내림 → 더 민감
                print(f"민감도 기준: {cfg.sensitivity:.0f}")
            elif key == ord("s"):
                d = save(out_dir, frame, view, glyphs, reads, text, stable, cfg)
                print(f"저장: {d}  (읽은 글자: {text or '-'})")
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


MIN_CONF = 0.5

if __name__ == "__main__":
    sys.exit(main())
