"""손글씨 판별 실시간 화면: 손글씨면 '1' 만 띄우고, 아니면 아무것도 띄우지 않습니다.

글자를 읽지(OCR) 않습니다. models/model_parameters.json 의 로지스틱 회귀(특징 8개, 가중치, bias)로
가이드 박스 안이 손글씨인지 아닌지만 판단합니다.

예)
  python check.py                 # iPhone 카메라 (기본 --device 0)
  python check.py --device 1      # Mac 내장 웹캠
  python check.py --image a.png   # 사진 한 장 판정
  python check.py --image a.png --cutout a_cutout.png   # 판정하고, 손글씨로 판단되면 손글씨만 남긴 투명 PNG 도 저장

누끼(cutout)는 손글씨로 판단해 1 이 떴을 때만 실행합니다. 손글씨가 아니면 누끼를 계산·저장하지 않습니다.

단축키: q/ESC 종료, d 판정 정보(확률·특징값) 보기/숨기기, s 현재 화면 저장(+ 글씨 누끼 cutout.png),
        m 글씨 누끼(종이는 투명) 미리보기 창 켜기/끄기,
        a 누끼 대상 전환(손글씨만 <-> 인쇄 포함), r 화면 90° 회전
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hw_demo.cutout import (CutoutConfig, LatestWorker, checkerboard_preview, cutout, handwritten_only, save_png,
                            to_rgba)
from hw_demo.camera import Camera, load_rotation, save_rotation
from hw_demo.keys import read_key
from hw_demo.hw_features import DEFAULT_PARAMS, FEATURE_KEYS, HandwritingClassifier

IPHONE_PARAMS = DEFAULT_PARAMS.with_name("model_parameters_iphone.json")   # train_weights.py 가 만드는 파일

WINDOW = "Handwriting Check"
CUTOUT_WINDOW = "Handwriting Cutout"
INK_MIN, INK_MAX = 12.0, 30.0   # 잉크 진하기 기준의 하한·상한 (화면 잡음의 5배를 이 범위로)
PAPER_MAX_STD = 15.0   # 획 주변 밝기 표준편차가 이보다 크면 '고른 종이' 가 아님
CONTRAST_RATIO = float("inf")   # (사용 안 함) 주변이 거친 곳의 획을 진하기로 인정하면 얼굴 머리카락도 통과해서 끔. 강판 색 펜은 채도로 잡음
MAX_PIECES = 30        # 글씨 조각이 이보다 많으면 글씨가 아님 (장면의 잡다한 선). iPhone 손글씨는 최대 19
STRAIGHT_RATIO = 0.08  # 곧은 직선 조각 비율이 이 이상이면 글씨가 아님. 장면 10~19%, 손글씨 0~5% (측정값)
SPECK_RATIO = 0.05     # 가장 큰 획 면적의 이 비율보다 작은 조각은 표면 결·먼지로 보고 버림
REASON_EN = {"글씨 없음": "no writing", "글씨 아님": "not writing"}   # 화면 표시용 (OpenCV 는 한글 불가)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="손글씨면 1, 아니면 아무것도 표시하지 않음")
    ap.add_argument("--device", type=int, default=0, help="카메라 번호 (이 Mac: 0 iPhone, 1 내장 웹캠)")
    ap.add_argument("--image", help="카메라 대신 사진 한 장 판정")
    ap.add_argument("--roi", type=float, nargs=2, default=[0.6, 0.45], metavar=("W", "H"),
                    help="판정할 가이드 박스 크기 (화면 대비 비율)")
    ap.add_argument("--params", default=None,
                    help="가중치 JSON (기본: iPhone 으로 다시 학습한 model_parameters_iphone.json 이 있으면 그것, 없으면 원래 JSON)")
    ap.add_argument("--cutout", metavar="PNG", help="--image 와 함께: 글씨 획만 남긴 투명 배경 PNG 저장 경로")
    ap.add_argument("--cutout-all", action="store_true",
                    help="누끼에 인쇄 글자도 포함 (기본: 손글씨로 판정된 묶음만 남김)")
    ap.add_argument("--hw-threshold", type=float, default=0.4,
                    help="누끼에서 묶음을 손글씨로 볼 확률 기준 (낮추면 수기를 덜 놓치고 인쇄가 섞임, 높이면 반대)")
    ap.add_argument("--smooth", type=int, default=8, help="최근 몇 프레임의 과반으로 표시할지 (깜빡임 방지)")
    ap.add_argument("--output", default="captures")
    return ap.parse_args(argv)


def has_ink(gray: np.ndarray) -> bool:
    """사진 모드용 간단 확인: 대비가 거의 없으면(빈 종이·벽) False."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    lo, mid, hi = np.percentile(g, [1, 50, 99])
    contrast = max(mid - lo, hi - mid)
    if contrast < 40:
        return False
    dark = mid - lo > hi - mid
    ink = (g < mid - contrast / 2) if dark else (g > mid + contrast / 2)
    return 0.003 <= ink.mean() <= 0.35


def ink_strength(img: np.ndarray, polarity: str = "dark") -> np.ndarray:
    """픽셀마다 '주변 바탕보다 얼마나 잉크 같은가' (0~255). 아래 두 값 중 큰 값.

    1. 어두운 정도: B·G·R 채널마다 '주변 바탕 - 픽셀' 을 재서 최댓값
       (닫힘 15px 로 가는 획을 지운 이미지 = 바탕). 검은 펜·연필·노란 펜(파랑 채널) 담당
    2. 색의 진한 정도(채도): '픽셀 채도 - 주변 바탕 채도'. 파랑·빨강 같은 색 펜 담당
       종이·강판·콘크리트처럼 바탕이 무채색이면 색 펜 글씨만 깨끗하게 드러남.
       어두운 정도만 쓰면 강판 표면의 결·점이 잉크로 섞이고 굵은 마커 획은 조각나서 글씨로 인정되지 않았음
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    chans = cv2.split(img) if img.ndim == 3 else [img]
    out = None
    for c in chans:
        if polarity == "dark":           # 밝은 바탕에 어두운 잉크
            d = cv2.morphologyEx(c, cv2.MORPH_CLOSE, k).astype(np.int16) - c
        else:                            # 어두운 바탕에 밝은 글씨
            d = c.astype(np.int16) - cv2.morphologyEx(c, cv2.MORPH_OPEN, k)
        out = d if out is None else np.maximum(out, d)
    if img.ndim == 3 and polarity == "dark":
        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
        k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))   # 굵은 마커 획도 지울 만큼 큰 창
        sat_bg = cv2.morphologyEx(sat, cv2.MORPH_OPEN, k_big)            # 열림: 가는 고채도 획을 지운 바탕 채도
        out = np.maximum(out, sat.astype(np.int16) - sat_bg)
    return np.clip(out, 0, 255).astype(np.uint8)


def _find_writing_one(img: np.ndarray, polarity: str):
    """카메라 화면에서 '종이 위에 쓴 글씨' 가 있는지 확인하고, 있으면 글씨 부분만 잘라 돌려줍니다.

    손글씨 판별 모델은 '정형 글자 vs 손글씨' 만 배웠고 '글씨 없음' 은 모릅니다 (bias 가 커서 애매하면 손글씨 쪽).
    그래서 방 안 물체·그림자·빈 종이를 모델에 넣으면 손글씨로 나옵니다. 이를 막기 위해 먼저 확인합니다.

    1. 잉크 진하기: 컬러 채널별로 '종이보다 얼마나 진한가' 를 재서 최댓값 (색 펜·연필 대응, ink_strength)
       → 진하기가 기준 이상인 '가는 선' 만 잉크. 큰 물체·그림자는 닫힘 연산 뒤에도 남아 잉크가 아님
       기준은 화면 잡음에 맞춰 자동: 잡음의 5배 (12~30 사이). 깨끗한 화면에서는 연한 연필도 잡힘
    2. 박스 테두리에 닿은 선(물체 모서리, 케이블, 종이 끝)은 제외
    3. 획 바로 주변이 고르게 밝은 종이인지 확인: 주변 밝기 표준편차가 크면(머리카락·눈·옷 주름처럼
       주변도 얼룩덜룩) 글씨가 아님. 실제 종이 손글씨 사진 100장에서 이 값은 중앙값 3.7, 얼굴 장면은 약 30
    4. 잉크 양, 획 굵기, 획다움(중심선 길이 × 굵기 ≈ 면적) 확인
    돌려주는 값: (글씨 부분 이미지 또는 None, 이유 문자열)
      글씨 부분 이미지는 잉크 색·진하기와 상관없이 '흰 바탕 + 진한 글씨' 로 맞춘 흑백 (판정 특징이 펜 종류에 덜 흔들리게)
    """
    s = min(1.0, 360 / img.shape[0])
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    im = cv2.GaussianBlur(im, (3, 3), 0)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    H, W = g.shape
    diff = ink_strength(im, polarity)
    noise = 1.4826 * float(np.median(np.abs(diff.astype(np.float32) - np.median(diff))))
    thr = float(np.clip(5 * noise, INK_MIN, INK_MAX))
    ink = (diff > thr).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    border_area = 0
    ink_grown = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
    ring_k = np.ones((11, 11), np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 15:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            border_area += a
            continue
        # 이 조각 주변 5px 띠(다른 잉크 제외)의 밝기가 고른지 = 종이 위인지
        x0, y0, x1, y1 = max(0, x - 6), max(0, y - 6), min(W, x + w + 6), min(H, y + h + 6)
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, ring_k) > 0) & ~ink_grown[y0:y1, x0:x1]
        if ring.sum() >= 20:
            rough = float(g[y0:y1, x0:x1][ring].std())
            strength = float(diff[y0:y1, x0:x1][comp > 0].mean())
            # 주변이 거칠어도(강판 결·녹) 획이 주변 거칠기보다 훨씬 진하면 글씨로 인정.
            # 머리카락·눈은 주변보다 약간 진한 정도라 여기서 걸러짐
            if rough > PAPER_MAX_STD and strength < CONTRAST_RATIO * rough:
                continue
        keep[i] = True
    # 가장 큰 획에 비해 아주 작은 조각(표면 결의 점, 먼지)은 버림
    if keep.any():
        big = st[keep, 4].max()
        keep &= st[:, 4] >= SPECK_RATIO * big
    m = keep[lab]
    area = int(m.sum())
    if area < 0.001 * H * W:
        return None, "글씨 없음 (잉크가 너무 적음)", 0.0
    if area > 0.15 * H * W:
        return None, "글씨 아님 (어두운 부분이 너무 많음)", 0.0
    if border_area > 2.0 * area:
        return None, "글씨 아님 (박스 밖까지 이어진 선이 많음)", 0.0
    if keep.sum() > MAX_PIECES:
        return None, "글씨 아님 (조각이 너무 많음)", 0.0
    # 곧은 직선 조각(창틀·난간·모서리·긁힘)이 많으면 글씨가 아님.
    # 손글씨 획은 휘어 있어 조각의 '두께/길이' 비(주성분 표준편차 비)가 크고, 직선은 0 에 가까움
    if keep.sum() >= 5:
        straight = 0
        for i in np.nonzero(keep)[0]:
            yy, xx = np.nonzero(lab == i)
            ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xx, yy]))))
            straight += np.sqrt(ev[0] / max(ev[1], 1e-6)) < 0.12
        if straight / keep.sum() >= STRAIGHT_RATIO:
            return None, "글씨 아님 (곧은 직선이 많음)", 0.0

    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    sk = _thin(m)
    width = 2 * float(np.median(dist[sk])) if sk.any() else 0.0
    likeness = sk.sum() * max(width, 1.0) / area     # 획이면 1 근처, 덩어리면 작음
    if width > 0.10 * H:
        return None, f"글씨 아님 (획이 너무 굵음 {width:.0f}px)", 0.0
    if not 0.5 <= likeness <= 2.0:
        return None, f"글씨 아님 (획 모양이 아님 {likeness:.2f})", 0.0

    ys, xs = np.nonzero(m)
    pad = int(0.1 * (ys.max() - ys.min() + 1)) + 4
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    hi = float(np.percentile(diff[m], 90)) if area else 255.0
    norm = 255 - np.clip(diff.astype(np.float32) * (200.0 / max(hi, 1.0)), 0, 255)   # 종이 255, 진한 잉크 ≈ 55
    clarity = float(np.median(diff[m])) / thr          # 잉크가 기준보다 얼마나 뚜렷한지
    return norm.astype(np.uint8)[y0:y1, x0:x1], "글씨 있음", clarity


def find_writing(img: np.ndarray, polarity: str = "auto"):
    """종이·강판 위 글씨 찾기. polarity="auto" 면 두 방향을 모두 시도합니다.
      dark : 밝은 바탕에 어두운(또는 색이 진한) 글씨 — 종이에 펜·연필
      light: 어두운 바탕에 밝은 글씨 — 강판에 흰색 페인트 마커
    둘 다 통과하면 잉크가 바탕과 더 뚜렷하게 구분되는 쪽을 씁니다.
    판정용 이미지는 어느 쪽이든 '흰 바탕 + 진한 글씨' 로 맞춰져 나옵니다."""
    if polarity != "auto":
        crop, reason, _ = _find_writing_one(img, polarity)
        return crop, reason
    dark = _find_writing_one(img, "dark")
    light = _find_writing_one(img, "light")
    passed = [r for r in (dark, light) if r[0] is not None]
    if not passed:
        return None, dark[1]
    best = max(passed, key=lambda r: r[2])
    return best[0], best[1] + (" (밝은 글씨)" if best is light else "")


def _thin(m: np.ndarray) -> np.ndarray:
    from hw_demo.hw_features import skeleton
    return skeleton(m)


def make_cutout(img: np.ndarray, clf, hand_only: bool = True, **cfg_kw):
    """(BGRA 누끼, 마스크). hand_only 면 손글씨로 판정된 묶음의 획만, 아니면 모든 획(인쇄 포함)."""
    cfg = CutoutConfig(**cfg_kw)
    if not hand_only:
        rgba, mask, _ = cutout(img, cfg)
        return rgba, mask
    mask = handwritten_only(img, clf, cfg)["mask"]
    return to_rgba(img, mask, cfg), mask


def idle_preview(w: int, h: int) -> np.ndarray:
    """누끼 창 대기 화면: 손글씨로 판단되지 않아 누끼를 하지 않는 동안 보여 줌."""
    out = checkerboard_preview(np.zeros((h, w, 4), np.uint8))
    text, font = "no cutout (not handwritten)", cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, 1.0, 2)
    scale = 0.8 * w / tw                                  # 글자 폭이 창 폭의 80% 가 되도록 크게
    thick = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = ((w - tw) // 2, (h + th) // 2)                  # 가운데
    cv2.putText(out, text, org, font, scale, (255, 255, 255), thick + 6, cv2.LINE_AA)   # 흰 테두리 (체크무늬 위에서도 잘 보이게)
    cv2.putText(out, text, org, font, scale, (60, 60, 60), thick, cv2.LINE_AA)
    return out


def judge(clf, img, live: bool = True):
    """(손글씨 여부, 판정 결과 또는 None, 이유)."""
    if live:
        crop, reason = find_writing(img)
        if crop is None:
            return False, None, reason
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if not has_ink(gray):
            return False, None, "글씨 없음"
        crop, reason = gray, "사진 전체"
    d = clf.decide(crop)
    return (d is not None and d.is_handwritten), d, reason


def draw(frame, roi, show_one, d, debug, reason=""):
    out = frame.copy()
    x, y, w, h = roi
    cv2.rectangle(out, (x, y), (x + w, y + h), (200, 200, 200), 1)      # 어디에 비출지 알려 주는 얇은 박스
    if show_one:
        H, W = out.shape[:2]
        scale = H / 120
        (tw, th), _ = cv2.getTextSize("1", cv2.FONT_HERSHEY_SIMPLEX, scale, int(scale * 3))
        cv2.putText(out, "1", ((W - tw) // 2, int(th * 1.3)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 220, 0),
                    int(scale * 3), cv2.LINE_AA)
    if debug:
        lines = [REASON_EN.get(reason.split(" (")[0], reason) if d is None else f"P(handwritten) = {d.prob:.3f}"]
        if d is not None:
            lines += [f"{k}: {d.features[k]:.3f}" for k in FEATURE_KEYS]
        for i, t in enumerate(lines):
            cv2.putText(out, t, (10, out.shape[0] - 20 - 22 * (len(lines) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    params = Path(args.params) if args.params else (IPHONE_PARAMS if IPHONE_PARAMS.exists() else DEFAULT_PARAMS)
    clf = HandwritingClassifier(params)
    print(f"가중치: {params.name}", file=sys.stderr)

    if args.image:
        g = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"[오류] 사진을 읽을 수 없습니다: {args.image}", file=sys.stderr)
            return 1
        hw, d, reason = judge(clf, g, live=False)
        print("1" if hw else "")
        if args.cutout and not hw:
            print(f"  (누끼 안 함: 손글씨로 판단되지 않음 → {args.cutout} 저장하지 않음)", file=sys.stderr)
        elif args.cutout:
            color = cv2.imread(args.image, cv2.IMREAD_COLOR)
            rgba, mask = make_cutout(color, clf, not args.cutout_all, drop_border=False,   # 사진 전체: 가이드 박스 없음
                                     hw_threshold=args.hw_threshold)
            save_png(args.cutout, rgba)
            print(f"  (누끼 저장: {args.cutout}, 글씨 픽셀 {int(mask.sum())}개)", file=sys.stderr)
        if d is not None:
            print(f"  (손글씨 확률 {d.prob:.3f}" + (f", 중앙값 대체: {', '.join(d.imputed)}" if d.imputed else "") + ")",
                  file=sys.stderr)
        return 0

    cam = Camera(args.device, rotate=load_rotation())   # collect.py 에서 r 로 맞춘 회전 각도를 그대로 사용
    try:
        cam.open()
    except RuntimeError as e:
        print(f"[오류] {e}", file=sys.stderr)
        return 1
    print(f"카메라 #{args.device} 시작. 박스 안에 비춘 것이 손글씨면 화면에 1 이 뜹니다.")
    print("단축키 (카메라 창을 클릭한 상태에서 입력):\n"
          "  q / ESC : 종료\n"
          "  m       : 글씨 누끼 미리보기 켜기/끄기 (종이는 체크무늬=투명, 1 이 떴을 때만 누끼)\n"
          "  a       : 누끼 대상 전환: 손글씨만 <-> 모든 글씨(인쇄 포함)\n"
          "  s       : 저장 (frame, view, result.json, 1 이 떴을 때만 cutout.png)\n"
          "  d       : 판정 정보(확률·특징값) 보기/숨기기\n"
          "  r       : 화면 90° 회전")

    history = deque(maxlen=args.smooth)
    debug, misses, show_cutout, hand_only = False, 0, False, not args.cutout_all
    # 누끼는 계산이 0.2~0.3초 걸려 메인 반복문에서 하면 카메라 화면이 끊긴다 -> 별도 스레드에서 최신 프레임만 계산
    worker = LatestWorker(lambda img, only: checkerboard_preview(
        make_cutout(img, clf, only, hw_threshold=args.hw_threshold)[0]))
    try:
        while True:
            frame = cam.read()
            if frame is None:
                misses += 1
                if misses >= 30:
                    print("[오류] 카메라 영상이 끊겼습니다.", file=sys.stderr)
                    return 1
                continue
            misses = 0
            H, W = frame.shape[:2]
            w, h = int(W * args.roi[0]), int(H * args.roi[1])
            roi = ((W - w) // 2, (H - h) // 2, w, h)
            x, y = roi[0], roi[1]
            hw, d, reason = judge(clf, frame[y:y + h, x:x + w])       # 컬러 그대로 (색 펜 대응)
            history.append(hw)
            show_one = sum(history) > len(history) / 2          # 최근 프레임 과반이 손글씨일 때만
            view = draw(frame, roi, show_one, d, debug, reason)
            cv2.imshow(WINDOW, view)
            if show_cutout:                                       # 켰을 때만. 계산은 스레드가 하고 여기서는 기다리지 않음
                if show_one:                                      # 손글씨로 판단해 1 이 떠 있을 때만 누끼 계산
                    worker.submit(frame[y:y + h, x:x + w].copy(), hand_only)
                    preview = worker.latest()
                else:                                             # 손글씨가 아니면 누끼를 하지 않음 (옛 결과도 지움)
                    worker.reset()
                    preview = idle_preview(w, h)
                if preview is not None:
                    cv2.imshow(CUTOUT_WINDOW, preview)

            key = read_key(1)                                    # 한글 입력 상태여도 동작
            if key in ("q", "esc"):
                break
            elif key == "d":
                debug = not debug
            elif key == "m":
                show_cutout = not show_cutout
                worker.reset()
                if not show_cutout:
                    cv2.destroyWindow(CUTOUT_WINDOW)
            elif key == "a":
                hand_only = not hand_only
                worker.reset()                                   # 옛 모드로 계산한 결과가 잠깐 보이지 않게
                print("누끼: " + ("손글씨만 (인쇄 제외)" if hand_only else "모든 글씨 (인쇄 포함)"))
            elif key == "r":
                save_rotation(cam.turn())
                print(f"화면 회전: {cam.rotate}°")
            elif key == "s":
                dd = Path(args.output) / ("check_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
                dd.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dd / "frame.png"), frame)
                cv2.imwrite(str(dd / "view.png"), view)
                if show_one:                                      # 1 이 떠 있을 때만 누끼 저장
                    save_png(dd / "cutout.png", make_cutout(frame[y:y + h, x:x + w], clf, hand_only, hw_threshold=args.hw_threshold)[0])      # 박스 안 글씨만, 배경 투명
                info = {"shown": "1" if show_one else "", "this_frame_handwritten": hw, "reason": reason,
                        "cutout": "cutout.png" if show_one else "안 함 (손글씨로 판단되지 않음)",
                        "prob": None if d is None else round(d.prob, 4),
                        "features": None if d is None else {k: round(v, 4) for k, v in d.features.items()}}
                (dd / "result.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"저장: {dd}")
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
