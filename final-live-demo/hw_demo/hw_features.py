"""손글씨 판별용 형태 특징 8개 + 로지스틱 회귀 (model_parameters.json 의 가중치 사용).

주의: model_parameters.json 에는 평균·표준편차·가중치만 있고 '특징을 어떻게 계산했는지' 는 없습니다.
그래서 아래 계산 방식은 특징 이름에 맞춰 이 프로젝트에서 정의한 것입니다.
학습 데이터(formal_handwritten_120_png)에서 계산한 평균·표준편차를 JSON 값과 비교해 확인했습니다 (README 참고).

계산 순서
  1. 글씨를 밝게 맞추고(어두운 글씨는 반전) Otsu 이진화 → 잉크 마스크
  2. 글자 높이를 일정하게(약 100px) 확대/축소 → 크기에 따라 특징값이 달라지지 않게
  3. 아래 8개 특징 계산. 계산할 수 없으면(예: 글자가 2개뿐이라 간격 편차를 못 구함) NaN → JSON 의 중앙값으로 대체
  4. z = (x - 평균) / 표준편차,  점수 = bias + Σ 가중치·z,  확률 = 1 / (1 + e^-점수)
  5. 확률 ≥ 0.5 이면 손글씨
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

DEFAULT_PARAMS = Path(__file__).resolve().parent.parent / "models" / "model_parameters.json"
TARGET_H = 100   # 특징 계산 전 글자(잉크 영역) 높이를 이 크기로 맞춤

FEATURE_KEYS = ["stroke_width_var", "spacing_dev", "slant_var", "baseline_wobble",
                "curvature_var", "cc_size_var", "contour_irregularity", "stroke_connectivity"]


# ---------------------------------------------------------------- 전처리
def bright_ink(gray: np.ndarray) -> np.ndarray:
    """배경(중앙값)에서 더 멀리 떨어진 쪽을 글씨로 보고, 글씨가 밝도록 맞춤."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    lo, mid, hi = np.percentile(g, [1, 50, 99])
    return 255 - gray if mid - lo > hi - mid else gray


def ink_mask(gray: np.ndarray) -> Optional[np.ndarray]:
    """잉크 마스크(bool), 잉크 영역만 잘라 높이 TARGET_H 로 맞춤. 잉크가 없으면 None."""
    b = bright_ink(gray)
    _, m = cv2.threshold(cv2.GaussianBlur(b, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    big = st[1:, 4].max()
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= max(4, 0.02 * big)     # 작은 잡음 점 제거
    m = keep[lab]
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = TARGET_H / m.shape[0]
    m = cv2.resize(m.astype(np.uint8) * 255, (max(1, round(m.shape[1] * s)), TARGET_H),
                   interpolation=cv2.INTER_LINEAR) > 127
    return np.pad(m, 4)


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 세선화로 1px 중심선을 만듭니다 (모서리에 잔가지가 적어 획 두께 측정에 유리)."""
    img = mask.astype(np.uint8).copy()
    while True:
        changed = False
        for it in range(2):
            p = np.pad(img, 1)
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9              # 이웃 잉크 수
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))  # 0→1 전환 수
            c1 = (P2 * P4 * P6 == 0) if it == 0 else (P2 * P4 * P8 == 0)
            c2 = (P4 * P6 * P8 == 0) if it == 0 else (P2 * P6 * P8 == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            return img > 0


def glyph_boxes(mask: np.ndarray):
    """연결요소를 가로로 겹치면 한 글자로 묶은 상자 목록 (왼쪽부터)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = sorted([tuple(int(v) for v in st[i, :4]) for i in range(1, n)], key=lambda c: c[0])
    boxes = []
    for x, y, w, h in comps:
        if boxes:
            bx, by, bw, bh = boxes[-1]
            if min(bx + bw, x + w) - max(bx, x) >= 0.5 * min(bw, w):
                nx, ny = min(bx, x), min(by, y)
                boxes[-1] = (nx, ny, max(bx + bw, x + w) - nx, max(by + bh, y + h) - ny)
                continue
        boxes.append((x, y, w, h))
    return boxes


def _cv(v) -> float:
    v = np.asarray(v, float)
    return float(v.std() / v.mean()) if len(v) and v.mean() > 0 else float("nan")


# ---------------------------------------------------------------- 특징 8개
def compute_features(gray: np.ndarray) -> Optional[Dict[str, float]]:
    m = ink_mask(gray)
    if m is None or m.sum() < 30:
        return None
    feats: Dict[str, float] = {}
    sk = skeleton(m)
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)

    # ① 획 두께 변화: 중심선 위 획 두께(거리변환)의 변동계수
    feats["stroke_width_var"] = _cv(dist[sk])

    boxes = glyph_boxes(m)
    heights = [b[3] for b in boxes]
    # ④ 문자 간격 편차: 이웃 글자 사이 간격(겹치면 0)의 변동계수. 글자 2개 이상, 간격 평균이 0 이면 계산 불가
    if len(boxes) >= 2:
        gaps = np.clip([boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]) for i in range(len(boxes) - 1)], 0, None)
        feats["spacing_dev"] = _cv(gaps)
    else:
        feats["spacing_dev"] = float("nan")

    # ⑤ 기울기 변화: 경계 방향 중 세로 획(세로에서 ±30° 이내)의 기울기 표준편차(도)
    #    인쇄 글자는 세로 획이 모두 같은 각도, 손글씨는 획마다 기울기가 다름
    mf = m.astype(np.float32)
    gx = cv2.Sobel(mf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mf, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy) > 0.5
    ang = np.degrees(np.arctan2(gy[edge], gx[edge])) % 180 - 90   # 0 = 경계 법선이 가로 = 세로 획
    vert = ang[np.abs(ang) < 30]
    feats["slant_var"] = float(vert.std()) if len(vert) > 10 else float("nan")

    # ⑥ Baseline 흔들림: 글자 아래끝 높이의 (최대 - 최소) / 글자 높이 중앙값. 글자 2개 이상
    if len(boxes) >= 2:
        bottoms = np.array([b[1] + b[3] for b in boxes], float)
        feats["baseline_wobble"] = float((bottoms.max() - bottoms.min()) / np.median(heights))
    else:
        feats["baseline_wobble"] = float("nan")

    # ⑧ 곡률 변화: 바깥 외곽선을 5px 간격으로 따라가며 잰 방향 변화(라디안, 부호 포함)의 표준편차
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    turns, hull_area = [], 0.0
    step = 5
    for c in contours:
        hull_area += cv2.contourArea(cv2.convexHull(c))
        pts = c[:, 0, :].astype(np.float32)
        if len(pts) < 3 * step:
            continue
        q = pts[::step]
        d = np.diff(np.vstack([q, q[:1]]), axis=0)
        th = np.arctan2(d[:, 1], d[:, 0])
        turns.extend(np.angle(np.exp(1j * np.diff(np.concatenate([th, th[:1]])))).tolist())
    feats["curvature_var"] = float(np.std(turns)) if len(turns) > 5 else float("nan")

    # ⑩ 연결요소 크기 분산: 연결요소 면적의 변동계수
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    feats["cc_size_var"] = _cv(st[1:, 4]) if n > 2 else 0.0

    # ⑪ 외곽선 불규칙도: 조각별 볼록 껍질 면적 합 / 잉크 면적 (오목하고 들쭉날쭉할수록 큼)
    feats["contour_irregularity"] = float(hull_area / m.sum()) if m.sum() else float("nan")

    # ⑫ 획 연결성: 1 / 연결요소 수 (획이 이어져 덩어리가 적을수록 큼)
    feats["stroke_connectivity"] = 1.0 / (n - 1) if n > 1 else float("nan")
    return feats


# ---------------------------------------------------------------- 로지스틱 회귀
@dataclass
class Decision:
    prob: float                 # 손글씨일 확률
    is_handwritten: bool
    features: Dict[str, float]  # 계산값 (NaN 은 중앙값으로 대체 전 값)
    imputed: List[str]          # 계산할 수 없어 중앙값으로 대체한 특징


class HandwritingClassifier:
    def __init__(self, params_path: Path = DEFAULT_PARAMS):
        p = json.loads(Path(params_path).read_text(encoding="utf-8"))
        self.bias = float(p["bias"])
        self.threshold = float(p.get("decision_threshold", 0.5))
        fs = sorted(p["features"], key=lambda f: f["index"])
        if len(fs) != len(FEATURE_KEYS):
            raise ValueError(f"특징 수가 다릅니다: JSON {len(fs)}개, 코드 {len(FEATURE_KEYS)}개")
        self.names = [f["name"] for f in fs]
        self.median = np.array([f["median_for_missing"] for f in fs])
        self.mean = np.array([f["mean"] for f in fs])
        self.std = np.array([f["std"] for f in fs])
        self.weight = np.array([f["weight"] for f in fs])

    def decide(self, gray: np.ndarray) -> Optional[Decision]:
        feats = compute_features(gray)
        if feats is None:
            return None
        x = np.array([feats[k] for k in FEATURE_KEYS], float)
        miss = ~np.isfinite(x)
        x[miss] = self.median[miss]
        z = (x - self.mean) / self.std
        score = self.bias + float(self.weight @ z)
        prob = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))
        return Decision(prob, prob >= self.threshold, feats,
                        [self.names[i] for i in np.nonzero(miss)[0]])
