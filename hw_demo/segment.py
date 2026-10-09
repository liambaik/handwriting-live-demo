"""카메라 영상에서 손글씨 글자를 찾아 한 글자씩 잘라, 모델이 학습한 형식으로 바꿉니다.

모델이 학습한 형식 (바꾸면 인식률이 떨어지니 그대로 유지)
  - 128x128 회색 배경(밝기 110) 위 밝은 글씨(밝기 225 근처)
  - 글자 외곽의 긴 변을 100px 로 맞춰 가운데 정렬 → 마지막에 64x64 로 축소
종이에 쓴 손글씨는 '흰 바탕 + 어두운 글씨' 라서 먼저 밝기를 반전합니다.

순서
  1. 극성 맞추기: 글씨가 항상 '밝게' 되도록 (종이 글씨는 반전)
  2. 잉크 마스크: 주변보다 확실히 밝은 픽셀 (조명이 고르지 않아도 되도록 적응형 이진화)
  3. 정리: 작은 점, 가이드 박스 테두리에 닿은 덩어리(종이 끝·손가락 그림자), 긴 선(줄 노트 선) 제거
  4. 글자 묶기: 가로로 겹치고 세로로 가까운 조각은 한 글자 (예: 끊긴 획)
  5. 줄 나누기: 세로 위치로 줄을 나누고, 줄 안에서는 왼쪽부터 정렬
  6. 글자 이미지: 그 글자의 잉크만 남기고 (옆 글자 조각 제거) 학습 형식으로 정규화
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

SIZE, BG, INK, FIT = 128, 110, 225, 100   # 학습 데이터 형식 (손글씨 CNN 과 같아야 함)
MODEL_SIZE = 64                           # 모델 입력 크기

Box = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class SegmentConfig:
    polarity: str = "auto"         # auto / dark(흰 종이에 어두운 글씨) / light(어두운 바탕에 밝은 글씨)
    sensitivity: float = 12.0      # 주변보다 이 값 이상 밝아야 잉크. 작을수록 흐린 글씨도 잡지만 잡음도 늘어남
    min_glyph_px: int = 12         # 글자 높이가 이보다 작으면 무시 (px)
    min_area_ratio: float = 0.0002 # 가이드 박스 면적 대비 이보다 작은 조각은 잡음
    max_glyphs: int = 20


@dataclass
class Glyph:
    box: Box                          # 가이드 박스 안 좌표
    line: int = 0                     # 몇 번째 줄인지 (0부터)
    image: np.ndarray = field(default=None, repr=False)  # 모델 입력 (64x64 uint8)


# ---------------------------------------------------------------- 1. 극성
def to_bright_ink(gray: np.ndarray, polarity: str = "auto") -> np.ndarray:
    """글씨가 밝도록 맞춘 흑백 이미지. auto: 배경(중앙값)에서 더 멀리 떨어진 쪽을 글씨로 봄."""
    if polarity == "dark":
        return 255 - gray
    if polarity == "light":
        return gray
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    lo, mid, hi = np.percentile(g, [0.5, 50, 99.5])
    return 255 - gray if mid - lo > hi - mid else gray


# ---------------------------------------------------------------- 2~3. 잉크 마스크
def ink_mask(bright: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """주변(적응형 평균)보다 sensitivity 이상 밝은 픽셀 = 잉크 (255)."""
    h, w = bright.shape
    block = max(15, (min(h, w) // 6) | 1)          # 글자보다 충분히 큰 창으로 '주변 밝기'를 계산
    g = cv2.GaussianBlur(bright, (3, 3), 0)
    mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -cfg.sensitivity)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def clean_components(mask: np.ndarray, cfg: SegmentConfig):
    """잡음·테두리·긴 선을 지운 연결 성분 목록 [(x, y, w, h, label)] 과 라벨 이미지."""
    H, W = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(6, cfg.min_area_ratio * H * W)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue                                   # 작은 점
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue                                   # 가이드 박스 테두리에 닿음 (종이 끝, 그림자)
        if (w > 0.5 * W and w > 6 * h) or (h > 0.8 * H and h > 6 * w):
            continue                                   # 줄 노트의 긴 선
        comps.append((int(x), int(y), int(w), int(h), i))
    return comps, labels


# ---------------------------------------------------------------- 4~5. 글자·줄
def group_glyphs(comps, cfg: SegmentConfig) -> List[dict]:
    """가로로 많이 겹치고 세로로 가까운 조각을 한 글자로 묶습니다."""
    groups = []
    for x, y, w, h, lab in sorted(comps, key=lambda c: c[0]):
        for g in groups:
            gx, gy, gw, gh = g["box"]
            overlap = min(gx + gw, x + w) - max(gx, x)
            vgap = max(gy, y) - min(gy + gh, y + h)           # 음수면 세로로도 겹침
            if overlap >= 0.5 * min(gw, w) and vgap < 0.5 * max(gh, h):
                nx, ny = min(gx, x), min(gy, y)
                g["box"] = (nx, ny, max(gx + gw, x + w) - nx, max(gy + gh, y + h) - ny)
                g["labels"].append(lab)
                break
        else:
            groups.append({"box": (x, y, w, h), "labels": [lab]})
    if not groups:
        return []
    med_h = float(np.median([g["box"][3] for g in groups]))
    groups = [g for g in groups if g["box"][3] >= max(cfg.min_glyph_px, 0.35 * med_h)]
    groups.sort(key=lambda g: -g["box"][2] * g["box"][3])
    return groups[:cfg.max_glyphs]


def assign_lines(groups: List[dict]) -> List[dict]:
    """세로 중심이 가까운 글자끼리 같은 줄. 줄은 위에서 아래, 줄 안은 왼쪽에서 오른쪽."""
    if not groups:
        return []
    med_h = float(np.median([g["box"][3] for g in groups]))
    groups = sorted(groups, key=lambda g: g["box"][1] + g["box"][3] / 2)
    line, last_cy = 0, None
    for g in groups:
        cy = g["box"][1] + g["box"][3] / 2
        if last_cy is not None and cy - last_cy > 0.6 * med_h:
            line += 1
        g["line"], last_cy = line, cy
    return sorted(groups, key=lambda g: (g["line"], g["box"][0]))


# ---------------------------------------------------------------- 6. 정규화
def normalize(g: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """밝은 글씨 이미지 g 와 잉크 마스크 ink(bool) → 128px 학습 형식 (손글씨 CNN 학습 때와 같은 계산)."""
    ys, xs = np.nonzero(ink)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        p = max(2, int(round(max(x1 - x0, y1 - y0) * 0.08)))
        H, W = g.shape
        x0, y0, x1, y1 = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
        g, ink = g[y0:y1, x0:x1], ink[y0:y1, x0:x1]
    g = g.astype(np.float32)
    bg = float(np.median(g[~ink])) if (~ink).sum() >= 16 else float(np.percentile(g, 10))
    hi = float(np.percentile(g[ink], 90)) if ink.sum() >= 16 else float(np.percentile(g, 99.5))
    g = BG + (g - bg) * (INK - BG) / max(hi - bg, 8.0)
    h, w = g.shape
    s = FIT / max(h, w)
    g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.full((SIZE, SIZE), BG, np.float32)
    h, w = g.shape
    y, x = (SIZE - h) // 2, (SIZE - w) // 2
    out[y:y + h, x:x + w] = g
    return np.clip(out, BG - 70, 255).astype(np.uint8)


def glyph_image(bright: np.ndarray, labels: np.ndarray, group: dict) -> np.ndarray:
    """그 글자의 잉크만 남긴 모델 입력 (64x64).
    옆 글자 조각이 함께 잘려 들어가면 다른 글자로 읽히므로, 다른 잉크는 배경 밝기로 덮습니다."""
    x, y, w, h = group["box"]
    p = max(4, int(0.25 * max(w, h)))
    H, W = bright.shape
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    crop = bright[y0:y1, x0:x1].copy()
    own = np.isin(labels[y0:y1, x0:x1], group["labels"])
    keep = cv2.dilate(own.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # 획 가장자리의 번짐까지 포함
    bg = float(np.median(crop[~keep])) if (~keep).any() else float(np.median(crop))
    crop[~keep] = bg
    img128 = normalize(crop, own)
    return cv2.resize(img128, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 전체
def find_glyphs(gray_roi: np.ndarray, cfg: SegmentConfig):
    """가이드 박스 안 흑백 이미지 → (글자 목록, 디버그용 잉크 마스크)."""
    bright = to_bright_ink(gray_roi, cfg.polarity)
    mask = ink_mask(bright, cfg)
    comps, labels = clean_components(mask, cfg)
    groups = assign_lines(group_glyphs(comps, cfg))
    glyphs = [Glyph(g["box"], g["line"], glyph_image(bright, labels, g)) for g in groups]
    return glyphs, mask
