"""손글씨 획 누끼(cutout): 종이·바탕은 투명하게, 글씨 획만 남깁니다.

hoonibbong 브랜치 PAC2/src/pac2/extraction.py 의 획 단위 후보 마스크(stroke_candidates)와 선형 구조 거르기
(group_components 의 long_structure)를 이 데모에 맞게 옮긴 것입니다. 이 데모의 기조에 맞춰 바꾼 점:
  - scikit-image·scipy·sklearn 없이 OpenCV·numpy 만 사용 (획 두께 측정용 세선화는 hw_features.skeleton 재사용)
  - 컬러 영상은 B·G·R 채널마다 응답을 재서 최댓값 사용 (check.py ink_strength 와 같은 방식: 노란 펜도 잡힘)
  - PAC2 의 P(handwritten) 로 덩어리를 거르는 단계는 가져오지 않음 (체크포인트·sklearn 필요).
    "이게 손글씨인가" 는 이 데모의 check.py 판별이 맡고, 여기서는 "획이 어디인가" 만 찾는다.

알고리즘
  1. 극성: 중앙값에서 가장 많이 벗어난 10% 픽셀이 어두우면 어두운 잉크(black-hat), 아니면 밝은 잉크(top-hat)
  2. 응답: 커널 9·15·25·41px 의 모폴로지 hat 응답의 최댓값. 커널보다 가는 구조만 반응하므로 녹 얼룩·조명 같은
     큰 덩어리는 반응하지 않는다
  3. 임계값: max(10, 중앙값 + 5 × 1.4826 × MAD) - 잉크가 화면의 큰 비율을 차지해도 흔들리지 않는 잡음 수준
  4. 성분 거르기: 면적 < 12px 제외, 면적 > 50% 제외, 두께/긴변 > 0.35 인 덩어리(blob) 제외 (획과 선은 유지),
     가장자리에 닿은 성분(drop_border), 길고 가는 선(변이 화면의 40% 초과, 종횡비 ≥ 4: 판 가장자리·긴 선)은 제외
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from hw_demo.hw_features import skeleton

MAX_LONG_SIDE = 1600       # 분석 해상도 상한 (긴 변). 더 큰 사진은 줄여서 마스크를 구한 뒤 원래 크기로 되돌림


@dataclass
class CutoutConfig:
    hat_kernels: tuple = (9, 15, 25, 41)
    thinness_max: float = 0.35      # 획 두께 / 긴 변 이 값 이하만 획 (덩어리는 제외)
    max_area_frac: float = 0.50
    min_area_px: int = 12
    noise_k: float = 5.0
    min_response: float = 10.0
    long_frac: float = 0.40         # 변이 화면의 이 비율을 넘고 종횡비 ≥ long_aspect 면 긴 선
    long_aspect: float = 4.0
    drop_border: bool = True        # 가이드 박스 테두리에 닿은 성분 제외 (박스에서 잘린 글씨·물체 모서리)
    drop_long: bool = True
    feather: int = 0                # >0 이면 알파 가장자리를 이 크기(홀수 px)로 부드럽게


def ink_polarity(gray: np.ndarray) -> str:
    """'dark' 면 밝은 바탕에 어두운 잉크."""
    med = float(np.median(gray))
    dev = gray.astype(np.float32) - med
    k = max(1, int(0.10 * dev.size))
    idx = np.argpartition(np.abs(dev).ravel(), -k)[-k:]
    return "dark" if dev.ravel()[idx].mean() < 0 else "bright"


def _hat_response(img: np.ndarray, op: int, kernels) -> np.ndarray:
    chans = cv2.split(img) if img.ndim == 3 else [img]
    resp = np.zeros(chans[0].shape, np.float32)
    for k in kernels:
        if k >= min(chans[0].shape):
            continue
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        for c in chans:
            resp = np.maximum(resp, cv2.morphologyEx(c, op, se).astype(np.float32))
    return resp


def stroke_mask(img: np.ndarray, cfg: CutoutConfig = CutoutConfig()) -> Dict[str, object]:
    """획 마스크. 반환: mask(bool, 입력과 같은 크기), polarity, threshold, n_rejected_blob."""
    H0, W0 = img.shape[:2]
    s = min(1.0, MAX_LONG_SIDE / max(H0, W0))
    work = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    pol = ink_polarity(gray)
    resp = _hat_response(work, cv2.MORPH_BLACKHAT if pol == "dark" else cv2.MORPH_TOPHAT, cfg.hat_kernels)
    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med)))
    thr = max(cfg.min_response, med + cfg.noise_k * 1.4826 * mad)
    raw = (resp > thr).astype(np.uint8)

    H, W = raw.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros(n, bool)
    rejected = 0
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < cfg.min_area_px:
            continue
        if cfg.drop_border and (x <= 0 or y <= 0 or x + w >= W or y + h >= H):
            continue
        if a > cfg.max_area_frac * raw.size:
            rejected += 1
            continue
        if cfg.drop_long and (w > cfg.long_frac * W or h > cfg.long_frac * H) and max(w, h) / max(1, min(w, h)) >= cfg.long_aspect:
            continue
        crop = np.pad(lab[y:y + h, x:x + w] == i, 1)       # 바깥에 배경 1px: 없으면 거리 변환이 무한대가 됨
        sk = skeleton(crop)
        dt = cv2.distanceTransform(crop.astype(np.uint8), cv2.DIST_L2, 3)
        width = 2.0 * float(np.median(dt[sk])) if sk.any() else 2.0 * float(dt.max())
        if width / max(w, h) <= cfg.thinness_max:
            keep[i] = True
        else:
            rejected += 1
    mask = keep[lab]
    if s < 1:
        mask = cv2.resize(mask.astype(np.uint8) * 255, (W0, H0), interpolation=cv2.INTER_LINEAR) > 127
    return {"mask": mask, "polarity": pol, "threshold": thr, "n_rejected_blob": rejected}


def to_rgba(img: np.ndarray, mask: np.ndarray, cfg: CutoutConfig = CutoutConfig()) -> np.ndarray:
    """원본 색을 유지한 채 마스크 밖을 투명하게 만든 BGRA."""
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    alpha = mask.astype(np.uint8) * 255
    if cfg.feather > 0:
        k = cfg.feather | 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return np.dstack([bgr, alpha])


def cutout(img: np.ndarray, cfg: CutoutConfig = CutoutConfig()):
    """(BGRA 누끼, 마스크, 상세). 글씨가 없으면 마스크는 전부 False."""
    r = stroke_mask(img, cfg)
    return to_rgba(img, r["mask"], cfg), r["mask"], r


def checkerboard_preview(rgba: np.ndarray, cell: int = 12) -> np.ndarray:
    """투명 영역을 체크무늬로 보여 주는 미리보기 (BGR)."""
    H, W = rgba.shape[:2]
    yy, xx = np.indices((H, W))
    bg = np.where(((yy // cell + xx // cell) % 2 == 0)[..., None], 235, 200).astype(np.float32).repeat(3, axis=2)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3].astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)


def save_png(path, rgba: np.ndarray) -> None:
    """한글 경로에서도 저장되도록 imencode 사용."""
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise ValueError("PNG 인코딩 실패")
    buf.tofile(str(path))
