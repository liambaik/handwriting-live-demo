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
from typing import Dict, List

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
    strong_pct: float = 98.0        # '확실한 잉크' 기준: 잡음 위 응답의 이 백분위수
    rel_thr: float = 0.4            # 픽셀은 확실한 잉크 응답의 이 비율 이상이어야 획 (표면 요철·흐린 얼룩은 응답이 약함)
    core_frac: float = 0.6          # 성분은 응답 상위 10% 가 확실한 잉크의 이 비율 이상이어야 유지 (약한 덩어리 통째로 제거)
    long_frac: float = 0.40         # 변이 화면의 이 비율을 넘고 종횡비 ≥ long_aspect 면 긴 선
    long_aspect: float = 4.0
    drop_border: bool = True        # 가이드 박스 테두리에 닿은 성분 제외 (박스에서 잘린 글씨·물체 모서리)
    drop_long: bool = True
    drop_skin: bool = True          # 손가락·손 등 피부색 영역 안의 성분 제외 (컬러 영상에서만)
    skin_min_frac: float = 0.02     # 피부색 덩어리가 화면의 이 비율 이상이어야 손으로 봄 (붉은 펜 점·벽 얼룩 같은 작은 조각 무시)
    skin_overlap: float = 0.5       # 성분 픽셀의 이 비율 이상이 피부색 영역 안이면 제외
    link_ratio: float = 0.8         # 수기 판별용 묶기: 상자 사이 틈이 (중앙 글자 높이 × 이 값) 이하면 한 묶음
    hw_threshold: float = 0.4       # 묶음의 손글씨 확률이 이 이상이면 유지 (판별기 기본 0.5 보다 낮춰 수기를 덜 놓치게)
    min_group_px: int = 40          # 이보다 잉크가 적은 묶음은 판별하지 않고 제외 (점·잔얼룩)
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


def skin_region(bgr: np.ndarray, min_frac: float = 0.02, open_k: int = 15, grow: int = 9) -> np.ndarray:
    """손가락·손 영역(bool). YCrCb 피부색 범위 → 큰 열기(획 같은 가는 것 제거) → 면적 작은 조각 제거 → 가장자리까지 넓힘.
    손가락 가장자리의 그림자 선이 획처럼 잡히므로 영역을 grow px 키워 가장자리도 포함한다."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y > 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_frac * m.size
    m = keep[lab].astype(np.uint8)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))) > 0


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
    cand = resp > thr
    strong = float(np.percentile(resp[cand], cfg.strong_pct)) if cand.any() else 0.0
    thr = max(thr, cfg.rel_thr * strong)                  # 상대 기준: 확실한 잉크보다 한참 약한 것(표면 요철)은 버림
    raw = (resp > thr).astype(np.uint8)

    H, W = raw.shape
    skin = skin_region(work, cfg.skin_min_frac) if cfg.drop_skin and work.ndim == 3 else None
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros(n, bool)
    rejected = 0
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < cfg.min_area_px:
            continue
        if cfg.drop_border and (x <= 0 or y <= 0 or x + w >= W or y + h >= H):
            continue
        if skin is not None and float(skin[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].mean()) >= cfg.skin_overlap:
            continue                                      # 손가락 위·가장자리 성분
        if a > cfg.max_area_frac * raw.size:
            rejected += 1
            continue
        if cfg.drop_long and (w > cfg.long_frac * W or h > cfg.long_frac * H) and max(w, h) / max(1, min(w, h)) >= cfg.long_aspect:
            continue
        sel = lab[y:y + h, x:x + w] == i
        if strong > 0 and float(np.percentile(resp[y:y + h, x:x + w][sel], 90)) < cfg.core_frac * strong:
            continue                                      # 응답이 전체적으로 약한 성분 = 요철·흐린 얼룩
        crop = np.pad(sel, 1)       # 바깥에 배경 1px: 없으면 거리 변환이 무한대가 됨
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


# ---------------------------------------------------------------- 수기만 남기기 (인쇄 글자 제외)
def group_strokes(mask: np.ndarray, link_ratio: float = 0.8, min_px: int = 40) -> List[dict]:
    """획 성분을 근접 묶음(글자·단어 단위)으로 묶는다. 상자 사이 틈이 중앙 높이 × link_ratio 이하면 같은 묶음.
    (PAC2 group_components 와 같은 방식.) 반환: [{"box": (x, y, w, h), "mask": 원 크기 bool, "pixels": n}]"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    idx = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 8]
    if not idx:
        return []
    gap = max(2.0, link_ratio * float(np.median([st[i, cv2.CC_STAT_HEIGHT] for i in idx])))
    parent = {i: i for i in idx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    box = {i: (st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3]) for i in idx}
    for ai, a in enumerate(idx):
        for b in idx[ai + 1:]:
            dx = max(0, box[b][0] - box[a][2], box[a][0] - box[b][2])
            dy = max(0, box[b][1] - box[a][3], box[a][1] - box[b][3])
            if dx <= gap and dy <= gap:
                parent[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        gm = np.isin(lab, members)
        if gm.sum() < min_px:
            continue
        ys, xs = np.nonzero(gm)
        out.append({"box": (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                    "mask": gm, "pixels": int(gm.sum())})
    return sorted(out, key=lambda g: (g["box"][1], g["box"][0]))


def handwritten_only(img: np.ndarray, clf, cfg: CutoutConfig = CutoutConfig()) -> Dict[str, object]:
    """획 마스크를 묶음별로 나눠 손글씨 판별기(clf: HandwritingClassifier)로 판정하고, 손글씨 묶음의 획만 남긴다.

    묶음 하나하나를 '그 묶음만 보이게'(나머지는 배경색으로 칠함) 잘라 clf.decide 에 넣는다.
    반환: mask(손글씨만), groups([{box, prob, keep, pixels}]), all_mask(인쇄 포함 전체 획), polarity, threshold
    """
    base = stroke_mask(img, cfg)
    all_mask = base["mask"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bg = int(np.median(gray))
    pad = 6
    H, W = gray.shape
    kept = np.zeros_like(all_mask)
    groups = []
    for g in group_strokes(all_mask, cfg.link_ratio, cfg.min_group_px):
        x, y, w, h = g["box"]
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        crop = gray[y0:y1, x0:x1].copy()
        near = cv2.dilate(g["mask"][y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        crop[~near] = bg                                  # 다른 묶음·종이 질감 지움: 이 묶음의 획만 남김
        d = clf.decide(crop)
        keep = d is not None and d.prob >= cfg.hw_threshold
        groups.append({"box": g["box"], "pixels": g["pixels"], "prob": None if d is None else d.prob, "keep": bool(keep)})
        if keep:
            kept |= g["mask"]
    return {"mask": kept, "all_mask": all_mask, "groups": groups, "polarity": base["polarity"],
            "threshold": base["threshold"]}


# ---------------------------------------------------------------- 백그라운드 계산 (카메라 화면이 끊기지 않게)
class LatestWorker:
    """무거운 계산(fn)을 별도 스레드에서 돌리는 '최신 입력만' 작업자.

    submit() 은 기다리지 않고 바로 돌아온다. 계산 중에 submit 이 여러 번 와도 가장 최근 입력 하나만 남기고
    나머지는 버린다 (밀린 프레임을 차례로 처리하면 화면이 점점 뒤처지므로). 결과는 latest() 로 가져간다.
    OpenCV 연산은 GIL 을 놓으므로 메인 스레드의 카메라·화면 갱신과 겹쳐 돈다.
    """

    def __init__(self, fn):
        import threading
        self._fn = fn
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job = None
        self._result = None
        self._error = None
        self._gen = 0                 # reset 할 때마다 올림: 계산 도중 reset 되면 그 결과는 버림
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="cutout-worker", daemon=True)
        self._thread.start()

    def submit(self, *args) -> None:
        with self._lock:
            self._job = args
        self._wake.set()

    def latest(self):
        """가장 최근에 끝난 결과 (아직 없으면 None)."""
        with self._lock:
            return self._result

    def error(self):
        with self._lock:
            return self._error

    def reset(self) -> None:
        """대기 중인 입력과 이전 결과를 버린다 (모드를 바꿨을 때 옛 결과가 보이지 않게)."""
        with self._lock:
            self._job, self._result = None, None
            self._gen += 1

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            if self._stop:
                return
            with self._lock:
                job, self._job = self._job, None
                gen = self._gen
                self._wake.clear()
            if job is None:
                continue
            try:
                out = self._fn(*job)
            except Exception as e:                    # 계산 오류가 카메라 화면을 죽이지 않게 기록만
                with self._lock:
                    self._error = e
                continue
            with self._lock:
                if gen == self._gen:
                    self._result = out
