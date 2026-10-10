"""화면 표시. OpenCV 는 한글을 그리지 못해 화면 글자는 영어로 씁니다."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .recognizer import Reading
from .segment import Box, Glyph

GREEN, YELLOW, RED, WHITE, GRAY, CYAN = (0, 200, 0), (0, 220, 255), (0, 0, 255), (255, 255, 255), (170, 170, 170), (255, 220, 0)
KEYS = "[q]quit [space]freeze [s]save [d]debug [p]polarity [ / ]sensitivity -/+"


def conf_color(c: float, min_conf: float) -> tuple:
    """확신도 색: 0.8 이상 초록, min_conf 이상 노랑, 그 아래 빨강."""
    return GREEN if c >= 0.8 else YELLOW if c >= min_conf else RED


def draw(frame: np.ndarray, roi: Box, glyphs: List[Glyph], reads: List[Reading], text: str, stable: str,
         info: str, min_conf: float, frozen: bool, debug_mask: Optional[np.ndarray] = None) -> np.ndarray:
    out = frame.copy()
    rx, ry, rw, rh = roi
    # 가이드 박스 바깥은 어둡게 → 어디에 글씨를 비춰야 하는지 보이게
    dim = (out * 0.45).astype(np.uint8)
    dim[ry:ry + rh, rx:rx + rw] = out[ry:ry + rh, rx:rx + rw]
    out = dim
    cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), CYAN, 2)
    cv2.putText(out, "Put handwriting here", (rx + 6, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)

    # 글자마다 박스 + 읽은 글자 + 확신도
    for g, r in zip(glyphs, reads):
        x, y, w, h = g.box
        x, y = x + rx, y + ry
        col = conf_color(r.conf, min_conf)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        label = r.char if r.conf >= min_conf else "?"
        cv2.putText(out, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
        cv2.putText(out, f"{r.conf:.2f}", (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # 위쪽 결과 패널
    cv2.rectangle(out, (0, 0), (out.shape[1], 74), (0, 0, 0), -1)
    cv2.putText(out, f"Read  : {text or '-'}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(out, f"Stable: {stable or '-'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)
    cv2.putText(out, info + ("  [FROZEN]" if frozen else ""), (out.shape[1] - 520, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW if frozen else GRAY, 1, cv2.LINE_AA)

    # 디버그: 잉크 마스크와 모델 입력 이미지
    if debug_mask is not None:
        m = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
        s = 220 / max(1, m.shape[0])
        m = cv2.resize(m, (int(m.shape[1] * s), 220))[:, : out.shape[1]]
        out[80:80 + m.shape[0], 0:m.shape[1]] = m
        cv2.putText(out, "ink mask", (6, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
        thumbs = [cv2.cvtColor(g.image, cv2.COLOR_GRAY2BGR) for g in glyphs][: out.shape[1] // 66]
        if thumbs:
            strip = np.hstack([cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=CYAN) for t in thumbs])
            y0 = out.shape[0] - 30 - strip.shape[0]
            out[y0:y0 + strip.shape[0], 0:strip.shape[1]] = strip

    H = out.shape[0]
    cv2.rectangle(out, (0, H - 24), (out.shape[1], H), (0, 0, 0), -1)
    cv2.putText(out, KEYS, (8, H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return out
