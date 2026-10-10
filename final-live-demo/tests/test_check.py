"""손글씨 판별(로지스틱 회귀) 테스트."""
import math

import cv2
import numpy as np

from check import has_ink
from hw_demo.hw_features import FEATURE_KEYS, HandwritingClassifier, compute_features


def _text_image(dark=True):
    img = np.full((200, 400), 220 if dark else 40, np.uint8)
    cv2.putText(img, "F12", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 3.5, 40 if dark else 220, 9, cv2.LINE_AA)
    return img


def test_features_have_8_values():
    f = compute_features(_text_image())
    assert list(f) == FEATURE_KEYS


def test_polarity_does_not_change_features():
    a, b = compute_features(_text_image(True)), compute_features(_text_image(False))
    for k in FEATURE_KEYS:
        if math.isfinite(a[k]):
            assert abs(a[k] - b[k]) < 0.15 * max(1.0, abs(a[k])), k


def test_logistic_regression_matches_json_formula():
    clf = HandwritingClassifier()
    d = clf.decide(_text_image())
    x = np.array([d.features[k] for k in FEATURE_KEYS], float)
    x[~np.isfinite(x)] = clf.median[~np.isfinite(x)]
    expect = 1 / (1 + math.exp(-(clf.bias + clf.weight @ ((x - clf.mean) / clf.std))))
    assert abs(d.prob - expect) < 1e-9 and d.is_handwritten == (expect >= 0.5)


def test_blank_page_shows_nothing():
    blank = np.full((200, 300), 200, np.uint8) + np.random.default_rng(0).integers(0, 6, (200, 300)).astype(np.uint8)
    assert not has_ink(blank)
    assert has_ink(_text_image())


def test_writing_gate_rejects_textured_scene_and_accepts_paper():
    from check import find_writing
    rng = np.random.default_rng(0)
    # 얼굴·머리카락처럼 얼룩덜룩한 배경 위의 가는 어두운 선 → 글씨 아님
    scene = cv2.GaussianBlur(rng.integers(30, 230, (360, 640)).astype(np.uint8), (0, 0), 1.2)  # 주변 표준편차 ≈ 30 (실제 얼굴 장면 측정값)
    for i in range(12):
        cv2.line(scene, (40 + 45 * i, 100), (60 + 45 * i, 260), 20, 2)
    assert find_writing(scene)[0] is None
    # 흰 종이 위 펜 글씨 → 글씨 있음
    paper = np.full((360, 640), 215, np.uint8)
    cv2.putText(paper, "V8 150", (120, 220), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, 50, 5, cv2.LINE_AA)
    assert find_writing(paper)[0] is not None


def test_yellow_pen_and_light_pencil_are_found():
    """흑백으로 바꾸면 사라지는 노란 펜, 연한 연필도 글씨로 찾아야 함."""
    from check import find_writing
    for color in [(40, 220, 245), (150, 150, 150)]:          # BGR: 노랑, 연한 연필 회색
        paper = np.full((360, 640, 3), 225, np.uint8)
        cv2.putText(paper, "V8 150", (120, 220), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, color, 5, cv2.LINE_AA)
        crop, reason = find_writing(paper)
        assert crop is not None, (color, reason)
        assert crop.min() < 120                               # 판정용 이미지는 흰 바탕 + 진한 글씨로 맞춰짐
