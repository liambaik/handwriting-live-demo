"""카메라 없이 실행되는 테스트. 실행: .venv/bin/python -m pytest -q"""
import cv2
import numpy as np

from hw_demo.recognizer import Recognizer
from hw_demo.segment import MODEL_SIZE, SegmentConfig, find_glyphs, to_bright_ink


def _paper(lines, gap=40):
    """흰 종이에 어두운 글씨 (cv2 필기체 글꼴로 흉내)."""
    img = np.full((300, 700), 230, np.uint8)
    for i, text in enumerate(lines):
        x = 60
        for ch in text:
            cv2.putText(img, ch, (x, 110 + i * 130), cv2.FONT_HERSHEY_SIMPLEX, 2.6, 40, 7, cv2.LINE_AA)
            x += 70 + gap
    return img


def test_dark_ink_is_inverted():
    img = _paper(["A"])
    assert to_bright_ink(img).mean() < 128          # 흰 종이 → 반전 후 어두운 배경
    assert to_bright_ink(255 - img).mean() < 128    # 이미 밝은 글씨면 그대로


def test_finds_glyphs_and_lines_in_order():
    glyphs, _ = find_glyphs(_paper(["AB3", "K7"]), SegmentConfig())
    assert len(glyphs) == 5
    assert [g.line for g in glyphs] == [0, 0, 0, 1, 1]
    xs = [g.box[0] for g in glyphs[:3]]
    assert xs == sorted(xs)                         # 줄 안에서 왼쪽부터
    assert all(g.image.shape == (MODEL_SIZE, MODEL_SIZE) for g in glyphs)


def test_neighbor_ink_is_removed():
    """글자 사이가 가까워도 각 글자 이미지에는 그 글자 잉크만 남아야 함 (옆 글자 조각 제거)."""
    glyphs, _ = find_glyphs(_paper(["HH"], gap=-15), SegmentConfig())
    assert len(glyphs) == 2
    for g in glyphs:
        img = g.image
        # 배경(110 근처)이 좌우 가장자리에 남아 있어야 함 = 옆 글자 획이 들어오지 않음
        assert img[:, :4].mean() < 130 and img[:, -4:].mean() < 130


def test_ignores_noise_and_border():
    img = np.full((300, 700), 230, np.uint8)
    cv2.circle(img, (300, 150), 2, 40, -1)          # 작은 점
    cv2.rectangle(img, (0, 0), (40, 299), 40, -1)   # 테두리에 닿은 그림자
    glyphs, _ = find_glyphs(img, SegmentConfig())
    assert glyphs == []


def test_recognizer_outputs_36_classes():
    rec = Recognizer(device="cpu")
    assert rec.chars == "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    glyphs, _ = find_glyphs(_paper(["7"]), SegmentConfig())
    r = rec.read([g.image for g in glyphs])
    assert len(r) == 1 and r[0].char in rec.chars and 0 <= r[0].conf <= 1
