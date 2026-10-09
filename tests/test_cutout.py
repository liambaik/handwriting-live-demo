"""손글씨 누끼(hw_demo/cutout.py) 테스트. torch 없이 실행됨."""
import cv2
import numpy as np

from hw_demo.cutout import CutoutConfig, checkerboard_preview, cutout, stroke_mask


def _truth(text="HK357", ink=40, paper=225, size=(220, 560), color=None, thick=7):
    """종이 + 글씨. (이미지, 정답 마스크)"""
    truth = np.zeros(size, np.uint8)
    cv2.putText(truth, text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, thick, cv2.LINE_AA)
    truth = truth > 127
    if color is None:
        img = np.full(size, paper, np.uint8)
        img[truth] = ink
    else:
        img = np.full(size + (3,), paper, np.uint8)
        img[truth] = color
    return img, truth


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_dark_ink_on_paper():
    img, truth = _truth()
    rgba, mask, _ = cutout(img)
    assert rgba.shape == img.shape + (4,) or rgba.shape == img.shape[:2] + (4,)
    assert _iou(mask, truth) > 0.7
    assert (rgba[..., 3][~cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)] == 0).all()


def test_bright_ink_on_dark_background():
    img, truth = _truth(ink=230, paper=40)
    assert stroke_mask(img)["polarity"] == "bright"
    assert _iou(stroke_mask(img)["mask"], truth) > 0.7


def test_yellow_pen_is_caught_in_color():
    img, truth = _truth(color=(40, 200, 235), paper=235)          # 노란 펜 (BGR): 흑백에서는 종이와 거의 같은 밝기
    assert _iou(stroke_mask(img)["mask"], truth) > 0.6


def test_large_shadow_and_border_objects_are_not_strokes():
    img, truth = _truth()
    cv2.circle(img, (470, 60), 45, 90, -1)                         # 큰 어두운 얼룩 (덩어리)
    img[:, :6] = 40                                                # 테두리에 닿은 세로 선
    cv2.line(img, (0, 200), (559, 200), 60, 3)                     # 화면을 가로지르는 긴 선
    m = stroke_mask(img)["mask"]
    assert not m[:, :8].any()
    assert not m[15:105, 425:515].any()
    assert not m[198:203, :].any()
    assert (m & truth).sum() / truth.sum() > 0.7                   # 글씨는 그대로 남음


def test_long_line_kept_when_option_off():
    img, _ = _truth()
    cv2.line(img, (20, 200), (539, 200), 60, 3)
    assert stroke_mask(img, CutoutConfig(drop_long=False))["mask"][199:202, 100:400].any()


def test_blank_paper_gives_empty_cutout():
    rng = np.random.default_rng(0)
    img = np.full((200, 300), 220, np.uint8) + rng.integers(0, 4, (200, 300)).astype(np.uint8)
    rgba, mask, _ = cutout(img)
    assert not mask.any() and (rgba[..., 3] == 0).all()


def test_preview_and_feather():
    img, _ = _truth()
    rgba, _, _ = cutout(img, CutoutConfig(feather=5))
    assert ((rgba[..., 3] > 0) & (rgba[..., 3] < 255)).any()       # 부드러운 가장자리
    prev = checkerboard_preview(rgba)
    assert prev.shape == rgba.shape[:2] + (3,) and prev.dtype == np.uint8


def test_large_photo_is_downscaled_but_mask_matches_input_size():
    img, _ = _truth(size=(220, 560))
    big = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    assert stroke_mask(big)["mask"].shape == big.shape[:2]
