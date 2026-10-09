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


# --- 수기만 남기기 (인쇄 글자 제외) -------------------------------------------------
class _FakeClf:
    """묶음을 위에서 아래 순서로 받아 verdicts 순서대로 손글씨 확률을 돌려줌 (판별 규칙이 아니라 배선 검증용)."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def decide(self, gray):
        from types import SimpleNamespace
        p = 0.9 if self.verdicts.pop(0) else 0.1
        return SimpleNamespace(prob=p, is_handwritten=p >= 0.5)


def _two_blocks():
    img = np.full((260, 700), 225, np.uint8)
    cv2.putText(img, "HK", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 40, 7, cv2.LINE_AA)       # 위쪽: 가짜 '수기'
    cv2.putText(img, "357", (330, 230), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 40, 7, cv2.LINE_AA)     # 아래 오른쪽: 가짜 '인쇄'
    return img


def test_group_strokes_splits_distant_text():
    from hw_demo.cutout import group_strokes
    m = stroke_mask(_two_blocks())["mask"]
    groups = group_strokes(m)
    assert len(groups) == 2
    assert groups[0]["box"][1] < groups[1]["box"][1]


def test_handwritten_only_keeps_only_groups_judged_handwritten():
    from hw_demo.cutout import handwritten_only
    img = _two_blocks()
    r = handwritten_only(img, _FakeClf([True, False]))            # 위쪽 묶음만 '수기'
    assert r["mask"][:130].any() and not r["mask"][130:].any()
    assert r["all_mask"][130:].any()                              # 전체 획에는 아래 글자도 있음
    assert [g["keep"] for g in r["groups"]] == [True, False]


def test_all_groups_rejected_gives_empty_cutout():
    from hw_demo.cutout import handwritten_only, to_rgba
    img = _two_blocks()
    r = handwritten_only(img, _FakeClf([False, False]))
    assert not r["mask"].any()
    assert (to_rgba(img, r["mask"])[..., 3] == 0).all()
