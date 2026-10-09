"""손글씨 글자 인식 모델 (ResNet18, EMNIST 숫자 0~9 + 대문자 A~Z = 36글자).

models/cnn_ocr36.pt 는 marking-recovery 프로젝트에서 학습한 모델을 '복사'해 온 파일입니다.
(이 데모는 그 프로젝트 코드를 import 하지 않고, 같은 구조의 모델을 여기서 다시 정의해 불러옵니다)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "cnn_ocr36.pt"
MEAN, STD = 0.449, 0.226   # 학습 때 쓴 입력 정규화 값 (바꾸면 안 됨)


@dataclass
class Reading:
    char: str          # 가장 가능성 높은 글자
    conf: float        # 확신도 (0~1)
    alt: str           # 두 번째 후보 글자
    alt_conf: float


def build_model(n_classes: int) -> nn.Module:
    """학습 때와 같은 구조: ResNet18 + 흑백 1채널 입력 + 출력 n_classes."""
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Recognizer:
    def __init__(self, model_path: Path = DEFAULT_MODEL, device: str = "auto"):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        self.device = pick_device() if device == "auto" else torch.device(device)
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        self.chars: str = ck["chars"]
        self.size: int = ck["size"]
        self.model = build_model(len(self.chars))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def read(self, images: List[np.ndarray]) -> List[Reading]:
        """64x64 글자 이미지 여러 장 → 글자별 결과."""
        if not images:
            return []
        x = torch.from_numpy(np.stack(images)).float().unsqueeze(1) / 255
        x = ((x - MEAN) / STD).to(self.device)
        p = F.softmax(self.model(x), 1).cpu()
        top = p.topk(2, dim=1)
        return [Reading(self.chars[i[0]], float(v[0]), self.chars[i[1]], float(v[1]))
                for v, i in zip(top.values.tolist(), top.indices.tolist())]
