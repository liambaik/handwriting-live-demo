"""카메라 열기 (OpenCV). 내장 카메라, USB 웹캠, iPhone 연속성 카메라 모두 같은 방식으로 열립니다."""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


class Camera:
    def __init__(self, device: int = 0, width: int = 1280, height: int = 720, rotate: int = 0):
        """rotate: 영상을 시계 방향으로 돌릴 각도 (도). 0 이면 회전 없음. r 키로 30° 씩 돌림."""
        self.device, self.width, self.height = device, width, height
        self.rotate = rotate % 360
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"카메라 #{self.device} 를 열 수 없습니다. macOS 라면 [시스템 설정 > 개인정보 보호 및 보안 > 카메라] 에서 "
                "VS Code 또는 터미널을 허용하고, 앱을 완전히 종료(⌘Q)했다가 다시 여세요. "
                "카메라 번호는 --list-cameras 로 확인할 수 있습니다.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 켠 직후에는 빈 프레임이 몇 장 올 수 있어 최대 3초 기다림
        end = time.time() + 3
        while time.time() < end:
            if self.cap.read()[0]:
                return
            time.sleep(0.05)
        raise RuntimeError(f"카메라 #{self.device} 는 열렸지만 영상이 들어오지 않습니다. 다른 앱이 쓰고 있는지 확인하세요.")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read() if self.cap else (False, None)
        if not ok:
            return None
        return rotate_frame(frame, self.rotate)

    def turn(self, step: int = 30) -> int:
        """시계 방향으로 step° 더 돌림 (기본 30°). 새 각도를 돌려줌."""
        self.rotate = (self.rotate + step) % 360
        return self.rotate

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
    """시계 방향으로 angle° 회전. 90° 단위는 그대로 돌리고, 그 밖의 각도(30°, 60° …)는
    잘리는 곳이 없도록 캔버스를 넓혀서 돌림 (빈 모서리는 검정)."""
    angle %= 360
    if angle == 0:
        return frame
    if angle in _ROT:
        return cv2.rotate(frame, _ROT[angle])
    h, w = frame.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)        # OpenCV 는 반시계가 + 라서 부호 반대
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(frame, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def list_cameras(max_index: int = 4) -> list:
    """0번부터 차례로 열어 보고 (번호, 해상도, 평균 밝기) 를 돌려줍니다.
    평균 밝기가 0 에 가까우면 렌즈가 가려졌거나 화면이 꺼진 iPhone 일 수 있습니다."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            frame = None
            for _ in range(15):          # 첫 몇 장은 비어 있거나 어두울 수 있음
                ok, f = cap.read()
                if ok:
                    frame = f
            if frame is not None:
                found.append((i, f"{frame.shape[1]}x{frame.shape[0]}", float(frame.mean())))
        cap.release()
    return found


# ---------------------------------------------------------------- 회전 설정 저장 (수집·판별 화면이 같이 씀)
import json as _json
from pathlib import Path as _Path

SETTINGS = _Path(__file__).resolve().parent.parent / "settings.json"


def load_rotation() -> int:
    try:
        return int(_json.loads(SETTINGS.read_text(encoding="utf-8")).get("rotate", 0)) % 360
    except (OSError, ValueError):
        return 0


def save_rotation(angle: int) -> None:
    try:
        data = _json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else {}
    except (OSError, ValueError):
        data = {}
    data["rotate"] = angle % 360
    SETTINGS.write_text(_json.dumps(data, indent=2), encoding="utf-8")
