"""키 입력 정리: 한글 입력 상태에서도 단축키가 동작하도록.

macOS 에서 한글 자판이 켜져 있으면 h 키가 'ㅗ' 로 들어와 OpenCV 가 다른 키 코드를 받습니다.
그래서 waitKeyEx 로 전체 코드를 받고, 한글 자모를 같은 자리의 영문 키로 바꿔 줍니다.
"""
from __future__ import annotations

import cv2

# 두벌식 자판: 한글 자모 → 같은 자리 영문 키
_HANGUL = dict(zip("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡㅃㅉㄸㄲㅆㅒㅖ",
                   "qwertyuiopasdfghjklzxcvbnmQWERTOP"))


def read_key(delay_ms: int = 1) -> str:
    """눌린 키를 영문 소문자 한 글자로 돌려줍니다 (없으면 ''). ESC 는 'esc', 스페이스는 ' '."""
    code = cv2.waitKeyEx(delay_ms)
    if code < 0:
        return ""
    if code == 27:
        return "esc"
    try:
        ch = chr(code) if code < 0x110000 else ""
    except ValueError:
        return ""
    ch = _HANGUL.get(ch, ch)
    if len(ch) != 1:
        return ""
    if not ch.isascii():                 # 알 수 없는 문자: 하위 바이트로 한 번 더 시도
        ch = chr(code & 0xFF)
    return ch.lower()
