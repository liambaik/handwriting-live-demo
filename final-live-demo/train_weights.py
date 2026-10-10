"""iPhone 사진으로 손글씨 판별 가중치 다시 학습하기.

원래 model_parameters.json 과 같은 방식입니다.
  - 특징 8개 (hw_demo/hw_features.py, 판별 화면과 똑같은 계산)
  - 로지스틱 회귀, 손실 = 클래스 균형 BCE + L2 정규화
  - λ 선택: 10회 반복 5-fold 층화 교차검증에서 평균 균형 log loss 가 가장 작은 값
결과는 models/model_parameters_iphone.json 에 원래 JSON 과 같은 형식으로 저장합니다 (원래 파일은 그대로).

데이터
  - data/iphone/handwritten/*.png, data/iphone/printed/*.png  (collect.py 로 모은 사진, 주 데이터)
  - --with-original 을 주면 Downloads 의 formal_handwritten_120_png 도 함께 사용

  python train_weights.py [--with-original PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from check import find_writing
from hw_demo.hw_features import FEATURE_KEYS, compute_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "iphone"
ORIG_PARAMS = ROOT / "models" / "model_parameters.json"
OUT = ROOT / "models" / "model_parameters_iphone.json"
LAMBDAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def features_of(gray, live=True):
    """판별 화면과 같은 순서: 글씨 확인 → 글씨 부분 잘라 특징 계산. 실패하면 None."""
    crop = find_writing(gray)[0] if live else gray
    if crop is None:
        return None
    f = compute_features(crop)
    return None if f is None else [f[k] for k in FEATURE_KEYS]


def load(with_original):
    X, y, src, skipped = [], [], [], []
    for cls, label in (("handwritten", 1), ("printed", 0)):
        for p in sorted((DATA / cls).glob("*.png")):
            f = features_of(cv2.imread(str(p)))   # 컬러 그대로 (색 펜 대응)
            if f is None:
                skipped.append(p.name)
                continue
            X.append(f), y.append(label), src.append(f"iphone/{cls}")
    if with_original:
        root = Path(with_original)
        for cls, label, q, inv in (("handwritten", 1, 70, False), ("formal", 0, 30, True)):
            for p in sorted((root / cls).glob("*.png")):
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                pad = g.shape[0] // 2   # 종이 위 글씨처럼 여백을 붙임 (원본은 글씨에 딱 맞게 잘린 이미지)
                g = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(np.percentile(g, q)))
                f = features_of(255 - g if inv else g)   # 정형(화면 캡처, 밝은 글씨)은 종이 인쇄처럼 반전
                if f is None:
                    skipped.append(p.name)
                    continue
                X.append(f), y.append(label), src.append(f"original/{cls}")
    return np.array(X, float), np.array(y, float), src, skipped


def fit(Xz, y, lam, iters=600):
    """균형 BCE + λ·||w||² 를 L-BFGS 로 최소화. (w, b) 반환."""
    X = torch.tensor(Xz, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    cw = torch.where(t == 1, 0.5 / t.mean(), 0.5 / (1 - t.mean()))
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (cw * torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, t, reduction="none")).sum() / cw.sum() \
            + lam * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def standardize(X, med=None, mu=None, sd=None):
    if med is None:
        med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, med)
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, med, mu, sd


def metrics(y, p):
    pred = p >= 0.5
    tpr = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
    tnr = (~pred & (y == 0)).sum() / max(1, (y == 0).sum())
    eps = 1e-12
    ll1 = -np.log(np.clip(p[y == 1], eps, 1)).mean() if (y == 1).any() else 0
    ll0 = -np.log(np.clip(1 - p[y == 0], eps, 1)).mean() if (y == 0).any() else 0
    return {"balanced_accuracy": (tpr + tnr) / 2, "recall_handwritten": tpr, "printed_shown_as_1": 1 - tnr,
            "balanced_log_loss": (ll1 + ll0) / 2}


def cross_validate(X, y, lam, repeats=10, k=5, seed=0):
    rng = np.random.default_rng(seed)
    res = []
    for r in range(repeats):
        folds = [[] for _ in range(k)]
        for label in (0, 1):                           # 층화: 클래스별로 고르게 나눔
            idx = rng.permutation(np.nonzero(y == label)[0])
            for i, j in enumerate(idx):
                folds[i % k].append(j)
        for f in folds:
            te = np.array(f)
            tr = np.setdiff1d(np.arange(len(y)), te)
            Ztr, med, mu, sd = standardize(X[tr])
            w, b = fit(Ztr, y[tr], lam)
            Zte = standardize(X[te], med, mu, sd)[0]
            res.append(metrics(y[te], 1 / (1 + np.exp(-(Zte @ w + b)))))
    return {k_: float(np.mean([r[k_] for r in res])) for k_ in res[0]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="iPhone 사진으로 손글씨 판별 가중치 다시 학습")
    ap.add_argument("--with-original", nargs="?", const=str(Path.home() / "Downloads" / "formal_handwritten_120_png"),
                    help="원래 120장 데이터셋도 함께 사용 (경로 생략 시 Downloads/formal_handwritten_120_png)")
    ap.add_argument("--min-per-class", type=int, default=10)
    ap.add_argument("--out", default=str(OUT), help="결과 JSON 경로")
    args = ap.parse_args(argv)
    out = Path(args.out)

    X, y, src, skipped = load(args.with_original)
    n_h, n_p = int((y == 1).sum()), int((y == 0).sum())
    print(f"학습 데이터: 손글씨 {n_h}장, 인쇄/정형 {n_p}장" + (f" (글씨 확인 실패로 제외 {len(skipped)}장)" if skipped else ""))
    n_iphone = {c: sum(s == f"iphone/{c}" for s in src) for c in ("handwritten", "printed")}
    if min(n_iphone.values()) < args.min_per_class:
        print(f"[중단] iPhone 사진이 부족합니다: 손글씨 {n_iphone['handwritten']}, 인쇄 {n_iphone['printed']} "
              f"(각각 {args.min_per_class}장 이상 필요). ./run_collect.sh 로 더 모아 주세요.")
        return 1

    print("λ 선택 (10회 반복 5-fold 교차검증, 평균 균형 log loss 최소):")
    cv = {}
    for lam in LAMBDAS:
        cv[lam] = cross_validate(X, y, lam)
        print(f"  λ={lam:<7g} 균형 log loss {cv[lam]['balanced_log_loss']:.4f}  균형 정확도 {cv[lam]['balanced_accuracy']:.1%}"
              f"  손글씨 재현 {cv[lam]['recall_handwritten']:.1%}  인쇄 오판 {cv[lam]['printed_shown_as_1']:.1%}")
    lam = min(LAMBDAS, key=lambda l: cv[l]["balanced_log_loss"])

    Z, med, mu, sd = standardize(X)
    w, b = fit(Z, y, lam)
    train = metrics(y, 1 / (1 + np.exp(-(Z @ w + b))))
    names = [f["name"] for f in sorted(json.loads(ORIG_PARAMS.read_text(encoding="utf-8"))["features"],
                                       key=lambda f: f["index"])]
    params = {
        "dataset": {"handwritten": n_h, "formal": n_p, "total": len(y),
                    "sources": {s: src.count(s) for s in sorted(set(src))}},
        "model": "PyTorch logistic regression",
        "loss": "balanced Binary Cross Entropy + L2 regularization",
        "selected_lambda": lam,
        "selection_rule": "10x repeated 5-fold stratified cross-validation; minimum mean balanced validation log loss",
        "decision_threshold": 0.5,
        "bias": b,
        "features": [{"index": i + 1, "name": names[i], "key": FEATURE_KEYS[i],
                      "median_for_missing": float(med[i]), "mean": float(mu[i]), "std": float(sd[i]),
                      "weight": float(w[i])} for i in range(len(FEATURE_KEYS))],
        "final_training_metrics": train,
        "cross_validation_at_selected_lambda": cv[lam],
        "created": datetime.now().isoformat(timespec="seconds"),
        "note": "hw_demo/hw_features.py 의 특징 계산으로 iPhone 사진에서 다시 학습한 가중치",
    }
    out.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    c = cv[lam]
    print(f"\n선택 λ={lam}: 교차검증 균형 정확도 {c['balanced_accuracy']:.1%}, 손글씨에 1 표시 {c['recall_handwritten']:.1%}, "
          f"인쇄 글자에 1 표시 {c['printed_shown_as_1']:.1%}")
    print(f"저장: {out}\n판별 화면은 이 파일이 있으면 자동으로 사용합니다: ./run_check.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
