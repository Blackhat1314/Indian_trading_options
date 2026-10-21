"""Gradient-boosted trees without scikit-learn on the server.

The 09:30 model may deploy sklearn's HistGradientBoostingClassifier (alone or blended with the network) when the
walk-forward picks it. Its split tables are exported to a small JSON file on the training PC (`export_hgb`) and
evaluated here with numpy (`predict_proba`), exactly as sklearn does: x[feature] <= threshold goes left, a missing
value follows the node's learnt direction, the raw score is the baseline plus every tree's leaf value, and the
probability is its logistic. A test checks the two agree to 1e-9.
"""
from __future__ import annotations

import numpy as np

FORMAT = "td-hgb-1"


def export_hgb(clf) -> dict:
    """A fitted binary HistGradientBoostingClassifier → plain JSON-able dict."""
    preds = getattr(clf, "_predictors", None)
    base = getattr(clf, "_baseline_prediction", None)
    if preds is None or base is None:
        raise RuntimeError("this scikit-learn version stores its trees differently — the tree export needs updating")
    if len(getattr(clf, "classes_", [])) != 2:
        raise ValueError("only binary classifiers are exported")
    trees = []
    for it in preds:
        if len(it) != 1:
            raise ValueError("expected one tree per boosting round (binary classification)")
        n = it[0].nodes
        if np.any(n["is_categorical"]):
            raise ValueError("categorical splits are not supported by the export")
        trees.append({
            "f": n["feature_idx"].astype(int).tolist(),
            "t": [float(x) for x in n["num_threshold"]],
            "l": n["left"].astype(int).tolist(),
            "r": n["right"].astype(int).tolist(),
            "leaf": n["is_leaf"].astype(bool).tolist(),
            "v": [float(x) for x in n["value"]],
            "ml": n["missing_go_to_left"].astype(bool).tolist(),
        })
    return {"format": FORMAT, "base": float(np.ravel(base)[0]), "n_features": int(clf.n_features_in_),
            "trees": trees}


def _arrays(model: dict):
    cache = model.get("_np")
    if cache is None:
        cache = [tuple(np.asarray(tr[k]) for k in ("f", "t", "l", "r", "leaf", "v", "ml")) for tr in model["trees"]]
        model["_np"] = cache
    return cache


def raw_score(model: dict, X: np.ndarray) -> np.ndarray:
    if model.get("format") != FORMAT:
        raise ValueError("not a tree file of this format — retrain with the current code")
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] != model["n_features"]:
        raise ValueError(f"the tree model expects {model['n_features']} inputs, got {X.shape[1]}")
    out = np.full(X.shape[0], model["base"], dtype=np.float64)
    rows = np.arange(X.shape[0])
    for f, t, lft, rgt, leaf, v, ml in _arrays(model):
        node = np.zeros(X.shape[0], dtype=np.int64)
        for _ in range(64):                                   # deeper than any tree sklearn grows here
            live = ~leaf[node]
            if not live.any():
                break
            idx = rows[live]
            nd = node[live]
            x = X[idx, f[nd]]
            go_left = np.where(np.isnan(x), ml[nd], x <= t[nd])
            node[live] = np.where(go_left, lft[nd], rgt[nd])
        out += v[node]
    return out


def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    """P(class 1) for each row."""
    return 1.0 / (1.0 + np.exp(-raw_score(model, X)))
