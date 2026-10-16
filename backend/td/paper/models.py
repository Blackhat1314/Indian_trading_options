"""Small, dependency-free learners for the paper trader (numpy only), with JSON persistence.

  Ridge   — standardised linear regression with an L2 penalty: stable, interpretable (per-feature bps).
  GBM     — gradient-boosted regression trees on quantile-binned features (histogram method, like LightGBM
            in miniature): picks up non-linear effects and interactions (e.g. "reversal works only when the
            stock is below VWAP"). Shrinkage, row/feature subsampling and minimum leaf sizes keep it from
            memorising noise.
  Ensemble — the average of both: a typical quant-desk choice, since the two make different mistakes.

All models predict the expected return of a trade (fraction of entry price, before costs). Training data is
clipped at its 1st / 99th percentiles so a handful of extreme days cannot steer the fit.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _clean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.where(np.isfinite(X), X, 0.0)


class Ridge:
    def __init__(self, lam: float = 0.05):
        self.lam = lam
        self.mu = self.sd = self.coef = None
        self.intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        X = _clean(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        n, k = Z.shape
        A = Z.T @ Z + self.lam * n * np.eye(k)
        yc = y - y.mean()
        self.coef = np.linalg.solve(A, Z.T @ yc)
        self.intercept = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept + ((_clean(X) - self.mu) / self.sd) @ self.coef

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Per-feature contribution to one prediction (same units as y)."""
        return ((_clean(x[None, :])[0] - self.mu) / self.sd) * self.coef

    def to_json(self) -> dict:
        return {"kind": "ridge", "lam": self.lam, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_json(cls, d: dict) -> "Ridge":
        m = cls(d["lam"])
        m.mu, m.sd, m.coef = (np.asarray(d[k], float) for k in ("mu", "sd", "coef"))
        m.intercept = float(d["intercept"])
        return m


class GBM:
    """Gradient boosting with depth-limited trees on binned features (squared loss)."""

    def __init__(self, n_trees: int = 200, depth: int = 3, lr: float = 0.05, min_leaf: int = 300,
                 bins: int = 32, subsample: float = 0.5, colsample: float = 0.7, l2: float = 50.0, seed: int = 7):
        self.n_trees, self.depth, self.lr, self.min_leaf = n_trees, depth, lr, min_leaf
        self.bins, self.subsample, self.colsample, self.l2, self.seed = bins, subsample, colsample, l2, seed
        self.base = 0.0
        self.trees: list[dict] = []          # each: feature[], threshold[], left[], right[], value[] (array tree)

    # -------------------------------------------------------------- binning
    def _bin(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        edges, Xb = [], np.zeros(X.shape, np.uint8)
        qs = np.linspace(0, 1, self.bins + 1)[1:-1]
        for j in range(X.shape[1]):
            e = np.unique(np.quantile(X[:, j], qs))
            edges.append(e)
            Xb[:, j] = np.searchsorted(e, X[:, j], side="left")
        return Xb, edges

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBM":
        X = _clean(X)
        y = np.asarray(y, np.float64)
        rng = np.random.default_rng(self.seed)
        Xb, edges = self._bin(X)
        n, k = X.shape
        self.base = float(y.mean())
        pred = np.full(n, self.base)
        self.trees = []
        for _ in range(self.n_trees):
            rows = np.where(rng.random(n) < self.subsample)[0]
            cols = np.where(rng.random(k) < self.colsample)[0]
            if len(cols) == 0:
                cols = np.array([rng.integers(k)])
            g = y - pred
            tree = self._grow(Xb, g, rows, cols, edges)
            self.trees.append(tree)
            pred += self.lr * self._apply(tree, X)
        return self

    def _grow(self, Xb, g, rows, cols, edges) -> dict:
        feat, thr, left, right, value = [], [], [], [], []

        def new_leaf(idx) -> int:
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1)
            value.append(float(g[idx].sum() / (len(idx) + self.l2)))
            return len(feat) - 1

        def split(node: int, idx: np.ndarray, depth: int) -> None:
            if depth >= self.depth or len(idx) < 2 * self.min_leaf:
                return
            G, Cn = g[idx].sum(), len(idx)
            parent = G * G / (Cn + self.l2)
            best = (0.0, -1, -1)
            for j in cols:
                b = Xb[idx, j]
                gs = np.bincount(b, weights=g[idx], minlength=self.bins)
                cs = np.bincount(b, minlength=self.bins).astype(float)
                gl, cl = np.cumsum(gs)[:-1], np.cumsum(cs)[:-1]
                gr, cr = G - gl, Cn - cl
                ok = (cl >= self.min_leaf) & (cr >= self.min_leaf)
                if not ok.any():
                    continue
                gain = np.where(ok, gl * gl / (cl + self.l2) + gr * gr / (cr + self.l2) - parent, -np.inf)
                bi = int(np.argmax(gain))
                if gain[bi] > best[0] and bi < len(edges[j]):
                    best = (float(gain[bi]), int(j), bi)
            if best[1] < 0:
                return
            _, j, bi = best
            go_left = Xb[idx, j] <= bi
            li, ri = idx[go_left], idx[~go_left]
            feat[node], thr[node] = j, float(edges[j][bi])
            left[node], right[node] = new_leaf(li), new_leaf(ri)
            split(left[node], li, depth + 1)
            split(right[node], ri, depth + 1)

        root = new_leaf(rows)
        split(root, rows, 0)
        return {"f": feat, "t": thr, "l": left, "r": right, "v": value}

    @staticmethod
    def _apply(tree: dict, X: np.ndarray) -> np.ndarray:
        f, t, l, r, v = (np.asarray(tree[k]) for k in ("f", "t", "l", "r", "v"))
        node = np.zeros(len(X), np.int64)
        for _ in range(16):                                   # depth bound
            internal = f[node] >= 0
            if not internal.any():
                break
            ni = node[internal]
            x = X[np.where(internal)[0], f[ni]]
            node[internal] = np.where(x <= t[ni], l[ni], r[ni])
        return v[node]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _clean(X)
        out = np.full(len(X), self.base)
        for tree in self.trees:
            out += self.lr * self._apply(tree, X)
        return out

    def importance(self, k: int) -> np.ndarray:
        imp = np.zeros(k)
        for tree in self.trees:
            for j in tree["f"]:
                if j >= 0:
                    imp[j] += 1
        return imp / max(imp.sum(), 1)

    def to_json(self) -> dict:
        return {"kind": "gbm", "lr": self.lr, "base": self.base, "depth": self.depth,
                "trees": [{k: [round(x, 9) if isinstance(x, float) else x for x in tr[k]] for k in tr}
                          for tr in self.trees]}

    @classmethod
    def from_json(cls, d: dict) -> "GBM":
        m = cls(lr=d["lr"], depth=d["depth"])
        m.base, m.trees = float(d["base"]), d["trees"]
        return m


class Ensemble:
    """Average of a ridge and a GBM fitted on the same data."""

    def __init__(self, ridge: Ridge | None = None, gbm: GBM | None = None, w_ridge: float = 0.5):
        self.ridge, self.gbm, self.w = ridge or Ridge(), gbm or GBM(), w_ridge

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ensemble":
        y = np.asarray(y, np.float64)
        lo, hi = np.nanquantile(y, 0.01), np.nanquantile(y, 0.99)
        y = np.clip(y, lo, hi)
        self.ridge.fit(X, y)
        self.gbm.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.w * self.ridge.predict(X) + (1 - self.w) * self.gbm.predict(X)

    def to_json(self) -> dict:
        return {"kind": "ensemble", "w": self.w, "ridge": self.ridge.to_json(), "gbm": self.gbm.to_json()}

    @classmethod
    def from_json(cls, d: dict) -> "Ensemble":
        return cls(Ridge.from_json(d["ridge"]), GBM.from_json(d["gbm"]), d["w"])
