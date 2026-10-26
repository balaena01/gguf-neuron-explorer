"""シート内の配置(別冊 UI 設計 §4.4)。ニューロンを 2D に埋め込み、シートの格子にスナップする。

- writes: W_down の列(単位ベクトル)→ PCA 50 → UMAP 2D。近くにあるニューロンは「少し似ていることがある」(弱い)。
- lens: steer lens の ± 上位語(無ければ logit lens)を疎なベクトルにして UMAP(cosine)。同じ語を書くニューロンが寄る。

格子へのスナップ: UMAP の y で並べて side 個ずつの行に切り、各行の中を x で並べる。返すのは cell[n](n → 格子の番号)。
"""

from __future__ import annotations

import numpy as np

from .steer import GneError

KINDS = ("writes", "lens")


def snap_to_grid(Y: np.ndarray, side: int) -> np.ndarray:
    """2D の座標 → 格子の番号。行は y の小さい順(上から)、行の中は x の小さい順。"""
    n = Y.shape[0]
    order = np.argsort(-Y[:, 1], kind="stable")                 # y の大きい方を上の行に
    cell = np.empty(n, np.int32)
    for r0 in range(0, n, side):
        row = order[r0:r0 + side]
        row = row[np.argsort(Y[row, 0], kind="stable")]
        cell[row] = np.arange(r0, r0 + row.size, dtype=np.int32)
    return cell


def _umap(X, *, metric: str, seed: int):
    try:
        import umap
    except ImportError as e:
        raise GneError("TOOL_MISSING", "umap-learn がありません", "pip install umap-learn(pyproject の ui extra)") from e
    import warnings
    with warnings.catch_warnings():                              # random_state を付けると 1 スレッドになる、などの警告
        warnings.simplefilter("ignore")
        return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric=metric, random_state=seed,
                         low_memory=True, verbose=False).fit_transform(X)


def _pca(X: np.ndarray, k: int = 50) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)                       # (d, d)。d ≪ n なので共分散の固有分解で
    w, V = np.linalg.eigh(C.astype(np.float64))
    V = V[:, np.argsort(-w)[:k]].astype(np.float32)
    return Xc @ V


def compute_layout(session, kind: str, L: int, *, seed: int = 0) -> tuple[np.ndarray, dict]:
    if kind not in KINDS:
        raise GneError("BAD_ARGS", f"配置の種類は {' / '.join(KINDS)} です")
    w = session.weights
    n = w.meta.n_ff_at(L)
    side = int(np.ceil(np.sqrt(n)))
    if kind == "writes":
        X = _pca(w.unit_matrix(L).T.astype(np.float32))
        Y = _umap(X, metric="euclidean", seed=seed)
        src = "W_down の列(単位ベクトル)、PCA 50"
    else:
        from scipy import sparse

        from .ui.data import UIData
        data = UIData(session)
        sl, ll = data.steer_lens(L), data.logit_lens(L)
        if sl is not None and sl.done.all():
            up, down, src = sl.up_ids, sl.down_ids, "steer lens の ± 上位語"
        elif ll is not None and ll.neurons.size == n:
            o = np.argsort(ll.neurons)
            up, down, src = ll.up_ids[o], ll.down_ids[o], "logit lens の ± 上位語"
        else:
            raise GneError("NOT_FOUND", f"層 {L} の lens が全ニューロンぶんありません",
                           "gne lens logit --layer L(数十秒)か lens steer で作ってください")
        V = len(session.vocab.pieces)
        rows, cols = [], []
        for i in range(n):                                       # + 極の語と − 極の語は別の特徴にする
            for t in up[i]:
                if t >= 0:
                    rows.append(i); cols.append(int(t))
            for t in down[i]:
                if t >= 0:
                    rows.append(i); cols.append(V + int(t))
        X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, 2 * V))
        Y = _umap(X, metric="cosine", seed=seed)
    cell = snap_to_grid(np.asarray(Y, np.float32), side)
    return cell, {"kind": kind, "n": n, "side": side, "source": src}
