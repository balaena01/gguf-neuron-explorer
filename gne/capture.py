"""cb_eval による中間テンソル捕捉。設計書 §6.3。

ggml の構造体レイアウトには依存せず、ggml の関数(ggml_get_name, ggml_nbytes, ggml_nrows,
ggml_backend_tensor_get 等)だけでテンソルを読む。

捕捉は ubatch(グラフの 1 回の計算)単位の step にまとめる。Engine は llama_decode の直後に end_step() を
呼ぶ。1 回の decode が複数の ubatch に分かれた場合は、グラフの先頭ノード(ask 段階で最初に見えた名前)が
もう一度来た時点で次の ubatch とみなして区切る。1 つのグラフに同じ名前のノードが 2 つある arch もある
(llama は FFN 出力と残差の和の両方に ffn_out-L を付ける)ので、step 内では先に来た方を採る。
take() は各 step の同名テンソルをトークン方向に連結して返す。行の並びは ubatch の並びで、llama_batch に
詰めた順とは限らない(ハイブリッド/回帰型は複数シーケンスをシーケンスごとの ubatch に並べ替える)ので、
probe ctx では 1 シーケンスずつ n_ubatch 以下に切って評価する(docs/verification.md の F1)。
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable

import numpy as np
from llama_cpp import llama_cpp as C
from llama_cpp._ggml import libggml

_vp = ctypes.c_void_p
libggml.ggml_get_name.restype = ctypes.c_char_p
libggml.ggml_get_name.argtypes = [_vp]
libggml.ggml_nbytes.restype = ctypes.c_size_t
libggml.ggml_nbytes.argtypes = [_vp]
libggml.ggml_nelements.restype = ctypes.c_int64
libggml.ggml_nelements.argtypes = [_vp]
libggml.ggml_nrows.restype = ctypes.c_int64
libggml.ggml_nrows.argtypes = [_vp]
libggml.ggml_element_size.restype = ctypes.c_size_t
libggml.ggml_element_size.argtypes = [_vp]
libggml.ggml_is_contiguous.restype = ctypes.c_bool
libggml.ggml_is_contiguous.argtypes = [_vp]
libggml.ggml_backend_tensor_get.restype = None
libggml.ggml_backend_tensor_get.argtypes = [_vp, _vp, ctypes.c_size_t, ctypes.c_size_t]


class Capture:
    """ggml_backend_sched_eval_callback の Python 実装。名前で選んだテンソルを float32 で回収する。"""

    def __init__(self, wanted: Callable[[str], bool] | None = None):
        self._wanted = wanted or (lambda name: False)
        self._step: dict[str, np.ndarray] = {}
        self._steps: list[dict[str, np.ndarray]] = []
        self.seen: dict[str, int] = {}      # record_names=True のとき、ask 段階で見えた名前 → 回数
        self.record_names = False
        self.n_splits = 0                   # 先頭ノードの再来で step を区切った回数(1 decode が複数 ubatch だった)
        self.duplicates: set[str] = set()   # 1 つのグラフに同名ノードが複数あった名前(先に来た方を採った)
        self._first: str | None = None      # いまのグラフの先頭ノードの名前
        self.skipped: dict[str, str] = {}   # F32 以外・非連続で回収できなかった名前 → 理由
        # 残差が複数本の arch(qwen4exp の hyper-connections)では層の出力が l_last-L = [n_embd, hc, T]。
        # 本数を入れると l_out-L という名前で流れの平均(n_tokens, n_embd)を返す。他のコードは今までどおり l_out-L を見ればよい
        self.hc_streams = 0
        # 同じ層を 2 回実行する arch(sensenova_loop。loop.py): 各回の中では残差が 2 本の流れで、グラフの名前が違う。
        # Engine が loop(LoopPlan)・loop_d(n_embd)・loop_full(full attention の層)を入れると、ふつうの名前で返す:
        #   l_out-L        = 2 本の平均(llama.cpp のグラフの l_out は、2 本の中の層では片方の流れの飛び飛びの view。
        #                    流れの中で control vector を足すように直した llama.cpp では、平均そのものがグラフの l_out になる)
        #   l_out_pre-L    = 層の最後の混ぜ合わせ(mhc_ffn_merged)の 2 本の平均(control vector を足す前)
        #   attn_residual-L = MLP が実際に読む正規化の前の入力(ルーターが 2 本を重みづけして足したもの)
        #   attn_post_norm-L・attn_norm-L・ffn_out-L・attn_output-L / linear_attn_out-L = 2 本の中の層で名前の違うもの
        #   l_out_streams-L・attn_merged_streams-L(2 本を並べたまま)と mhc_ffn_gates-L / mhc_attn_gates-L(ルーターの書き込みの
        #   重み 2 つと混ぜる行列の 4 つの値)は、その名前を欲しいと言ったときだけ返す
        self.loop = None
        self.loop_d = 0
        self.loop_full: frozenset[int] = frozenset()
        self._cb = C.ggml_backend_sched_eval_callback(self._on_tensor)  # 参照を保持(GC 対策)

    def install(self, cparams: C.llama_context_params) -> None:
        cparams.cb_eval = self._cb
        cparams.cb_eval_user_data = None

    def set_wanted(self, wanted: Callable[[str], bool]) -> None:
        self._wanted = wanted

    def end_step(self) -> None:
        """Engine が llama_decode の直後に呼ぶ。"""
        self._first = None
        self._close_step()

    def _close_step(self) -> None:
        if self._step:
            if self.loop is not None:
                self._loop_fill(self._step)
            self._steps.append(self._step)
            self._step = {}

    def _loop_fill(self, st: dict) -> None:
        """2 本の中の層の l_out-L が無ければ(グラフの l_out が飛び飛びの view のとき)、l_out_pre-L(2 本の平均)で埋める。"""
        for name in [n for n in st if n.startswith("l_out_pre-")]:
            L = name[len("l_out_pre-"):]
            if f"l_out-{L}" not in st and self.loop.inner(int(L)) and self._wanted(f"l_out-{L}"):
                st[f"l_out-{L}"] = st[name]
        for name in [n for n in st if not self._wanted(n)]:      # 別の名前のために取っただけのもの
            del st[name]

    def reset(self) -> None:
        self._step, self._steps, self._first = {}, [], None

    def take(self) -> dict[str, np.ndarray]:
        """直前の reset 以降に回収したもの。{name: (n_tokens, ne0) float32}(step 方向に連結)。"""
        self.end_step()
        steps, self._steps = self._steps, []
        out: dict[str, list[np.ndarray]] = {}
        for st in steps:
            for name, arr in st.items():
                out.setdefault(name, []).append(arr)
        return {k: (v[0] if len(v) == 1 else np.concatenate(v, axis=0)) for k, v in out.items()}

    def _canon(self, name: str) -> str:
        if self.loop is not None:
            return self._loop_canon(name)
        return "l_out-" + name[7:] if self.hc_streams and name.startswith("l_last-") else name

    _LOOP_NAMES = {"mhc_ffn_merged": "l_out_pre", "mhc_attn_merged": "attn_merged", "mhc_ffn_packed": "attn_residual",
                   "mhc_attn_packed": "attn_in", "mhc_ffn_norm": "attn_post_norm", "mhc_attn_norm": "attn_norm",
                   "mhc_ffn_out": "ffn_out"}

    def _loop_canon(self, name: str) -> str:
        base, sep, L = name.rpartition("-")
        if not sep or not L.isdigit():
            return name
        if base == "mhc_attn_out":
            return f"attn_output-{L}" if int(L) in self.loop_full else f"linear_attn_out-{L}"
        b = self._LOOP_NAMES.get(base)
        return f"{b}-{L}" if b else name

    def _want(self, name: str) -> bool:
        """捕捉するか。l_out_pre-L は、2 本の中の層の l_out-L を欲しいときにも取る(グラフの l_out が view なら、それで埋める)。"""
        if self._wanted(name):
            return True
        if self.loop is None:
            return False
        base, _, L = name.rpartition("-")
        if base == "l_out_pre" and self.loop.inner(int(L)) and self._wanted(f"l_out-{L}"):
            return True
        derived = self._LOOP_DERIVED.get(base)
        return bool(derived and self._wanted(f"{derived}-{L}"))

    # 並べたままの値・ルーターの重みの名前(元になる捕捉の名前 → 別に返す名前)
    _LOOP_DERIVED = {"l_out_pre": "l_out_streams", "attn_merged": "attn_merged_streams", "attn_residual": "mhc_ffn_gates",
                     "attn_in": "mhc_attn_gates"}

    def _ask(self, name: str) -> bool:
        if self._first is None:
            self._first = name
        elif name == self._first:            # 先頭ノードがもう一度 = 次の ubatch のグラフ
            self.n_splits += 1
            self._close_step()
        if self.record_names:
            self.seen[name] = self.seen.get(name, 0) + 1
        return bool(self._want(name))

    def _store(self, name: str, arr: np.ndarray) -> None:
        if self.loop is not None:
            arr = self._loop_shape(name, arr)
        if name in self._step:               # 同じグラフの 2 つ目の同名ノード
            self.duplicates.add(name)
            return
        self._step[name] = arr

    def _loop_shape(self, name: str, arr: np.ndarray) -> np.ndarray:
        """2 本の流れを並べたテンソル([2d] の行)は平均に、ルーターの出力([d + 6] の行)は前の d 列にする。
        並べたままの値・ルーターの重みは、その名前を欲しいと言ったときだけ別の名前で残す。"""
        d = self.loop_d
        base = name.rpartition("-")[0]
        if base in ("l_out_pre", "attn_merged") and arr.shape[1] == 2 * d:
            sname = name.replace(base, self._LOOP_DERIVED[base], 1)
            if self._wanted(sname) and sname not in self._step:
                self._step[sname] = arr
            return arr.reshape(arr.shape[0], 2, d).mean(axis=1)
        if base in ("attn_residual", "attn_in") and arr.shape[1] == d + 6:
            gname = name.replace(base, self._LOOP_DERIVED[base], 1)
            if self._wanted(gname) and gname not in self._step:
                self._step[gname] = np.ascontiguousarray(arr[:, d:])
            return np.ascontiguousarray(arr[:, :d])
        return arr

    def _on_tensor(self, t: int, ask: bool, _ud) -> bool:
        try:
            name = self._canon(libggml.ggml_get_name(t).decode("utf-8", "replace"))
            if ask:
                return self._ask(name)
            if not self._want(name):
                return True
            nbytes = libggml.ggml_nbytes(t)
            nelem = libggml.ggml_nelements(t)
            if self.loop is not None and name.startswith("l_out-") and not libggml.ggml_is_contiguous(t):
                return True                      # 2 本の中の層のグラフの l_out(片方の流れの view)。l_out_pre-L の平均で埋める
            if libggml.ggml_element_size(t) != 4 or nbytes != nelem * 4:
                self.skipped[name] = "not F32"
                return True
            if not libggml.ggml_is_contiguous(t):
                self.skipped[name] = "not contiguous"
                return True
            nrows = libggml.ggml_nrows(t)
            # 要素 4 バイトでも I32 のテンソル(MoE の選んだ専門家の番号)は名前で見分ける(CHANGE-05 0 項 3)
            buf = np.empty(nelem, dtype=np.int32 if name.split("-")[0] in INT_NAMES else np.float32)
            libggml.ggml_backend_tensor_get(t, buf.ctypes.data, 0, nbytes)   # GPU でもホストでも動く
            # ggml は ne0 が最内側: [ne0, ne1(, ...)] → numpy (nrows, ne0)
            arr = buf.reshape(nrows, nelem // nrows)
            if self.hc_streams and name.startswith("l_out-"):
                arr = arr.reshape(nrows // self.hc_streams, self.hc_streams, -1).mean(axis=1)
            self._store(name, arr)
            return True
        except Exception as e:  # noqa: BLE001  コールバック内の例外は C 側に投げられない
            self.skipped["<callback error>"] = repr(e)
            return True


# I32 で出るテンソル(llama-graph.cpp build_moe_ffn の ggml_argsort_top_k / ggml_argsort)
INT_NAMES = {"ffn_moe_topk", "ffn_moe_argsort"}

# ffn_down の入力(ニューロン活性 a)になるテンソル名の候補。build_ffn で最後に付く名前ほど優先する
# (gate_par は活性化の後の gate ⊙ up、act は act_scales での割り算)。src/llama-graph.cpp build_ffn。
ACT_CANDIDATES = (
    "ffn_gate_par", "ffn_act", "ffn_swiglu_limited", "ffn_swiglu", "ffn_geglu", "ffn_reglu",
    "ffn_swiglu_oai", "ffn_sqr(relu)", "ffn_gelu", "ffn_silu", "ffn_relu",
)


def pick_act_name(names: set[str], L: int) -> str | None:
    """層 L の ffn_down 入力のテンソル名を探す。見つからなければ None。"""
    for base in ACT_CANDIDATES:
        if f"{base}-{L}" in names:
            return f"{base}-{L}"
    return None
