"""Inkling 2(arch inkling2_loop / inkling2、Thinking Machines Lab)の構造の読み取りと、まだ対応していない機能の断り。
docs/inkling2.md。

Inkling-2-Mini は保存されている 24 層のうち 6〜17 を 3 回続けて実行する(1 回の計算で 48 段)。GGUF の block_count は保存されている
層の数(24)で、繰り返しは {arch}.loop.start / end / count で表す(SenseNova-Loop の GGUF は block_count が実行順の層の数で、
形が違う。loop.py は SenseNova の形だけを扱う)。llama.cpp はキャッシュの番号(グラフの名前の -N)を、1 回目は保存されている
層の番号、2 回目以降は 24 から順に振る(2 回目 24〜35、3 回目 36〜47)。繰り返しの後の層 18〜23 は 18〜23 のままなので、
番号の順と実行の順が一致しない。gne の表示の「段」は実行の順(0〜47)。

残差は 1 本ではない。6 層ごと(層 0・6・12・18)に新しい流れが始まり、それまでの流れは取っておかれて、各副層の入力は
「それまでの流れと今の流れを RMS 正規化して、学習された問い(attn_res_query・ffn_res_query)で重みづけして足したもの」
(深さ方向の注意)になる。2 回目・3 回目は層 5 の出力からやり直し、前の回の流れ(流れ 1・2)を、Q・K・V・MLP の入力に
低ランクのゲート(gain・forget)で混ぜる。attention と MLP の出力は短い畳み込み(4 タップ)を通ってから流れに足される。
このため、gne が残差・ヘッド・ニューロンを読む前提(1 本の残差に各層の出力がそのまま足される)が成り立たない。
いまの gne は、読み込み・情報・マップ・会話だけを扱い、中を測る・書き換える機能は理由を書いて止める(下の refuse)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .steer import GneError

ARCHS = ("inkling2_loop", "inkling2")


def is_inkling(arch: str | None) -> bool:
    return (arch or "") in ARCHS


@dataclass(frozen=True)
class InklingPlan:
    arch: str
    n_stored: int                  # 保存されている層の数(block_count)
    core_start: int                # 繰り返す層 [core_start, core_end)(保存されている層の番号)。繰り返しが無ければ 0, 0
    core_end: int
    n_passes: int                  # 繰り返しの回数(inkling2 は 1)
    block: int                     # 残差の流れが新しく始まる間隔(attn_res.block_size)
    swa: tuple[bool, ...]          # 保存されている層ごと: True = 窓つきの attention(sliding window)、False = 全体を見る attention
    window: int                    # 窓の長さ(トークン)
    gate_rank: int                 # 前の回を混ぜるゲートのランク
    varying: bool                  # 回数を変えて学習したか({arch}.loop.varying)

    @property
    def width(self) -> int:
        return self.core_end - self.core_start

    @property
    def n_steps(self) -> int:
        """1 回の計算で実行する段の数(48)。llama.cpp の llama_model_n_layer もこの数。"""
        return self.n_stored + (self.n_passes - 1) * self.width

    def step(self, i: int) -> dict:
        """実行順の段 i → {step, layer(保存されている層), pass(1〜n_passes。繰り返しの外は 0), slot(llama.cpp のキャッシュの番号 =
        グラフの名前の -N), stream(書き込む流れの番号。0 = 層 0〜5、繰り返しの中は回の中の番号、繰り返しの後は続きの番号),
        new_stream(この段で新しい流れが始まるか), full(全体を見る attention か)}。"""
        i = int(i)
        if not 0 <= i < self.n_steps:
            raise GneError("BAD_ARGS", f"段 {i} は範囲外です(0〜{self.n_steps - 1})")
        lo, w = self.core_start, self.width
        if i < lo or not w:
            layer, p, slot = i, 0, i
        elif i < lo + self.n_passes * w:
            k = i - lo
            p, layer = k // w + 1, lo + k % w
            slot = layer if p == 1 else self.n_stored + (p - 2) * w + (layer - lo)
        else:
            layer = i - (self.n_passes - 1) * w
            p, slot = 0, layer
        return {"step": i, "layer": layer, "pass": p, "slot": slot, "stream": layer // self.block,
                "new_stream": layer % self.block == 0, "full": not self.swa[layer]}

    def steps(self) -> list[dict]:
        return [self.step(i) for i in range(self.n_steps)]

    def steps_of(self, layer: int) -> list[int]:
        """保存されている層 layer が実行される段(繰り返しの層は n_passes 個)。"""
        return [s["step"] for s in self.steps() if s["layer"] == int(layer)]

    def full_layers(self) -> list[int]:
        return [L for L, s in enumerate(self.swa) if not s]

    def step_groups(self) -> list[dict]:
        """段の表を、続いている段のまとまり(繰り返しの前・各回・繰り返しの後)にしたもの。gne info の表に使う。"""
        out: list[dict] = []
        for s in self.steps():
            g = out[-1] if out else None
            if g is None or g["pass"] != s["pass"] or s["slot"] != g["slots"][1] + 1:
                out.append({"pass": s["pass"], "steps": [s["step"], s["step"]], "layers": [s["layer"], s["layer"]],
                            "slots": [s["slot"], s["slot"]], "streams": [s["stream"]], "full": []})
                g = out[-1]
            else:
                g["steps"][1], g["layers"][1], g["slots"][1] = s["step"], s["layer"], s["slot"]
                if s["stream"] not in g["streams"]:
                    g["streams"].append(s["stream"])
            if s["full"]:
                g["full"].append(s["layer"])
        return out

    def summary(self) -> str:
        """1 行の説明(gne info の警告・UI の凡例と同じ中身)。"""
        if self.n_passes > 1:
            a, b = self.core_start, self.core_end - 1
            loop = (f"層 {a}〜{b} を {self.n_passes} 回続けて実行します(実行順の段 {self.n_steps}。2 回目・3 回目は層 {a - 1} の出力から"
                    f"やり直し、前の回の流れを低ランク(ランク {self.gate_rank})のゲートで Q・K・V・MLP の入力に混ぜます)。")
        else:
            loop = ""
        starts = ", ".join(str(L) for L in range(0, self.n_stored, self.block))
        full = ", ".join(map(str, self.full_layers()))
        return (loop + f"残差は 1 本ではなく、層 {starts} で新しい流れが始まり、各副層はそれまでの流れを学習された重みで混ぜて読みます"
                f"(深さ方向の注意)。attention は窓つき(sliding window、{self.window} トークン)で、層 {full} だけ全体を見ます。"
                "attention と MLP の出力は短い畳み込みを通ってから流れに足されます")

    def to_dict(self) -> dict:
        return {"arch": self.arch, "n_stored": self.n_stored, "core_start": self.core_start, "core_end": self.core_end,
                "n_passes": self.n_passes, "block": self.block, "window": self.window, "gate_rank": self.gate_rank,
                "varying": self.varying, "n_steps": self.n_steps, "full_layers": self.full_layers(),
                "swa": list(self.swa), "steps": self.steps(), "step_groups": self.step_groups(), "summary": self.summary()}


def expert_tensors(tensors: dict, n_stored: int) -> dict:
    """専門家の重みの型と、専門家ごとの倍率(ffn_*_exps.scale)のある層。テンソルの見出し(名前・型・形)だけを見て、中身は読まない。
    tensors はテンソル名 → FastTensor。"""
    kinds: dict[str, dict[str, list[int]]] = {}
    scaled: dict[str, list[int]] = {}
    for L in range(n_stored):
        for k in ("ffn_gate_up_exps", "ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"):
            t = tensors.get(f"blk.{L}.{k}.weight")
            if t is not None:
                kinds.setdefault(k, {}).setdefault(t.tensor_type.name, []).append(L)
            if f"blk.{L}.{k}.scale" in tensors:
                scaled.setdefault(k, []).append(L)
    return {"types": kinds, "scaled_layers": scaled}


def plan_of(get, arch: str | None) -> InklingPlan | None:
    """GGUF のメタデータを読む関数 get(key) から。Inkling 2 の arch でなければ None。形が読めなければ BAD_MODEL。"""
    if not is_inkling(arch):
        return None
    n = int(get(f"{arch}.block_count", 0) or 0)
    block = int(get(f"{arch}.attn_res.block_size", 0) or 0)
    pat = get(f"{arch}.attention.sliding_window_pattern")
    window = int(get(f"{arch}.attention.sliding_window", 0) or 0)
    if arch == "inkling2_loop":
        start, end, count = (get(f"{arch}.loop.{k}") for k in ("start", "end", "count"))
        if start is None or end is None or count is None:
            raise GneError("BAD_MODEL", f"{arch} の GGUF に loop.start / loop.end / loop.count がありません")
        start, end, count = int(start), int(end), int(count)
        rank = int(get(f"{arch}.loop.gate_rank", 0) or 0)
        varying = bool(get(f"{arch}.loop.varying", False))
    else:
        start = end = 0
        count, rank, varying = 1, 0, False
    if not isinstance(pat, list) or len(pat) != n or not block or not n:
        raise GneError("BAD_MODEL", f"{arch} の層の並び(block_count {n}、attn_res.block_size {block}、"
                       f"sliding_window_pattern {len(pat) if isinstance(pat, list) else '無し'})を読めません")
    if arch == "inkling2_loop" and (count < (1 if varying else 2) or not 0 < start < end <= n or start % block or end % block):
        raise GneError("BAD_MODEL", f"{arch} の繰り返し(層 {start}〜{end - 1} × {count}、流れの間隔 {block}、層 {n})は、"
                       "llama.cpp が読む形ではありません")
    return InklingPlan(arch, n, start, end, count, block, tuple(bool(x) for x in pat), window, rank, varying)


# --- 返事の区切り ---------------------------------------------------------------------------------------------------

THINK_TOKEN, TEXT_TOKEN = "<|content_thinking|>", "<|content_text|>"
THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n\n"


class ReplyMarks:
    """Inkling 2 は返事の前に思考(reasoning)を書く: <|message_model|><|content_thinking|> 思考 <|end_message|><|message_model|>
    <|content_text|> 回答 <|end_message|><|content_model_end_sampling|>。区切りは特別なトークンなので、ふつうの表示では消えて
    思考と回答がつながる。表示では思考を <think> … </think> で囲み、会話の履歴には回答だけを入れる。1 つの返事ごとに作る。"""

    def __init__(self, engine):
        def one(text: str) -> int | None:
            ids = engine.tokenize(text, parse_special=True)
            return ids[0] if len(ids) == 1 else None
        self.think, self.text = one(THINK_TOKEN), one(TEXT_TOKEN)
        self.open = False

    def piece(self, tok: int) -> str | None:
        """区切りのトークンなら表示する文字(思考の始まり・終わり)、ほかは None。"""
        if self.think is not None and tok == self.think:
            self.open = True
            return THINK_OPEN
        if self.text is not None and tok == self.text and self.open:
            self.open = False
            return THINK_CLOSE
        return None

    def render(self, engine, token_ids: list[int]) -> str:
        """返事のトークン → 表示の文字列(思考を <think> … </think> で囲む)。"""
        self.open = False
        out, run = [], []
        for t in token_ids:
            if engine.is_eog(int(t)):
                continue
            mark = self.piece(int(t))
            if mark is None:
                run.append(int(t))
                continue
            out.append(engine.detokenize(run))
            run = []
            out.append(mark)
        out.append(engine.detokenize(run))
        if self.open:                       # 回答の前で終わった(長さの上限など)
            out.append(THINK_CLOSE.rstrip("\n"))
        self.open = False
        return "".join(out)


def answer_part(text: str) -> str:
    """表示の文字列から回答だけ(思考の欄を除く)。会話の履歴に入れる。"""
    if text.startswith(THINK_OPEN):
        i = text.find(THINK_CLOSE.strip("\n"))
        return text[i + len(THINK_CLOSE.strip("\n")):].lstrip("\n") if i >= 0 else ""
    return text


# --- まだ対応していない機能の断り ----------------------------------------------------------------------------------

FIX = ("このモデルで使えるのは、読み込み・gne info・マップ・会話(gne chat、gne serve のチャット、OpenAI 互換 API)と、"
       "レシピを当てない gne behave です(docs/inkling2.md)")


# CLI で使えるコマンド(gne_command の名前)。ほかは読み込む前に止める。読み込み・情報・会話・名前の一覧と、ファイルだけを扱うもの。
# 低い層の止め方(engine.py の捕捉・control vector・LoRA、weights.py の重みの読み出し)は、MCP・UI のジョブにも効く
CLI_OK = frozenset({"info", "probe", "verify", "bench", "chat", "behave", "worker", "recipe-combine", "recipe-scale",
                    "direction keys"})


def check_command(plan: InklingPlan, command: str | None) -> None:
    """CLI のコマンドが Inkling 2 で使えるか。使えなければ refuse。command が None(CLI 以外から)なら何もしない。"""
    if command is not None and command not in CLI_OK:
        raise refuse(plan, f"gne {command}")


def refuse(plan: InklingPlan, what: str) -> GneError:
    """中を測る・書き換える機能を止める例外。what は「残差の捕捉(中間のテンソルを読むこと)」のような、止めたものの名前。"""
    loop = (f"層 {plan.core_start}〜{plan.core_end - 1} を {plan.n_passes} 回実行し、" if plan.n_passes > 1 else "")
    why = (f"Inkling 2(arch {plan.arch})は、残差が 1 本ではなく({plan.block} 層ごとに新しい流れが始まり、各副層はそれまでの流れを"
           f"学習された重みで混ぜて読む)、{loop}attention と MLP の出力が短い畳み込みを通ってから足されるので、gne が残差・ヘッド・"
           f"ニューロンを読む前提が成り立ちません。{what}は、このモデルではまだ使えません(止めました)。")
    return GneError("ARCH_UNSUPPORTED", why, FIX)
