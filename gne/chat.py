"""−/0/+ の 3 列比較チャット。設計書 §6.10。

同じ会話履歴を列ごとに生成する。ユーザーの発話は全列で共有し、assistant の返答は列ごとに別に持つ。
control vector はコンテキスト単位なので、列は 1 本ずつ順に(set_cvec を切り替えて)生成する。
"""

from __future__ import annotations

import codecs
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from .steer import AblateSpec, GneError, Recipe, Unit


@dataclass
class Column:
    label: str
    recipe: Recipe | None          # None = 無介入


@dataclass
class ChatState:
    columns: list[Column]
    system: str | None = None
    histories: list[list[dict]] = field(default_factory=list)   # 列ごとの会話(system を除く)

    def __post_init__(self):
        if not self.histories:
            self.histories = [[] for _ in self.columns]

    def reset(self) -> None:
        self.histories = [[] for _ in self.columns]

    def messages(self, i: int) -> list[dict]:
        return ([{"role": "system", "content": self.system}] if self.system else []) + self.histories[i]


def _with_dose(recipe: Recipe, dose: float | None, sign: float) -> Recipe:
    """dose を指定したら各ユニットの大きさをそろえる(符号は保つ)。sign で列の向き(+1 / −1)を掛ける。"""
    # gain は − の列で「1 を挟んで反対側」の倍率にする(×2 なら ×0、×0 なら ×2)。dose はそろえない
    # ablate(§6.17)は − の列で scale を反転する(負の scale はその方向を強める)
    # query(ヘッドの条件付きクエリ操作)は − の列では外す(scale 0)
    # route(専門家のルーターのバイアス、CHANGE-05)は − の列で delta の符号を反転する
    # proj(ヘッド・ニューロンの方向成分の射影、CHANGE-06)は − の列で s の符号を反転する(方向成分を強める)
    # keyed(鍵つきの書き込み、CHANGE-07)は − の列で α の符号を反転する(鍵が反応する状態でその方向を打ち消す)。α は raw なので dose はそろえない
    units = [replace(u, k=None if u.k is None else u.k * sign)
             if u.mode in ("keyed", "nw") else
             Unit(u.layer, u.neuron, 0.0, u.kind, u.seed, "gain", u.k if sign > 0 else 2 - u.k, head=u.head, expert=u.expert)
             if u.mode == "gain" else
             replace(u, k=u.k * sign)
             if u.mode == "proj" else
             Unit(u.layer, None, 0.0, "head", None, "query", u.k if sign > 0 else 0.0, u.direction, u.head, u.seek)
             if u.mode == "query" else
             Unit(u.layer, None, u.dose * sign, "expert", mode="route", expert=u.expert)
             if u.mode == "route" else
             Unit(u.layer, u.neuron, (abs(dose) if dose is not None else abs(u.dose)) * (1 if u.dose >= 0 else -1) * sign,
                  u.kind, u.seed, direction=u.direction, expert=u.expert) for u in recipe.units]
    ab = recipe.ablate
    if ab is not None and sign < 0:
        ab = AblateSpec(ab.direction, -ab.scale, ab.layers, ab.from_layer)
    return recipe.like(units, ablate=ab, sampling=None)


def effort_choices(template: str | None) -> list[str] | None:
    """チャットテンプレートが受け付ける reasoning_effort の名前(テンプレートの effort_map の鍵。Inkling 2 は none / minimal / low /
    medium / high / xhigh / max)。reasoning_effort を使わないテンプレートは None、使うが名前の表が無ければ []。"""
    import re
    if not template or "reasoning_effort" not in template:
        return None
    m = re.search(r"effort_map\s*=\s*\{([^}]*)\}", template)
    return re.findall(r"[\"']([\w-]+)[\"']\s*:", m.group(1)) if m else []


def effort_value(template: str | None, value):
    """--reasoning-effort の値をテンプレートに渡す形にする。None・空はテンプレートの既定(渡さない)。名前の表があるテンプレートは、
    その名前か 0〜0.99 の数(Inkling 2 のテンプレートの範囲)。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    choices = effort_choices(template)
    if choices is None:
        raise GneError("BAD_ARGS", "このモデルのチャットテンプレートは reasoning_effort を使いません")
    v = value.strip() if isinstance(value, str) else value
    try:
        num = float(v)
    except (TypeError, ValueError):
        num = None
    if num is not None:
        if choices and not 0.0 <= num <= 0.99:
            raise GneError("BAD_ARGS", f"reasoning_effort の数は 0〜0.99 です({v})")
        return num
    if choices and v not in choices:
        raise GneError("BAD_ARGS", f"reasoning_effort は {' / '.join(choices)} か 0〜0.99 の数です({v!r})")
    return v


def make_columns(recipe: Recipe | None, *, dose: float | None = None, left: Recipe | None = None,
                 right: Recipe | None = None, center: Recipe | None = None) -> list[Column]:
    """既定は [−レシピ, 無介入, +レシピ]。--left / --right / --center で任意のレシピに差し替えられる。"""
    if recipe is None and (left is None or right is None):
        raise GneError("BAD_ARGS", "--recipe / --unit か、--left と --right の両方を指定してください")

    def label(r: Recipe | None, default: str) -> str:
        if r is None:
            return default
        return r.name or ", ".join(u.label() for u in r.units)

    lr = left if left is not None else _with_dose(recipe, dose, -1)
    rr = right if right is not None else _with_dose(recipe, dose, +1)
    return [Column(label(lr, "−"), lr), Column(label(center, "0(無介入)"), center), Column(label(rr, "+"), rr)]


def chat_turn(session, state: ChatState, user: str, *, max_tokens: int = 256,
              on_text: Callable[[int, str], None] | None = None,
              on_column: Callable[[int, Column], None] | None = None,
              when_override: str | None = None, sampling=None, lora_when: str = "all", hide_reasoning: bool = False) -> list[dict]:
    """ユーザーの発話を全列に足し、列ごとに返答を生成して履歴に足す。on_text(列, 文字列の断片)で逐次表示できる。
    sampling(engine.Sampling)を渡すと、どの列も同じ seed で引く(列の違いは介入の違いだけ)。
    答えの前に思考を書くモデル(Inkling 2)では、思考を <think> … </think> で囲んで出す。hide_reasoning なら思考は出さず、
    「(思考中… N トークン)」と回答だけを出す(返事の text も回答だけ)。返事には answer(回答だけ)と reasoning_tokens も入れる。"""
    from .when import generate_when
    e = session.engine
    ctx = e.steer_ctx()
    replies = []
    ink = getattr(getattr(session, "weights", None), "inkling", None) is not None    # Inkling 2: 思考の欄を <think> … </think> で見せる(inkling.py)
    for i, col in enumerate(state.columns):
        state.histories[i].append({"role": "user", "content": user})
        vecs = session.vectors(col.recipe)[0] if col.recipe is not None else None
        ids = e.tokenize_prompt(e.render_chat(state.messages(i)))
        if len(ids) + max_tokens > e.n_ctx:
            raise GneError("CONTEXT_FULL", f"会話が長すぎます({len(ids)} トークン、n_ctx {e.n_ctx})",
                           "--n-ctx を大きくするか、会話をやり直してください(/reset)")
        if on_column:
            on_column(i, col)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        marks = None
        if ink:
            from .inkling import ReplyMarks
            marks = ReplyMarks(e)
        cb = None
        think_n = [0]
        if on_text is not None:
            def cb(_si, tok, _i=i, _dec=dec):
                mark = marks.piece(tok) if marks is not None else None
                if hide_reasoning and marks is not None:
                    if mark is not None:                       # 思考の始まり・終わり: 中身の代わりに長さだけ
                        _dec.decode(b"", final=True)
                        on_text(_i, "(思考中…" if marks.open else f" {think_n[0]} トークン)\n\n")
                        return
                    if marks.open:
                        think_n[0] += 1
                        _dec.decode(e.token_bytes(tok))
                        return
                piece = _dec.decode(b"", final=True) + mark if mark is not None else _dec.decode(e.token_bytes(tok))
                if piece:
                    on_text(_i, piece)
        when = (when_override or col.recipe.when) if col.recipe is not None else "all"
        r, _ = generate_when(e, ctx, state.messages(i), vecs, when, max_tokens=max_tokens, on_token=cb,
                             lora=session.lora_for(col.recipe), sampling=sampling, lora_when=lora_when)
        if on_text is not None:
            if hide_reasoning and marks is not None and marks.open:     # 回答の前で終わった
                dec.decode(b"", final=True)
                on_text(i, f" {think_n[0]} トークン、回答の前に上限で切れました)")
            tail = dec.decode(b"", final=True)
            if tail:
                on_text(i, tail)
        text, content, rt = r.text, r.text, None
        if marks is not None:
            from .inkling import answer_part, reasoning_tokens
            rt = reasoning_tokens(marks, r.token_ids)
            text = marks.render(e, r.token_ids)
            content = answer_part(text)                # 履歴には回答だけ(思考の欄は入れない)
            if hide_reasoning:
                text = content
        state.histories[i].append({"role": "assistant", "content": content})
        replies.append({"column": col.label, "text": text, "finish": r.finish, "n_gen": r.n_gen,
                        **({"answer": content, "reasoning_tokens": rt} if marks is not None else {})})
    return replies
