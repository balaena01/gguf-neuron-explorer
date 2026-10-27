"""−/0/+ の 3 列比較チャット。設計書 §6.10。

同じ会話履歴を列ごとに生成する。ユーザーの発話は全列で共有し、assistant の返答は列ごとに別に持つ。
control vector はコンテキスト単位なので、列は 1 本ずつ順に(set_cvec を切り替えて)生成する。
"""

from __future__ import annotations

import codecs
from collections.abc import Callable
from dataclasses import dataclass, field

from .steer import GneError, Recipe, Unit


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
    units = [Unit(u.layer, u.neuron, (abs(dose) if dose is not None else abs(u.dose)) * (1 if u.dose >= 0 else -1) * sign,
                  u.kind, u.seed) for u in recipe.units]
    return Recipe(units, recipe.model_id, recipe.dose_unit, recipe.name, recipe.when)


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
              when_override: str | None = None) -> list[dict]:
    """ユーザーの発話を全列に足し、列ごとに返答を生成して履歴に足す。on_text(列, 文字列の断片)で逐次表示できる。"""
    from .when import generate_when
    e = session.engine
    ctx = e.steer_ctx()
    replies = []
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
        cb = None
        if on_text is not None:
            def cb(_si, tok, _i=i, _dec=dec):
                piece = _dec.decode(e.token_bytes(tok))
                if piece:
                    on_text(_i, piece)
        when = (when_override or col.recipe.when) if col.recipe is not None else "all"
        r, _ = generate_when(e, ctx, state.messages(i), vecs, when, max_tokens=max_tokens, on_token=cb)
        if on_text is not None:
            tail = dec.decode(b"", final=True)
            if tail:
                on_text(i, tail)
        state.histories[i].append({"role": "assistant", "content": r.text})
        replies.append({"column": col.label, "text": r.text, "finish": r.finish, "n_gen": r.n_gen})
    return replies
