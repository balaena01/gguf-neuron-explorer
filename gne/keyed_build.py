"""鍵つきの書き込み(keyed write、CHANGE-07)の鍵の作成 `gne direction key` と、分離の確認 `gne direction key-check`。

数式・鍵のファイル・LoRA の行列は gne/keyed.py。ここは書き込み行列 W の入力活性 x の捕捉と、鍵 κ の作成・確認。

側 A(効かせたい状態)の作り方は 2 通り(どちらも同じ関数で、positions で決まる):

- (a) system / レシピの状態(mode "system"): 対照集合の system_a・recipe_a で作った文脈の活性を側 A、system_b・recipe_b の
  文脈の活性を側 B にする。推論時には system_a が無いので、この鍵は推論時にはふつう発火しない。推論時に自然に起きる状態を
  打ち消す(d = −r)用途や、レシピを当てた状態を鍵にする用途向け。
- (b) 推論時と同じ文脈の返事の位置(mode "same"、positions = before:<正規表現>): 側 B と同じ文脈(system なし、または
  system_b・recipe_b)でモデル自身の返事(生成するか、与えた返事を読ませる)を読み、正規表現に一致する箇所を書く直前の位置を
  側 A、同じ返事のそれ以外の位置(not-before:<正規表現>)を側 B にする。語尾・口調(ずんだもん・ツンデレ)の例はこちら。

位置(positions。列の番号 t の活性が t+1 のトークンを決める):

- response(既定): 返事の各トークンを読んだ位置(EOG は除く。列の位置 0 は attention sink なので使わない)。
- assistant: 生成プロンプト(assistant の見出し)のトークン。last: 生成プロンプトの最後のトークン。last:K: 最後の K トークン。
  この 3 つは返事を作らない(生成なし、速い)。
- before:<re>: 返事の中で正規表現に一致する箇所(重ならない一致すべて)の、始まりの文字を含むトークンを決める位置
  (そのトークンの 1 つ前の位置。返事の最初のトークンなら生成プロンプトの最後のトークン)。文字の位置はトークン片の
  バイト長の累積でトークンに戻す(日本語の 1 文字が複数のトークンにまたがることがある)。
- not-before:<re>: 返事の各トークンを決める位置のうち before:<re> 以外(Q 用の補集合)。
- skip=K は返事の先頭 K トークン(を読む位置・決める位置)を落とす。

Q(触りたくない状態の基底)の作り方 q_mode: means(群ごとの平均、側 B と dont_touch の群の数)/ examples(例ごとの平均、既定)/
samples(トークンの活性。q_max_samples まで間引き、keyed.orth_basis が Gram 行列の固有分解で基底を作る)。

擬似鍵(ランダム対照): 例ごとの平均を A・B まとめて、ラベルを例の単位で並べ替えて同じ手順で作った鍵(kappa_ctrl_L、N 本)。
key-check は本物の鍵の AUC が擬似鍵の AUC の p95 を超えるかを見る(beats_pseudo)。

書き込み先の差し替え口: 活性の捕捉は書き込み先(target)ごとの TargetCapture(names: 捕捉するテンソル名、rows: 選んだ位置の
入力 x とゲート、rms: post-norm の補正用の rms)にまとめてある:
- ffn: dense の ffn_down の入力。
- attn: 書き込み行列 attn_output / ssm_out の入力 = ヘッド出力の連結(CHANGE-04/06 の head_out と同じテンソル: attention の
  層は attn_gated(出力ゲートの後)か kqv_out、線形注意の層は final_output、bitnet は attn_sub_norm)。
- shared: MoE の層の shared expert(ffn_down_shexp)の入力。MoE の層で build_ffn が付ける名前(ffn_swiglu など)は
  shared expert のもの。ゲート(ffn_shexp_gate / shared_expert_gate_sigmoid、トークンごと)を gate として返す → 鍵の gate_a。
- expert:e: ffn_down_exps の専門家 e の入力(ffn_moe_swiglu など、選ばれた専門家の枠ごと)。ffn_moe_argsort で専門家 e が
  選ばれた位置だけに絞り、その枠の実効のルーターの重み(moe.moe_names の weights)を gate として返す → 鍵の gate_a。
rows が層ごとに行を絞る書き込み先(専門家)では、例の並びが層ごとに違う(collect_acts の index_l)。build_keys は層ごとの
例の数で擬似鍵の並べ替えと samples の例の番号を合わせる(全層で同じ数なら全層共通の並べ替え)。例のハッシュ(hashes_a・
hashes_b)は、どれかの層で行があった例すべて。専門家の鍵は側 A の位置でその専門家が選ばれた行だけでできるので、
expert_usage で側 A の位置でよく選ばれる専門家を見てから選ぶ。
"""

from __future__ import annotations

import bisect
import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import keyed as K
from .reverse import ContrastSet, _conv, check_questions, question_text
from .steer import GneError

Progress = Callable[[int, int, str], None]
Q_MODES = ("means", "examples", "samples")
SUBSETS = ("even", "odd", "all")
# 返事の書き出し: 返事の最初の START_TOKENS トークンを決める位置(生成プロンプトの最後のトークンと、返事の最初の 2 トークンを
# 読んだ位置)。書き出しの状態はどの質問でも似ているので、側 A がここに偏ると鍵がどの返事の書き出しにも合いやすい
# (verification.md F134・F138)。
START_TOKENS = 3
# 鍵の作成(mode same)で、μ に占める書き出しの位置の重み(側 A の例ごとの「書き出しの行 ÷ 行」の平均。μ は例ごとの平均の
# 平均)がこれ以上なら警告する。書き出しの状態の Q の外の成分が、側 A の残りの位置の成分と同じ大きさで直交すると、どの返事の
# 書き出しでも ρ ≈ w / (w² + (1 − w)²) になり、w = 0.25 で 0.4(下の START_RHO_WARN)に当たる。
START_SHARE_WARN = 0.25
# key-check で、側 B・dont_touch の返事の書き出しの ρ の中央値(書き出しの位置ごと)がこれ以上なら警告する。実測では書き出しで
# 合ってしまった鍵が 0.65〜1.6(F134 の dont_touch が計算・事実だけの鍵、F138 の attn の鍵)、書き出しを Q に入れた鍵と ffn の鍵は
# −0.28〜0.26 だった(ρ = 1 が側 A の平均の書き込み)。
START_RHO_WARN = 0.4
# 側 A が少ないときの警告: 位置のある例が MIN_A_EXAMPLES 未満か、行(位置)が MIN_A_ROWS 未満。μ が数位置の活性の平均になり、
# 鍵が状態ではなくそのトークン・例の本文を拾いやすく、key-check の選択性も数例で揺れる。専門家の鍵は層ごとの行(rows_l)で見る。
MIN_A_EXAMPLES = 2
MIN_A_ROWS = 8
# 鍵の作成で、dont_touch の各群の質問の「返事の冒頭」を、その群の positions とは別の行として Q に入れる(既定。切るのは
# build_keys の dont_touch_start=False か、対照集合の dont_touch_start: false)。入れるのは返事の最初のトークンを決める位置
# = 生成プロンプトの最後のトークン(positions の last と同じ。q_mode examples では例ごとに 1 行)。返事の最初の 2・3 トークンを
# 決める位置(上の START_TOKENS の残り)は入れない: Qwen3-0.6B のツンデレでは、側 A の書き出しの位置はどれも最初のトークンの
# 直前で、dont_touch の返事の書き出しで鍵が合ったのもそこだけだった(2・3 番目は入れなくても ρ の中央値 0.13 以下)。3 位置の
# 平均を 1 行にすると、最初のトークンの直前の状態を打ち消し切れず、保留した質問の書き出しの ρ が 0.25〜1.15 残った
# (最初のトークンの直前だけなら 0.05〜0.55、docs/verification.md F142)。
# 群の positions が生成プロンプトの位置(last・last:K・assistant)なら、その群の行がもう返事の冒頭の状態なので重ねない。
# 側 B(鍵を作った場面の質問)の返事の冒頭は入れない(側 A が返事の冒頭にあるとき、側 A そのものを打ち消してしまう)。
DT_START_POSITIONS = "last"
_POS_RE = re.compile(r"(response|assistant|last)|last:(\d+)")
_KEY_NAME_RE = re.compile(r"[\w.][\w.\-]*")
_DT_KEYS = {"name", "system", "prompts", "replies", "positions"}
_PROMPT_END = ("last", "assistant")          # 生成プロンプトの位置(返事を作らない)。この群の行は返事の冒頭の状態そのもの


# --- 位置 ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Positions:
    """捕捉する位置の指定(parse_positions で作る)。"""
    kind: str                       # response / assistant / last / before / not-before
    k: int = 1                      # last:K
    pattern: str | None = None      # before / not-before の正規表現
    text: str = "response"          # 元の書き方(鍵の npz に残す)

    @property
    def needs_reply(self) -> bool:
        """返事(生成するか、与えたもの)が要るか。assistant・last・last:K は生成プロンプトだけで測る。"""
        return self.kind in ("response", "before", "not-before")


def parse_positions(spec) -> Positions:
    """positions の書き方 → Positions。response / assistant / last / last:K / before:<re> / not-before:<re>。"""
    if isinstance(spec, Positions):
        return spec
    s = "response" if spec is None else str(spec)
    if s.lstrip().startswith(("before:", "not-before:")):
        kind, _, pat = s.lstrip().partition(":")            # 正規表現の中の空白はそのまま
    else:
        m = _POS_RE.fullmatch(s.strip())
        if not m:
            raise GneError("BAD_ARGS", f"positions は response / assistant / last / last:K / before:<正規表現> / "
                                       f"not-before:<正規表現> です({s!r})")
        if m.group(2) is None:
            return Positions(m.group(1), text=m.group(1))
        k = int(m.group(2))
        if k < 1:
            raise GneError("BAD_ARGS", f"last:K の K は 1 以上です({s!r})")
        return Positions("last", k=k, text=f"last:{k}")
    if not pat:
        raise GneError("BAD_ARGS", f"positions {s!r} に正規表現がありません")
    try:
        rx = re.compile(pat)
    except re.error as e:
        raise GneError("BAD_ARGS", f"positions {s!r} の正規表現が不正です: {e}") from e
    if rx.fullmatch(""):
        raise GneError("BAD_ARGS", f"positions {s!r} の正規表現が空の文字列に一致します(位置が決まりません)")
    return Positions(kind, pattern=pat, text=f"{kind}:{pat}")


def byte_offsets(pieces: Sequence[bytes]) -> list[int]:
    """トークン片(バイト列)の累積バイト位置。out[i] = トークン i の始まり、out[-1] = 全体の長さ(長さ len(pieces) + 1)。"""
    out = [0]
    for p in pieces:
        out.append(out[-1] + len(p))
    return out


def match_tokens(pieces: Sequence[bytes], pattern) -> list[int]:
    """正規表現に一致する箇所(重ならない一致すべて、長さ 0 の一致は除く)の始まりの文字を含むトークンの番号(昇順、重複なし)。
    文字はトークン片を連結したバイト列の UTF-8 で数える(途中で切れたバイトは surrogateescape で 1 文字にして位置を保つ)。
    1 文字が複数のトークンにまたがる(バイト単位の BPE)ときは、その文字の最初のバイトを含むトークン。"""
    rx = re.compile(pattern) if isinstance(pattern, str) else pattern
    offs = byte_offsets(pieces)
    text = b"".join(pieces).decode("utf-8", "surrogateescape")
    cb = [0]                                                   # 文字の番号 → バイト位置
    for ch in text:
        cb.append(cb[-1] + len(ch.encode("utf-8", "surrogateescape")))
    out = set()
    for m in rx.finditer(text):
        if m.end() == m.start():
            continue
        # 同じバイト位置で始まるトークンが複数(長さ 0 の片)なら、中身のある最後のものを採る
        out.add(bisect.bisect_right(offs, cb[m.start()]) - 1)
    return sorted(out)


def select_positions(engine, reply_ids: Sequence[int], positions="response", skip: int = 0, *, n_prompt: int = 0,
                     start: int | None = None, pieces: Sequence[bytes] | None = None) -> np.ndarray:
    """会話(プロンプト n_prompt トークン + 返事 reply_ids)の中で捕捉する位置(列の番号、昇順の int64)。
    返事の末尾の EOG は除く。n_prompt = 0 なら返事の中の番号で返す(before が返事の最初のトークンに当たると -1 =
    生成プロンプトの最後のトークン)。n_prompt > 0 なら列の位置 0(attention sink)は除く。
    start は生成プロンプトの始まり(assistant で使う。無ければ最後のトークンだけ)。pieces は返事のトークン片(省くと
    engine.token_bytes で引く)。engine は is_eog と token_bytes だけを使う。"""
    pos = parse_positions(positions)
    r = list(reply_ids)
    while r and engine.is_eog(r[-1]):
        r.pop()
    R, P, skip = len(r), int(n_prompt), max(0, int(skip or 0))
    if pos.kind == "response":
        idx = [P + j for j in range(skip, R)]
    elif pos.kind == "assistant":
        s0 = P - 1 if start is None else int(start)
        idx = list(range(max(s0, 0), P))
    elif pos.kind == "last":
        idx = list(range(max(P - pos.k, 0), P))
    else:
        pc = [engine.token_bytes(t) for t in r] if pieces is None else list(pieces)[:R]
        hit = {j for j in match_tokens(pc, pos.pattern) if j >= skip}
        js = sorted(hit) if pos.kind == "before" else [j for j in range(skip, R) if j not in hit]
        idx = [P + j - 1 for j in js]
    if P > 0:
        idx = [i for i in idx if i >= 1]
    return np.asarray(sorted(set(idx)), dtype=np.int64)


def start_positions(engine, reply_ids: Sequence[int], positions="response", skip: int = 0, *, n_prompt: int,
                    k: int = START_TOKENS, pieces: Sequence[bytes] | None = None) -> np.ndarray:
    """返事の書き出しの位置(列の番号、昇順の int64): 返事の最初の k トークンを決める位置 P−1 … P−2+k(P = n_prompt。
    生成プロンプトの最後のトークン P−1 はいつも、ほかは返事の長さまで)。書き出しの位置 j = 位置 − (P−1) は返事の j 番目の
    トークンを決める。positions の種類で絞る:
    - before: 自分の位置(一致の直前、skip より後)のうち書き出しに入るもの(鍵の作成で、側 A が書き出しに偏っていないか)。
    - not-before: 書き出しから一致の位置を除いたもの(skip に関係なく。推論時は書き出しも通るので、key-check で測る)。
    - response・assistant・last: 書き出しのすべて(response の自分の位置は返事のトークンを読んだ位置なので P−1 を含まない)。"""
    pos = parse_positions(positions)
    r = list(reply_ids)
    while r and engine.is_eog(r[-1]):
        r.pop()
    R, P = len(r), int(n_prompt)
    js = [0] + list(range(1, min(int(k), R)))
    if pos.pattern:
        pc = [engine.token_bytes(t) for t in r] if pieces is None else list(pieces)[:R]
        hit = set(match_tokens(pc, pos.pattern))
        js = [j for j in js if j in hit and j >= max(0, int(skip or 0))] if pos.kind == "before" else [j for j in js if j not in hit]
    return np.asarray([P - 1 + j for j in js if P - 1 + j >= 1], dtype=np.int64)


# --- 鍵の対照集合(dont_touch つき)-------------------------------------------------------------------------

@dataclass
class DontTouch:
    """触りたくない状態の追加の群(Q に足す)。system を省く(null)と system なしの会話(素の口調)。
    空の文字列は空の system メッセージになる(ContrastSet と同じ)。replies を与えると生成せずにその返事を読ませる。"""
    name: str
    system: str | None
    prompts: list
    replies: list[str] | None = None
    positions: str | None = None             # 省くと側 B の位置

    def convs(self, idx: Sequence[int]) -> list[list[dict]]:
        return [_conv(self.system, self.prompts[i]) for i in idx]


@dataclass
class KeySpec:
    """load_key_spec の中身に、鍵の作成の既定(positions・positions_b・skip)と対照集合に書いた返事を足したもの。"""
    contrast: ContrastSet
    dont_touch: list[DontTouch] = field(default_factory=list)
    positions: str | None = None
    positions_b: str | None = None
    skip: int | None = None
    replies: dict[str, list[str] | None] = field(default_factory=dict)   # "same"(replies)・"A"(replies_a)・"B"(replies_b)
    dont_touch_start: bool | None = None     # dont_touch の質問の返事の冒頭を Q に入れるか(None = 既定の True)


def _rel(base: Path | None, s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() or base is None else base / p


def _reply_list(spec, prompts: list, base: Path | None, where: str) -> list[str] | None:
    """返事: None / リスト / {プロンプト: 返事} / ファイルのパス(対照集合のファイルからの相対、writers.load_replies の形)。"""
    if spec is None:
        return None
    if isinstance(spec, str):
        from .writers import load_replies
        p = _rel(base, spec)
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"{where}: 返事のファイル {spec} がありません")
        return load_replies(p, prompts)
    if isinstance(spec, dict):
        keys = [question_text(q) for q in prompts]
        missing = [k for k in keys if k not in spec]
        if missing:
            raise GneError("BAD_CONTRAST", f"{where}: 返事の無いプロンプトがあります({missing[0]!r} など {len(missing)} 個)")
        return [str(spec[k]) for k in keys]
    if isinstance(spec, list):
        if len(spec) != len(prompts):
            raise GneError("BAD_CONTRAST", f"{where}: 返事の数 {len(spec)} がプロンプトの数 {len(prompts)} と違います")
        return [str(x) for x in spec]
    raise GneError("BAD_CONTRAST", f"{where}: 返事はリスト・{{プロンプト: 返事}}・ファイルのパスのどれかです")


def _parse_dont_touch(raw, base: Path | None, where: str) -> list[DontTouch]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise GneError("BAD_CONTRAST", f"{where}: dont_touch は群のリストです(各群は {{system?, prompts, replies?, positions?}})")
    out: list[DontTouch] = []
    for i, g in enumerate(raw):
        w = f"{where}: dont_touch の {i + 1} 番目"
        if not isinstance(g, dict):
            raise GneError("BAD_CONTRAST", f"{w} は {{system?, prompts, replies?, positions?}} の表です")
        extra = set(g) - _DT_KEYS
        if extra:
            raise GneError("BAD_CONTRAST", f"{w} に知らないキー {sorted(extra)} があります(name・system・prompts・replies・positions)")
        system = g.get("system")
        if system is not None and not isinstance(system, str):
            raise GneError("BAD_CONTRAST", f"{w}: system は文字列です(system なしにするにはキーを省きます)")
        pr = g.get("prompts")
        if isinstance(pr, str):
            from .api import load_prompt_list
            p = _rel(base, pr)
            try:
                prompts = list(load_prompt_list(p))
            except GneError as e:
                raise GneError(e.code, f"{w}: {e.message}", e.hint) from e
        elif isinstance(pr, list) and pr:
            check_questions(pr, f"{w}: prompts")
            prompts = list(pr)
        else:
            raise GneError("BAD_CONTRAST", f"{w}: prompts は質問のリストか、プロンプトの組の YAML のパス(対照集合のファイルからの相対)です")
        replies = _reply_list(g.get("replies"), prompts, base, w)
        pos = g.get("positions")
        if pos is not None:
            pos = parse_positions(pos).text
        out.append(DontTouch(str(g.get("name") or f"dt{i}"), system, prompts, replies, pos))
    names = [d.name for d in out]
    if len(set(names)) != len(names):
        raise GneError("BAD_CONTRAST", f"{where}: dont_touch の name が重なっています({names})")
    return out


def read_key_spec(path: str | Path) -> KeySpec:
    """鍵の対照集合の YAML を読む。ContrastSet の書き方に次を足したもの(どれも任意):

    - dont_touch: [{name?, system?, prompts: [...] | パス, replies?: [...] | パス, positions?}](パスは対照集合のファイルからの相対)
    - dont_touch_start: dont_touch の質問の返事の冒頭も Q に入れるか(既定 true。build_keys の引数が優先)
    - positions / positions_b / skip: 鍵の作成の既定(build_keys の引数が優先)
    - replies(mode same の返事)/ replies_a / replies_b: 与える返事(リストかパス)

    側 A・側 B の文脈が同じ(system も recipe も質問も同じ)対照集合も読める(before:<re> の位置で側 A を作る mode same 用)。"""
    p = Path(path)
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"対照集合 {p} がありません")
    text = p.read_text(encoding="utf-8")
    d = yaml.safe_load(text) or {}
    if not isinstance(d, dict):
        raise GneError("BAD_CONTRAST", f"{p}: 対照集合は YAML の表です")
    try:
        cs = ContrastSet.load(p)
    except GneError as e:
        same_ctx = (e.code == "BAD_CONTRAST" and d.get("recipe_a") is None and d.get("recipe_b") is None
                    and d.get("prompts_a") is None and d.get("prompts_b") is None and d.get("system_a") == d.get("system_b"))
        if not same_ctx or not d.get("user_prompts"):
            raise
        check_questions(d.get("user_prompts"), f"{p}: user_prompts")
        sb = d.get("system_b")
        cs = ContrastSet(d.get("name") or p.stem, sb, sb, list(d["user_prompts"]),
                         hashlib.sha256(text.encode()).hexdigest()[:16], None, None, p.resolve().parent, None, None)
    base = p.resolve().parent
    for k in ("positions", "positions_b"):
        if d.get(k) is not None:
            parse_positions(d[k])
    skip = d.get("skip")
    if skip is not None and (not isinstance(skip, int) or skip < 0):
        raise GneError("BAD_CONTRAST", f"{p}: skip は 0 以上の整数です")
    dts = d.get("dont_touch_start")
    if dts is not None and not isinstance(dts, bool):
        raise GneError("BAD_CONTRAST", f"{p}: dont_touch_start は true か false です")
    replies = {"same": _reply_list(d.get("replies"), cs.prompts("B"), base, f"{p}: replies"),
               "A": _reply_list(d.get("replies_a"), cs.prompts("A"), base, f"{p}: replies_a"),
               "B": _reply_list(d.get("replies_b"), cs.prompts("B"), base, f"{p}: replies_b")}
    return KeySpec(cs, _parse_dont_touch(d.get("dont_touch"), base, str(p)), d.get("positions"), d.get("positions_b"),
                   skip, replies, dts)


def load_key_spec(path: str | Path) -> tuple[ContrastSet, list[DontTouch]]:
    """鍵の対照集合 → (ContrastSet, dont_touch の群)。既定の位置や返事も要るときは read_key_spec。"""
    s = read_key_spec(path)
    return s.contrast, s.dont_touch


def _as_spec(contrast) -> KeySpec:
    if isinstance(contrast, KeySpec):
        return contrast
    if isinstance(contrast, ContrastSet):
        return KeySpec(contrast)
    if isinstance(contrast, tuple) and len(contrast) == 2 and isinstance(contrast[0], ContrastSet):
        return KeySpec(contrast[0], list(contrast[1]))
    return read_key_spec(contrast)


# --- 介入(鍵を他の介入を当てた状態で作る・確かめる)-----------------------------------------------------------

def intervention(session, iv, warn: Callable[[str], None] | None = None) -> tuple[dict | None, list | None, dict | None]:
    """iv → (control vector, LoRA, 記録用の説明)。iv はレシピ(Recipe)、ユニットの書き方のリスト、レシピの YAML のパスか
    保存したレシピの名前、または (vecs, lora) の組。生成と捕捉の全位置に当てる(when は使わない)。専門家全体の gain・route
    (スケールパッチ)は当てられない。"""
    if iv is None:
        return None, None, None
    if isinstance(iv, tuple) and len(iv) == 2 and not isinstance(iv[0], str):
        vecs, lora = iv
        return (vecs or None), (list(lora) if lora else None), {"vectors": sorted(int(L) for L in (vecs or {})),
                                                                 "lora": len(lora or [])}
    if hasattr(iv, "units"):
        r = iv
    elif isinstance(iv, (list, tuple)):
        r = session.recipe(units=[str(u) for u in iv], name="iv")
    else:
        p = Path(str(iv))
        r = session.recipe(path=p if p.suffix in (".yaml", ".yml") and p.exists() else session.recipe_path(str(iv)))
    if session.patches_for(r) is not None:
        raise GneError("BAD_ARGS", "iv: 専門家全体の gain・route(スケールパッチ)は鍵の作成・確認には当てられません")
    if getattr(r, "when", "all") not in (None, "all"):
        (warn or session.warn)(f"iv: when={r.when} は鍵の捕捉では使いません(生成と捕捉の全位置に当てます)")
    vecs = session.vectors(r)[0] if r.add_units else None
    lora = session.lora_for(r)
    return (vecs or None), (lora or None), r.to_dict()


def _combine(side_iv, vecs, lora):
    from .reverse import combine
    return combine(side_iv, vecs, lora)


# --- 書き込み先ごとの捕捉(差し替え口)-------------------------------------------------------------------------

class TargetCapture:
    """書き込み先ごとの活性の捕捉。派生は names・rows(と post-norm なら rms)を実装して _TARGETS に登録する。
    names(seen, L) は捕捉するテンソル名(list_names で見えた名前 seen から選ぶ)、rows(got, L, sel) は選んだ位置 sel
    (列の番号)での書き込み先の入力 x(n × in、float32)と、書き込みに掛かるゲート((n,)、無ければ None)。専門家の鍵なら
    rows が専門家 e の選ばれた位置だけに絞る(n は len(sel) より少なくてよい)。rms(x, L) は post-norm の層で
    rms(W x)(eps 込み、行ごと)、post-norm でなければ None。"""
    target = "?"

    def __init__(self, session, layers: Sequence[int]):
        self.session, self.layers = session, list(layers)
        self.in_dim: dict[int, int] = {}                     # 層 → 書き込み先の入力の次元(派生が埋める)

    def bind(self, seen: set[str]) -> None:
        """捕捉できる名前を見て準備する(名前が無ければ ARCH_UNSUPPORTED)。"""

    def names(self, L: int) -> list[str]:
        raise NotImplementedError

    def rows(self, got: dict[str, np.ndarray], L: int, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def post_norm(self, L: int) -> bool:
        return False

    def rms(self, x: np.ndarray, L: int) -> np.ndarray | None:
        return None

    def token_rows(self, nm: str, arr: np.ndarray, T: int) -> np.ndarray:
        """捕捉したテンソル nm を (T, 幅) の行に直す(既定はそのまま)。3 次元で出る名前([幅/ヘッド, ヘッド, T] など)を
        トークンの行にそろえる派生が上書きする。"""
        return arr


def rms_rows(x: np.ndarray, W: np.ndarray, *, scale: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    """post-norm の補正用: 行ごとの rms(y) = sqrt(mean(y²) + eps)、y = scale · W x(x は (n, in)、W は (out, in))。"""
    y = (np.asarray(x, dtype=np.float32) @ np.asarray(W, dtype=np.float32).T).astype(np.float64) * float(scale)
    return np.sqrt(np.mean(np.square(y), axis=1) + float(eps))


class FfnCapture(TargetCapture):
    """ffn: dense の ffn_down の入力(ニューロン活性、n_ff)。writers・direction と同じ pick_act_name の名前。bitnet は
    ffn_sub_norm の後が ffn_down の入力なので ffn_sub_norm-L を採る。鍵は補正しない生の x で作る(LoRA が読むのはこれ)。"""
    target = "ffn"

    def __init__(self, session, layers):
        super().__init__(session, layers)
        w = session.weights
        self.in_dim = {L: K.writer_in_dim(w, L, "ffn") for L in self.layers}
        self.gamma = {L: K.post_norm_gamma(w, L, "ffn") for L in self.layers}
        self.scale = dict(w.meta.ffn_down_scale or {})
        self._name: dict[int, str] = {}

    def bind(self, seen: set[str]) -> None:
        from .capture import pick_act_name
        from .moe import moe_names
        for L in self.layers:
            if "argsort" in moe_names(seen, L):
                raise GneError("BAD_ARGS", f"層 {L} は MoE です。ffn の鍵は dense の層だけです")
            nm = f"ffn_sub_norm-{L}" if f"ffn_sub_norm-{L}" in seen else pick_act_name(seen, L)
            if nm is None:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L} の ffn_down の入力(ニューロン活性)のテンソルが捕捉できません")
            self._name[L] = nm

    def names(self, L: int) -> list[str]:
        return [self._name[L]]

    def rows(self, got, L, sel):
        x = got[self._name[L]]
        if x.shape[1] != self.in_dim[L]:
            raise GneError("ARCH_UNSUPPORTED", f"層 {L}: 捕捉した {self._name[L]} の幅 {x.shape[1]} が ffn_down の入力の次元 "
                                               f"{self.in_dim[L]} と違います")
        return x[sel], None

    def post_norm(self, L: int) -> bool:
        return self.gamma.get(L) is not None

    def rms(self, x, L):
        if self.gamma.get(L) is None:
            return None
        from .writers import _post_norm
        _, eps = _post_norm(self.session.weights, L)
        return rms_rows(x, self.session.weights.down_matrix(L), scale=self.scale.get(L, 1.0), eps=eps)


class AttnCapture(TargetCapture):
    """attn: ヘッドの書き込み行列(attention の層は attn_output、線形注意の層は ssm_out)の入力 = ヘッド出力の連結
    (n_head·head_dim)。名前は CHANGE-04/06 の head_out(attn.head_out_names: attention の層は attn_gated(Qwen3.5 系の
    出力ゲートの後)か kqv_out、線形注意の層(Gated DeltaNet)は final_output(gated norm の後))。どれも llama.cpp で
    build_lora_mm(wo / ssm_out, x) に入る x そのもので、keyed の LoRA の A が読むのと同じ。bitnet は kqv_out の後に
    attn_sub_norm があるので attn_sub_norm-L を採る(ffn の ffn_sub_norm と同じ)。
    post-norm の arch(keyed.post_norm_gamma が断る)、書き込み行列の .scale の掛かり方が分からない組み合わせ
    (keyed.writer_out_scale が断る)、ヘッドの書き込み行列が無い層(Mamba 系など)は ARCH_UNSUPPORTED。ゲートは無い。"""
    target = "attn"

    def __init__(self, session, layers):
        super().__init__(session, layers)
        w = session.weights
        for L in self.layers:
            li = w.arch.layers[L]
            if li.mixer not in ("attention", "linear_attention") or not li.head_writer or not li.head_dim:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L}({li.mixer})にはヘッドの出力を残差に書き込む重み行列(attn_output / ssm_out)が"
                                                   "ありません。attn の鍵は attention・線形注意の層だけです")
            K.post_norm_gamma(w, L, "attn")                       # post-norm の arch は断る
            K.writer_out_scale(w, L, "attn")                      # .scale と LoRA の掛かり方が分からなければ断る
            n_in = K.writer_in_dim(w, L, "attn")
            if n_in != li.n_head * li.head_dim:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L}: {li.head_writer} の入力 {n_in} が n_head·head_dim = "
                                                   f"{li.n_head}·{li.head_dim} と違います")
            self.in_dim[L] = n_in
        self._name: dict[int, str] = {}
        self._layer_of: dict[str, int] = {}

    def bind(self, seen: set[str]) -> None:
        from .attn import head_out_names
        heads = head_out_names(self.session, seen, self.layers)
        arch = self.session.weights.arch
        self._name, self._layer_of = {}, {}
        for L in self.layers:
            nm = f"attn_sub_norm-{L}" if (arch.layers[L].mixer == "attention" and f"attn_sub_norm-{L}" in seen) else heads.get(L)
            if nm is None:
                li = arch.layers[L]
                raise GneError("ARCH_UNSUPPORTED", f"層 {L} の {li.head_writer} の入力(ヘッド出力の連結 "
                                                   f"{' / '.join(f'{b}-{L}' for b in li.head_out)})が捕捉できません")
            self._name[L] = nm
            self._layer_of[nm] = L

    def names(self, L: int) -> list[str]:
        return [self._name[L]]

    def token_rows(self, nm, arr, T):
        L = self._layer_of.get(nm)
        if L is None or arr.shape[0] == T:
            return arr
        n = self.in_dim[L]
        if arr.size == T * n:                    # [head_dim, n_head, T] → (T, n_head·head_dim)(ヘッド h は列 h·head_dim から)
            return arr.reshape(T, n)
        return arr

    def rows(self, got, L, sel):
        x = got[self._name[L]]
        if x.shape[1] != self.in_dim[L]:
            raise GneError("ARCH_UNSUPPORTED", f"層 {L}: 捕捉した {self._name[L]} の幅 {x.shape[1]} が重み行列の入力の次元 "
                                               f"{self.in_dim[L]} と違います")
        return x[sel], None


class SharedCapture(TargetCapture):
    """shared: MoE の層の shared expert の書き込み行列 ffn_down_shexp の入力(n_ff_shexp)。MoE の層では build_ffn の
    名前(capture.pick_act_name の ffn_swiglu など)は shared expert のもの(専門家は ffn_moe_*)。幅が n_ff_shexp と違えば
    ARCH_UNSUPPORTED。出力にゲート(ffn_gate_inp_shexp の sigmoid、トークンごと)が掛かる arch(qwen2moe・qwen3next・
    qwen35moe・qwen4exp)はゲートの値を gate として返す(鍵の gate_a。write_vector が B を割って、側 A の平均のゲートで
    α を書くようにする)。ゲートの捕捉名が見つからなければ gate_a を保存しない(書く大きさは α·ゲート、keyed_plan が警告)。
    post-norm のある MoE と ffn_down_shexp.scale は ARCH_UNSUPPORTED(keyed.post_norm_gamma・writer_out_scale)。"""
    target = "shared"

    def __init__(self, session, layers):
        super().__init__(session, layers)
        w = session.weights
        self.gated: dict[int, bool] = {}
        for L in self.layers:
            K.writer_tensor_name(w, L, "shared")                 # shared expert が無い層は BAD_RECIPE
            K.post_norm_gamma(w, L, "shared")
            K.writer_out_scale(w, L, "shared")
            self.in_dim[L] = K.writer_in_dim(w, L, "shared")
            self.gated[L] = w.has_tensor(f"blk.{L}.ffn_gate_inp_shexp.weight")
        self._name: dict[int, str] = {}
        self._gate: dict[int, str | None] = {}

    def bind(self, seen: set[str]) -> None:
        from .capture import pick_act_name
        from .moe import moe_names
        missing = []
        for L in self.layers:
            if "argsort" not in moe_names(seen, L):
                raise GneError("ARCH_UNSUPPORTED", f"層 {L} の MoE の捕捉名(ffn_moe_argsort)が見えません。shared の鍵は MoE の層だけです")
            nm = pick_act_name(seen, L)
            if nm is None:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L} の shared expert の活性(ffn_down_shexp の入力)が捕捉できません")
            self._name[L] = nm
            g = next((f"{b}-{L}" for b in K.SHEXP_GATE_NAMES if f"{b}-{L}" in seen), None) if self.gated[L] else None
            if self.gated[L] and g is None:
                missing.append(L)
            self._gate[L] = g
        if missing:
            self.session.warn(f"L{missing}: shared expert のゲートの捕捉名({' / '.join(K.SHEXP_GATE_NAMES)})が見えないので、"
                              "鍵に gate_a を保存しません(書く大きさは α·ゲート になります)")

    def names(self, L: int) -> list[str]:
        return [self._name[L]] + ([self._gate[L]] if self._gate[L] else [])

    def rows(self, got, L, sel):
        x = got[self._name[L]]
        if x.shape[1] != self.in_dim[L]:
            raise GneError("ARCH_UNSUPPORTED", f"層 {L}: 捕捉した {self._name[L]} の幅 {x.shape[1]} が ffn_down_shexp の入力の次元 "
                                               f"{self.in_dim[L]} と違います(shared expert の活性ではありません)")
        g = self._gate[L]
        return x[sel], (None if g is None else np.asarray(got[g], dtype=np.float32).reshape(-1)[sel])


class ExpertCapture(TargetCapture):
    """expert:e: ffn_down_exps の専門家 e の入力(n_ff_exp)。捕捉は選ばれた専門家の枠ごと(ffn_moe_swiglu など
    [n_ff_exp, n_used, T])と、選んだ専門家の番号(ffn_moe_argsort の先頭 n_used 列)、実効のルーターの重み
    (moe.moe_names の weights、[n_used, T])。rows は選んだ位置のうち専門家 e が選ばれたトークンだけを返し(層ごとに行の数が
    違う → collect_acts の index_l)、gate にその枠のルーターの重みを返す(鍵の gate_a。専門家の出力はルーターの重みを
    掛けてから足されるので、write_vector が B を割って、側 A の平均の重みで α を書くようにする)。ルーターの重みを入力側に
    掛ける arch(llama4)は gate を返さない(LoRA の出力に重みは掛からない)。post-norm のある MoE と ffn_down_exps.scale は
    ARCH_UNSUPPORTED。"""

    def __init__(self, session, layers, expert: int):
        super().__init__(session, layers)
        w = session.weights
        self.expert = int(expert)
        self.target = K.target_of("expert", self.expert)
        self.n_used: dict[int, int] = {}
        for L in self.layers:
            li = w.moe_layer(L)                                    # MoE でない層は BAD_RECIPE
            K.writer_tensor_name(w, L, "expert", self.expert)      # 専門家の番号の範囲
            K.post_norm_gamma(w, L, "expert")
            K.writer_out_scale(w, L, "expert", self.expert)
            self.in_dim[L] = K.writer_in_dim(w, L, "expert", self.expert)
            self.n_used[L] = int(li.n_expert_used)
            if not self.n_used[L]:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L} の選ぶ専門家の数(expert_used_count)が分かりません")
        self.weight_before = w.meta.arch in K._ROUTER_BEFORE_FFN
        self._nm: dict[int, dict[str, str]] = {}
        self._kind_of: dict[str, tuple[int, str]] = {}

    def bind(self, seen: set[str]) -> None:
        from .moe import moe_names
        self._nm, self._kind_of = {}, {}
        for L in self.layers:
            nm = moe_names(seen, L)
            need = ("act", "argsort") if self.weight_before else ("act", "argsort", "weights")
            miss = [k for k in need if k not in nm]
            if miss:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L} の MoE の捕捉名({', '.join(miss)})が見えません({sorted(nm.values())})")
            self._nm[L] = {k: nm[k] for k in need}
            for k, v in self._nm[L].items():
                self._kind_of[v] = (L, k)

    def names(self, L: int) -> list[str]:
        return list(self._nm[L].values())

    def token_rows(self, nm, arr, T):
        L, k = self._kind_of.get(nm, (None, None))
        if L is None or arr.shape[0] == T:
            return arr
        u = self.n_used[L]
        if arr.shape[0] != T * u:
            raise GneError("ARCH_UNSUPPORTED", f"{nm} の形 {arr.shape} が (トークン {T} × 専門家 {u}, 幅) と違います")
        return arr.reshape(T, -1)                                  # (T·n_used, 幅) → (T, n_used·幅)(重みは (T, n_used))

    def rows(self, got, L, sel):
        from .moe import selected
        nm, u = self._nm[L], self.n_used[L]
        act = got[nm["act"]]
        T = act.shape[0]
        act = act.reshape(T, u, -1)
        if act.shape[2] != self.in_dim[L]:
            raise GneError("ARCH_UNSUPPORTED", f"層 {L}: 捕捉した {nm['act']} の幅 {act.shape[2]} が ffn_down_exps の入力の次元 "
                                               f"{self.in_dim[L]} と違います")
        top = selected(np.asarray(got[nm["argsort"]]), u)[sel]   # (len(sel), n_used)
        ti, tj = np.nonzero(top == self.expert)                  # 1 トークンで同じ専門家は 1 枠だけ
        x = act[sel[ti], tj]
        if "weights" not in nm:
            return x, None
        wts = np.asarray(got[nm["weights"]], dtype=np.float32).reshape(T, u)
        return x, wts[sel[ti], tj]


_TARGETS: dict[str, type[TargetCapture]] = {"ffn": FfnCapture, "attn": AttnCapture, "shared": SharedCapture,
                                            "expert": ExpertCapture}


def target_capture(session, target: str, layers: Sequence[int]) -> TargetCapture:
    """書き込み先 target(ffn / attn / shared / expert:e)の捕捉。"""
    kind, expert = K.parse_target(target)
    cls = _TARGETS.get(kind)
    if cls is None:
        raise GneError("ARCH_UNSUPPORTED", f"鍵の作成(活性の捕捉)は {' / '.join(_TARGETS)} だけです(target {target})")
    return cls(session, layers, expert) if kind == "expert" else cls(session, layers)


# --- 捕捉 ------------------------------------------------------------------------------------------------

class _Reservoir:
    """トークンの行を cap 行まで一様に残す(reservoir sampling)。乱数は数えた行の数だけで決まるので、同じ seed なら
    層が違っても同じ行を残す。"""

    def __init__(self, cap: int, seed: int):
        self.cap, self.rng, self.n = int(cap), np.random.default_rng(seed), 0
        self.rows: list[np.ndarray] = []
        self.ex: list[int] = []

    def add(self, x: np.ndarray, ex: int) -> None:
        for row in x:
            if len(self.rows) < self.cap:
                self.rows.append(np.array(row, dtype=np.float32))
                self.ex.append(ex)
            else:
                j = int(self.rng.integers(0, self.n + 1))
                if j < self.cap:
                    self.rows[j] = np.array(row, dtype=np.float32)
                    self.ex[j] = ex
            self.n += 1

    def array(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.rows:
            return np.zeros((0, width), np.float32), np.zeros(0, np.int64)
        return np.stack(self.rows), np.asarray(self.ex, dtype=np.int64)


def _capture(e):
    from .capture import Capture
    cap = getattr(e, "_keyed_capture", None)
    if cap is None:
        cap = e._keyed_capture = Capture()
    return cap


def check_capture(cap, wanted) -> None:
    """捕捉で回収できなかったテンソル(F32 以外・非連続・コールバックの例外)があれば ARCH_UNSUPPORTED。Capture は
    回収できない名前を skipped に入れて黙って飛ばすので、鍵の作成では必ずこれで確かめる(レビュー P8)。"""
    bad = {k: v for k, v in cap.skipped.items() if k in wanted or k.startswith("<")}
    if bad:
        k, v = next(iter(bad.items()))
        raise GneError("ARCH_UNSUPPORTED", f"活性 {k} を捕捉できません({v})", "F32 で計算されるバックエンド・型で読み込んでください")


def _as_conv(c) -> list[dict]:
    return [{"role": "user", "content": c}] if isinstance(c, str) else list(c)


def collect_acts(session, convs, *, layers, positions="response", skip: int = 0, max_tokens: int = 96,
                 replies: Sequence[str] | None = None, iv=None, target: str = "ffn", samples: int = 0,
                 project: Mapping[int, np.ndarray] | None = None, rms_groups: Sequence[str] = ("A",), seed: int = 0,
                 start: int = 0, tick: Callable[[str], None] | None = None) -> dict:
    """会話 convs(メッセージのリストか、1 ターンの質問の文字列)を読み、指定した層の書き込み先の入力活性を位置で選んで集める。
    返事: replies を与えればそれを読ませる(生成しない)。無ければ iv を当てて greedy で max_tokens まで生成する(位置が
    返事を要らないもの(assistant・last)だけなら生成しない)。iv(intervention を参照)は生成と捕捉の両方に当てる。
    positions は 1 つの書き方(群の名前 "A")か {群の名前: 書き方}(同じ返事から複数の群を一度の捕捉で取る。mode same の
    側 A・側 B)。層は keyed.check_layer_ok で確かめる(MTP の層と最終層は断る)。

    戻り値 {"groups": {群: {"positions", "index"(位置が 1 つ以上あった例の番号), "n_pos", "rows",
    "means": {L: (n, in) float64 の例ごとの平均}, "samples": {L: (m, in) float32}(samples > 0 のとき、群ごとに samples 行まで
    一様に間引き), "sample_ex": {L: (m,) 例の番号}, "proj": {L: [例ごとの (n_i, k) = x @ project[L].T]}(project のとき),
    "rms": {L: 位置での rms(W x) の平均}(post-norm の層で、rms_groups の群), "gate": {L: ゲートの平均}(rows がゲートを
    返す書き込み先で、rms_groups の群), "index_l": {L: その層で行があった例の番号}(専門家のように層ごとに行を絞る
    書き込み先では index と違う。means[L] の行はこれに並ぶ), "rows_l": {L: その層の行の数},
    "count_l": {L: [例ごとの行の数](index_l に並ぶ)}, "start": 返事の書き出し(start > 0 のとき、無ければ None)}},
    "texts", "replies"(given / generated / None), "n_prompt", "n_reply", "layers", "iv"}。
    start = k(> 0)なら、返事の最初の k トークンを決める位置(start_positions。群の位置の種類で絞る)も見て、群の "start" に
    {"k", "n_l": {L: [例ごとの書き出しの行の数](index_l に並ぶ)}, "proj": {L: [例ごとの (書き出しの位置 j の (m,), (m, k) の
    x @ project[L].T)]}(project のとき)} を入れる。鍵の作成は側 A が書き出しに偏っていないかを、key-check は側 B・dont_touch の
    書き出しで鍵が合っていないかを見る。
    target は書き込み先の文字列(ffn / attn / shared / expert:e)か、TargetCapture(層は呼ぶ側が確かめたもの)。
    tick(msg) は会話 1 本(と生成 1 回)ごとに呼ぶ。"""
    t0 = time.perf_counter()
    e, w = session.engine, session.weights
    Ls = sorted({int(L) for L in layers})
    if not Ls:
        raise GneError("BAD_ARGS", "鍵の層(layers)を 1 つ以上指定してください")
    if isinstance(target, TargetCapture):                    # 捕捉を直接渡す(expert_usage の専門家の選ばれ方など)
        tc = target
        for L in Ls:
            if K.is_mtp_layer(w, L) or L == K.last_main_layer(w):
                raise GneError("BAD_ARGS", f"層 {L} は MTP の層か最終層なので捕捉できません")
    else:
        kind, expert = K.parse_target(target)
        for L in Ls:
            K.check_layer_ok(w, L, kind, expert)
        tc = target_capture(session, target, Ls)
    groups = {"A": parse_positions(positions)} if not isinstance(positions, Mapping) else \
        {str(g): parse_positions(p) for g, p in positions.items()}
    convs = [_as_conv(c) for c in convs]
    n = len(convs)
    if replies is not None and len(replies) != n:
        raise GneError("BAD_ARGS", f"返事の数 {len(replies)} が会話の数 {n} と違います")
    vecs, lora, iv_desc = intervention(session, iv)
    need_reply = any(p.needs_reply for p in groups.values())
    need_start = any(p.kind == "assistant" for p in groups.values())
    tick = tick or (lambda msg: None)

    pids = [e.tokenize_prompt(e.render_chat(c)) for c in convs]
    starts: list[int | None] = [None] * n
    if need_start:
        for i, c in enumerate(convs):
            head = e.tokenize_prompt(e.render_chat(c, add_generation_prompt=False))
            starts[i] = len(head) if pids[i][:len(head)] == head else len(pids[i]) - 1
    mode = None
    if not need_reply:
        resp: list[list[int]] = [[] for _ in convs]
        texts = ["" for _ in convs]
    elif replies is not None:
        resp = [e.tokenize(str(r), add_special=False, parse_special=False) for r in replies]
        texts, mode = [str(r) for r in replies], "given"
    else:
        from .evalx import _generate
        outs = _generate(session, [(c, int(max_tokens)) for c in convs], vecs, lora, "all", lambda k: None)
        resp = [[t for t in o["token_ids"] if not e.is_eog(t)] for o in outs]
        texts, mode = [o["text"] for o in outs], "generated"
        tick("返事の生成")
    seqs = [p + r for p, r in zip(pids, resp, strict=True)]
    need = max(len(s) for s in seqs) if seqs else 1
    if need > e.n_ctx:
        raise GneError("CONTEXT_FULL", f"{need} トークンの会話が n_ctx={e.n_ctx} に入りません")

    cap = _capture(e)
    seen = e.list_names(seqs[0][:16]) if seqs else set()
    tc.bind(seen)
    wanted = {nm for L in Ls for nm in tc.names(L)}
    cap.set_wanted(lambda nm: nm in wanted)
    cap.skipped = {}

    acc = {g: {"index": [], "index_l": {L: [] for L in Ls}, "n_pos": [], "means": {L: [] for L in Ls}, "rows": 0,
               "rows_l": {L: 0 for L in Ls}, "count_l": {L: [] for L in Ls},
               "res": {L: _Reservoir(samples, seed + 7919 * k) for L in Ls} if samples else None,
               "proj": {L: [] for L in Ls} if project else None,
               "st_n": {L: [] for L in Ls} if start else None,
               "st_proj": {L: [] for L in Ls} if start and project else None,
               "rms": {L: [0.0, 0] for L in Ls}, "gate": {L: [0.0, 0] for L in Ls}}
           for k, g in enumerate(groups)}
    proj = {L: np.asarray(project[L], dtype=np.float64) for L in Ls} if project else None
    ctx = e.probe_ctx(cap, need=need)
    e.set_cvec(ctx, vecs or None)
    e.set_lora(ctx, lora or None)
    try:
        for i, ids in enumerate(seqs):
            pieces = [e.token_bytes(t) for t in resp[i]] if any(p.pattern for p in groups.values()) else None
            sel = {g: select_positions(e, resp[i], p, skip, n_prompt=len(pids[i]), start=starts[i], pieces=pieces)
                   for g, p in groups.items()}
            if not any(s.size for s in sel.values()):
                tick("捕捉")
                continue
            e.clear_memory(ctx)
            e.prefill(ctx, [ids])
            got = cap.take()
            check_capture(cap, wanted)
            for nm in wanted:
                if nm not in got:
                    raise GneError("ARCH_UNSUPPORTED", f"{nm} が捕捉できません")
                got[nm] = tc.token_rows(nm, got[nm], len(ids))
                if got[nm].shape[0] != len(ids):
                    raise GneError("ARCH_UNSUPPORTED", f"{nm} の行の数 {got[nm].shape[0]} が列の長さ {len(ids)} と違います"
                                                       "(最終層など、行が間引かれる層は鍵に使えません)")
            for g, s in sel.items():
                if not s.size:
                    continue
                a = acc[g]
                xs, gates = {}, {}
                for L in Ls:
                    xs[L], gates[L] = tc.rows(got, L, s)
                cnt = {L: int(xs[L].shape[0]) for L in Ls}
                if not any(cnt.values()):
                    continue
                a["index"].append(i)
                a["n_pos"].append(max(cnt.values()))
                a["rows"] += max(cnt.values())
                st = (start_positions(e, resp[i], groups[g], skip, n_prompt=len(pids[i]), k=start, pieces=pieces)
                      if start else None)
                for L in Ls:
                    x = xs[L]
                    if not x.shape[0]:
                        continue
                    a["index_l"][L].append(i)
                    a["rows_l"][L] += int(x.shape[0])
                    a["count_l"][L].append(int(x.shape[0]))
                    if st is not None:                    # 返事の書き出しの行(専門家は選ばれた位置だけなので 1 位置ずつ)
                        offs, xr = [], []
                        for p0 in st:
                            r0, _ = tc.rows(got, L, np.asarray([p0], dtype=np.int64))
                            if r0.shape[0]:
                                offs.append(int(p0) - (len(pids[i]) - 1))
                                xr.append(np.asarray(r0[0], dtype=np.float64))
                        a["st_n"][L].append(len(offs))
                        if a["st_proj"] is not None:
                            pj = np.stack(xr) @ proj[L].T if xr else np.zeros((0, proj[L].shape[0]))
                            a["st_proj"][L].append((np.asarray(offs, dtype=np.int64), pj))
                    a["means"][L].append(x.astype(np.float64).mean(axis=0))
                    if a["res"] is not None:
                        a["res"][L].add(x, i)
                    if proj is not None:
                        a["proj"][L].append(x.astype(np.float64) @ proj[L].T)
                    if g in rms_groups and tc.post_norm(L):
                        rr = tc.rms(x, L)
                        a["rms"][L][0] += float(rr.sum())
                        a["rms"][L][1] += int(rr.size)
                    if g in rms_groups and gates[L] is not None:            # 専門家・shared expert のゲート → 鍵の gate_a
                        a["gate"][L][0] += float(np.sum(gates[L]))
                        a["gate"][L][1] += int(np.size(gates[L]))
            tick("捕捉")
    finally:
        e.set_cvec(ctx, None)
        e.set_lora(ctx, None)

    out_groups = {}
    width = {L: int(tc.in_dim.get(L, 0)) for L in Ls}
    for g, a in acc.items():
        res = {L: a["res"][L].array(width[L]) for L in Ls} if a["res"] is not None else None
        out_groups[g] = {
            "positions": groups[g].text, "index": a["index"], "index_l": a["index_l"], "n_pos": a["n_pos"], "rows": a["rows"],
            "rows_l": a["rows_l"], "count_l": a["count_l"],
            "start": None if a["st_n"] is None else {"k": int(start), "n_l": a["st_n"], "proj": a["st_proj"]},
            "means": {L: (np.stack(a["means"][L]) if a["means"][L] else np.zeros((0, width[L]))) for L in Ls},
            "samples": None if res is None else {L: res[L][0] for L in Ls},
            "sample_ex": None if res is None else {L: res[L][1] for L in Ls},
            "proj": a["proj"],
            "rms": {L: a["rms"][L][0] / a["rms"][L][1] for L in Ls if a["rms"][L][1]} or None,
            "gate": {L: a["gate"][L][0] / a["gate"][L][1] for L in Ls if a["gate"][L][1]} or None}
    return {"groups": out_groups, "texts": texts, "replies": mode, "n_prompt": [len(p) for p in pids],
            "n_reply": [len(r) for r in resp], "layers": Ls, "iv": iv_desc, "seconds": time.perf_counter() - t0}


# --- 作成と確認の共通: どの会話をどの位置で読むか --------------------------------------------------------------

@dataclass
class _Set:
    """1 回の捕捉(同じ文脈・同じ返事)で読む会話と、そこから取る群。"""
    label: str
    convs: list
    hashes: list[str]
    groups: dict[str, str]                   # 群の名前 → 位置の書き方
    replies: list[str] | None
    iv: tuple                                # (vecs, lora)


def _pick(xs, idx):
    return None if xs is None else [xs[i] for i in idx]


def _resolve_mode(pos_a: Positions, pos_b_spec) -> tuple[str, Positions]:
    if pos_a.kind == "not-before":
        raise GneError("BAD_ARGS", "書き込みたい位置の positions に not-before は使えません(それは書き込みたくない位置用の補集合です)",
                       "書き込みたい位置は before:<正規表現>(mode same)か response / assistant / last(mode system)です")
    mode = "same" if pos_a.kind == "before" else "system"
    if pos_b_spec is None:
        pos_b = parse_positions(f"not-before:{pos_a.pattern}") if mode == "same" else pos_a
    else:
        pos_b = parse_positions(pos_b_spec)
    return mode, pos_b


def _plan(session, spec: KeySpec, mode: str, pos_a: Positions, pos_b: Positions, subset: str, replies_a, replies_b,
          vecs0, lora0, warn: Callable[[str], None], dt_start: bool = False) -> tuple[list[_Set], dict]:
    """作成・確認で読む会話の組(_Set のリスト)と、記録用の情報。引数で渡す返事のパスは今の作業場所からの相対
    (対照集合に書いた返事のパスは対照集合のファイルからの相対)。dt_start なら、dont_touch の各群に、その質問の返事の冒頭
    (DT_START_POSITIONS = 生成プロンプトの最後のトークン)の群 "dts:<名前>" を同じ捕捉で足す(鍵の作成だけ。群の positions が
    生成プロンプトの位置なら足さない)。"""
    from .direction import _subset
    from .reverse import side_interventions
    cs = spec.contrast
    side_iv = side_interventions(session, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
    sets: list[_Set] = []
    notes: list[str] = []

    def replies_of(param, side, prompts):
        if param is not None:
            return _reply_list(str(param) if isinstance(param, Path) else param, prompts, None, f"replies_{side.lower()}")
        return spec.replies.get(side)

    if mode == "system":
        if spec.replies.get("same") is not None:
            warn("対照集合の replies は positions が before:<正規表現> のとき(mode same)の返事です。mode system では使いません"
                 "(側ごとの返事は replies_a / replies_b)")
        ra = replies_of(replies_a, "A", cs.prompts("A"))
        rb = replies_of(replies_b, "B", cs.prompts("B"))
        if (not cs.has_recipes and cs.system_a == cs.system_b and not cs.separate_prompts and ra == rb):
            raise GneError("BAD_CONTRAST", "書き込みたい側と書き込みたくない側が同じです(system も recipe も質問も返事も違いがありません)",
                           "推論時と同じ文脈の返事の中で書き込みたい位置を取るなら positions を before:<正規表現> にしてください")
        for side, pos, rep in (("A", pos_a, ra), ("B", pos_b, rb)):
            prompts, convs = cs.prompts(side), cs.convs(side)
            idx = _subset(len(prompts), subset)
            system = cs.system(side)
            sets.append(_Set(side, [convs[i] for i in idx],
                             [K.example_hash(system, question_text(prompts[i])) for i in idx],
                             {side: pos.text}, _pick(rep, idx), _combine(side_iv[side], vecs0, lora0)))
    else:
        if replies_a is not None and replies_b is not None and replies_a != replies_b:
            raise GneError("BAD_ARGS", "positions が before:<正規表現> のとき(書き込みたい位置と書き込みたくない位置が同じ返事にある)は返事を 1 組だけ渡してください")
        param = replies_b if replies_b is not None else replies_a
        if param is None and spec.replies.get("A") is not None and spec.replies.get("same") is None:
            warn("対照集合の replies_a は mode same では使いません(書き込みたい位置も書き込みたくない位置と同じ返事から取ります。返事は replies に書きます)")
        prompts, convs = cs.prompts("B"), cs.convs("B")
        rep = replies_of(param, "B", prompts) if param is not None else (spec.replies.get("same") or spec.replies.get("B"))
        idx = _subset(len(prompts), subset)
        if cs.system_a != cs.system_b or cs.recipe_a is not None:
            notes.append("書き込みたい位置も書き込みたくない位置と同じ文脈(system_b・recipe_b)の返事から取りました(system_a・recipe_a は方向用で、鍵では使いません)")
        sets.append(_Set("AB", [convs[i] for i in idx],
                         [K.example_hash(cs.system_b, question_text(prompts[i])) for i in idx],
                         {"A": pos_a.text, "B": pos_b.text}, _pick(rep, idx), _combine(side_iv["B"], vecs0, lora0)))
    dts = []
    for dt in spec.dont_touch:
        idx = _subset(len(dt.prompts), subset)
        if dt.system == "":
            warn(f"dont_touch {dt.name}: system が空の文字列なので空の system メッセージで読みます(system なしにするにはキーを省きます)")
        pos = dt.positions or pos_b.text
        groups = {f"dt:{dt.name}": pos}
        start = None
        if dt_start and parse_positions(pos).kind not in _PROMPT_END:        # 返事の冒頭を別の行として Q に入れる
            groups[f"dts:{dt.name}"] = start = DT_START_POSITIONS
        sets.append(_Set(f"dt:{dt.name}", dt.convs(idx),
                         [K.example_hash(dt.system, question_text(dt.prompts[i])) for i in idx],
                         groups, _pick(dt.replies, idx), _combine(None, vecs0, lora0)))
        dts.append({"name": dt.name, "system": "none" if dt.system is None else ("empty" if dt.system == "" else "set"),
                    "positions": pos, "replies": "given" if dt.replies is not None else "generated"}
                   | ({"start": start} if dt_start else {}))
    return sets, {"notes": notes, "dont_touch": dts}


def _run_plan(session, sets: list[_Set], *, layers, target, skip, max_tokens, samples, project=None, seed=0,
              rms_groups: Sequence[str] = ("A",), progress: Progress | None = None, extra_steps: int = 0,
              start: int = 0) -> dict:
    """_plan の会話の組を順に捕捉して、群 → collect_acts の群(と、群の例のハッシュ hashes)にまとめる。start は
    collect_acts の返事の書き出しの位置の数(0 なら見ない)。"""
    def gen_steps(s: _Set) -> int:
        return int(s.replies is None and any(parse_positions(p).needs_reply for p in s.groups.values()))
    total = sum(len(s.convs) + gen_steps(s) for s in sets) + extra_steps
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(min(done, total), total, f"key: {msg}")
    out: dict[str, dict] = {}
    texts: dict[str, list[str]] = {}
    reply_mode: dict[str, str | None] = {}
    for s in sets:
        if not s.convs:
            continue
        r = collect_acts(session, s.convs, layers=layers, positions=s.groups, skip=skip, max_tokens=max_tokens,
                         replies=s.replies, iv=s.iv, target=target, samples=samples, project=project, seed=seed,
                         rms_groups=rms_groups, start=start, tick=tick)
        for g, d in r["groups"].items():
            d["hashes"] = [s.hashes[i] for i in d["index"]]
            d["n_examples"] = len(s.convs)
            out[g] = d
        texts[s.label] = r["texts"]
        reply_mode[s.label] = r["replies"]
    return {"groups": out, "texts": texts, "replies": reply_mode, "tick": tick}


# --- 鍵の作成 ----------------------------------------------------------------------------------------------

def _cap_rows(X: np.ndarray, cap: int, rng) -> np.ndarray:
    if cap and X.shape[0] > cap:
        return X[np.sort(rng.choice(X.shape[0], cap, replace=False))]
    return X


def _q_rows(q_mode: str, eb: np.ndarray, dt_means: list[np.ndarray], sb: np.ndarray | None,
            dt_samples: list[np.ndarray], cap: int, rng) -> np.ndarray:
    """Q を張る行: means は群ごとの平均、examples は例ごとの平均、samples はトークンの行(cap 行まで)。"""
    if q_mode == "means":
        rows = [m.mean(axis=0) for m in [eb] + list(dt_means) if m.shape[0]]
        return np.stack(rows) if rows else np.zeros((0, eb.shape[1]))
    if q_mode == "examples":
        return np.concatenate([eb] + [m for m in dt_means if m.shape[0]], axis=0)
    X = np.concatenate([sb] + [m for m in dt_samples if m.shape[0]], axis=0)
    return _cap_rows(X, cap, rng).astype(np.float64)          # 間引いてから float64 にする


def balanced_split(rng, n_a: int, n_b: int) -> tuple[np.ndarray, np.ndarray]:
    """擬似鍵のラベルの並べ替え: 例(A の n_a 個、B の n_b 個、番号は A が 0..n_a−1、B が n_a..)を、擬似 A(n_a 個)と
    擬似 B(n_b 個)に分ける。どちらにも本物の A が同じ割合(n_a/(n_a+n_b))で入るように釣り合わせる(並べ替えの混ざり方を
    期待値に固定する)。釣り合わせない並べ替えは本物のラベルに近いものを数 % 含み、例が少ないと擬似鍵の p95 が本物の鍵と
    ほとんど同じになる(Qwen3-0.6B の 8 + 8 例で確かめた)。"""
    ka = min(n_a, int(round(n_a * n_a / max(n_a + n_b, 1))))
    ka = max(ka, n_a - n_b)                                   # 擬似 A に入れる B が足りないとき
    pa, pb = rng.permutation(n_a), rng.permutation(n_b) + n_a
    return (np.concatenate([pa[:ka], pb[:n_a - ka]]).astype(np.int64),
            np.concatenate([pa[ka:], pb[n_a - ka:]]).astype(np.int64))


def _basis(q_mode: str, X: np.ndarray, energy: float, max_rank: int | None) -> np.ndarray:
    if q_mode == "means":                                      # 群の平均は数本なので全部使う(打ち切ると漏れる)
        return K.orth_basis(X, energy=1.0)
    return K.orth_basis(X, energy=energy, max_rank=max_rank)


def few_side_a(n_examples: int, n_rows: int) -> bool:
    """側 A が少ないか(位置のある例が MIN_A_EXAMPLES 未満か、行が MIN_A_ROWS 未満)。鍵の作成と key-check で同じ基準。"""
    return int(n_examples) < MIN_A_EXAMPLES or int(n_rows) < MIN_A_ROWS


def few_side_a_message(n_examples: int, n_rows: int, *, mode: str = "same", where: str = "", check: bool = False) -> str:
    """書き込みたい位置(側 A)が少ないときの警告の文。check なら key-check(確かめる側の数)。"""
    head = (f"{where}key-check の書き込みたい位置が {n_examples} 例の {n_rows} 位置だけです" if check
            else f"{where}書き込みたい位置が {n_examples} 例の {n_rows} 位置だけです")
    head += f"(目安は {MIN_A_EXAMPLES} 例・{MIN_A_ROWS} 位置以上)。"
    if check:
        return head + "key-check の選択性・AUC が数例で揺れるので、比較用の鍵との比べは参考にとどめ、質問を増やして測り直してください"
    how = ("質問を増やすか、素のモデルが実際に書く言い方に正規表現を合わせて(結果の samples と matched を見る)、一致する位置を"
           "増やしてください" if mode == "same" else "質問を増やしてください")
    return head + "μ が数位置の活性の平均になるので、鍵が状態ではなくそのトークンや例の本文を拾いやすく、key-check の数字も揺れます。" + how


def _start_share(g: dict, L: int) -> tuple[float | None, int]:
    """群 g(collect_acts の群、start つき)の層 L で、μ に占める返事の書き出しの行の重み(例ごとの「書き出しの行 ÷ 行」の平均。
    μ は例ごとの平均の平均なので、この重みで書き出しの状態が μ に入る)と、書き出しの行の数。"""
    st = g.get("start")
    if not st:
        return None, 0
    n, c = st["n_l"].get(L) or [], g["count_l"].get(L) or []
    fr = [a / b for a, b in zip(n, c) if b]
    return (float(np.mean(fr)) if fr else None), int(sum(n))


def _starts_groups(spec: KeySpec, auto: Sequence[str] = ()) -> list[str]:
    """dont_touch の群のうち、返事の書き出しの状態を Q に入れたもの: 生成プロンプトの位置(last・last:K・assistant)の群と、
    返事の冒頭を自動で足した群(auto)。dont_touch の並び順。"""
    auto = set(auto)
    return [dt.name for dt in spec.dont_touch
            if dt.name in auto or (dt.positions and parse_positions(dt.positions).kind in _PROMPT_END)]


def _reply_start_info(spec: KeySpec, ga: dict, Ls: Sequence[int], mode: str, kind: str,
                      auto: Sequence[str] | None = None) -> dict | None:
    """鍵の作成(mode same)で、側 A の位置のうち返事の書き出しにあるもの: {k, share(μ に占める重み。専門家は層の最大),
    rows(書き出しの行の数)、rows_a、layer(share が最大の層)、dont_touch_starts(書き出しの状態を Q に入れた dont_touch の群)、
    dont_touch_start(dont_touch の質問の返事の冒頭を自動で Q に入れたか)}。auto は返事の冒頭を自動で足した群(None = 切った)。"""
    if mode != "same" or not ga.get("start"):
        return None
    best = None
    for L in (Ls if kind == "expert" else Ls[:1]):           # 専門家のほかは全層で行が同じ
        sh, n = _start_share(ga, L)
        if sh is not None and (best is None or sh > best[0]):
            best = (sh, n, L, int(ga["rows_l"][L]))
    if best is None:
        return None
    return {"k": int(ga["start"]["k"]), "share": round(best[0], 4), "rows": best[1], "rows_a": best[3],
            "layer": best[2] if kind == "expert" else None, "dont_touch_starts": _starts_groups(spec, auto or ()),
            "dont_touch_start": auto is not None}


def _reply_start_message(info: dict, n_dont_touch: int = 0) -> str:
    """書き込みたい位置(側 A)が返事の冒頭に偏っているときの文。返事の冒頭の状態を Q に入れた dont_touch の群があれば注、
    無ければ警告(dont_touch の群が無いなら質問を足す案内、返事を書き始める直前の位置の自動を切っているなら戻す案内)。
    n_dont_touch は dont_touch の群の数。"""
    where = f"L{info['layer']} の" if info.get("layer") is not None else ""
    head = (f"{where}書き込みたい位置の {info['share'] * 100:.0f}%(μ に占める重み。{info['rows']} / {info['rows_a']} 位置)が"
            f"返事の冒頭(返事の最初の {info['k']} トークンを決める位置。返事の最初で正規表現に一致すると、書き込みたい位置は"
            "返事を書き始める直前の位置になる)にあります。")
    if info["dont_touch_starts"]:
        return (head + f"dont_touch の群({', '.join(info['dont_touch_starts'])})の質問の、返事を書き始める直前の位置は Q に"
                "入れてあります。それでも書いた後に状態が書き込みたい位置の状態から抜けずに鍵が反応し続け、繰り返しに落ちやすく"
                "なります(keyed-write.md §14)。key-check の返事の冒頭での ρ(start_rho)と、生成した返事で確かめてください")
    risk = ("返事の冒頭の状態はどの質問でも似ているので、どの返事の冒頭でも鍵が反応しやすくなります(書き込みたくない場面でも"
            "返事の最初で方向を書く)。")
    if n_dont_touch and not info.get("dont_touch_start", True):
        return (head + risk + "dont_touch の質問の、返事を書き始める直前の位置を Q に入れない設定(dont_touch_start: false)に"
                "なっています。入れる設定に戻して鍵を作り直してください。key-check の返事の冒頭での ρ(start_rho)でも確かめられます")
    return (head + risk + "dont_touch の群がありません。ふつうの質問(計算・事実の質問など、ここでは書き込みたくない場面)を"
            "dont_touch に足して鍵を作り直してください(dont_touch の質問の、返事を書き始める直前の位置も Q に入ります)。"
            "key-check の返事の冒頭での ρ(start_rho)でも確かめられます")


def build_keys(session, contrast, *, name: str | None = None, layers, positions: str | None = None,
               positions_b: str | None = None, skip: int | None = None, max_tokens: int = 96, q_mode: str = "examples",
               q_energy: float = 0.99, q_max_rank: int | None = None, q_max_samples: int = 4096, subset: str = "even",
               pseudo: int = 16, replies_a=None, replies_b=None, iv=None, target: str = "ffn", seed: int = 0,
               dont_touch_start: bool | None = None, progress: Progress | None = None) -> dict:
    """対照集合(contrast: パス・ContrastSet・KeySpec)から層ごとの鍵 κ を作り、<root>/keys/<model_id>/<name>.npz に保存する。

    positions(側 A の位置。省くと対照集合の positions、無ければ response)が before:<re> なら mode same(推論時と同じ文脈の
    返事の位置、側 B は positions_b、既定 not-before:<re>)、それ以外は mode system(system_a・recipe_a の文脈 対 system_b・
    recipe_b の文脈、側 B の位置は既定で positions と同じ)。skip も省くと対照集合の skip、無ければ 0。
    subset(even / odd / all)で使う質問を選ぶ(既定 even。key-check は odd で確かめる)。dont_touch の群は Q に足す。
    dont_touch_start(省くと対照集合の dont_touch_start、無ければ True)なら、dont_touch の各群の質問の返事の冒頭(生成
    プロンプトの最後のトークン)も、群の positions とは別の行として Q に足す(側 B の返事の冒頭は足さない)。
    q_mode・q_energy・q_max_rank・q_max_samples は Q の作り方。pseudo 本の擬似鍵(例の単位のラベルの並べ替え、seed で決まる)も
    保存する。iv はレシピ(intervention)を生成と捕捉の全位置に当てた状態で作る(他の介入と重ねるとき)。
    target は書き込み先: ffn(既定、ffn_down の入力)、attn(attn_output / ssm_out の入力 = ヘッド出力の連結。ハイブリッドの
    モデルでは attention の層と線形注意の層を 1 つの鍵に混ぜてよく、層ごとに κ の長さが違ってよい)、shared(MoE の層の
    shared expert の ffn_down_shexp の入力、ゲートの平均を gate_a に)、expert:e(ffn_down_exps の専門家 e の入力。側 A・B とも
    専門家 e が選ばれた位置の行だけで作り、ルーターの重みの平均を gate_a に。層ごとに行と例の数が違う → info の counts_l・
    rows_l。expert_usage で側 A の位置でよく選ばれる専門家を選ぶ)。
    μ・Q が鍵にならない層(keyed.make_key の BAD_KEY)は警告して外す。戻り値は層ごとの kn・gain・q_rank・作成に使った例での
    AUC などの要約と path。"""
    t0 = time.perf_counter()
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        session.warn(msg)
    if q_mode not in Q_MODES:
        raise GneError("BAD_ARGS", f"q_mode は {' / '.join(Q_MODES)} です({q_mode!r})")
    if subset not in SUBSETS:
        raise GneError("BAD_ARGS", f"subset は {' / '.join(SUBSETS)} です({subset!r})")
    if pseudo < 0 or q_max_samples < 1:
        raise GneError("BAD_ARGS", "pseudo は 0 以上、q_max_samples は 1 以上です")
    if not 0 < q_energy <= 1:
        raise GneError("BAD_ARGS", f"q_energy は 0 より大きく 1 以下です({q_energy})")
    if not layers:
        raise GneError("BAD_ARGS", "鍵の層(layers)を指定してください(Q の基底は層ごとに作るので、全層は既定にしません)")
    Ls = sorted({int(L) for L in layers})
    kind, expert = K.parse_target(target)
    for L in Ls:
        K.check_layer_ok(session.weights, L, kind, expert)
    target_capture(session, target, Ls)                     # まだ捕捉できない書き込み先はここで断る
    spec = _as_spec(contrast)
    cs = spec.contrast
    name = name or cs.name
    if not _KEY_NAME_RE.fullmatch(name):
        raise GneError("BAD_ARGS", f"鍵の名前 {name!r} が不正です(英数字・_・.・- だけ)")
    pos_a = parse_positions(positions or spec.positions or "response")
    mode, pos_b = _resolve_mode(pos_a, positions_b or spec.positions_b)
    skip = int(skip if skip is not None else (spec.skip or 0))
    vecs0, lora0, iv_desc = intervention(session, iv, warn)
    dts_on = bool(dont_touch_start if dont_touch_start is not None else
                  (spec.dont_touch_start if spec.dont_touch_start is not None else True))
    sets, pinfo = _plan(session, spec, mode, pos_a, pos_b, subset, replies_a, replies_b, vecs0, lora0, warn,
                        dt_start=dts_on)
    samples = q_max_samples if q_mode == "samples" else 0
    run = _run_plan(session, sets, layers=Ls, target=target, skip=skip, max_tokens=max_tokens, samples=samples,
                    seed=seed, progress=progress, extra_steps=len(Ls), start=START_TOKENS if mode == "same" else 0)
    G = run["groups"]
    ga, gb = G.get("A"), G.get("B")
    dt_keys = [g for g in G if g.startswith("dt:")]
    dts_keys = [g for g in G if g.startswith("dts:") and G[g]["index"]]      # dont_touch の質問の返事の冒頭(Q だけに使う)
    dt_start_rec = {"on": dts_on, "positions": DT_START_POSITIONS if dts_on else None,
                    "groups": {g[4:]: int(G[g]["rows"]) for g in dts_keys}, "rows": sum(int(G[g]["rows"]) for g in dts_keys)}
    if ga is None or not ga["index"]:
        hint = ("返事に正規表現が一致していません。返事(texts)を見て正規表現を直すか、一致する返事を replies で与えてください"
                if mode == "same" else "返事が空か、位置の指定に合うトークンがありません")
        raise GneError("BAD_CONTRAST", f"書き込みたい位置がありません(positions {pos_a.text})", hint)
    if (gb is None or not gb["index"]) and not any(G[g]["index"] for g in dt_keys):
        raise GneError("BAD_CONTRAST", f"書き込みたくない位置がありません(positions {pos_b.text})")
    n_a = len(ga["index"])
    n_b = len(gb["index"]) if gb else 0
    if kind != "expert" and few_side_a(n_a, int(ga["rows"])):      # 専門家の鍵は層ごとの行で見る(下の層のループ)
        warn(few_side_a_message(n_a, int(ga["rows"]), mode=mode))
    starts_info = _reply_start_info(spec, ga, Ls, mode, kind, list(dt_start_rec["groups"]) if dts_on else None)
    if starts_info is not None and starts_info["share"] >= START_SHARE_WARN:
        msg = _reply_start_message(starts_info, len(spec.dont_touch))
        if starts_info["dont_touch_starts"]:
            pinfo["notes"].append(msg)
        else:
            warn(msg)
    rng = np.random.default_rng(seed)
    perms = [balanced_split(rng, n_a, n_b) for _ in range(pseudo)] if n_b else []   # 層をまたいで同じ並べ替え
    if pseudo and not n_b:
        warn("書き込みたくない側の例が無い(dont_touch だけ)ので比較用の鍵は作りません")

    kappa, mu, kn, gain, qrank, exa, exb, ctrl, rms_a, gate_a = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    rows_q, dropped, build_auc, weak, counts_l = {}, {}, {}, [], {}
    for L in Ls:
        # 層ごとの例(rows が層ごとに行を絞る書き込み先では index と違う。means[L] の行は index_l[L] に並ぶ)
        idx_a = ga["index_l"][L]
        idx_b = gb["index_l"][L] if gb else []
        na, nb = len(idx_a), len(idx_b)
        if not na:
            dropped[L] = "書き込みたい位置でこの層の書き込み先の行がありません"
            warn(f"L{L} は鍵にならないので外しました: {dropped[L]}")
            run["tick"](f"L{L}")
            continue
        same_n = (na, nb) == (n_a, n_b)
        if not same_n:
            counts_l[str(L)] = {"a": na, "b": nb}
        if kind == "expert" and few_side_a(na, int(ga["rows_l"][L])):
            warn(few_side_a_message(na, int(ga["rows_l"][L]), mode=mode,
                                    where=f"L{L}(専門家 {expert} が選ばれた位置だけ): "))
        if same_n:
            perms_l = perms
        elif nb and pseudo:                                     # 層ごとの例の数で並べ替え(層 L ごとに決まった乱数)
            rng_l = np.random.default_rng([seed, L])
            perms_l = [balanced_split(rng_l, na, nb) for _ in range(pseudo)]
        else:
            perms_l = []
        EA = ga["means"][L]
        EB = gb["means"][L] if gb else np.zeros((0, EA.shape[1]))
        DTm = [G[g]["means"][L] for g in dt_keys + dts_keys]
        SB = gb["samples"][L] if (samples and gb) else np.zeros((0, EA.shape[1]), np.float32)
        DTs = [G[g]["samples"][L] for g in dt_keys + dts_keys] if samples else []
        m = EA.mean(axis=0)
        X = _q_rows(q_mode, EB, DTm, SB, DTs, q_max_samples, np.random.default_rng(seed + 1 + L))
        try:
            Q = _basis(q_mode, X, q_energy, q_max_rank)
            kap, info = K.make_key(m, Q, warn=warn, label=f"L{L}")
        except GneError as ex:
            if ex.code != "BAD_KEY":
                raise
            dropped[L] = ex.message
            warn(f"L{L} は鍵にならないので外しました: {ex.message}")
            run["tick"](f"L{L}")
            continue
        kappa[L], mu[L], kn[L], gain[L], qrank[L] = kap, m, info["kn"], info["gain"], info["q_rank"]
        rows_q[L] = int(X.shape[0])
        if info["weak"]:
            weak.append(L)
        exa[L] = EA.astype(np.float32)
        if nb:
            exb[L] = EB.astype(np.float32)
        if ga["rms"] and L in ga["rms"]:
            rms_a[L] = float(ga["rms"][L])
        if ga["gate"] and L in ga["gate"]:
            gate_a[L] = float(ga["gate"][L])
        build_auc[L] = K.auc(EA @ kap, EB @ kap) if nb else None
        # 擬似鍵: 例ごとの平均(A・B)のラベルを釣り合わせて並べ替え(balanced_split)、同じ手順で作る。dont_touch はいつも Q に入れる
        if perms_l:
            pool = np.concatenate([EA, EB], axis=0)
            if samples:
                SA = ga["samples"][L]
                pid_a = {ex: k for k, ex in enumerate(idx_a)}
                pid_b = {ex: na + k for k, ex in enumerate(idx_b)}
                spool = np.concatenate([SA, SB], axis=0)
                sid = np.concatenate([np.array([pid_a[int(x)] for x in ga["sample_ex"][L]], dtype=np.int64),
                                      np.array([pid_b[int(x)] for x in gb["sample_ex"][L]], dtype=np.int64)])
            C = np.zeros((len(perms_l), kap.size), dtype=np.float64)
            for j, perm in enumerate(perms_l):
                for attempt in range(4):
                    ia, ib = perm if attempt == 0 else balanced_split(np.random.default_rng(seed + 1000 * (j + 1) + attempt),
                                                                      na, nb)
                    sb_j = spool[np.isin(sid, ib)] if samples else None
                    Xj = _q_rows(q_mode, pool[ib], DTm, sb_j, DTs, q_max_samples,
                                 np.random.default_rng(seed + 7 * j + L))
                    try:
                        C[j] = K.make_key(pool[ia].mean(axis=0), _basis(q_mode, Xj, q_energy, q_max_rank))[0]
                        break
                    except GneError as ex:
                        if ex.code != "BAD_KEY":
                            raise
                else:
                    warn(f"L{L}: 比較用の鍵 {j} を作れませんでした(0 にしました)")
            ctrl[L] = C.astype(np.float32)
        run["tick"](f"L{L}")
    if not kappa:
        raise GneError("BAD_KEY", "どの層でも鍵を作れませんでした",
                       "q_mode を examples / means にするか、q_energy・q_max_rank を下げるか、例を増やしてください")
    counts = {"a": n_a, "b": n_b, "rows_a": int(ga["rows"]), "rows_b": int(gb["rows"]) if gb else 0,
              "dt": sum(len(G[g]["index"]) for g in dt_keys), "rows_dt": sum(int(G[g]["rows"]) for g in dt_keys),
              "examples_a": int(ga["n_examples"]), "examples_b": int(gb["n_examples"]) if gb else 0}
    if starts_info is not None:
        counts["rows_a_start"] = int(starts_info["rows"])
    counts["rows_dt_start"] = int(dt_start_rec["rows"])     # dont_touch の質問の返事の冒頭として Q に入れた行(切れば 0)
    info = {"mode": mode, "positions_b": pos_b.text, "q_energy": q_energy, "q_max_rank": q_max_rank,
            "q_max_samples": q_max_samples, "pseudo": len(perms), "pseudo_method": "balanced", "seed": seed,
            "contrast": cs.name,
            "replies": run["replies"], "dont_touch": pinfo["dont_touch"], "iv": iv_desc,
            "dropped": {str(L): v for L, v in dropped.items()}, "weak": weak,
            "q_rows": {str(L): v for L, v in rows_q.items()},
            "build_auc": {str(L): v for L, v in build_auc.items()}}
    if counts_l:                                            # 層ごとに例の数が違う(専門家など)ときだけ記録する
        info["counts_l"] = counts_l
    if starts_info is not None:                             # mode same: 側 A のうち返事の書き出しにある位置
        info["reply_start"] = starts_info
    info["dont_touch_start"] = dt_start_rec                 # これが無い鍵は、返事の冒頭を自動で入れる前に作ったもの
    if kind == "expert":                                    # 専門家の鍵: 層ごとの行(専門家 e が選ばれた位置)の数
        info["rows_l"] = {str(L): {"a": int(ga["rows_l"][L]), "b": int(gb["rows_l"][L]) if gb else 0,
                                   "of_a": int(ga["rows"]), "of_b": int(gb["rows"]) if gb else 0} for L in Ls}
    ks = K.KeySet(target=K.target_of(kind, expert), kappa=kappa, mu=mu, kn=kn, gain=gain, q_rank=qrank, exa=exa, exb=exb,
                  kappa_ctrl=ctrl, rms_a=rms_a, gate_a=gate_a, q_mode=q_mode, positions=pos_a.text, skip=skip, max_tokens=int(max_tokens),
                  subset=subset, contrast_sha=cs.sha, hashes_a=list(ga["hashes"]), hashes_b=list(gb["hashes"]) if gb else [],
                  counts=counts, info=info, model_id=session.meta.model_id, name=name)
    path = ks.save(K.key_dir(session) / f"{name}.npz")
    rows = []
    for L in ks.layers:
        row = {"layer": L, "kn": kn[L], "gain": gain[L], "q_rank": qrank[L], "q_rows": rows_q[L], "weak": L in weak,
               "kappa_mu": float(np.asarray(kappa[L], dtype=np.float64) @ mu[L]),
               "build_auc": build_auc[L], "rms_a": rms_a.get(L)}
        if kind in ("shared", "expert"):                   # MoE: ゲートの平均と、層ごとの行・例の数(ffn・attn の形は変えない)
            row.update({"gate_a": gate_a.get(L), "rows_a": int(ga["rows_l"][L]), "rows_b": int(gb["rows_l"][L]) if gb else 0,
                        "examples_a": len(ga["index_l"][L]), "examples_b": len(gb["index_l"][L]) if gb else 0})
            if kind == "expert" and starts_info is not None:       # 専門家の鍵: 層ごとの書き出しの行と重み
                sh, nst = _start_share(ga, L)
                row.update({"rows_a_start": nst, "start_share": sh})
        rows.append(row)
    texts = run["texts"]
    sample_side = "AB" if mode == "same" else "A"
    return {"name": name, "path": str(path), "target": ks.target, "mode": mode, "positions": pos_a.text,
            "positions_b": pos_b.text, "skip": skip, "max_tokens": int(max_tokens), "subset": subset, "q_mode": q_mode,
            "contrast": cs.name, "contrast_sha": cs.sha, "layers": rows,
            "dropped": [{"layer": L, "error": v} for L, v in dropped.items()], "counts": counts,
            "n_pseudo": ks.n_pseudo, "replies": run["replies"], "dont_touch": pinfo["dont_touch"], "iv": iv_desc,
            "matched": {"A": n_a, "of": int(ga["n_examples"])}, "reply_start": starts_info, "dont_touch_start": dt_start_rec,
            "samples": [t[:240] for t in texts.get(sample_side, [])[:4]],
            "notes": pinfo["notes"] + ["build_auc は作成に使った例での値です(甘い)。key-check を保留集合(subset odd か別の対照集合)で"
                                       "回してください"],
            "warnings": warnings, "sha": ks.sha, "seconds": time.perf_counter() - t0}


# --- 専門家の選び方(expert:e の鍵の前に)---------------------------------------------------------------------

class _RouteCapture(TargetCapture):
    """専門家の選ばれ方(expert_usage 用): 選んだ位置ごとに、選ばれた専門家を 1・それ以外を 0 にした行(n_expert)と、
    実効のルーターの重みを選ばれた専門家に散らした行(n_expert)を横に並べて返す(幅 2·n_expert)。例ごとの平均が
    その例での選ばれた割合と、重みの平均(選ばれなければ 0)。"""
    target = "route"

    def __init__(self, session, layers):
        super().__init__(session, layers)
        w = session.weights
        self.n_expert, self.n_used = {}, {}
        for L in self.layers:
            li = w.moe_layer(L)
            self.n_expert[L], self.n_used[L] = int(li.n_expert), int(li.n_expert_used)
            self.in_dim[L] = 2 * self.n_expert[L]
        self._nm: dict[int, dict[str, str]] = {}
        self._layer_of: dict[str, int] = {}

    def bind(self, seen):
        from .moe import moe_names
        for L in self.layers:
            nm = moe_names(seen, L)
            if "argsort" not in nm:
                raise GneError("ARCH_UNSUPPORTED", f"層 {L} の ffn_moe_argsort が見えません")
            self._nm[L] = {k: nm[k] for k in ("argsort", "weights") if k in nm}
            for v in self._nm[L].values():
                self._layer_of[v] = L

    def names(self, L):
        return list(self._nm[L].values())

    def token_rows(self, nm, arr, T):
        L = self._layer_of.get(nm)
        if L is None or arr.shape[0] == T:
            return arr
        return arr.reshape(T, -1)

    def rows(self, got, L, sel):
        from .moe import selected
        u, n = self.n_used[L], self.n_expert[L]
        top = selected(np.asarray(got[self._nm[L]["argsort"]]), u)[sel]
        x = np.zeros((len(sel), 2 * n), np.float32)
        r = np.repeat(np.arange(len(sel)), u)
        x[r, top.reshape(-1)] = 1.0
        if "weights" in self._nm[L]:
            wts = np.asarray(got[self._nm[L]["weights"]], dtype=np.float32).reshape(-1, u)[sel]
            x[r, n + top.reshape(-1)] = wts.reshape(-1)
        return x, None


def expert_usage(session, contrast, *, layers, positions: str | None = None, positions_b: str | None = None,
                 skip: int | None = None, max_tokens: int = 96, subset: str = "even", replies_a=None, replies_b=None,
                 iv=None, top: int = 8, progress: Progress | None = None) -> dict:
    """expert:e の鍵の専門家を選ぶための、側 A・側 B の位置での専門家の選ばれ方(build_keys と同じ対照集合・位置・返事)。
    専門家の鍵は側 A の位置で専門家 e が選ばれた行だけでできるので、側 A でよく選ばれる専門家ほど行が多く、鍵が例の本文を
    拾いにくい。層ごとに、側 A で選ばれた割合 rate_a(例ごとの割合の平均)の大きい順に top 個:
    {expert, rate_a, rate_b, diff(a − b), lift(a / b), weight_a(選ばれたときのルーターの重みの平均), rows_a(側 A で選ばれた
    行の数の見込み = rate_a × 側 A の位置の数)}。dont_touch の群は rate_dt に出す。"""
    t0 = time.perf_counter()
    if subset not in SUBSETS:
        raise GneError("BAD_ARGS", f"subset は {' / '.join(SUBSETS)} です({subset!r})")
    if not layers:
        raise GneError("BAD_ARGS", "層(layers)を指定してください")
    Ls = sorted({int(L) for L in layers})
    tc = _RouteCapture(session, Ls)                          # MoE でない層は BAD_RECIPE
    spec = _as_spec(contrast)
    pos_a = parse_positions(positions or spec.positions or "response")
    mode, pos_b = _resolve_mode(pos_a, positions_b or spec.positions_b)
    skip = int(skip if skip is not None else (spec.skip or 0))
    vecs0, lora0, iv_desc = intervention(session, iv, session.warn)
    sets, pinfo = _plan(session, spec, mode, pos_a, pos_b, subset, replies_a, replies_b, vecs0, lora0, session.warn)
    run = _run_plan(session, sets, layers=Ls, target=tc, skip=skip, max_tokens=max_tokens, samples=0, rms_groups=(),
                    progress=progress)
    G = run["groups"]
    if "A" not in G or not G["A"]["index"]:
        raise GneError("BAD_CONTRAST", f"書き込みたい位置がありません(positions {pos_a.text})")
    out_layers = []
    for L in Ls:
        n = tc.n_expert[L]

        def rate(g, L=L, n=n):
            m = G[g]["means"][L] if g in G else np.zeros((0, 2 * n))
            if not m.shape[0]:
                return np.zeros(n), np.zeros(n)
            mm = m.mean(axis=0)
            return mm[:n], mm[n:]
        ra, wa = rate("A")
        rb, _ = rate("B")
        dts = {g[3:]: rate(g)[0] for g in G if g.startswith("dt:")}
        n_pos_a = int(G["A"]["rows_l"][L])
        order = np.argsort(-ra, kind="stable")[:max(1, int(top))]
        rows = []
        for ex in order:
            rows.append({"expert": int(ex), "rate_a": float(ra[ex]), "rate_b": float(rb[ex]), "diff": float(ra[ex] - rb[ex]),
                         "lift": float(ra[ex] / rb[ex]) if rb[ex] > 0 else None,
                         "weight_a": float(wa[ex] / ra[ex]) if ra[ex] > 0 else None,
                         "rows_a": round(float(ra[ex]) * n_pos_a, 1),
                         "rate_dt": {k: float(v[ex]) for k, v in dts.items()}})
        out_layers.append({"layer": L, "n_expert": n, "n_used": tc.n_used[L], "positions_a": n_pos_a,
                           "positions_b": int(G["B"]["rows_l"][L]) if "B" in G else 0,
                           "base_rate": tc.n_used[L] / n, "top": rows})
    return {"contrast": spec.contrast.name, "mode": mode, "positions": pos_a.text, "positions_b": pos_b.text,
            "subset": subset, "layers": out_layers, "replies": run["replies"], "iv": iv_desc, "notes": pinfo["notes"],
            "seconds": time.perf_counter() - t0}


# --- 鍵の確認 ----------------------------------------------------------------------------------------------

def held_out_info(key, contrast_sha: str | None, subset: str | None, hashes: Sequence[str]) -> dict:
    """鍵の作成に使った例と、確かめる例の重なり(レビュー P19・V34)。同じ対照集合(contrast_sha が同じ)で subset が重なる
    (同じ subset か、どちらかが all)か、例のハッシュ((system, プロンプト))が作成の例と重なれば held_out は False。"""
    same = bool(key.contrast_sha) and contrast_sha == key.contrast_sha
    sub = bool(same and (subset in (None, "all") or key.subset in (None, "", "all") or subset == key.subset))
    built = set(key.hashes_a) | set(key.hashes_b)
    hs = set(hashes)
    n_hash = len(hs & built)
    return {"held_out": not sub and n_hash == 0, "same_contrast": same, "subset_overlap": sub, "hash_overlap": n_hash,
            "n_checked": len(hs), "built_subset": key.subset, "subset": subset}


def _stats(v) -> dict | None:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if not v.size:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "abs_median": float(np.median(np.abs(v)))}


def _hist(groups: dict[str, np.ndarray], bins: int = 20) -> dict | None:
    allv = np.concatenate([v for v in groups.values() if v.size]) if any(v.size for v in groups.values()) else None
    if allv is None:
        return None
    lo, hi = np.percentile(allv, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(allv.min()) - 0.5, float(allv.max()) + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    return {"edges": [float(x) for x in edges],
            **{g: np.histogram(np.clip(v, lo, hi), bins=edges)[0].astype(int).tolist() for g, v in groups.items()}}


def _b_split_key(ks, L: int) -> np.ndarray | None:
    """参考値: 側 B の例を 2 つに分けて作った擬似差の鍵(偶数番目を μ、奇数番目を Q、q_mode に関係なく例ごとの平均で作る)。"""
    eb = ks.exb.get(L)
    if eb is None or eb.shape[0] < 4:
        return None
    eb = eb.astype(np.float64)
    try:
        return K.make_key(eb[0::2].mean(axis=0), K.orth_basis(eb[1::2], energy=float(ks.info.get("q_energy", 0.99))))[0]
    except GneError:
        return None


def key_check(session, key, *, contrast=None, subset: str = "odd", layers=None, positions: str | None = None,
              positions_b: str | None = None, skip: int | None = None, max_tokens: int | None = None, replies_a=None,
              replies_b=None, iv=None, strict: bool = False, progress: Progress | None = None) -> dict:
    """鍵の分離の確認(生成なしで済む位置なら生成なし)。各例・各トークンで ρ = κᵀx を測り、層ごとに側 A 対 側 B の AUC
    (例単位が主、トークン単位も)、dont_touch の群の ρ の中央値、擬似鍵の AUC の p50・p95 と beats_pseudo(本物の AUC >
    擬似鍵の p95)、側 B を 2 分した擬似差の鍵の AUC(参考値)を出す。位置・skip・max_tokens・mode の既定は鍵のファイルから。
    鍵の作成に使った例と重なる(同じ対照集合で subset が重なる、または例のハッシュが重なる)と警告して held_out: false に
    する(strict なら BAD_ARGS)。contrast を省くと、鍵に保存した例ごとの平均(作成に使った例)で測る(捕捉なし、甘い値)。
    key は鍵の名前(NAME~j なら擬似鍵 j を主に測る)か KeySet。iv を渡すとその介入を当てた状態で測る。"""
    t0 = time.perf_counter()
    warnings: list[str] = []
    notes: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        session.warn(msg)
    ks = key if isinstance(key, K.KeySet) else K.load_key(session, str(key))
    if ks.model_id and ks.model_id != session.meta.model_id:
        warn(f"鍵 {ks.name} は別のモデル({ks.model_id})で作られています")
    if subset not in SUBSETS:
        raise GneError("BAD_ARGS", f"subset は {' / '.join(SUBSETS)} です({subset!r})")
    Ls = sorted({int(L) for L in layers}) if layers else ks.layers
    for L in Ls:
        if L not in ks.kappa:
            raise GneError("BAD_ARGS", f"鍵 {ks.name} に層 {L} がありません(ある層: {ks.layers})")
    n_ps = ks.n_pseudo
    bsk = {L: _b_split_key(ks, L) for L in Ls}
    # 測る鍵の行: 0 = 主の鍵、1 = B の 2 分割(無ければ 0 ベクトル)、2.. = 擬似鍵
    M = {}
    for L in Ls:
        rows = [ks.kappa_of(L).astype(np.float64)[None, :], (bsk[L] if bsk[L] is not None else np.zeros(ks.kappa_of(L).size))[None, :]]
        if n_ps:
            rows.append(ks.kappa_ctrl[L][:n_ps].astype(np.float64))
        M[L] = np.vstack(rows)
    if ks.info.get("iv") and iv is None:
        notes.append("この鍵は介入(iv)を当てた状態で作られています。同じ状態で確かめるには iv を渡してください")
    if contrast is None:
        notes.append("対照集合を渡していないので、鍵に保存した作成時の例ごとの平均で測りました(作成に使った例での値で甘い。"
                     "保留集合で確かめてください)")
        warn("key-check: 作成に使った例で測っています(held_out: false)")
        groups_ex = {}
        for L in Ls:
            ea = ks.exa.get(L)
            eb = ks.exb.get(L)
            if ea is None:
                raise GneError("BAD_KEY", f"鍵 {ks.name} の層 {L} に例ごとの平均(exa)がありません", "対照集合を渡してください")
            groups_ex[L] = {"A": ea.astype(np.float64) @ M[L].T,
                            "B": (eb.astype(np.float64) @ M[L].T) if eb is not None else np.zeros((0, M[L].shape[0]))}
        layers_out = [_layer_result(ks, L, groups_ex[L], None, n_ps) for L in Ls]
        ov = {"held_out": False, "same_contrast": True, "subset_overlap": True, "hash_overlap": len(set(ks.hashes_a) | set(ks.hashes_b)),
              "n_checked": len(set(ks.hashes_a) | set(ks.hashes_b)), "built_subset": ks.subset, "subset": ks.subset}
        return _check_summary(ks, Ls, layers_out, ov, None, None, ks.positions, ks.info.get("positions_b"), ks.skip,
                              ks.max_tokens, ks.info.get("mode", "system"), {}, notes, warnings, t0, {})

    spec = _as_spec(contrast)
    cs = spec.contrast
    pos_a = parse_positions(positions or ks.positions or "response")
    mode, pos_b = _resolve_mode(pos_a, positions_b or (ks.info.get("positions_b") if positions is None else None))
    skip = int(ks.skip if skip is None else skip)
    max_tokens = int(ks.max_tokens if max_tokens is None else max_tokens)
    vecs0, lora0, iv_desc = intervention(session, iv, warn)
    sets, pinfo = _plan(session, spec, mode, pos_a, pos_b, subset, replies_a, replies_b, vecs0, lora0, warn)
    ov = held_out_info(ks, cs.sha, subset, [h for s in sets for h in s.hashes])
    if not ov["held_out"]:
        why = []
        if ov["subset_overlap"]:
            why.append(f"同じ対照集合で subset が重なっています(作成 {ks.subset}・確認 {subset})")
        if ov["hash_overlap"]:
            why.append(f"作成に使った例と同じ (system, プロンプト) が {ov['hash_overlap']} 個あります")
        msg = "key-check: 作成に使った例と重なっています(" + "、".join(why) + ")。AUC は甘くなります"
        if strict:
            raise GneError("BAD_ARGS", msg, "鍵を subset even で作ったなら subset odd か、別の対照集合で確かめてください")
        warn(msg)
        notes.append("作成に使った例を含む集合で測りました(held_out: false、甘い値)")
    run = _run_plan(session, sets, layers=Ls, target=ks.target, skip=skip, max_tokens=max_tokens, samples=0,
                    project=M, rms_groups=(), progress=progress, start=START_TOKENS)
    G = run["groups"]
    layers_out = []
    for L in Ls:
        ex = {g: (np.stack([p.mean(axis=0) for p in G[g]["proj"][L]]) if G[g]["proj"][L] else np.zeros((0, M[L].shape[0])))
              for g in G}
        tok = {g: (np.concatenate(G[g]["proj"][L], axis=0) if G[g]["proj"][L] else np.zeros((0, M[L].shape[0])))
               for g in G}
        res = _layer_result(ks, L, ex, tok, n_ps)
        res["start_rho"] = _start_rho(G, L)
        layers_out.append(res)
    counts = {g: {"examples": len(G[g]["index"]), "rows": int(G[g]["rows"]), "of": int(G[g]["n_examples"])} for g in G}
    ga = G.get("A")
    if ks.target.startswith("expert"):                      # 専門家の鍵は層ごとの行(専門家 e が選ばれた位置だけ)
        for L in Ls:
            na, ra = (len(ga["index_l"][L]), int(ga["rows_l"][L])) if ga else (0, 0)
            if few_side_a(na, ra):
                warn(few_side_a_message(na, ra, where=f"L{L}: ", check=True))
    else:
        na, ra = (len(ga["index"]), int(ga["rows"])) if ga else (0, 0)
        if few_side_a(na, ra):
            warn(few_side_a_message(na, ra, check=True))
    hot = _start_hot(layers_out, {g: G[g]["positions"] for g in G},
                     auto=((ks.info.get("dont_touch_start") or {}).get("groups") or {}))
    if hot:
        warn(_start_hot_message(hot))
    return _check_summary(ks, Ls, layers_out, ov, cs.name, cs.sha, pos_a.text, pos_b.text, skip, max_tokens, mode, counts,
                          pinfo["notes"] + notes, warnings, t0, run["replies"], iv_desc)


def _start_rho(G: dict, L: int) -> dict | None:
    """key-check の層 L の、側 B と dont_touch の群の返事の書き出しでの ρ(主の鍵): 書き出しの位置 j(返事の j 番目のトークンを
    決める位置、0 = 生成プロンプトの最後のトークン)ごとの中央値・最大・例の数。{"B": {...}, "dont_touch": {名前: {...}}}。
    dont_touch の ρ の中央値(dont_touch_median)は返事の全位置の中央値なので、書き出しの 1 位置で合っても目立たない(F138)。"""
    out: dict = {}
    for g, d in G.items():
        st = d.get("start")
        if (g != "B" and not g.startswith("dt:")) or not st or st.get("proj") is None:
            continue
        k = int(st["k"])
        per: list[list[float]] = [[] for _ in range(k)]
        for offs, pj in st["proj"].get(L) or []:
            for j, v in zip(offs, pj[:, 0]):
                if 0 <= int(j) < k:
                    per[int(j)].append(float(v))
        row = {"median": [float(np.median(v)) if v else None for v in per],
               "max": [float(np.max(v)) if v else None for v in per], "n": [len(v) for v in per]}
        if g == "B":
            out["B"] = row
        else:
            out.setdefault("dont_touch", {})[g[3:]] = row
    return out or None


def _start_hot(layers_out: list[dict], positions: dict[str, str], auto=()) -> list[dict]:
    """書き出しで鍵が合っている(書き出しの位置ごとの ρ の中央値が START_RHO_WARN 以上、例 2 つ以上)層と群。群ごとに中央値が
    最大の位置 1 つ。starts は書き出しを Q に入れた群(positions last・assistant の群か、鍵の作成で返事の冒頭を自動で足した
    群 auto。名前で照らすので、key-check の対照集合の同じ名前の群)か。"""
    auto = set(auto)
    hot = []
    for r in layers_out:
        sr = r.get("start_rho") or {}
        groups = ([("B", "B", sr["B"])] if sr.get("B") else []) + [(f"dt:{n}", n, v) for n, v in (sr.get("dont_touch") or {}).items()]
        for g, label, v in groups:
            cand = [(m, j) for j, (m, n) in enumerate(zip(v["median"], v["n"])) if m is not None and n >= 2 and m >= START_RHO_WARN]
            if cand:
                m, j = max(cand)
                pos = positions.get(g)
                hot.append({"layer": r["layer"], "group": label, "j": j, "median": m,
                            "starts": (g != "B" and label in auto) or (bool(pos) and parse_positions(pos).kind in _PROMPT_END)})
    return hot


def _start_hot_message(hot: list[dict]) -> str:
    """key-check で、書き込みたくない位置(側 B)・dont_touch の返事の冒頭で鍵が反応しているときの警告の文。dont_touch の
    冒頭で反応するのは、書き込みたくない場面への漏れ(F134・F138)。書き込みたくない位置(鍵を作った場面の、書き込みたい位置
    以外)の冒頭だけで反応するのは、その場面の返事の冒頭で方向を書くということで、書き込みたい位置が冒頭に偏った鍵では書いた
    後も鍵が反応し続けて繰り返しに落ちやすい(F137)。"""
    def fmt(hs):
        return "、".join(f"L{h['layer']} {'書き込みたくない位置' if h['group'] == 'B' else h['group']} {h['median']:+.2f}"
                        f"(位置 {h['j']})" for h in hs[:6]) + ("、…" if len(hs) > 6 else "")
    dts = [h for h in hot if h["group"] != "B"]
    bs = [h for h in hot if h["group"] == "B"]
    fix = ("ふつうの質問を dont_touch に足して、鍵を作り直してください(dont_touch の質問の、返事を書き始める直前の位置は、"
           "鍵の作成で自動で Q に入ります。dont_touch_start を切っていれば戻します)")
    if dts:
        msg = (f"dont_touch の返事の冒頭で鍵が反応しています(返事の冒頭での ρ の中央値 {fmt(dts)}。位置 0 は返事を書き始める"
               "直前の位置)。書き込みたくない場面でも、返事の最初で方向を書いて言い回しが変わります(dont_touch の ρ の中央値は"
               "返事の全位置の中央値なので、返事の冒頭の 1 位置だけで反応しても目立ちません)。" + fix)
        if bs:
            msg += f"。書き込みたくない位置の返事の冒頭でも反応しています({fmt(bs)})"
        if any(h["starts"] for h in dts):
            msg += "。返事の冒頭を Q に入れた群でも保留した例で反応しているので、その群の質問を増やしてください"
        return msg
    return (f"書き込みたくない位置(鍵を作った場面の、書き込みたい位置以外)の返事の冒頭で鍵が反応しています(返事の冒頭での ρ の"
            f"中央値 {fmt(bs)}。位置 0 は返事を書き始める直前の位置)。その場面の返事では、書き込みたい位置でなくても返事の冒頭で"
            "方向を書きます。書き込みたい位置が返事の冒頭に偏った鍵では、書いた後も状態が書き込みたい位置の状態から抜けずに鍵が"
            "反応し続け、繰り返しに落ちやすくなります(keyed-write.md §14)。dont_touch の返事の冒頭で反応していなければ、"
            "ほかの場面には漏れていません。生成した返事で繰り返しを数えてください")


def _pct(v: list, q: float) -> float | None:
    return float(np.percentile(v, q)) if v else None


def _layer_result(ks, L: int, ex: dict[str, np.ndarray], tok: dict[str, np.ndarray] | None, n_ps: int) -> dict:
    """層 L の確認の結果。ex・tok は群 → (n, 2 + 擬似鍵の本数) の ρ(列 0 = 主の鍵、1 = B の 2 分割、2.. = 擬似鍵)。
    選択性 selectivity = 例ごとの ρ の平均の差(側 A − 側 B)。どの鍵も自分の μ で κᵀμ = 1 なので、本物の鍵と擬似鍵で
    同じ物差しになる(書き込む量 α(κᵀx)d の A と B の差、仕様 E.2 の「効きの差」)。beats_pseudo は選択性が擬似鍵の p95 を
    超えること(と AUC > 0.5)。例単位の AUC は A・B の位置がよく分かれる対照集合では擬似鍵でも 1 に張り付くので、AUC での
    比べは beats_pseudo_auc(トークン単位があればトークン単位)に分けて出す。"""
    A = ex.get("A", np.zeros((0, 2 + n_ps)))
    B = ex.get("B", np.zeros((0, 2 + n_ps)))
    both = bool(A.shape[0] and B.shape[0])
    a_ex = K.auc(A[:, 0], B[:, 0])
    has_tok = tok is not None and tok.get("A") is not None and tok.get("B") is not None and tok["A"].shape[0] and tok["B"].shape[0]
    a_tok = K.auc(tok["A"][:, 0], tok["B"][:, 0]) if has_tok else None
    sel = float(A[:, 0].mean() - B[:, 0].mean()) if both else None
    ps_ex = [x for x in (K.auc(A[:, 2 + j], B[:, 2 + j]) for j in range(n_ps)) if x is not None] if both else []
    ps_tok = [x for x in (K.auc(tok["A"][:, 2 + j], tok["B"][:, 2 + j]) for j in range(n_ps)) if x is not None] if has_tok else []
    ps_sel = [float(A[:, 2 + j].mean() - B[:, 2 + j].mean()) for j in range(n_ps)] if both else []
    bs = K.auc(A[:, 1], B[:, 1]) if both and np.any(A[:, 1] != 0) else None
    dts = {g[3:]: _stats(v[:, 0]) for g, v in ex.items() if g.startswith("dt:")}
    hist_src = tok if tok is not None else ex
    p95_sel = _pct(ps_sel, 95)
    ref_auc, ref_ps = (a_tok, ps_tok) if has_tok else (a_ex, ps_ex)
    rho = {"A": _stats(A[:, 0]), "B": _stats(B[:, 0]), "dont_touch": dts}
    if tok is not None:
        rho["A_token"] = _stats(tok["A"][:, 0]) if tok.get("A") is not None else None
        rho["B_token"] = _stats(tok["B"][:, 0]) if tok.get("B") is not None else None
    return {"layer": L, "auc": a_ex, "auc_token": a_tok, "selectivity": sel,
            "pseudo_auc": {"n": len(ps_ex), "p50": _pct(ps_ex, 50), "p95": _pct(ps_ex, 95), "values": ps_ex},
            "pseudo_auc_token": {"n": len(ps_tok), "p50": _pct(ps_tok, 50), "p95": _pct(ps_tok, 95)},
            "pseudo_selectivity": {"n": len(ps_sel), "p50": _pct(ps_sel, 50), "p95": p95_sel, "values": ps_sel},
            "beats_pseudo": None if p95_sel is None or sel is None else bool(sel > p95_sel and (a_ex or 0) > 0.5),
            "beats_pseudo_auc": None if not ref_ps or ref_auc is None else bool(ref_auc > _pct(ref_ps, 95)),
            "b_split_auc": bs, "rho": rho,
            "dont_touch_median": {g: (s["median"] if s else None) for g, s in dts.items()},
            "values": {g: [float(x) for x in v[:, 0]] for g, v in ex.items()},
            "hist": _hist({g: v[:, 0] for g, v in hist_src.items()}),
            "kn": ks.kn.get(L), "gain": ks.gain.get(L)}


def _check_summary(ks, Ls, layers_out, ov, cname, csha, pos_a, pos_b, skip, max_tokens, mode, counts, notes, warnings,
                   t0, replies, iv_desc=None) -> dict:
    def rank(r):                       # 例単位の AUC、トークン単位の AUC、選択性の擬似鍵 p95 からの余裕の順(AUC は 1 に張り付くことがある)
        ps = r["pseudo_selectivity"]["p95"]
        return (r["auc"], r["auc_token"] or 0.0, (r["selectivity"] or 0.0) - (ps or 0.0), r["layer"])
    aucs = [(rank(r), r["layer"]) for r in layers_out if r["auc"] is not None]
    beats = [r["beats_pseudo"] for r in layers_out if r["beats_pseudo"] is not None]
    return {"key": ks.spec or ks.name, "sha": ks.sha, "target": ks.target, "contrast": cname, "contrast_sha": csha,
            "subset": ov.get("subset"), "mode": mode, "positions": pos_a, "positions_b": pos_b, "skip": skip,
            "max_tokens": max_tokens, "held_out": bool(ov["held_out"]), "overlap": ov,
            "built_on": {"contrast_sha": ks.contrast_sha, "subset": ks.subset, "q_mode": ks.q_mode},
            "layers": layers_out, "best_layer": max(aucs)[1] if aucs else None,
            "all_beat_pseudo": (all(beats) if beats else None), "n_pseudo": ks.n_pseudo, "counts": counts,
            "replies": replies, "iv": iv_desc, "notes": notes, "warnings": warnings,
            "seconds": time.perf_counter() - t0}


# --- 評価(gne/keyed_eval.py、CHANGE-07 E)からの入口 ------------------------------------------------------------

def opposite_subset(subset: str | None) -> str:
    """鍵の作成に使った subset の反対側(even ↔ odd)。all・不明は all(作成に使った例と重なる)。"""
    return {"even": "odd", "odd": "even"}.get(subset or "", "all")


def eval_plan(session, key, contrast, *, subset: str | None = None, replies_a=None, replies_b=None,
              warn: Callable[[str], None] | None = None) -> dict:
    """評価(CHANGE-07 E)用の公開の入口: 鍵の作成の条件(positions・positions_b・skip・max_tokens・mode)で、対照集合
    (contrast: パス・ContrastSet・KeySpec)のどの会話をどの位置の群で読むか(_Set のリスト)を返す。捕捉はしない(呼ぶ側が
    同じ列を介入ごとに読み直す)。subset を省くと鍵の作成と反対側(even ↔ odd)。key は鍵の名前か KeySet。
    戻り値 {"sets", "mode", "positions", "positions_b", "skip", "max_tokens", "subset", "overlap"(held_out_info),
    "notes", "dont_touch", "contrast", "contrast_sha"}。"""
    ks = key if isinstance(key, K.KeySet) else K.load_key(session, str(key))
    warn = warn or session.warn
    spec = _as_spec(contrast)
    cs = spec.contrast
    sub = opposite_subset(ks.subset) if subset is None else subset
    if sub not in SUBSETS:
        raise GneError("BAD_ARGS", f"subset は {' / '.join(SUBSETS)} です({sub!r})")
    pos_a = parse_positions(ks.positions or "response")
    mode, pos_b = _resolve_mode(pos_a, ks.info.get("positions_b"))
    sets, pinfo = _plan(session, spec, mode, pos_a, pos_b, sub, replies_a, replies_b, None, None, warn)
    ov = held_out_info(ks, cs.sha, sub, [h for s in sets for h in s.hashes])
    return {"sets": sets, "mode": mode, "positions": pos_a.text, "positions_b": pos_b.text, "skip": int(ks.skip or 0),
            "max_tokens": int(ks.max_tokens), "subset": sub, "overlap": ov, "notes": pinfo["notes"],
            "dont_touch": pinfo["dont_touch"], "contrast": cs.name, "contrast_sha": cs.sha}
