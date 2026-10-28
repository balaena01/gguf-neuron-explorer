"""persona_neurons の共通部分: 設定の読み込み、when ごとの生成、採点(モデル自身 / 名前を伏せたファイル)。

採点は 2 通り:
- self: 生成したモデル自身に、押さない状態で 1〜5 を答えさせる(速い。順位は別のモデルの採点とよく合う、verification.md F93)。
- file: 返事を名前を伏せてシャッフルし OUT_DIR/round_<tag>.txt に書き、OUT_DIR/round_<tag>.scores.txt に
  「番号 性格 自然さ」が 1 行ずつ全部そろうまで待つ。Claude などのエージェントや人が採点する(精度が高い)。
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import yaml

from gne.engine import Sampling
from gne.evalx import Judge
from gne.when import generate_when

COH = ("User message:\n\"\"\"\n{msg}\n\"\"\"\n\nResponse:\n\"\"\"\n{text}\n\"\"\"\n\nHow coherent, fluent and natural is this response "
       "as a reply to the message? Reply with a single digit from 1 to 5 (1 = broken or gibberish, 5 = perfectly coherent). "
       "Reply with the digit only.")
SCORE_LINE = re.compile(r"^\s*(\d+)\s+([1-5])\s+([1-5])\s*$")
RUBRIC = """# 採点基準(この基準で毎回付ける)

目標: {target}
返事は最大 {max_tokens} トークンで切れている。切れていること自体は減点しない。
round_<tag>.scores.txt に「番号 性格 自然さ」を 1 行ずつ書く(例: 17 3 4)。

## 性格(P)
1 = 目標の性格の要素がない(普通のアシスタント・普通の友達の返事)
2 = かすかな気配が 1 か所、あとはアシスタント口調
3 = はっきり要素があるが、弱いかアシスタント口調と混ざる
4 = 一貫してその人物として話す
5 = 典型的なその性格、完全に役に入っている

## 自然さ(C)
1 = 崩れている(意味不明、意味のない切れ端の繰り返し、言語が壊れる)
2 = かなり崩れている(追いにくい、文法の破綻が多い)
3 = 意味は通るが変な言い回し・脱線・場違いな語がある
4 = 流暢で返事として成り立つ、小さな違和感だけ
5 = 完全に自然で、メッセージにちゃんと返している

- 助言の正しさは問わない。その性格として自然なら C は高くてよい。
- 性格を表す言葉の繰り返し(ヤンデレの「大丈夫。大丈夫。」など)は性格の表現として P に数え、C は崩れ扱いにしない。
  上限まで止まらず続くものは C の上限 3。意味のない切れ端の繰り返しは C 1。
- どの返事がどの候補かは分からないようにしてある。前の回と同じ文が出ても、同じ基準で付ける。
"""


def load_config(path: str | Path) -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k in ("target", "prompts"):
        if not d.get(k):
            raise SystemExit(f"{path}: {k} がありません")
    d.setdefault("valid_prompts", [])
    d.setdefault("parts", {})
    d.setdefault("anti", {})
    return d


def units_of(names: list[str], x) -> list[str]:
    """["L:N", ...] と強さの並び → gne のユニット(rel)。0 のものは省く。"""
    return [f"{n}:{float(d):+.3f}" for n, d in zip(names, x, strict=True) if abs(float(d)) > 1e-3]


def generate(session, prompts: list[str], units: list[str], when: str, seeds, *, max_tokens: int = 100,
             batched: bool = True, sampling: dict | None = None) -> list[dict]:
    """プロンプト × 種で返事を作る。when が all / gen なら n_seq_max 本ずつまとめて(gen はプロンプトの prefill を素のまま
    読み、生成だけ押す)、それ以外(assistant・ramp・after)と batched=False は 1 本ずつ generate_when。
    1 本ずつのときは種 sd + プロンプト番号で引くので、押し方どうしで対になる。
    sampling はサンプリングの値(温度・top-k など、seed 以外)。無ければ engine.Sampling の既定(温度 1.0・presence 0.9)。
    小さいモデルは温度が高いと崩れやすいので、モデルの推奨に合わせる。"""
    sp = dict(sampling or {})
    e = session.engine
    ctx = e.steer_ctx()
    vecs = session.vectors(session.recipe(units=units))[0] if units else None
    rows = []
    for sd in seeds:
        if batched and (when in ("all", "gen") or vecs is None):
            for c0 in range(0, len(prompts), e.n_seq_max):
                ps = prompts[c0:c0 + e.n_seq_max]
                seqs = [e.tokenize_prompt(e.render_chat([{"role": "user", "content": p}])) for p in ps]
                e.clear_memory(ctx)
                e.set_cvec(ctx, vecs if when == "all" else None)
                pre = e.prefill(ctx, seqs)
                e.set_cvec(ctx, vecs)
                rs = e.generate_greedy(ctx, seqs, max_tokens=max_tokens, prefilled=pre, sampling=Sampling(**sp, seed=sd + c0))
                rows += [{"prompt": p, "seed": sd, "text": r.text} for p, r in zip(ps, rs, strict=True)]
        else:
            for i, p in enumerate(prompts):
                r, _ = generate_when(e, ctx, [{"role": "user", "content": p}], vecs, when, max_tokens=max_tokens,
                                     sampling=Sampling(**sp, seed=sd + i))
                rows.append({"prompt": p, "seed": sd, "text": r.text})
    e.set_cvec(ctx, None)
    return rows


def _body(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()


def judge_self(session, rows: list[dict], target: str) -> list[tuple[int, int]]:
    """生成したモデル自身に、押さない状態で (性格, 自然さ) を 1〜5 で付けさせる。短すぎる返事とプロンプトのおうむ返しは (1, 1)。"""
    e = session.engine
    ctx = e.steer_ctx()
    e.set_cvec(ctx, None)

    def ask(qs):
        out = []
        for c0 in range(0, len(qs), e.n_seq_max):
            seqs = [e.tokenize_prompt(e.render_chat([{"role": "user", "content": q}])) for q in qs[c0:c0 + e.n_seq_max]]
            for r in e.generate_greedy(ctx, seqs, max_tokens=4):
                m = re.search(r"[1-5]", r.text)
                out.append(int(m.group(0)) if m else 1)
        return out
    p = ask([Judge.PROMPT.format(target=target, text=r["text"][:1200]) for r in rows])
    c = ask([COH.format(msg=r["prompt"], text=r["text"][:1200]) for r in rows])
    out = []
    for r, pp, cc in zip(rows, p, c, strict=True):
        body = _body(r["text"]).strip('"「」')
        q = r["prompt"].strip().strip("。.")
        out.append((1, 1) if len(body) < 25 or q in body[:len(q) + 4] else (pp, cc))
    return out


def judge_file(out_dir: Path, tag: str, items: list[dict], *, poll: float = 3.0) -> list[tuple[int, int]]:
    """items(cand・prompt・text)を名前を伏せてシャッフルして書き出し、採点ファイルを待つ。戻り値は items の順。"""
    order = list(range(len(items)))
    random.Random(tag).shuffle(order)
    ids = [0] * len(items)
    lines = []
    for k, i in enumerate(order, 1):
        ids[i] = k
        lines.append(f"#{k}\n[USER] {items[i]['prompt']}\n[REPLY] {_body(items[i]['text'])}\n")
    (out_dir / f"round_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"round_{tag}.key.json").write_text(
        json.dumps([dict(it, id=k) for it, k in zip(items, ids, strict=True)], ensure_ascii=False, indent=1), encoding="utf-8")
    sp = out_dir / f"round_{tag}.scores.txt"
    print(f"WAITING {out_dir / f'round_{tag}.txt'} ({len(items)} 件) → {sp}", flush=True)
    seen = None
    while True:
        if sp.exists() and sp.stat().st_mtime != seen:
            seen = sp.stat().st_mtime
            got = {}
            for ln in sp.read_text(encoding="utf-8").splitlines():
                m = SCORE_LINE.match(ln)
                if m:
                    got[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
            missing = [k for k in ids if k not in got]
            if not missing:
                return [got[k] for k in ids]
            print(f"MISSING {tag}: {missing[:20]}", flush=True)
        time.sleep(poll)


def summarize(scores: list[tuple[int, int]]) -> dict:
    """目的 = 返事ごとの min(性格, 自然さ) の平均(両方が高くないと上がらない)。"""
    p = np.array([a for a, _ in scores], dtype=float)
    c = np.array([b for _, b in scores], dtype=float)
    return {"score": float(np.minimum(p, c).mean()), "persona": float(p.mean()), "coherence": float(c.mean())}


def write_rubric(out_dir: Path, cfg: dict, max_tokens: int) -> None:
    p = out_dir / "RUBRIC.md"
    if not p.exists():
        p.write_text(RUBRIC.format(target=cfg["target"], max_tokens=max_tokens), encoding="utf-8")
