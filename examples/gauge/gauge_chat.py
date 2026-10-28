"""配信のゲージなどに合わせて、発言ごとにレシピの強さと押し方を変える例(gne serve --openai に送る)。

ゲージ(0〜100)は発言のたびにファイルから読む。配信のオーバーレイや別のスクリプトがこのファイルを書き換える想定。
発言は標準入力から 1 行ずつ読み、返事はストリーミングで表示する。会話の履歴はこのスクリプトが持つ
(サーバは状態を持たないので、毎回の強さがそのまま効く)。標準ライブラリだけで動く。

ゲージ → 強さと押し方(plan)は、ヤンデレのニューロンのレシピ(configs/recipes/yandere_qwen38_27b.yaml)で測った結果から
(docs/verification.md F94・F96。元の Qwen3.8-27B、倍率 1):
  ramp:0.5-1     能力の低下 0、文はほとんど崩れない、性格は控えめ
  ramp:0.5-1.5   能力の低下 0、性格は assistant と同じくらい
  all            性格がいちばん出るが、能力 −8.3pt、半分ほどの返事が崩れる
別のレシピでは、gne eval と examples/persona_neurons/compare.py で測って plan の境目と上限を決め直す。

使い方:
  gne serve -m MODEL --openai                     # 別の端末で(レシピは ~/.gne/recipes に置く)
  echo 30 > gauge.txt
  python gauge_chat.py --recipe yandere_qwen38_27b --gauge-file gauge.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def plan(g: float) -> dict:
    """ゲージ(0〜100)→ gne の指定 {when, scale}。scale 0 は押さない(レシピは送るので、レシピの sampling はそのまま使う)。"""
    g = max(0.0, min(100.0, g))
    if g < 10:
        return {"when": "all", "scale": 0.0}
    if g < 60:                                   # 能力を落とさずに、少しずつ強める
        return {"when": "ramp:0.5-1", "scale": round(0.3 + 0.7 * (g - 10) / 50, 3)}
    if g < 85:
        return {"when": "ramp:0.5-1.5", "scale": 1.0}
    return {"when": "all", "scale": round(0.8 + 0.2 * (g - 85) / 15, 3)}      # 濃いが崩れやすい。倍率 1 を上限にする


def read_gauge(path: Path) -> float:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0.0


def complete(url: str, body: dict) -> str:
    """/v1/chat/completions にストリーミングで送り、届いた文字をそのまま表示して、返事の全文を返す。"""
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    text = ""
    try:
        with urllib.request.urlopen(req) as res:
            for raw in res:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if "error" in d:
                    raise SystemExit(f"エラー: {d['error'].get('message')}")
                for ch in d.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        text += piece
                        print(piece, end="", flush=True)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"エラー {e.code}: {e.read().decode('utf-8', 'replace')}") from e
    print()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, help="~/.gne/recipes に保存したレシピの名前")
    ap.add_argument("--gauge-file", type=Path, required=True, help="ゲージ(0〜100)を書いたファイル。発言のたびに読む")
    ap.add_argument("--url", default="http://127.0.0.1:8791")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=200, help="ramp は max_tokens をかけて強さを上げるので、使う長さで決める")
    ap.add_argument("--history", type=int, default=6, help="残す往復の数(長いほど毎回の読み直しが遅くなる)")
    ap.add_argument("--temperature", type=float, default=None, help="無ければレシピの sampling(それも無ければ greedy)")
    a = ap.parse_args()
    turns: list[dict] = []
    print("発言を入力(Ctrl-D で終了)", file=sys.stderr)
    for line in sys.stdin:
        user = line.strip()
        if not user:
            continue
        g = read_gauge(a.gauge_file)
        p = plan(g)
        print(f"[ゲージ {g:g} → " + (f"{p['when']} ×{p['scale']:g}]" if p["scale"] else "押さない]"), file=sys.stderr)
        msgs = ([{"role": "system", "content": a.system}] if a.system else []) + turns[-2 * a.history:] + \
            [{"role": "user", "content": user}]
        body = {"model": "gne", "messages": msgs, "max_tokens": a.max_tokens, "stream": True}
        if a.temperature is not None:
            body["temperature"] = a.temperature
        body["gne"] = {"recipe": a.recipe, **p}
        reply = complete(a.url, body)
        turns += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
