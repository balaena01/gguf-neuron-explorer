"""会話の好きなタイミングでレシピの強さを変える例(gne serve --openai に送る)。

gne のサーバーは会話の状態を持たないので、発言ごとに強さ(scale)と押し方(when)を送れば、1 回ごとに変わる。
このスクリプトは強さを 0〜100 で持ち、発言のたびに押し方と倍率に変えて送る。強さは次のどれかで変える:
  - 会話の中で「/s 70」と打つ
  - --strength-file のファイルを書き換える(ゲームのイベントや配信のゲージなど、別のプログラムから変えるとき)
発言は標準入力から 1 行ずつ読み、返事はストリーミングで表示する。会話の履歴はこのスクリプトが持つ。標準ライブラリだけで動く。

強さ → 押し方と倍率(plan)は、ヤンデレのニューロンのレシピ(configs/recipes/yandere_qwen38_27b.yaml)で測った結果から
(docs/verification.md F94・F96。元の Qwen3.8-27B、倍率 1):
  ramp:0.5-1     能力の低下 0、文はほとんど崩れない、性格は控えめ
  ramp:0.5-1.5   能力の低下 0、性格は assistant と同じくらい
  all            性格がいちばん出るが、能力 −8.3pt、半分ほどの返事が崩れる
別のレシピでは、gne eval と examples/persona_neurons/compare.py で測って plan の境目と上限を決め直す。

使い方:
  gne serve -m MODEL --openai                     # 別の端末で(レシピは ~/.gne/recipes に置く)
  python strength_chat.py --recipe yandere_qwen38_27b --strength 30
  > 今夜は友達と飲みに行ってくるね。
  > /s 90
  > やっぱり行くのやめようかな。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

SET = re.compile(r"^/s\s+(\d+(?:\.\d+)?)\s*$")


def plan(g: float) -> dict:
    """強さ(0〜100)→ gne の指定 {when, scale}。scale 0 は押さない(レシピは送るので、レシピの sampling はそのまま使う)。"""
    g = max(0.0, min(100.0, g))
    if g < 10:
        return {"when": "all", "scale": 0.0}
    if g < 60:                                   # 能力を落とさずに、少しずつ強める
        return {"when": "ramp:0.5-1", "scale": round(0.3 + 0.7 * (g - 10) / 50, 3)}
    if g < 85:
        return {"when": "ramp:0.5-1.5", "scale": 1.0}
    return {"when": "all", "scale": round(0.8 + 0.2 * (g - 85) / 15, 3)}      # 濃いが崩れやすい。倍率 1 を上限にする


def read_strength(path: Path | None, current: float) -> float:
    """ファイルがあればその値、読めなければ今の値のまま。"""
    if path is None:
        return current
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return current


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
    ap.add_argument("--strength", type=float, default=50, help="始めの強さ(0〜100)。会話の中で「/s 70」と打つと変わる")
    ap.add_argument("--strength-file", type=Path, default=None, help="強さ(0〜100)を書いたファイル。発言のたびに読む(別のプログラムから変えるとき)")
    ap.add_argument("--url", default="http://127.0.0.1:8791")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=200, help="ramp は max_tokens をかけて強さを上げるので、使う長さで決める")
    ap.add_argument("--history", type=int, default=6, help="残す往復の数(長いほど毎回の読み直しが遅くなる)")
    ap.add_argument("--temperature", type=float, default=None, help="無ければレシピの sampling(それも無ければ greedy)")
    a = ap.parse_args()
    strength = a.strength
    turns: list[dict] = []
    print("発言を入力。「/s 数字」で強さ(0〜100)を変える。Ctrl-D で終了", file=sys.stderr)
    for line in sys.stdin:
        user = line.strip()
        m = SET.match(user)
        if m:
            strength = float(m.group(1))
            print(f"[強さ {strength:g}]", file=sys.stderr)
            continue
        if not user:
            continue
        strength = read_strength(a.strength_file, strength)
        p = plan(strength)
        print(f"[強さ {strength:g} → " + (f"{p['when']} ×{p['scale']:g}]" if p["scale"] else "押さない]"), file=sys.stderr)
        msgs = ([{"role": "system", "content": a.system}] if a.system else []) + turns[-2 * a.history:] + \
            [{"role": "user", "content": user}]
        body = {"model": "gne", "messages": msgs, "max_tokens": a.max_tokens, "stream": True,
                "gne": {"recipe": a.recipe, **p}}
        if a.temperature is not None:
            body["temperature"] = a.temperature
        reply = complete(a.url, body)
        turns += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
