"""実地テスト(gne fieldtest): 本番と同じ動かし方で、モデルのファイルが実際に使えるかを確かめて、ファイルどうしを比べる。

gne eval は同じファイルにレシピの倍率を掛けて、短い問題で壊れていないかを速く見る。こちらは llama-server を立て、思考あり・
サンプリングありで、次を測る。短い問題では出ない崩れ(長い思考の途中のループ、上限まで終わらない思考、ツール呼び出しの崩れ)を見る。

- 長い作業: 上限で切れた割合、短い周期と長い周期のループ、本文が空の割合
- ツール呼び出し: 呼ぶべきときに呼ぶか、名前と引数は正しいか、呼ばなくてよいときに呼ばないか、結果を返した後に続けられるか
- 自分の質問集(任意): ループ、本文が空の割合、上限で切れた割合、正規表現の一致率
- ベンチマーク(任意、GSM8K・MMLU-Pro): 正答率と、片方のファイルだけが正解した問題の数。データは取得して手元にだけ置く
- perplexity: 参照テキストをそのまま読ませて測る(gne で読むので、焼き込んだファイルも直接測れる)

1 つのファイルずつ順に GPU に載せる。生成した文章は別のファイルに書き、結果の表には数字だけを出す。"""

from __future__ import annotations

import atexit
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from .reverse import check_questions, turns
from .steer import GneError

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "fieldtest"
LONG_TASKS = CONFIG / "long_tasks.json"
TOOL_TASKS = CONFIG / "tool_tasks.json"
ChatFn = Callable[..., tuple[dict, str, int]]


def find_llama_server(path: str | None = None) -> str | None:
    if path:
        return path if Path(path).exists() else None
    env = os.environ.get("GNE_LLAMA_SERVER")
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve().parent.parent / "build" / "llama.cpp" / "build" / "bin" / "llama-server"
    if here.exists():
        return str(here)
    return shutil.which("llama-server")


# --- 崩れの物差し ------------------------------------------------------------------------------------------

def long_repeat(s: str, n: int = 64, step: int = 32) -> float:
    """後半の 64 文字の塊のうち、前にも出てきた割合。末尾の窓より周期の長いループ(段落の繰り返し)を見つける。ふつうの文章は 0 近く。"""
    if len(s) < 4 * n:
        return 0.0
    half = len(s) // 2
    seen = {s[i:i + n] for i in range(0, half - n + 1)}
    chunks = [s[i:i + n] for i in range(half, len(s) - n + 1, step)]
    return sum(c in seen for c in chunks) / len(chunks) if chunks else 0.0


def tail_distinct(s: str, n: int = 10, tail: int = 1500) -> float:
    """末尾 tail 文字の n 文字の並びのうち、異なるものの割合。ふつうの文章は 1 近く、ループすると下がる。"""
    s = s[-tail:]
    g = [s[i:i + n] for i in range(len(s) - n + 1)]
    return len(set(g)) / len(g) if g else 1.0


def _mean(xs: Sequence) -> float | None:
    return float(np.mean([float(x) for x in xs])) if len(xs) else None


# --- 並列に回す(進み具合と中止はジョブのスレッドで見る) ---------------------------------------------------------

def _map(fn: Callable, items: Sequence, n_parallel: int, tick: Callable[[int], None]) -> list:
    out: list = [None] * len(items)
    ex = ThreadPoolExecutor(max(1, n_parallel))
    try:
        futs = {ex.submit(fn, i, x): i for i, x in enumerate(items)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            tick(1)
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)      # 中止: サーバを止めれば、走っている要求もすぐ終わる
        raise
    ex.shutdown(wait=True)
    return out


# --- 測る区画(chat を受け取るので、サーバなしでも試せる) ---------------------------------------------------------

def run_long(chat: ChatFn, prompts: Sequence, max_tokens: int, n_parallel: int, tick: Callable[[int], None],
             seed: int = 1000) -> tuple[dict, list[dict]]:
    def one(i, p):
        msg, fin, ntok = chat(turns(p), max_tokens, seed + i)
        think, body = msg.get("reasoning_content") or "", msg.get("content") or ""
        all_ = think + "\n" + body
        td, lr = tail_distinct(all_), long_repeat(all_)
        return {"prompt": p, "finish": fin, "tokens": ntok, "think_chars": len(think), "body_chars": len(body),
                "tail_distinct": td, "long_repeat": lr, "loop": td < 0.5, "long_loop": lr >= 0.5,
                "reasoning": think, "content": body}
    recs = _map(one, list(prompts), n_parallel, tick)
    return {"n": len(recs), "hit_limit": _mean([r["finish"] == "length" for r in recs]), "loop": _mean([r["loop"] for r in recs]),
            "long_loop": _mean([r["long_loop"] for r in recs]), "any_loop": _mean([r["loop"] or r["long_loop"] for r in recs]),
            "empty": _mean([r["body_chars"] == 0 for r in recs]), "tail_distinct": _mean([r["tail_distinct"] for r in recs]),
            "tokens": _mean([r["tokens"] for r in recs]), "error": _mean([r["finish"] == "error" for r in recs])}, recs


def run_tools(chat: ChatFn, spec: dict, max_tokens: int, n_parallel: int, tick: Callable[[int], None],
              seed: int = 2000) -> tuple[dict, list[dict]]:
    tools = spec["tools"]
    required = {t["function"]["name"]: t["function"]["parameters"].get("required", []) for t in tools}
    results = spec.get("results") or {}

    def one(i, t):
        msgs = turns(t["prompt"])
        msg, fin, ntok = chat(msgs, max_tokens, seed + i, tools)
        calls = msg.get("tool_calls") or []
        rec = {"prompt": t["prompt"], "expect": t.get("expect"), "finish": fin, "tokens": ntok, "n_calls": len(calls),
               "first": msg, "name_ok": False, "args_ok": False, "followup": None}
        if calls:
            f = calls[0].get("function") or {}
            rec["name_ok"] = f.get("name") == t.get("expect")
            try:
                args = json.loads(f.get("arguments") or "")
                rec["args_ok"] = isinstance(args, dict) and all(k in args for k in required.get(f.get("name"), []))
            except (TypeError, ValueError):
                rec["args_ok"] = False
        if t.get("expect") and calls:                     # 仮の結果を返して、答えるか次のツールを呼ぶかを見る
            msgs = msgs + [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}]
            for c in calls:
                name = (c.get("function") or {}).get("name")
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(results.get(name, {"ok": True}))})
            m2, fin2, ntok2 = chat(msgs, max_tokens, seed + 1000 + i, tools)
            body2 = m2.get("content") or ""
            rec["followup"] = {"finish": fin2, "tokens": ntok2, "n_calls": len(m2.get("tool_calls") or []), "body_chars": len(body2),
                               "tail_distinct": tail_distinct((m2.get("reasoning_content") or "") + "\n" + body2), "message": m2}
        return rec
    recs = _map(one, list(spec["tasks"]), n_parallel, tick)
    exp = [r for r in recs if r["expect"]]
    none = [r for r in recs if not r["expect"]]
    fu = [r["followup"] for r in exp if r["followup"]]
    return {"n": len(recs), "n_expect": len(exp), "n_none": len(none), "n_followup": len(fu),
            "called": _mean([r["n_calls"] > 0 for r in exp]), "name_ok": _mean([r["name_ok"] for r in exp]),
            "args_ok": _mean([r["args_ok"] for r in exp]), "first_hit_limit": _mean([r["finish"] == "length" for r in recs]),
            "no_tool_ok": _mean([r["n_calls"] == 0 for r in none]),
            # 仮の結果の後は、答えるか次のツールを呼べばよい。上限まで終わらないか、空で止まると壊れとみなす
            "followup_ok": _mean([f["finish"] not in ("length", "error") and (f["body_chars"] > 0 or f["n_calls"] > 0) for f in fu]),
            "followup_hit_limit": _mean([f["finish"] == "length" for f in fu]),
            "error": _mean([r["finish"] == "error" for r in recs] + [f["finish"] == "error" for f in fu])}, recs


def run_prompts(chat: ChatFn, prompts: Sequence, max_tokens: int, n_parallel: int, tick: Callable[[int], None],
                match: str | None = None, seed: int = 4000) -> tuple[dict, list[dict]]:
    rx = re.compile(match, re.IGNORECASE) if match else None

    def one(i, p):
        msg, fin, ntok = chat(turns(p), max_tokens, seed + i)
        think, body = msg.get("reasoning_content") or "", msg.get("content") or ""
        all_ = think + "\n" + body
        td, lr = tail_distinct(all_), long_repeat(all_)
        return {"prompt": p, "finish": fin, "tokens": ntok, "think_chars": len(think), "body_chars": len(body), "tail_distinct": td,
                "long_repeat": lr, "loop": td < 0.5 or lr >= 0.5, "match": bool(rx.search(body)) if rx else None,
                "reasoning": think, "content": body}
    recs = _map(one, list(prompts), n_parallel, tick)
    return {"n": len(recs), "match": _mean([r["match"] for r in recs]) if rx else None, "loop": _mean([r["loop"] for r in recs]),
            "empty": _mean([r["body_chars"] == 0 for r in recs]), "hit_limit": _mean([r["finish"] == "length" for r in recs]),
            "tokens": _mean([r["tokens"] for r in recs]), "error": _mean([r["finish"] == "error" for r in recs])}, recs


# --- ベンチマーク(能力を配布ファイルで測る) -----------------------------------------------------------------

BENCHES = {"gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test", "n": 250},
           "mmlu_pro": {"dataset": "TIGER-Lab/MMLU-Pro", "config": "default", "split": "test", "n": 300}}
_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"
_GSM_ANSWER = re.compile(rf"[Aa]nswer\s*[:：]\s*[*_\s]*\$?\s*({_NUM})")
_GSM_BOXED = re.compile(rf"\\boxed\{{\s*\\?\$?\s*({_NUM})")
_MC_ANSWER = re.compile(r"[Aa]nswer\s*[:：]\s*[*_\s]*\(?([A-J])\)?(?![A-Za-z])")
_MC_BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?\(?([A-J])\)?")


def _number(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


def extract_answer(bench: str, text: str) -> str | None:
    """返事の本文から答えを取り出す(「Answer: …」か \\boxed の最後のもの)。無ければ None(不正解に数える)。"""
    pats = (_GSM_ANSWER, _GSM_BOXED) if bench == "gsm8k" else (_MC_ANSWER, _MC_BOXED)
    for pat in pats:
        got = pat.findall(text or "")
        if got:
            return got[-1]
    return None


def is_correct(bench: str, answer: str | None, gold: str) -> bool:
    if answer is None:
        return False
    if bench == "gsm8k":
        a, b = _number(answer), _number(gold)
        return a is not None and b is not None and abs(a - b) < 1e-6
    return answer == gold


def bench_item(bench: str, idx: int, row: dict) -> dict:
    if bench == "gsm8k":
        gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        prompt = f"{row['question']}\n\nSolve the problem. End your reply with a final line of the form: Answer: <number>"
    else:
        opts = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(row["options"]))
        gold = str(row["answer"]).strip()
        prompt = f"{row['question']}\n\n{opts}\n\nChoose the best option. End your reply with a final line of the form: Answer: <letter>"
    return {"id": str(idx), "prompt": prompt, "gold": gold}


def _fetch_rows(dataset: str, config: str, split: str, offset: int, length: int, retries: int = 8) -> dict:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    err: Exception | None = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{_ROWS_URL}?{q}", timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, OSError, ValueError) as e:      # 429 は言われた時間だけ待つ
            err = e
            after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            wait = float(after) if after and str(after).replace(".", "", 1).isdigit() else 5.0 * 2 ** i
            time.sleep(min(120.0, wait))
    raise GneError("DOWNLOAD_FAILED", f"{dataset} を取得できません: {err}", "ネットワークを確かめてください(一度取得すれば手元の写しを使います)")


def load_bench(bench: str, cache_dir: Path, n: int | None = None, seed: int = 0, fetch: Callable = _fetch_rows,
               pace: float = 1.0) -> list[dict]:
    """ベンチマークの問題を、種を固定して n 問選ぶ。最初の 1 回だけ取得し、<cache_dir>/ に写しを置く(リポジトリには入れない)。"""
    if bench not in BENCHES:
        raise GneError("BAD_ARGS", f"ベンチマーク {bench!r} はありません({', '.join(BENCHES)})")
    b = BENCHES[bench]
    n = int(n or b["n"])
    path = Path(cache_dir) / f"{bench}-n{n}-s{seed}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    first = fetch(b["dataset"], b["config"], b["split"], 0, 1)
    total = int(first["num_rows_total"])
    idx = sorted(int(x) for x in np.random.default_rng(seed).choice(total, size=min(n, total), replace=False))
    rows: dict[int, dict] = {}
    i = 0
    while i < len(idx):                                 # 1 回で 100 行まで。近い番号はまとめて取る
        start = idx[i]
        if rows:
            time.sleep(pace)                            # 続けて取るとレート制限(429)に当たる
        got = fetch(b["dataset"], b["config"], b["split"], start, 100)
        for r in got["rows"]:
            rows[int(r["row_idx"])] = r["row"]
        while i < len(idx) and idx[i] < start + 100:
            i += 1
    items = [bench_item(bench, k, rows[k]) for k in idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8")
    return items


def run_bench(chat: ChatFn, bench: str, items: Sequence[dict], max_tokens: int, n_parallel: int, tick: Callable[[int], None],
              seed: int = 5000) -> tuple[dict, dict, list[dict]]:
    def one(i, it):
        msg, fin, ntok = chat(turns(it["prompt"]), max_tokens, seed + i)
        think, body = msg.get("reasoning_content") or "", msg.get("content") or ""
        ans = extract_answer(bench, body)
        return {"id": it["id"], "finish": fin, "tokens": ntok, "body_chars": len(body), "answer": ans, "gold": it["gold"],
                "correct": is_correct(bench, ans, it["gold"]), "reasoning": think, "content": body}
    recs = _map(one, list(items), n_parallel, tick)
    m = {"n": len(recs), "accuracy": _mean([r["correct"] for r in recs]), "no_answer": _mean([r["answer"] is None for r in recs]),
         "empty": _mean([r["body_chars"] == 0 for r in recs]), "hit_limit": _mean([r["finish"] == "length" for r in recs]),
         "tokens": _mean([r["tokens"] for r in recs]), "error": _mean([r["finish"] == "error" for r in recs])}
    return m, {"ids": [r["id"] for r in recs], "correct": [bool(r["correct"]) for r in recs]}, recs


def mcnemar_p(only_a: int, only_b: int) -> float:
    """片方だけが正解した問題の数からの、両側の正確な検定(差が無いなら半々に分かれるはず)。"""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# --- ファイルどうしの比較 ---------------------------------------------------------------------------------

# (区画, 指標, 表の名前, よい向き)。よい向き: -1 = 小さいほどよい、+1 = 大きいほどよい、0 = 参考
LONG_ROWS = [("hit_limit", "上限で切れた", -1), ("loop", "短い周期のループ", -1), ("long_loop", "長い周期のループ", -1),
             ("empty", "本文が空", -1), ("tail_distinct", "末尾の多様さ", 1), ("tokens", "トークン数", 0), ("error", "エラー", -1)]
TOOL_ROWS = [("called", "呼ぶべきときに呼んだ", 1), ("name_ok", "名前が正しい", 1), ("args_ok", "引数がそろっている", 1),
             ("no_tool_ok", "不要なときに呼ばなかった", 1), ("followup_ok", "結果の後に続けられた", 1),
             ("first_hit_limit", "最初の返事が上限で切れた", -1), ("followup_hit_limit", "結果の後が上限で切れた", -1), ("error", "エラー", -1)]
PROMPT_ROWS = [("match", "一致率", 0), ("loop", "ループ", -1), ("empty", "本文が空", -1), ("hit_limit", "上限で切れた", -1),
               ("tokens", "トークン数", 0), ("error", "エラー", -1)]
BENCH_ROWS = [("no_answer", "答えを取り出せなかった", -1), ("empty", "本文が空", -1), ("hit_limit", "上限で切れた", -1),
              ("tokens", "トークン数", 0), ("error", "エラー", -1)]
PPL_TOL = 0.01                                      # perplexity は 1% まで同程度
TAIL_TOL = 0.05


def compare(ref: dict, tgt: dict) -> list[dict]:
    """参照のファイルと比べた行。割合は「2 問ぶん」、perplexity は 1% までの差を同程度とみなし、それより悪い行に worse を付ける
    (種は同じだが、1 問の違いはサンプリングの揺れで起きる)。"""
    rows: list[dict] = []

    def add(section, key, label, sign, a, b, n, tol=None):
        if a is None and b is None:
            return
        delta = None if a is None or b is None else b - a
        t = tol if tol is not None else (2 / n if n else 0.0)
        rows.append({"section": section, "metric": key, "label": label, "reference": a, "target": b, "delta": delta, "tol": t if sign else None,
                     "worse": bool(sign and delta is not None and sign * delta < -t - 1e-9)})
    for key, label, sign in LONG_ROWS:
        tol = TAIL_TOL if key == "tail_distinct" else None
        add("long", key, label, sign, ref["long"].get(key), tgt["long"].get(key), tgt["long"].get("n"), tol)
    for key, label, sign in TOOL_ROWS:
        n = {"called": "n_expect", "name_ok": "n_expect", "args_ok": "n_expect", "no_tool_ok": "n_none",
             "followup_ok": "n_followup", "followup_hit_limit": "n_followup"}.get(key, "n")
        add("tools", key, label, sign, ref["tools"].get(key), tgt["tools"].get(key), tgt["tools"].get(n), None)
    for name in sorted(set(ref.get("prompts") or {}) | set(tgt.get("prompts") or {})):
        a, b = (ref.get("prompts") or {}).get(name) or {}, (tgt.get("prompts") or {}).get(name) or {}
        for key, label, sign in PROMPT_ROWS:
            add(f"prompts:{name}", key, label, sign, a.get(key), b.get(key), b.get("n") or a.get("n"), None)
    for name in sorted(set(ref.get("bench") or {}) & set(tgt.get("bench") or {})):
        a, b = ref["bench"][name], tgt["bench"][name]
        ia, ib = (ref.get("bench_items") or {}).get(name) or {}, (tgt.get("bench_items") or {}).get(name) or {}
        ca, cb = dict(zip(ia.get("ids", []), ia.get("correct", []))), dict(zip(ib.get("ids", []), ib.get("correct", [])))
        both = set(ca) & set(cb)                    # 同じ問題どうしで比べる。悪化は、正答率が下がり、片方だけ正解の偏りが有意なときだけ
        only_r, only_t = sum(ca[k] and not cb[k] for k in both), sum(cb[k] and not ca[k] for k in both)
        p = mcnemar_p(only_r, only_t)
        delta = None if a.get("accuracy") is None or b.get("accuracy") is None else b["accuracy"] - a["accuracy"]
        rows.append({"section": f"bench:{name}", "metric": "accuracy", "label": "正答率", "reference": a.get("accuracy"), "target": b.get("accuracy"),
                     "delta": delta, "tol": None, "only_reference": only_r, "only_target": only_t, "p": p,
                     "worse": bool(delta is not None and delta < 0 and p < 0.05)})
        for key, label, sign in BENCH_ROWS:
            add(f"bench:{name}", key, label, sign, a.get(key), b.get(key), b.get("n"), None)
    pa, pb = ref.get("ppl") or {}, tgt.get("ppl") or {}
    for k in sorted(set(pa) | set(pb)):
        a, b = pa.get(k), pb.get(k)
        tol = PPL_TOL * a if a else None
        add("ppl", k, "全体(幾何平均)" if k == "mean" else k, -1, a, b, 1, tol if tol is not None else 0.0)
    return rows


# --- llama-server ---------------------------------------------------------------------------------------

def _die_with_parent() -> None:                     # 親が落ちてもサーバが GPU に残らないように(Linux)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)
    except OSError:
        pass


class LlamaServer:
    """llama-server を立てて止める。127.0.0.1 だけで待ち受ける。"""

    def __init__(self, binary: str, model: str | Path, *, lora: str | None, n_parallel: int, n_ctx_slot: int, ngl: int,
                 template_kwargs: dict | None, log: Path, extra: Sequence[str] = (), start_timeout: float = 900.0):
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            self.port = so.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [binary, "-m", str(model), "-ngl", str(999 if ngl < 0 else ngl), "-fa", "on", "-c", str(n_parallel * n_ctx_slot),
               "-np", str(n_parallel), "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "--host", "127.0.0.1", "--port", str(self.port)]
        if template_kwargs:
            cmd += ["--chat-template-kwargs", json.dumps(template_kwargs)]
        if lora:
            cmd += ["--lora-scaled", lora]
        cmd += list(extra)
        self.log, self.cmd, self.timeout = log, cmd, start_timeout
        self.p: subprocess.Popen | None = None

    def __enter__(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "w", encoding="utf-8") as f:
            self.p = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=f, preexec_fn=_die_with_parent)
        atexit.register(self.stop)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.timeout:
            if self.p.poll() is not None:
                tail = self.log.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise GneError("TOOL_FAILED", f"llama-server が止まりました(rc={self.p.returncode})", tail)
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        return self
            except (OSError, ValueError):
                pass
            time.sleep(1)
        self.stop()
        raise GneError("TOOL_FAILED", f"llama-server が {self.timeout:.0f} 秒で立ち上がりません", str(self.log))

    def stop(self) -> None:
        p, self.p = self.p, None
        if p is None:
            return
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        atexit.unregister(self.stop)

    def __exit__(self, *exc):
        self.stop()

    def chat_fn(self, sampling: dict, timeout: float = 3600.0) -> ChatFn:
        def chat(messages, max_tokens, seed, tools=None):
            body = {"messages": messages, "max_tokens": max_tokens, "seed": seed, **sampling}
            if tools:
                body["tools"] = tools
            req = urllib.request.Request(f"{self.url}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    r = json.load(resp)
            except (urllib.error.URLError, OSError, ValueError):   # 出力を解釈できない(崩れたツール呼び出しなど)か、サーバを止めた
                return {}, "error", 0
            c = r["choices"][0]
            return c["message"], c.get("finish_reason") or "stop", int((r.get("usage") or {}).get("completion_tokens") or 0)
        return chat


# --- 入力を読む --------------------------------------------------------------------------------------------

def _load_any(p: str | Path):
    path = Path(p)
    if not path.is_file():
        raise GneError("FILE_NOT_FOUND", f"{path} がありません")
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def load_long_tasks(p: str | Path | None = None) -> list:
    d = _load_any(p or LONG_TASKS)
    check_questions(d, "長い作業の課題")
    if not d:
        raise GneError("BAD_ARGS", "長い作業の課題がありません")
    return d


def load_tool_tasks(p: str | Path | None = None) -> dict:
    d = _load_any(p or TOOL_TASKS)
    if not isinstance(d, dict) or not d.get("tools") or not d.get("tasks"):
        raise GneError("BAD_ARGS", "ツールの課題は {tools: [...], tasks: [{prompt, expect}, ...], results: {...}} の形です")
    names = {t["function"]["name"] for t in d["tools"]}
    for t in d["tasks"]:
        if not isinstance(t.get("prompt"), str) or (t.get("expect") is not None and t["expect"] not in names):
            raise GneError("BAD_ARGS", f"ツールの課題が読めません: {json.dumps(t, ensure_ascii=False)[:120]}")
    return d


def parse_prompt_sets(sets: dict | Sequence[str] | None) -> dict[str, list]:
    """{"名前": ファイルか質問のリスト} か ["名前=ファイル", ...] → {"名前": 質問のリスト}。"""
    if not sets:
        return {}
    if not isinstance(sets, dict):
        pairs = {}
        for s in sets:
            if "=" not in str(s):
                raise GneError("BAD_ARGS", f"--prompts は 名前=ファイル の形です({s})")
            k, v = str(s).split("=", 1)
            pairs[k.strip()] = v.strip()
        sets = pairs
    out = {}
    for k, v in sets.items():
        if not re.fullmatch(r"[\w.\-]+", str(k)):
            raise GneError("BAD_ARGS", f"質問集の名前 {k!r} は英数字・_・-・. だけにしてください")
        q = v if isinstance(v, list) else _load_any(v)
        check_questions(q, f"質問集 {k}")
        if not q:
            raise GneError("BAD_ARGS", f"質問集 {k} が空です")
        out[str(k)] = list(q)
    return out


# --- 本体 ---------------------------------------------------------------------------------------------------

def measure_ppl(session, path: str | Path, lora: tuple[str, float] | None, tick: Callable[[int], None]) -> dict:
    """gne で読み直して参照テキストの perplexity を測る(eval の ppl と同じ、512 トークンずつ)。"""
    from .api import Session
    from .evalx import _ppl, load_suite
    kw = session._engine_kw
    s = Session(path, root=session.store.root, n_gpu_layers=kw["n_gpu_layers"], n_ctx=kw["n_ctx"])
    try:
        got = _ppl(s, load_suite(None).ppl, None, [lora] if lora else None, tick)
    finally:
        s.close()
    if got:
        got["mean"] = float(np.exp(np.mean(np.log(list(got.values())))))
    return got


def fieldtest(session, model: str | Path | None = None, *, against: str | Path | None = None, lora: str | None = None,
              name: str | None = None, long_tasks: str | Path | None = None, tool_tasks: str | Path | None = None,
              prompts: dict | Sequence[str] | None = None, match: str | None = None, long_tokens: int = 3072, tool_tokens: int = 1536,
              prompt_tokens: int = 1536, temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20, presence_penalty: float = 0.0,
              n_parallel: int = 4, reasoning_effort: str | None = "medium", template_kwargs: dict | None = None,
              llama_server: str | None = None, server_args: Sequence[str] | str | None = None, ppl: bool = True,
              skip_long: bool = False, skip_tools: bool = False, bench: Sequence[str] | str | None = None, bench_n: int | None = None,
              bench_tokens: int = 4096) -> dict:
    """model(既定はいま読んでいるモデル)を本番と同じ動かし方で測り、against のファイルと比べる。lora は model に当てる LoRA
    (パスか パス:倍率)。against を省いて lora を渡すと、同じファイルの LoRA なしと比べる。1 つずつ順に GPU に載せる。"""
    t0 = time.perf_counter()
    binary = find_llama_server(llama_server)
    if binary is None:
        raise GneError("TOOL_MISSING", "llama-server が見つかりません",
                       "--llama-server でパスを渡すか、環境変数 GNE_LLAMA_SERVER を設定してください"
                       "(llama.cpp で cmake --build <build> --target llama-server)")
    target = Path(model) if model else Path(session.path)
    if not target.is_file():
        raise GneError("FILE_NOT_FOUND", f"{target} がありません")
    ref = Path(against) if against else (target if lora else None)
    if ref is not None and not ref.is_file():
        raise GneError("FILE_NOT_FOUND", f"{ref} がありません")
    lora_arg, lora_pair = None, None
    if lora:
        fn, _, sc = str(lora).partition(":")
        if not Path(fn).is_file():
            raise GneError("FILE_NOT_FOUND", f"LoRA {fn} がありません")
        try:
            scale = float(sc) if sc else 1.0
        except ValueError:
            raise GneError("BAD_ARGS", f"LoRA は パス か パス:倍率 です({lora})") from None
        lora_arg, lora_pair = f"{fn}:{scale:g}", (fn, scale)
    if match:
        try:
            re.compile(match)
        except re.error as e:
            raise GneError("BAD_ARGS", f"--match の正規表現が読めません: {e}") from e
    long_q = None if skip_long else load_long_tasks(long_tasks)
    tool_spec = None if skip_tools else load_tool_tasks(tool_tasks)
    sets = parse_prompt_sets(prompts)
    names = [x.strip() for x in bench.split(",") if x.strip()] if isinstance(bench, str) else list(bench or [])
    benches = {b: load_bench(b, session.store.fieldtest_dir() / "data", bench_n) for b in names}   # 取得の失敗は GPU を動かす前に返す
    if not long_q and not tool_spec and not sets and not benches and not ppl:
        raise GneError("BAD_ARGS", "測るものがありません")
    nm = name or re.sub(r"[^\w.\-]+", "_", target.name.removesuffix(".gguf"))
    if not re.fullmatch(r"[\w.\-]+", nm):
        raise GneError("BAD_ARGS", f"名前 {nm!r} は英数字・_・-・. だけにしてください")
    if isinstance(server_args, str):
        import shlex
        server_args = shlex.split(server_args)
    tkw = dict(template_kwargs or {})
    if reasoning_effort:
        tkw.setdefault("reasoning_effort", reasoning_effort)
    sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "presence_penalty": presence_penalty}
    n_ctx_slot = max(6144, max(long_tokens, tool_tokens, prompt_tokens, bench_tokens if benches else 0) + 2048)
    out_dir = session.store.fieldtest_dir()
    from .evalx import load_suite
    n_ppl = (len(load_suite(None).ppl) if ppl else 0)
    plan = ([("reference", ref, None)] if ref is not None else []) + [("target", target, lora_arg)]
    per_run = (len(long_q) if long_q else 0) + (len(tool_spec["tasks"]) if tool_spec else 0) + sum(len(v) for v in sets.values()) + sum(len(v) for v in benches.values()) + n_ppl
    total, done = per_run * len(plan), 0
    prog = session.progress

    def step(stage: str) -> Callable[[int], None]:
        def tick(n: int) -> None:
            nonlocal done
            done += n
            if prog:
                prog(done, total, stage)
        return tick
    runs, replies = [], {}
    # 実地テストの間は GPU を llama-server に渡す。読み込んでいたモデルを下ろし、対話的な処理(チャットなど)には譲らない
    session.close()
    session.no_yield = True
    try:
        for label, path, lo in plan:
            tr = time.perf_counter()
            run: dict = {"label": label, "model": str(path), "file": Path(path).name, "lora": lo}
            rec: dict = {}
            if long_q or tool_spec or sets or benches:
                with LlamaServer(binary, path, lora=lo, n_parallel=n_parallel, n_ctx_slot=n_ctx_slot, ngl=session._engine_kw["n_gpu_layers"],
                                 template_kwargs=tkw or None, log=out_dir / f"{nm}.{label}.server.log", extra=server_args or ()) as srv:
                    chat = srv.chat_fn(sampling)
                    if long_q:
                        run["long"], rec["long"] = run_long(chat, long_q, long_tokens, n_parallel, step(f"{label}: 長い作業"))
                    if tool_spec:
                        run["tools"], rec["tools"] = run_tools(chat, tool_spec, tool_tokens, n_parallel, step(f"{label}: ツール"))
                    run["prompts"] = {}
                    for k, q in sets.items():
                        run["prompts"][k], rec[f"prompts:{k}"] = run_prompts(chat, q, prompt_tokens, n_parallel,
                                                                             step(f"{label}: 質問集 {k}"), match)
                    run["bench"], run["bench_items"] = {}, {}
                    for k, q in benches.items():
                        run["bench"][k], run["bench_items"][k], rec[f"bench:{k}"] = run_bench(chat, k, q, bench_tokens, n_parallel,
                                                                                              step(f"{label}: {k}"))
            if ppl:
                run["ppl"] = measure_ppl(session, path, lora_pair if lo else None, step(f"{label}: perplexity"))
            run["seconds"] = time.perf_counter() - tr
            runs.append(run)
            replies[label] = rec
    finally:
        session.no_yield = False
    tgt_run = runs[-1]
    ref_run = runs[0] if len(runs) > 1 else None
    empty = {"long": {}, "tools": {}, "prompts": {}, "ppl": {}, "bench": {}, "bench_items": {}}
    rows = compare({**empty, **ref_run}, {**empty, **tgt_run}) if ref_run else None
    d = {"name": nm, "target": {k: tgt_run[k] for k in ("model", "file", "lora")},
         "reference": None if ref_run is None else {k: ref_run[k] for k in ("model", "file", "lora")},
         "settings": {"long_tokens": long_tokens, "tool_tokens": tool_tokens, "prompt_tokens": prompt_tokens, **sampling,
                      "n_parallel": n_parallel, "template_kwargs": tkw, "match": match, "llama_server": binary,
                      "n_long": len(long_q or []), "n_tools": len((tool_spec or {}).get("tasks") or []),
                      "prompts": {k: len(v) for k, v in sets.items()}, "bench": {k: len(v) for k, v in benches.items()},
                      "bench_tokens": bench_tokens},
         "runs": runs, "compare": rows, "same_level": None if rows is None else not any(r["worse"] for r in rows),
         "worse": [] if rows is None else [f"{r['section']}.{r['metric']}" for r in rows if r["worse"]],
         "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": time.perf_counter() - t0}
    path = out_dir / f"{nm}.json"
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    rp = out_dir / f"{nm}.replies.json"                  # 生成した文章は別のファイルに(表には出さない)
    rp.write_text(json.dumps(replies, ensure_ascii=False, indent=1), encoding="utf-8")
    d["path"], d["replies_path"] = str(path), str(rp)
    return d


def list_fieldtests(session) -> list[dict]:
    out = []
    for p in sorted(session.store.fieldtest_dir().glob("*.json"), key=lambda q: -q.stat().st_mtime):
        if p.name.endswith(".replies.json"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"name": d.get("name", p.stem), "target": (d.get("target") or {}).get("file"),
                    "reference": (d.get("reference") or {}).get("file"), "same_level": d.get("same_level"),
                    "n_worse": len(d.get("worse") or []), "updated": d.get("updated"), "seconds": d.get("seconds")})
    return out
