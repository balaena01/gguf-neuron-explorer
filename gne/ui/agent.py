"""エージェント連携の受け口(別冊 UI 設計 §11、U5)。

`gne mcp --ui-url http://127.0.0.1:8791` の `ui_highlight` / `ui_note` / `ui_open` が `POST /api/ui/{action}` を叩き、
接続中のブラウザに SSE(`GET /api/ui/events`)で届ける。マップはそれぞれ §4.8 の marks / notes、§5 のパネル開閉に写す。

状態(強調と注釈)はサーバが持ち、あとから開いたブラウザにも最初に全部を送る。
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time

from ..steer import GneError

MAX_HIGHLIGHT = 200
MAX_NOTES = 200
MAX_TEXT = 400
_UNIT = re.compile(r"^\s*L?(\d+)\s*[:\s]\s*N?(\d+)(?:\s*:\s*([+-]?[\d.]+))?\s*$", re.I)


class AgentBus:
    def __init__(self, n_layer: int, n_ff_at):
        self.n_layer, self.n_ff_at = n_layer, n_ff_at
        self.highlights: list[dict] = []          # [{layer, neuron, dose|None}]
        self.label: str | None = None             # 強調のまとまりの見出し
        self.notes: dict[str, dict] = {}          # "L:n" → {layer, neuron, text, t}
        self.last: dict | None = None             # 最後の操作(画面の状態表示用)
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()

    # --- 状態 --------------------------------------------------------------------------------------------
    def _check(self, L, n) -> tuple[int, int]:
        try:
            L, n = int(L), int(n)
        except (TypeError, ValueError) as e:
            raise GneError("BAD_ARGS", "layer と neuron が要ります") from e
        if not 0 <= L < self.n_layer or not 0 <= n < self.n_ff_at(L):
            raise GneError("BAD_ARGS", f"L{L} N{n} は範囲外です")
        return L, n

    def snapshot(self) -> dict:
        return {"type": "state", "highlights": self.highlights, "label": self.label,
                "notes": list(self.notes.values()), "last": self.last}

    def highlight(self, units: list[str] | None, label: str | None = None) -> dict:
        """units: ["22:2742", "L14 N8949", "14:8949:+1"]。空なら強調を消す。"""
        out = []
        for u in units or []:
            m = _UNIT.match(str(u))
            if not m:
                raise GneError("BAD_ARGS", f"ユニット {u!r} が読めません", '"22:2742" や "14:8949:+1" の形で渡してください')
            L, n = self._check(m.group(1), m.group(2))
            out.append({"layer": L, "neuron": n, "dose": float(m.group(3)) if m.group(3) else None})
        if len(out) > MAX_HIGHLIGHT:
            raise GneError("BAD_ARGS", f"強調は {MAX_HIGHLIGHT} 個までです")
        self.highlights, self.label = out, (label or None) and str(label)[:80]
        self.last = {"action": "highlight", "n": len(out), "label": self.label, "t": time.time()}
        ev = {"type": "highlight", "highlights": out, "label": self.label, "last": self.last}
        self._publish(ev)
        return {"ok": True, "n": len(out), "browsers": len(self._subs)}

    def note(self, layer, neuron, text: str | None) -> dict:
        """text が空なら、そのニューロンの注釈を消す。"""
        L, n = self._check(layer, neuron)
        key = f"{L}:{n}"
        text = (text or "").strip()[:MAX_TEXT]
        if text:
            if key not in self.notes and len(self.notes) >= MAX_NOTES:
                raise GneError("BAD_ARGS", f"注釈は {MAX_NOTES} 個までです(空の text で消せます)")
            self.notes[key] = {"layer": L, "neuron": n, "text": text, "t": time.time()}
        else:
            self.notes.pop(key, None)
        self.last = {"action": "note", "layer": L, "neuron": n, "text": text, "t": time.time()}
        self._publish({"type": "note", "layer": L, "neuron": n, "text": text, "last": self.last})
        return {"ok": True, "notes": len(self.notes), "browsers": len(self._subs)}

    def open(self, layer, neuron, tab: str | None = None) -> dict:
        L, n = self._check(layer, neuron)
        if tab not in (None, "examples", "chat", "details", "web"):
            raise GneError("BAD_ARGS", "tab は examples / chat / details / web です")
        self.last = {"action": "open", "layer": L, "neuron": n, "t": time.time()}
        self._publish({"type": "open", "layer": L, "neuron": n, "tab": tab, "last": self.last})
        return {"ok": True, "browsers": len(self._subs)}

    def clear(self) -> dict:
        self.highlights, self.label, self.notes = [], None, {}
        self.last = {"action": "clear", "t": time.time()}
        self._publish({**self.snapshot(), "last": self.last})
        return {"ok": True}

    # --- 配信(SSE) ---------------------------------------------------------------------------------------
    def _publish(self, ev: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            loop.call_soon_threadsafe(q.put_nowait, ev)

    async def events(self):
        loop, q = asyncio.get_running_loop(), asyncio.Queue()
        with self._lock:
            self._subs.append((loop, q))
        try:
            yield _sse(self.snapshot())                                         # あとから開いたブラウザにも今の状態を
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield _sse(ev)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            with self._lock:
                self._subs = [s for s in self._subs if s[1] is not q]


def _sse(d: dict) -> str:
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
