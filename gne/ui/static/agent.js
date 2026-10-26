// エージェント連携(別冊 §11、U5)。gne mcp --ui-url で繋いだエージェントの ui_highlight / ui_note / ui_open を
// SSE(/api/ui/events)で受け、地図の強調・注釈・NEURON パネルに写す。header の「agent」で一覧と消去。
import { postJSON } from "./api.js";
import { esc } from "./panels.js";

const ACT = { highlight: "強調", note: "注釈", open: "開いた", clear: "強調と注釈を消した" };

export class AgentLink {
  constructor({ button, pop, onState, onOpen }) {
    this.btn = button;
    this.pop = pop;
    this.onState = onState;            // (state) → 地図に写す
    this.onOpen = onOpen;              // ({ layer, n, tab }) → NEURON を開く
    this.state = { highlights: [], notes: [], label: null, last: null };
    this.connected = false;
    this.btn.addEventListener("click", () => { this.pop.classList.toggle("hidden"); this.renderPop(); });
    this.pop.addEventListener("click", (ev) => {
      const o = ev.target.closest("[data-open]");
      if (o) { const [L, n] = o.dataset.open.split(":").map(Number); this.onOpen({ layer: L, n }); return; }
      if (ev.target.closest("#ag-clear")) postJSON("/api/ui/clear", {});
      if (ev.target.closest("#ag-close")) this.pop.classList.add("hidden");
    });
    this.connect();
  }

  connect() {
    const es = new EventSource("/api/ui/events");
    es.onopen = () => { this.connected = true; this.renderButton(); };
    es.onerror = () => { this.connected = false; this.renderButton(); };   // EventSource は自分でつなぎ直す
    es.onmessage = (m) => this.handle(JSON.parse(m.data));
    this.es = es;
  }

  handle(ev) {
    const s = this.state;
    if (ev.type === "state") Object.assign(s, { highlights: ev.highlights, notes: ev.notes, label: ev.label });
    else if (ev.type === "highlight") Object.assign(s, { highlights: ev.highlights, label: ev.label });
    else if (ev.type === "note") {
      s.notes = s.notes.filter((x) => !(x.layer === ev.layer && x.neuron === ev.neuron));
      if (ev.text) s.notes.push({ layer: ev.layer, neuron: ev.neuron, text: ev.text });
    } else if (ev.type === "open") this.onOpen({ layer: ev.layer, n: ev.neuron, tab: ev.tab || null });
    if (ev.last) s.last = ev.last;
    this.onState(s, ev.type === "state" ? null : ev);
    this.renderButton();
    if (!this.pop.classList.contains("hidden")) this.renderPop();
  }

  noteOf(L, n) {
    const x = this.state.notes.find((y) => y.layer === L && y.neuron === n);
    return x ? x.text : null;
  }

  lastText() {
    const l = this.state.last;
    if (!l) return "";
    const what = l.action === "note" ? (l.text ? "に注釈" : "の注釈を消した") : l.action === "open" ? "を開いた"
      : l.action === "highlight" ? (l.n ? "を強調" : "強調を消した") : ACT[l.action] || l.action;
    const who = l.layer !== undefined ? `L${l.layer} N${l.neuron} ` : l.action === "highlight" && l.n ? `${l.n} 個 ` : "";
    return `エージェント: ${who}${what}`;
  }

  renderButton() {
    const s = this.state, n = s.highlights.length + s.notes.length;
    this.btn.textContent = `agent${s.highlights.length ? ` ◌${s.highlights.length}` : ""}${s.notes.length ? ` ✎${s.notes.length}` : ""}`;
    this.btn.classList.toggle("on", n > 0);
    this.btn.classList.toggle("off", !this.connected);
    this.btn.title = this.connected
      ? `エージェント連携: gne mcp --ui-url ${location.origin} で繋ぐと、ui_highlight / ui_note / ui_open がここに届く`
      : "エージェント連携の受け口に繋がっていません(サーバを再起動したなら自動でつなぎ直します)";
  }

  renderPop() {
    const s = this.state;
    const hl = s.highlights.map((h) => `<div class="ag-row" data-open="${h.layer}:${h.neuron}"><span class="ag-dot"></span>L${h.layer} N${h.neuron}${
      h.dose === null || h.dose === undefined ? "" : ` <span class="${h.dose >= 0 ? "plus" : "minus"}">${h.dose >= 0 ? "+" : "−"}${Math.abs(h.dose)}</span>`}</div>`).join("");
    const nt = s.notes.map((x) => `<div class="ag-row" data-open="${x.layer}:${x.neuron}"><b>L${x.layer} N${x.neuron}</b> ${esc(x.text)}</div>`).join("");
    const ago = s.last ? Math.round((Date.now() / 1000 - s.last.t) / 60) : null;
    this.pop.innerHTML = `<div class="ag-head"><b>エージェント</b><button class="small" id="ag-close">閉じる</button>
        <span class="muted">${this.connected ? "接続中" : "未接続"}${s.last ? ` · 最後: ${esc(this.lastText().replace(/^エージェント: /, ""))}(${ago < 1 ? "1 分以内" : `${ago} 分前`})` : ""}</span></div>
      <p class="muted note">Claude Code などから <code>gne mcp --ui-url ${esc(location.origin)}</code> で繋ぐと、エージェントの ui_highlight(強調)・ui_note(注釈)・ui_open(開く)がここと地図に出ます。行のクリックでそのニューロンを開きます。</p>
      <h4>強調 ${s.highlights.length} 個${s.label ? ` — ${esc(s.label)}` : ""}</h4>${hl || '<p class="muted note">なし</p>'}
      <h4>注釈 ${s.notes.length} 個</h4>${nt || '<p class="muted note">なし</p>'}
      ${s.highlights.length + s.notes.length ? '<button class="small" id="ag-clear">強調と注釈をすべて消す</button>' : ""}`;
  }
}
