// エージェント連携(別冊 §11、U5)。gne mcp --ui-url で繋いだエージェントの ui_highlight / ui_note / ui_open を
// SSE(/api/ui/events)で受け、地図の強調・注釈・NEURON パネルに写す。header の「agent」で一覧と消去。
import { postJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";



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
    const id = l.layer !== undefined ? `L${l.layer} N${l.neuron}` : "";
    const what = l.action === "note" ? t(l.text ? "{id} に注釈" : "{id} の注釈を消した", { id })
      : l.action === "open" ? t("{id} を開いた", { id })
      : l.action === "highlight" ? (l.n ? t("{n} 個を強調", { n: l.n }) : t("強調を消した"))
      : l.action === "clear" ? t("強調と注釈を消した") : l.action;
    return `${t("エージェント")}: ${what}`;
  }

  renderButton() {
    const s = this.state, n = s.highlights.length + s.notes.length;
    this.btn.textContent = `agent${s.highlights.length ? ` ◌${s.highlights.length}` : ""}${s.notes.length ? ` ✎${s.notes.length}` : ""}`;
    this.btn.classList.toggle("on", n > 0);
    this.btn.classList.toggle("off", !this.connected);
    this.btn.title = this.connected
      ? t("エージェント連携: gne mcp --ui-url {url} で繋ぐと、ui_highlight / ui_note / ui_open がここに届く", { url: location.origin })
      : t("エージェント連携の受け口に繋がっていません(サーバを再起動したなら自動でつなぎ直します)");
  }

  renderPop() {
    const s = this.state;
    const hl = s.highlights.map((h) => `<div class="ag-row" data-open="${h.layer}:${h.neuron}"><span class="ag-dot"></span>L${h.layer} N${h.neuron}${
      h.dose === null || h.dose === undefined ? "" : ` <span class="${h.dose >= 0 ? "plus" : "minus"}">${h.dose >= 0 ? "+" : "−"}${Math.abs(h.dose)}</span>`}</div>`).join("");
    const nt = s.notes.map((x) => `<div class="ag-row" data-open="${x.layer}:${x.neuron}"><b>L${x.layer} N${x.neuron}</b> ${esc(x.text)}</div>`).join("");
    const ago = s.last ? Math.round((Date.now() / 1000 - s.last.t) / 60) : null;
    this.pop.innerHTML = `<div class="ag-head"><b>${t("エージェント")}</b><button class="small" id="ag-close">${t("閉じる")}</button>
        <span class="muted">${t(this.connected ? "接続中" : "未接続")}${s.last ? ` · ${t("最後")}: ${esc(this.lastText().replace(`${t("エージェント")}: `, ""))} (${ago < 1 ? t("1 分以内") : t("{n} 分前", { n: ago })})` : ""}</span></div>
      <p class="muted note">${t("Claude Code などから {cmd} で繋ぐと、エージェントの ui_highlight(強調)・ui_note(注釈)・ui_open(開く)がここと地図に出ます。行のクリックでそのニューロンを開きます。", { cmd: `<code>gne mcp --ui-url ${esc(location.origin)}</code>` })}</p>
      <h4>${t("強調 {n} 個", { n: s.highlights.length })}${s.label ? ` — ${esc(s.label)}` : ""}</h4>${hl || `<p class="muted note">${t("なし")}</p>`}
      <h4>${t("注釈 {n} 個", { n: s.notes.length })}</h4>${nt || `<p class="muted note">${t("なし")}</p>`}
      ${s.highlights.length + s.notes.length ? `<button class="small" id="ag-clear">${t("強調と注釈をすべて消す")}</button>` : ""}`;
  }
}
