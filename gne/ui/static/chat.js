// NEURON の Chat タブ(別冊 §5.3)。writes = −d / 0 / +d の 3 列で押して生成(多ターン、SSE で逐次表示)、
// reads you = 文章を流して、このニューロンの活性をトークンごとに色で見る(§5.5 の 1 ニューロン版)。
// 会話の状態はブラウザが持ち、毎ターン列ごとの履歴をサーバに送る(サーバは状態を持たない)。
import { postJSON, postSSE } from "./api.js";
import { t as tr } from "./i18n.js";                 // t は会話のターンの変数名に使っているので tr
import { esc, tokHTML } from "./panels.js";

// "14:8949:+1" → "L14 N8949 +1"
const unitLabel = (spec) => { const [L, n, d] = spec.split(":"); return `L${L} N${n} ${(+d >= 0 ? "+" : "−") + Math.abs(+d)}`; };
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));

export class ChatPanel {
  constructor({ recipe, onStatus }) {
    this.recipe = recipe;
    this.onStatus = onStatus || (() => {});
    this.mode = "writes";
    this.target = "neuron";
    this.dose = 0.8;
    this.scale = 1.0;
    this.maxTokens = 128;
    this.when = "all";                 // 足す位置(本体設計 §6.15): all / gen / assistant / after / ramp
    this.whenArg = "";                 // after の文字列、ramp の a-b
    this.turns = [];                   // [{ user, labels, replies: [], finish: [] }]
    this.neuron = null;                // { layer, n }
    this.busy = false;
    this.sid = null;
    this.readsText = "";
    this.readsOut = null;
  }

  // NEURON パネルが別のニューロンを開いたとき。ニューロンを相手にした会話はやり直す
  setNeuron(L, n, detail) {
    if (!this.neuron || this.neuron.layer !== L || this.neuron.n !== n) {
      if (this.target === "neuron" && !this.busy) this.turns = [];
      this.readsOut = null;
    }
    this.neuron = { layer: L, n };
    this.detail = detail;
  }

  useRecipe() { this.target = "recipe"; this.mode = "writes"; }

  columns() {
    if (this.target === "recipe") {                // − 列はレシピ全体の符号を反転したもの(− のユニットだけ、ではない)
      const s = this.scale;
      return [{ label: tr("レシピ×({s})", { s: `−${fmt(s, 2)}` }), units: this.recipe.specs(-s) }, { label: tr("0(無介入)"), units: [] },
              { label: tr("レシピ×({s})", { s: `+${fmt(s, 2)}` }), units: this.recipe.specs(s) }];
    }
    const { layer: L, n } = this.neuron, d = this.dose;
    return [{ label: `−${fmt(d, 2)}`, units: [`${L}:${n}:-${d}`] }, { label: tr("0(無介入)"), units: [] },
            { label: `+${fmt(d, 2)}`, units: [`${L}:${n}:+${d}`] }];
  }

  mount(el) {
    this.el = el;
    this.render();
  }

  render() {
    if (!this.el) return;
    const d = this.detail, nR = this.recipe.units.length;
    const lens = [];
    if (d && d.sweep) lens.push(`${tr("語")} ${d.sweep.words.slice(0, 5).map(([w]) => esc(w)).join(" · ")}`);
    if (d && d.steer_lens) lens.push(`steer+ ${d.steer_lens.up.slice(0, 4).map(tokHTML).join("")}`);
    if (d && d.logit_lens) lens.push(`logit+ ${d.logit_lens.up.slice(0, 4).map(tokHTML).join("")}`);
    const seg = (name, items, cur) => `<span class="seg">${items.map(([v, t, dis]) =>
      `<button data-${name}="${v}"${v === cur ? ' class="on"' : ""}${dis ? " disabled" : ""}>${t}</button>`).join("")}</span>`;
    this.el.innerHTML = `
      <div class="ch-tools">${seg("mode", [["writes", "writes"], ["reads", "reads you"]], this.mode)}
        ${this.mode === "writes" ? seg("target", [["neuron", tr("このニューロン")], ["recipe", tr("レシピ({n})", { n: nR }), !nR]], this.target) : ""}</div>
      ${this.mode === "writes" ? this.writesHTML() : this.readsHTML()}
      ${lens.length && this.target === "neuron" ? `<div class="ch-lens">${lens.join("<span class='muted'> | </span>")}</div>` : ""}`;
    for (const b of this.el.querySelectorAll("[data-mode]")) b.addEventListener("click", () => { this.mode = b.dataset.mode; this.render(); });
    for (const b of this.el.querySelectorAll("[data-target]")) b.addEventListener("click", () => { this.target = b.dataset.target; this.render(); });
    if (this.mode === "writes") this.wireWrites(); else this.wireReads();
  }

  // --- writes ----------------------------------------------------------------------------------------
  writesHTML() {
    const cols = this.columns();
    const amount = this.target === "recipe"
      ? `${tr("倍率")} <button class="small" data-step="-0.25" title="${tr("小さく")}">▼</button><input id="ch-dose" type="number" step="0.25" value="${this.scale}"><button class="small" data-step="0.25" title="${tr("大きく")}">▲</button>`
      : `dose <button class="small" data-step="-0.1" title="${tr("小さく")}">▼</button><input id="ch-dose" type="number" step="0.1" value="${this.dose}"><button class="small" data-step="0.1" title="${tr("大きく")}">▲</button>
         <span class="seg"><button class="on" title="${tr("本体設計の rel dose(残差ノルムの中央値に対する大きさ)を足す")}">add</button><button disabled title="${tr("活性の倍率は本体設計で未対応")}">×</button></span>`;
    const turns = this.turns.map((t, ti) => `
      <div class="ch-user"><span class="muted">${tr("あなた")}</span> ${esc(t.user)}${t.labels.join() !== cols.map((c) => c.label).join() ? `<span class="muted note">(${t.labels.map(esc).join(" / ")})</span>` : ""}</div>
      <div class="ch-row">${t.replies.map((r, ci) => `<div class="ch-cell ${["minus", "", "plus"][ci] || ""}" data-t="${ti}" data-c="${ci}">${this.cellHTML(t, ci)}</div>`).join("")}</div>`).join("");
    return `
      <div class="ch-tools">${amount}<span class="muted">max</span><input id="ch-max" type="number" min="1" max="2048" value="${this.maxTokens}">
        <span class="muted" title="${tr("足す位置: all = 全位置 / gen = 生成中だけ(プロンプトの理解を壊さない)/ assistant = assistant の発話だけ / after = 出力にその文字列が出てから / ramp = 倍率を a から b へ変える")}">when</span><select id="ch-when">${
          ["all", "gen", "assistant", "after", "ramp"].map((w) => `<option value="${w}"${this.when === w ? " selected" : ""}>${w}</option>`).join("")}</select>${
          this.when === "after" || this.when === "ramp" ? `<input id="ch-when-x" size="8" value="${esc(this.whenArg)}" placeholder="${this.when === "after" ? tr("結論:") : "1.5-0.5"}">` : ""}
        <button class="small" id="ch-reset"${this.busy ? " disabled" : ""}>${tr("会話を消す")}</button></div>
      <div class="ch-log">
        <div class="ch-row ch-head">${cols.map((c, i) => `<div class="${["minus", "", "plus"][i]}">${esc(c.label)}${
          this.target === "recipe" && c.units.length ? `<div class="ch-units muted">${c.units.map((u) => `<span class="u">${esc(unitLabel(u))}</span>`).join(" · ")}</div>` : ""}</div>`).join("")}</div>
        ${turns || `<p class="muted note">${tr("同じ発話を 3 列で生成します(左から −、無介入、+)。返答は列ごとに別の履歴になります。")}</p>`}
      </div>
      <div class="ch-input"><textarea id="ch-text" rows="2" placeholder="${tr("メッセージ(Ctrl+Enter で送る)")}"${this.busy ? " disabled" : ""}></textarea>
        <button id="ch-send"${this.busy ? " disabled" : ""}>${tr("送る")}</button><button id="ch-stop"${this.busy ? "" : " disabled"}>${tr("止める")}</button></div>
      <div class="ch-status muted note" id="ch-status">${esc(this.status || "")}</div>`;
  }

  cellHTML(t, ci) {
    const f = t.finish[ci];
    const badge = f === "length" ? `<span class="badge" title="${tr("max_tokens で打ち切り")}">${tr("打ち切り")}</span>` : f === "stopped" ? `<span class="badge">${tr("止めた")}</span>` : "";
    const cur = this.busy && t === this.turns[this.turns.length - 1] && this.activeCol === ci ? `<span class="caret">▍</span>` : "";
    return `${badge}<div class="txt">${esc(t.replies[ci])}${cur}</div>`;
  }

  whenSpec() {
    if (this.when === "after") return this.whenArg ? `after:${this.whenArg}` : "all";
    if (this.when === "ramp") return `ramp:${this.whenArg || "1.5-0.5"}`;
    return this.when;
  }

  wireWrites() {
    const $ = (s) => this.el.querySelector(s);
    const dose = $("#ch-dose");
    dose.addEventListener("change", () => {
      const v = Math.abs(+dose.value);
      if (Number.isFinite(v)) { if (this.target === "recipe") this.scale = v; else this.dose = v; }
      this.render();
    });
    for (const b of this.el.querySelectorAll("[data-step]")) {
      b.addEventListener("click", () => {
        const k = this.target === "recipe" ? "scale" : "dose";
        this[k] = Math.max(0, +(this[k] + +b.dataset.step).toFixed(3));
        this.render();
      });
    }
    $("#ch-max").addEventListener("change", (ev) => { this.maxTokens = Math.max(1, Math.min(2048, +ev.target.value || 128)); });
    $("#ch-when").addEventListener("change", (ev) => { this.when = ev.target.value; this.render(); });
    const wx = $("#ch-when-x");
    if (wx) wx.addEventListener("change", (ev) => { this.whenArg = ev.target.value.trim(); });
    $("#ch-reset").addEventListener("click", () => { this.turns = []; this.status = ""; this.render(); });
    const text = $("#ch-text");
    text.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) this.send(text.value); });
    $("#ch-send").addEventListener("click", () => this.send(text.value));
    $("#ch-stop").addEventListener("click", () => { if (this.sid) postJSON(`/api/steer/${this.sid}/stop`, {}); });
    const log = $(".ch-log");
    log.scrollTop = log.scrollHeight;
  }

  async send(user) {
    user = (user || "").trim();
    if (!user || this.busy) return;
    const cols = this.columns();
    const turn = { user, labels: cols.map((c) => c.label), replies: cols.map(() => ""), finish: cols.map(() => null) };
    const prev = this.turns.slice();
    this.turns.push(turn);
    this.busy = true;
    this.activeCol = null;
    this.status = tr("GPU を待っています(ジョブが走っていればその切れ目まで)…");
    this.failed = false;
    this.render();
    const t0 = performance.now();
    const body = {
      max_tokens: this.maxTokens,
      when: this.whenSpec(),
      columns: cols.map((c, ci) => ({
        label: c.label, units: c.units,
        messages: [...prev.flatMap((t) => [{ role: "user", content: t.user }, { role: "assistant", content: t.replies[ci] }]),
                   { role: "user", content: user }],
      })),
    };
    const cell = (ci) => this.el && this.el.querySelector(`.ch-cell[data-t="${this.turns.length - 1}"][data-c="${ci}"]`);
    const paint = (ci) => { const c = cell(ci); if (c) c.innerHTML = this.cellHTML(turn, ci); };
    try {
      await postSSE("/api/steer", body, (ev) => {
        if (ev.type === "start") this.sid = ev.id;
        else if (ev.type === "gpu") {
          this.waited = ev.waited_s;
          this.setStatus(tr("生成中(GPU を待った時間 {s} 秒)", { s: fmt(ev.waited_s, 1) }));
        } else if (ev.type === "column") { const old = this.activeCol; this.activeCol = ev.col; if (old !== null) paint(old); paint(ev.col); }
        else if (ev.type === "text") { turn.replies[ev.col] += ev.text; paint(ev.col); }
        else if (ev.type === "end") {
          if (ev.text !== null) turn.replies[ev.col] = ev.text;
          turn.finish[ev.col] = ev.finish;
          paint(ev.col);
        } else if (ev.type === "warn") this.setStatus(ev.warnings.join(" / "));
        else if (ev.type === "error") { this.status = `${tr("エラー")}: ${ev.error.message}`; this.failed = true; }
        else if (ev.type === "done") {
          if (!this.failed) this.status = tr("{s} 秒で 3 列を生成(うち GPU 待ち {w} 秒)", { s: fmt(ev.seconds, 1), w: fmt(this.waited, 1) });
        }
      });
    } catch (e) { this.status = `${tr("エラー")}: ${e.message}`; }
    this.busy = false;
    this.sid = null;
    this.activeCol = null;
    this.render();
  }

  setStatus(s) {
    this.status = s;
    const el = this.el && this.el.querySelector("#ch-status");
    if (el) el.textContent = s;
  }

  // --- reads you -------------------------------------------------------------------------------------
  readsHTML() {
    const o = this.readsOut;
    const mx = o ? Math.max(1, ...o.score.map(Math.abs)) : 1;            // 濃さはこの文章の中の最大(1 未満なら 1)に対する割合
    const toks = o ? o.tokens.map((t, i) => {
      const s = o.score[i], a = Math.min(1, Math.abs(s) / mx);
      const bg = s >= 0 ? `rgba(242,210,75,${(a * 0.85).toFixed(3)})` : `rgba(201,79,167,${(a * 0.85).toFixed(3)})`;
      return `<span class="rd-tok" style="background:${bg}" title="${tr("スコア {s} · 活性 {a}", { s: fmt(s, 3), a: fmt(o.act[i], 4) })}">${tokHTML(t).replace(/^<span class="tok">|<\/span>$/g, "")}</span>`;
    }).join("<wbr>") : "";
    const top = o ? o.tokens.map((t, i) => [Math.abs(o.score[i]), t, o.score[i]]).sort((a, b) => b[0] - a[0]).slice(0, 5) : [];
    const first = o ? `<span class="rd-tok first" title="${tr("先頭のトークン(注意の受け皿になりやすいので除く)")}">${tokHTML(o.first).replace(/^<span class="tok">|<\/span>$/g, "")}</span><wbr>` : "";
    const base = o && o.normalization === "text_p99" ? tr("文章の中の層内 p99(この層は活性統計が無い)") : tr("この層の基準(活性統計の q999)");
    return `
      <div class="ch-input"><textarea id="rd-text" rows="4" placeholder="${tr("文章を貼る(このニューロンがどこで光るかを見る)")}">${esc(this.readsText)}</textarea>
        <button id="rd-run">${tr("流す")}</button></div>
      <p class="muted note">${tr("スコア = 活性 ÷ {base}。濃さはこの文章の中の最大 |スコア|{mx}に対する割合(fires の地図の色は絶対値)。黄 = 正、赤紫 = 負。先頭のトークン(灰色)は除く。マウスを当てると値。", { base, mx: o ? ` (${fmt(mx, 1)})` : "" })}</p>
      <div class="rd-out">${first}${toks}</div>
      ${top.length ? `<p class="muted note">${tr("強い順")}: ${top.map(([, t, s]) => `${tokHTML(t)}${fmt(s, 2)}`).join(" ")}</p>` : ""}
      <div class="ch-status muted note" id="ch-status">${esc(this.readsStatus || "")}</div>`;
  }

  wireReads() {
    const t = this.el.querySelector("#rd-text");
    t.addEventListener("input", () => { this.readsText = t.value; });
    this.el.querySelector("#rd-run").addEventListener("click", async () => {
      if (!this.readsText.trim()) return;
      this.readsStatus = tr("流しています…");
      this.render();
      try {
        const { layer, n } = this.neuron;
        this.readsOut = await postJSON("/api/reads", { layer, neuron: n, text: this.readsText });
        this.readsStatus = tr("{n} トークン · 基準 {b}", { n: this.readsOut.tokens.length, b: this.readsOut.normalization });
      } catch (e) { this.readsStatus = `${tr("エラー")}: ${e.message}`; }
      this.render();
    });
  }
}
