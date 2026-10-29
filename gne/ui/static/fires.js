// LIST の fires タブ(別冊 §5.5)。文章を流してトークンを並べ、トークンを選ぶと、そのトークンで最も光るニューロンの
// 一覧(層の範囲・符号・閾値・lens の語で絞る)と、マップの色(全層の値)を出す。「全トークン」は各ニューロンの |最大|。
// 「ヘッド」を入れておくと、板の上辺のヘッドも、選んだトークンへの注目(後ろの位置からの平均)で色付けする(CHANGE-04 G の fires_heads)。
// MoE(CHANGE-05 G): 選んだトークンで各 MoE の層が選んだ専門家を一覧にし、地図のタイルに枠を付ける(onRouted)。選ばれなかった専門家の
// ニューロンは 0。一覧のニューロンは E37 N5 の形。
import { getJSON, postJSON } from "./api.js";
import { t as tr } from "./i18n.js";                 // t はトークンの変数名に使っているので tr
import { esc, tokHTML } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));

export class FiresPanel {
  constructor(root, { nLayer, onOpen, onField, onRouted }) {
    this.root = root;
    this.nLayer = nLayer;
    this.onOpen = onOpen;
    this.onField = onField;            // (fid, tok, label, thr, heads) → マップの色を切り替える
    this.onRouted = onRouted || (() => {});   // (Map(層 → [専門家]) | null) → 地図のタイルの枠
    this.heads = true;                 // ヘッドの注目も色にする(flash attention を切った prefill を 1 回足す)
    this.text = "";
    this.res = null;
    this.tok = -1;
    this.filt = { lmin: 0, lmax: nLayer - 1, sign: "both", thr: 1.0, q: "" };
    this.list = null;
    this.render();
  }

  render() {
    const r = this.res, f = this.filt;
    const topMax = r ? Math.max(1e-6, ...r.top.map((x) => Math.abs((x.neurons[0] || {}).score || 0))) : 1;
    const chips = r ? [`<button class="chip-t${this.tok === -1 ? " on" : ""}" data-tok="-1">${tr("全トークン")}</button>`,
      ...r.tokens.slice(1).map((t, i) => {
        const top = r.top[i] && r.top[i].neurons[0];
        const a = top ? Math.abs(top.score) / topMax : 0;                             // 最も光ったニューロンのスコア(文章の中の最大に対する割合)
        return `<button class="chip-t${this.tok === i + 1 ? " on" : ""}" data-tok="${i + 1}" style="border-color:rgba(232,176,74,${(0.2 + 0.8 * a).toFixed(2)})"
          title="${tr("最大スコア {s}", { s: fmt(top && top.score, 2) })}">${tokHTML(t).replace(/^<span class="tok">|<\/span>$/g, "")}</button>`;
      })].join("") : "";
    this.root.innerHTML = `
      <div class="ch-input"><textarea id="fi-text" rows="4" placeholder="${tr("文章を貼る(トークンごとに、どの層のどのニューロンが光るか)")}">${esc(this.text)}</textarea>
        <button id="fi-run">${tr("流す")}</button></div>
      <div class="ch-status muted note">${esc(this.status || tr("スコア = 活性 ÷ その層の基準(活性統計の q999)。先頭のトークンは除く"))}</div>
      ${r ? `<div class="fi-toks">${chips}</div>${this.routedHTML()}
      <div class="list-tools fi-filt">
        <label class="muted">${tr("層")} <input id="fi-lmin" type="number" min="0" max="${this.nLayer - 1}" value="${f.lmin}">〜<input id="fi-lmax" type="number" min="0" max="${this.nLayer - 1}" value="${f.lmax}"></label>
        <select id="fi-sign"><option value="both">±</option><option value="+">+</option><option value="-">−</option></select>
        <label class="muted">|s| ≥ <input id="fi-thr" type="number" step="0.5" min="0" value="${f.thr}"></label>
        <input id="fi-q" placeholder="${tr("lens の語")}" value="${esc(f.q)}" spellcheck="false">
        <label class="muted" title="${tr("板の上辺のヘッドを、選んだトークンへの注目(そのトークンより後ろの位置からの注目の平均)で色付けする")}"><input type="checkbox" id="fi-heads"${this.heads ? " checked" : ""}> ${tr("ヘッド")}</label></div>
      <div class="rank-wrap"><table class="rank mono fi-tab${r.routed ? " moe" : ""}"><thead><tr><th>${tr("層")}</th><th>N</th><th>${tr("スコア")}</th><th title="${tr("光った向きの lens の極。S = steer lens、L = logit lens")}">${tr("lens(向きの極)")}</th></tr></thead>
        <tbody id="fi-list">${this.listHTML()}</tbody></table></div>` : ""}`;
    const $ = (s) => this.root.querySelector(s);
    $("#fi-text").addEventListener("input", (ev) => { this.text = ev.target.value; });
    $("#fi-run").addEventListener("click", () => this.run());
    if (!r) return;
    $("#fi-sign").value = f.sign;
    $("#fi-heads").addEventListener("change", (ev) => { this.heads = ev.target.checked; this.onField(r.id, this.tok, this.label(), this.filt.thr, this.heads); });
    for (const b of this.root.querySelectorAll("[data-tok]")) b.addEventListener("click", () => this.pick(+b.dataset.tok));
    for (const [id, k] of [["#fi-lmin", "lmin"], ["#fi-lmax", "lmax"], ["#fi-thr", "thr"], ["#fi-sign", "sign"], ["#fi-q", "q"]]) {
      $(id).addEventListener("change", (ev) => {
        this.filt[k] = k === "sign" || k === "q" ? ev.target.value : +ev.target.value;
        this.loadList();
        if (k === "thr") this.onField(r.id, this.tok, this.label(), this.filt.thr, this.heads);
      });
    }
    $("#fi-list").addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr[data-l]");
      if (tr) this.onOpen({ layer: +tr.dataset.l, n: +tr.dataset.n });
    });
  }

  listHTML() {
    if (!this.list) return `<tr><td class="muted">${tr("読み込み中…")}</td></tr>`;
    if (!this.list.neurons.length) return `<tr><td class="muted">${tr("条件に合うニューロンはありません")}</td></tr>`;
    return this.list.neurons.map((x) => {
      const weak = x.norm === "text_p99";
      const lens = x.lens ? `<span class="${x.lens.pole === "+" ? "plus" : "minus"}" title="${tr("{src} lens の {pole} 極", { src: x.lens.src, pole: x.lens.pole })}">${x.lens.pole === "+" ? "↑" : "↓"}${x.lens.src === "steer" ? "S" : "L"}</span> ${x.lens.tokens.map(tokHTML).join("")}` : `<span class="muted">${tr("lens なし")}</span>`;
      return `<tr data-l="${x.layer}" data-n="${x.neuron}" class="${weak ? "weak" : ""}"${weak ? ` title="${tr("この層は活性統計が無いので、文章の中の層内 p99 で割った(ほかの層と比べにくい)")}"` : ""}>
        <td>L${x.layer}${weak ? '<span class="badge">p99</span>' : ""}</td><td${x.expert !== undefined ? ` title="${tr("層の中の通し番号 {n}", { n: x.neuron })}"` : ""}>${
          x.expert !== undefined ? `E${x.expert} N${x.expert_neuron}` : `N${x.neuron}`}</td>
        <td class="${x.score >= 0 ? "plus" : "minus"}">${fmt(x.score, 2)}</td><td class="w">${lens}</td></tr>`;
    }).join("")
      + (this.list.n_hits > this.list.neurons.length ? `<tr><td colspan="4" class="muted">${tr("ほか {n} 個", { n: this.list.n_hits - this.list.neurons.length })}</td></tr>` : "");
  }

  // 選んだトークンで MoE の層ごとに選ばれた専門家(routed の行 i = 位置 i + 1)。全トークンや MoE でなければ null
  routedAt(tok) {
    const ro = this.res && this.res.routed;
    if (!ro || tok < 1) return null;
    return new Map(Object.entries(ro).filter(([, rows]) => rows[tok - 1]).map(([L, rows]) => [+L, rows[tok - 1]]));
  }

  routedHTML() {
    const ro = this.res && this.res.routed;
    if (!ro) return "";
    const m = this.routedAt(this.tok);
    if (!m) return `<p class="muted note fi-routed">${tr("トークンを選ぶと、MoE の層ごとに選ばれた専門家を出します(地図のタイルに水色の枠)")}</p>`;
    const rows = [...m].sort((a, b) => a[0] - b[0]).map(([L, es]) => `<div><span class="muted">L${L}</span> ${es.map((e) => `E${e}`).join(" ")}</div>`).join("");
    return `<details class="fi-routed" open><summary class="muted note">${tr("選ばれた専門家(このトークン、{n} 層、選ばれた順。地図のタイルに水色の枠)", { n: m.size })}</summary>
      <div class="mono fi-routed-list">${rows}</div></details>`;
  }

  label() {
    if (!this.res) return "";
    return this.tok === -1 ? tr("全トークン") : `「${this.res.tokens[this.tok]}」`;
  }

  async run() {
    if (!this.text.trim()) return;
    this.status = tr("流しています(GPU を待つことがあります)…");
    this.render();
    try {
      this.res = await postJSON("/api/fires", { text: this.text, top: 50 });
      const weak = Object.entries(this.res.normalization).filter(([, v]) => v === "text_p99").map(([L]) => `L${L}`);
      this.status = tr("{n} トークン · 層 {l}", { n: this.res.n_tokens, l: this.res.layers.length }) +
        (weak.length ? ` · ${tr("{layers} は活性統計が無いので文章の中の層内 p99 で割った(ほかの層と比べにくい。gne stats act --include-last-layer で作れる)", { layers: weak.join(", ") })}` : "");
      this.tok = -1;
      this.list = null;
      this.render();
      this.onField(this.res.id, -1, this.label(), this.filt.thr, this.heads);
      this.onRouted(null);
      this.loadList();
    } catch (e) { this.status = `${tr("エラー")}: ${e.message}`; this.render(); }
  }

  pick(tok) {
    this.tok = tok;
    this.list = null;
    this.render();
    this.onField(this.res.id, tok, this.label(), this.filt.thr, this.heads);
    this.onRouted(this.routedAt(tok));
    this.loadList();
  }

  async loadList() {
    const f = this.filt, r = this.res;
    const tok = this.tok;
    try {
      const x = await getJSON(`/api/fires/${r.id}/neurons`, { tok, lmin: f.lmin, lmax: f.lmax, sign: f.sign, thr: f.thr, q: f.q, limit: 200 });
      if (r !== this.res || tok !== this.tok) return;
      this.list = x;
    } catch (e) { this.list = { neurons: [], n_hits: 0 }; this.status = `${tr("エラー")}: ${e.message}`; }
    const tb = this.root.querySelector("#fi-list");
    if (tb) tb.innerHTML = this.listHTML();
  }
}
