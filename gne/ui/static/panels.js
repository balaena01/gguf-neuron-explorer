// LIST(rank タブ)と NEURON パネル(Examples / Details)。別冊 §2・§5。
import { getJSON, postJSON } from "./api.js";
import { t } from "./i18n.js";

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);

// トークン 1 個を見える形に(語頭・語末の空白は ␣、改行は ↵)
export function tokHTML(s) {
  if (s === null || s === undefined) return "";
  let t = esc(s).replace(/\n/g, "↵").replace(/\t/g, "⇥");
  const lead = t.match(/^ +/), trail = t.match(/ +$/);
  if (lead && lead[0].length === t.length) return `<span class="tok"><i>${"␣".repeat(t.length)}</i></span>`;
  if (lead) t = `<i>${"␣".repeat(lead[0].length)}</i>${t.slice(lead[0].length)}`;
  if (trail) t = `${t.slice(0, t.length - trail[0].length)}<i>${"␣".repeat(trail[0].length)}</i>`;
  return `<span class="tok">${t || "∅"}</span>`;
}
export const toksHTML = (arr) => (arr || []).filter((x) => x !== null && x !== undefined).map(tokHTML).join("");

// --- LIST: rank ---------------------------------------------------------------------------------------------
export class RankList {
  constructor(root, { onOpen }) {
    this.root = root;
    this.onOpen = onOpen;
    this.data = null;
    this.layer = null;
    this.run = null;
    this.sort = { key: "rank", dir: 1 };
    this.selected = null;
    root.innerHTML = `
      <div class="list-head mono" id="rank-head"></div>
      <div class="list-tools">
        <input id="rank-filter" placeholder="${t("語・番号で絞り込み")}" spellcheck="false">
        <label class="muted"><input type="checkbox" id="rank-random"> random</label>
      </div>
      <div class="rank-wrap"><table class="rank mono"><thead><tr>
        <th data-k="rank">#</th><th data-k="neuron">N</th><th data-k="mass">mass</th><th data-k="null_pct">null</th>
        <th data-k="words">${t("語")}</th></tr></thead><tbody></tbody></table></div>
      <p class="muted note" id="rank-empty"></p>`;
    root.querySelector("#rank-filter").addEventListener("input", () => this.render());
    root.querySelector("#rank-random").addEventListener("change", () => this.render());
    for (const th of root.querySelectorAll("th")) {
      th.addEventListener("click", () => {
        const k = th.dataset.k;
        this.sort = { key: k, dir: this.sort.key === k ? -this.sort.dir : (k === "mass" || k === "null_pct" ? -1 : 1) };
        this.render();
      });
    }
    root.querySelector("tbody").addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr");
      if (tr) this.onOpen({ layer: this.layer, n: +tr.dataset.n, kind: tr.dataset.kind, run: this.run });
    });
  }

  async show(L, units, coverage) {
    this.layer = L;
    this.data = units;
    const runs = units.runs;
    if (!runs.some((r) => r.run === this.run)) this.run = runs.length ? runs[0].run : null;
    const head = this.root.querySelector("#rank-head");
    if (!runs.length) {
      const with_ = coverage.filter((c) => c.sweep_runs.length).map((c) => `<a href="#" data-l="${c.layer}">L${c.layer}</a>`);
      head.innerHTML = `L${L}`;
      this.root.querySelector("#rank-empty").innerHTML = t("この層の sweep はまだありません({cmd})。", { cmd: `<code>gne sweep --layer ${L}</code>` }) +
        (with_.length ? `<br>${t("sweep のある層")}: ${with_.join(" ")}` : "");
      this.root.querySelector("tbody").innerHTML = "";
      return;
    }
    const sel = runs.length > 1 ? `<select id="rank-run">${runs.map((r) => `<option${r.run === this.run ? " selected" : ""}>${esc(r.run)}</option>`).join("")}</select>` : esc(this.run);
    head.innerHTML = `L${L} · ${sel}`;
    const rs = head.querySelector("#rank-run");
    if (rs) rs.addEventListener("change", () => { this.run = rs.value; this.render(); });
    this.render();
  }

  select(L, n, kind = "neuron") {
    this.selected = { L, n, kind };
    for (const tr of this.root.querySelectorAll("tbody tr")) {
      tr.classList.toggle("sel", +tr.dataset.n === n && tr.dataset.kind === kind && L === this.layer);
    }
  }

  render() {
    if (!this.data) return;
    const r = this.data.runs.find((x) => x.run === this.run);
    if (!r) return;
    const q = this.root.querySelector("#rank-filter").value.trim().toLowerCase();
    const showRandom = this.root.querySelector("#rank-random").checked;
    let rows = r.units.filter((u) => u.kind === "neuron" || showRandom);
    if (q) rows = rows.filter((u) => String(u.neuron) === q.replace(/^n/, "") || u.words.some((w) => String(w).toLowerCase().includes(q))
                                     || (u.name || "").toLowerCase().includes(q));
    const { key, dir } = this.sort;
    const val = (u) => (key === "words" ? u.words.join(" ") : key === "rank" ? (u.rank ?? 1e9) : u[key] ?? -1e9);
    rows.sort((a, b) => (val(a) < val(b) ? -dir : val(a) > val(b) ? dir : 0));
    const sel = this.selected;
    for (const th of this.root.querySelectorAll("th")) {                  // 並べ替えの印
      th.dataset.sort = th.dataset.k === key ? (dir > 0 ? "▲" : "▼") : "⇅";
      th.classList.toggle("on", th.dataset.k === key);
    }
    this.root.querySelector("#rank-empty").innerHTML =
      t("{a} / {b} ニューロン · 対照 {c}(床 = 対照の最大 mass {f})", { a: rows.filter((u) => u.kind === "neuron").length, b: r.n_neurons, c: r.n_controls, f: fmt(r.floor, 1) });
    this.root.querySelector("tbody").innerHTML = rows.map((u) => {
      const rnd = u.kind === "random";
      const above = !rnd && r.floor !== null && u.mass > r.floor;
      const cls = [rnd ? "rnd" : "", above ? "above" : "", sel && sel.L === this.layer && sel.n === u.neuron && sel.kind === u.kind ? "sel" : ""].join(" ");
      return `<tr class="${cls}" data-n="${u.neuron}" data-kind="${u.kind}">
        <td>${rnd ? "R" : u.rank ?? ""}</td><td>${rnd ? `r${u.neuron}` : u.neuron}</td><td>${fmt(u.mass, 1)}</td>
        <td>${u.null_pct === null ? "—" : fmt(u.null_pct, 2)}</td>
        <td class="w" title="${esc(u.words.join(" · "))}">${u.name ? `<b>${esc(u.name)}</b> ` : ""}${u.words.map(esc).join(" · ")}</td></tr>`;
    }).join("");
  }
}

// --- NEURON パネル ------------------------------------------------------------------------------------------
export class NeuronPanel {
  constructor(root, { onNamed, onRecipe, inRecipe, onChat, hasConnections, onGoto, agentNote, onEval }) {
    this.agentNote = agentNote || (() => null);
    this.hasConnections = hasConnections || (() => false);
    this.onGoto = onGoto;
    this.onEval = onEval;              // (eval の名前) → eval ビューで開く
    this.links = undefined;            // connections(undefined = 読み込み中、null = 無い)
    this.webParams = { hops: 2, first: 8, k: 4, zmin: 2 };
    this.webSum = null;
    this.root = root;
    this.onNamed = onNamed;
    this.onRecipe = onRecipe;          // (L, n, dose) → レシピに足す
    this.inRecipe = inRecipe;          // (L, n) → レシピの中の dose か null
    this.onChat = onChat;              // (要素, 調書) → Chat タブの中身を描く
    this.tab = "examples";
    this.d = null;
    this._token = 0;
  }

  async open({ layer, n, kind = "neuron", run = null, tab = null }) {
    if (tab) this.tab = tab;
    const tok = ++this._token;
    this.root.innerHTML = `<p class="muted">${t("{id} を読み込み中…", { id: `L${layer} ${kind === "random" ? `random ${n}` : `N${n}`}` })}</p>`;
    let d;
    try {
      d = await getJSON("/api/unit", { layer, n, kind, ...(run ? { run } : {}) });
    } catch (e) {
      if (tok === this._token) this.root.innerHTML = `<p class="err">${esc(e.message)}</p>`;
      return;
    }
    if (tok !== this._token) return;
    this.d = d;
    this.links = undefined;
    this.webSum = null;
    this.render();
    if (kind === "neuron" && this.hasConnections()) {
      getJSON("/api/connections", { layer, n, k: 14 }).then((x) => { if (tok === this._token) { this.links = x; this.render(); } })
        .catch(() => { if (tok === this._token) { this.links = null; this.render(); } });
    } else this.links = null;
  }

  render() {
    if (!this.d) return;                                                     // まだ読み込んでいない
    const d = this.d, s = d.sweep;
    const rnd = d.kind === "random";
    const id = rnd ? `L${d.layer} random ${d.neuron}` : `L${d.layer} N${d.neuron}`;
    const stats = s ? `${s.rank ? `#${s.rank}/${s.n_ranked} · ` : ""}mass ${fmt(s.mass, 1)} · null ${s.null_pct === null ? "—" : pct(s.null_pct)} · z ${fmt(s.z, 2)}`
      : t("sweep なし");
    this.root.innerHTML = `
      <div class="n-head">
        <div class="n-id mono">${id}</div>
        <div class="n-stats mono muted">${stats}${s ? ` · <span title="${esc(s.design)}">${esc(s.run)}</span>` : ""}</div>
        ${d.note ? `<p class="muted note">${esc(d.note_args ? t("run {run} には順位表が無いので、この層のほかの run({other})の結果を出しています", d.note_args) : d.note)}</p>` : ""}
        ${!rnd && this.agentNote(d.layer, d.neuron) ? `<div class="ag-box"><span class="muted">✎ ${t("エージェントの注釈")}</span> ${esc(this.agentNote(d.layer, d.neuron))}</div>` : ""}
        ${rnd ? `<p class="muted note">${t("ランダム方向の対照(名前は付けられません)")}</p>` : `
        <div class="n-row"><input class="n-name" placeholder="${t("名前(入力して Enter か欄の外で保存)")}" value="${esc(d.name || "")}" spellcheck="false">
          ${this.recipeButtons(d)}</div>`}
      </div>
      <div class="tabs">
        <button data-tab="examples">Examples</button><button data-tab="chat"${rnd ? ` disabled title="${t("ランダム方向の対照ではチャットできません")}"` : ""}>Chat</button>
        <button data-tab="details">Details</button><button data-tab="web"${rnd || !this.hasConnections() ? ` disabled title="${t("connections が要ります(層の右クリック →「connections」)")}"` : ""}>Web</button>
      </div>
      <div class="n-tab" id="n-tab"></div>`;
    for (const b of this.root.querySelectorAll(".tabs button")) {
      b.classList.toggle("on", b.dataset.tab === this.tab);
      b.addEventListener("click", () => { this.tab = b.dataset.tab; this.render(); });
    }
    const name = this.root.querySelector(".n-name");
    if (name) {
      const save = async () => {
        if ((name.value.trim() || null) === (d.name || null)) return;
        const r = await postJSON("/api/name", { layer: d.layer, neuron: d.neuron, name: name.value });
        d.name = r.name;
        name.classList.add("saved");
        setTimeout(() => name.classList.remove("saved"), 800);
        this.onNamed(r.names);
      };
      name.addEventListener("keydown", (ev) => { if (ev.key === "Enter") save(); });
      name.addEventListener("blur", save);
    }
    for (const b of this.root.querySelectorAll("[data-rc]")) {
      b.addEventListener("click", () => { this.onRecipe(d.layer, d.neuron, +b.dataset.rc); this.render(); });
    }
    const body = this.root.querySelector("#n-tab");
    if (this.tab === "chat" && !rnd) { this.onChat(body, d); return; }
    if (this.tab === "web" && !rnd && this.hasConnections()) { this.web(body); return; }
    body.innerHTML = this.tab === "details" ? this.details() : this.examples();
    for (const tr of body.querySelectorAll("tr[data-go]")) {
      tr.addEventListener("click", () => { const [L, n] = tr.dataset.go.split(":").map(Number); this.onGoto({ layer: L, n }); });
    }
    for (const a of body.querySelectorAll("a[data-eval]")) a.addEventListener("click", (ev) => { ev.preventDefault(); this.onEval(a.dataset.eval); });
    for (const c of body.querySelectorAll(".cell[data-full]")) c.addEventListener("click", () => c.classList.toggle("open"));
  }

  // connections(§5.4): 書き込む先(下流)と駆動するもの(上流)。↑ = 相手を + に押す / このニューロンを + に押す
  linksHTML() {
    const lk = this.links;
    if (this.d.kind !== "neuron") return "";
    if (lk === undefined) return `<h4>connections</h4><p class="muted">${t("読み込み中…")}</p>`;
    if (lk === null) return `<h4>connections</h4><p class="muted">${t("まだありません(層の番号を右クリック →「connections」。重みだけから全層ペアを CPU で計算)")}</p>`;
    const sg = (e) => `<span class="sgb ${e.sign >= 0 ? "plus" : "minus"}">${e.sign >= 0 ? "+" : "−"}</span>`;
    const rows = (xs, down) => xs.map((e) => `<tr data-go="${e.layer}:${e.neuron}"><td>${sg(e)}</td>
      <td>L${e.layer} N${e.neuron}</td><td>${fmt(e.z, 1)}</td><td class="w">${e.name ? `<b>${esc(e.name)}</b> ` : ""}${
        down ? `<span class="muted">${t(e.sign >= 0 ? "+極" : "−極")} </span>` : ""}${toksHTML(e.lens)}</td></tr>`).join("");
    const head = (a, d) => `<tr class="hd"><th>${a}</th><th>${t("相手")}</th><th>z</th><th>${d}</th></tr>`;
    return `<h4>${t("書き込む先(下流 {n})", { n: lk.down.length })}</h4>
      <table class="kv mono links">${head(t("押す向き"), t("相手がこのとき書く語(押される側の極)"))}${rows(lk.down, true)}</table>
      <h4>${t("駆動するもの(上流 {n})", { n: lk.up.length })}</h4>
      <table class="kv mono links">${head(t("押す向き"), t("相手が + に光ったときに書く語"))}${rows(lk.up, false)}</table>
      <p class="muted note">${t("押す向き = 相手(下流)を、またはこのニューロン(上流)を + / − のどちらへ押すか。z = その相手が、すべての層の候補の中でどれだけ飛び抜けているか。重みだけからの近似(SwiGLU の gate·up)。上流の語は、相手が + 側に光ると仮定したもの(2 次の近似なので、押す向きは相手の光る向きによらない)。")}</p>`;
  }

  // Web タブ(§5.6): 多段の web の要約。toward = 下流が書く語、away = 抑える語、fed_by / held_by = 上流で + / − に駆動する語
  web(el) {
    const p = this.webParams, s = this.webSum, d = this.d;
    const words = (xs) => (xs && xs.length ? xs.map((x) => `<span class="chip${x.both ? " both" : ""}" title="${t("{n} ニューロン", { n: x.count })} · lift ${x.lift}${
        x.both ? ` · ${t("toward と away の両方に出る語")}` : ""}">${tokHTML(x.token).replace(/^<span class="tok">|<\/span>$/g, "")}<i>${x.count}</i>${x.both ? "<b>±</b>" : ""}</span>`).join("")
      : `<span class="muted">${t("なし")}</span>`);
    el.innerHTML = `
      <div class="ch-tools mono"><span class="grp"><span class="muted">${t("段数")}</span><input id="wb-hops" type="number" min="1" max="4" value="${p.hops}"></span>
        <span class="grp"><span class="muted">${t("1 段目の本数")}</span><input id="wb-first" type="number" min="1" max="24" value="${p.first}"></span>
        <span class="grp"><span class="muted">${t("2 段目以降の本数")}</span><input id="wb-k" type="number" min="1" max="24" value="${p.k}"></span>
        <span class="grp"><span class="muted">z ≥</span><input id="wb-zmin" type="number" step="0.5" value="${p.zmin}"></span><button class="small" id="wb-run">${t("更新")}</button></div>
      <p class="muted note">${t("開いたニューロンから下流・上流へたどった配線(header の「線」を web にすると地図に出る)。ここでの符号は経路で掛け合わせたもの(下流の 2 段目以降は途中のニューロンが、上流は相手が、+ 側で光ると仮定)。数字 = その語を lens に持つニューロンの数。並びは、層の中でその語を持つニューロンの割合から見た期待に対する比(lift)の順で、2 ニューロン以上の語が先。大文字・語頭の空白の違いはまとめる。± = toward と away の両方に出る語。")}</p>
      ${s ? `<p class="muted note mono">${t("{n} ニューロン · {e} 本", { n: s.n_neurons, e: s.n_edges })}</p>
      <h4>${t("toward(下流が書く語)")}</h4><div class="chips">${words(s.toward)}</div>
      <h4>${t("away(下流で抑えられる語)")}</h4><div class="chips">${words(s.away)}</div>
      <h4>${t("fed_by(上流で + に駆動する語)")}</h4><div class="chips">${words(s.fed_by)}</div>
      <h4>${t("held_by(上流で − に駆動する語)")}</h4><div class="chips">${words(s.held_by)}</div>` : `<p class="muted">${this.webBusy ? t("計算中…") : ""}</p>`}`;
    el.querySelector("#wb-run").addEventListener("click", () => {
      for (const k of ["hops", "first", "k", "zmin"]) this.webParams[k] = +el.querySelector(`#wb-${k}`).value;
      this.webSum = null; this.loadWeb(el, d);
    });
    if (!s && !this.webBusy) this.loadWeb(el, d);
  }

  async loadWeb(el, d) {
    this.webBusy = true;
    const tok = this._token;
    try { this.webSum = await getJSON("/api/web/summary", { layer: d.layer, n: d.neuron, ...this.webParams }); }
    catch (e) { this.webSum = { n_neurons: 0, n_edges: 0, error: e.message }; }
    this.webBusy = false;
    if (tok === this._token && this.tab === "web") this.web(el);
  }

  recipeButtons(d) {
    const cur = this.inRecipe(d.layer, d.neuron);
    const on = (sign) => cur !== null && Math.sign(cur) === sign;
    const b = (sign, label) => `<button data-rc="${sign}" class="${on(sign) ? `on ${sign > 0 ? "plus" : "minus"}` : ""}"
      title="${on(sign) ? t("もう一度押すとレシピから外す") : t("レシピに dose {d} で足す(dose は recipe タブで変えられる)", { d: sign > 0 ? "+1" : "−1" })}">${
        on(sign) ? `${t("レシピ")} ${cur > 0 ? "+" : "−"}${Math.abs(cur).toFixed(2)}` : label}</button>`;
    return `${b(1, t("+ レシピ"))}${b(-1, t("− レシピ"))}`;
  }

  examples() {
    const s = this.d.sweep;
    if (!s) return `<p class="muted">${t("この層の sweep にこのユニットはありません。rollout は sweep を回すと出ます({cmd})。", { cmd: `<code>gne sweep --layer ${this.d.layer}</code>` })}</p>`;
    const ex = s.examples, cols = ex.columns;
    const cell = (c, base) => {
      if (!c) return `<td class="cell empty"></td>`;
      const capped = c.finish === "length" ? `<span class="badge" title="${t("max_tokens で打ち切り")}">${t("打ち切り")}</span>` : "";
      return `<td class="cell${base ? " base" : ""}" data-full="1">${capped}<div class="txt">${esc(c.text)}</div></td>`;
    };
    const head = `<tr><th></th><th>0<div class="muted">${t("ベースライン")}</div></th>${cols.map((c) =>
      `<th class="${+c.dose >= 0 ? "plus" : "minus"}">${c.dose}<div class="muted">rel ${+c.dose < 0 ? "−" : "+"}${fmt(c.rel_push, 2)}</div></th>`).join("")}</tr>`;
    // プロンプト固有の dose(free × +3 など)は空の列をまたいだ 1 つのセルにする(§5.2)
    const span = (e) => `<td class="cell span" colspan="${Math.max(1, cols.length)}" data-full="1"><span class="badge ${e.dose >= 0 ? "plus" : "minus"}">${t("このプロンプトだけの dose")} ${e.dose > 0 ? "+" : ""}${e.dose} · rel ${e.dose < 0 ? "−" : "+"}${fmt(e.rel_push, 2)}</span>${
      e.finish === "length" ? `<span class="badge" title="${t("max_tokens で打ち切り")}">${t("打ち切り")}</span>` : ""}<div class="txt">${esc(e.text)}</div></td>`;
    const rows = ex.rows.map((r) => {
      const first = !Object.keys(r.cells).length && r.extra.length ? r.extra[0] : null;
      const main = `<tr><th class="p" title="${esc(r.text)}">${esc(r.prompt)}<div class="muted">${esc(r.text)}</div></th>${cell(r.baseline, true)}${
        first ? span(first) : cols.map((c) => cell(r.cells[c.dose])).join("")}</tr>`;
      return main + r.extra.slice(first ? 1 : 0).map((e) => `<tr><th></th><td></td>${span(e)}</tr>`).join("");
    }).join("");
    return `<p class="muted note">${t("行 = プロンプト、列 = dose(rel)。セルをクリックすると全文。")}</p>
      <div class="ex-wrap"><table class="ex">${head}${rows}</table></div>`;
  }

  details() {
    const d = this.d, s = d.sweep, out = [];
    if (s) {
      out.push(`<h4>${t("語(sweep の z の順。数字 = 出現 rollout 数、マウスを当てると z とプロンプト)")}</h4><div class="chips">${s.words.map(([w, c, z, ps]) =>
        `<span class="chip" title="${c} rollout · z ${fmt(z, 2)} · ${esc(ps.join(", "))}">${esc(w)}<i>${c}</i></span>`).join("")}</div>`);
      out.push(`<h4>${t("dose ごと")}</h4><table class="kv mono"><tr><th>dose</th><th>rel_push</th><th>distinct-4</th><th title="${t("max_tokens で打ち切られた割合")}">${t("打ち切り")}</th><th>logprob</th></tr>${
        s.dose_table.map((r) => `<tr><td>${esc(r.dose)}</td><td>${String(r.dose).startsWith("-") ? "−" : ""}${fmt(r.rel_push)}</td><td>${fmt(r.distinct4, 3)}</td><td>${pct(r.capped)}</td><td>${fmt(r.logprob, 3)}</td></tr>`).join("")}</table>`);
    }
    const sl = d.steer_lens;
    if (sl) {
      out.push(`<h4>${t("steer lens(dose ±{d}、KL の対照 95% 点 {k})", { d: fmt(sl.dose, 1), k: fmt(sl.kl_null_p95, 3) })}</h4>
        <div class="lens"><span class="sg plus">+</span>${toksHTML(sl.up)}<span class="kl">KL ${fmt(sl.kl_up, 3)} · ${t("対照の {p} 超", { p: pct(sl.pct_up) })}</span></div>
        <div class="lens"><span class="sg minus">−</span>${toksHTML(sl.down)}<span class="kl">KL ${fmt(sl.kl_down, 3)} · ${t("対照の {p} 超", { p: pct(sl.pct_down) })}</span></div>`);
    } else if (d.kind === "neuron") out.push(`<h4>steer lens</h4><p class="muted">${t("未計算({cmd})", { cmd: `<code>gne lens steer --layer ${d.layer}</code>` })}</p>`);
    const ll = d.logit_lens;
    if (ll) {
      out.push(`<h4>${t("logit lens(最大 |z| {z})", { z: fmt(ll.peak_z, 1) })}</h4>
        <div class="lens"><span class="sg plus">+</span>${toksHTML(ll.up)}</div><div class="lens"><span class="sg minus">−</span>${toksHTML(ll.down)}</div>`);
    } else if (d.kind === "neuron") out.push(`<h4>logit lens</h4><p class="muted">${t("未計算({cmd})", { cmd: `<code>gne lens logit --layer ${d.layer}</code>` })}</p>`);
    const dirs = Object.entries(d.directions || {});
    if (dirs.length) {
      out.push(`<h4>${t("逆引きの方向との cos")}</h4><table class="kv mono">${dirs.map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td class="${v.cos >= 0 ? "plus" : "minus"}">${fmt(v.cos, 4)}</td><td class="muted">${t("層内 {r} 位", { r: v.rank_in_layer })}</td></tr>`).join("")}</table>`);
    }
    const a = d.activation;
    if (a) out.push(`<h4>${t("活性(統計コーパス)")}</h4><table class="kv mono"><tr><td>p99.9</td><td>${fmt(a.q999, 3)}</td><td>absmax</td><td>${fmt(a.absmax, 3)}</td></tr><tr><td>rms</td><td>${fmt(a.rms, 3)}</td><td>${t("平均")}</td><td>${fmt(a.mean, 4)}</td></tr></table>`);
    if (d.evals && d.evals.length) {
      const how = (e) => (e.mode === "gain" ? `×${fmt(e.k, 2)}` : `${e.dose >= 0 ? "+" : "−"}${fmt(Math.abs(e.dose), 2)}`);
      out.push(`<h4>${t("eval(このニューロンを含むもの、新しい順)")}</h4>${d.evals.map((e) => `<div class="n-eval">
        <a href="#" data-eval="${esc(e.name)}">${esc(e.name)}</a> <span class="mono">${how(e)}</span>${e.n_units > 1 ? ` <span class="muted">${t("ほか {n}", { n: e.n_units - 1 })}</span>` : ""} <span class="muted">when=${esc(e.when || "all")}</span>
        <div class="muted note mono"><span title="${t("配布可の最大倍率(能力 −5pt・ppl +10%・混入なし)")}">${e.max_shippable_dose === null ? `<span class="minus">${t("配布可なし")}</span>` : t("配布可 {d}", { d: fmt(e.max_shippable_dose, 2) })}</span>${
          e.at_max ? ` · <span title="${t("最大の倍率での能力の低下(pt)と性格の変化量")}">@${fmt(e.at_max.dose, 2)} −${fmt(e.at_max.capability_drop_pt, 1)}pt · ${t("新語")} ${fmt(e.at_max.novelty, 2)}</span>` : ""}</div></div>`).join("")}`);
    }
    if (d.gain_k && d.gain_k.length) out.push(`<p class="muted note mono">${t("当てたことのある gain")}: ${d.gain_k.map((k) => `×${fmt(k, 2)}`).join(" ")}</p>`);
    if (d.down_norm !== null) out.push(`<p class="muted note mono">‖W_down[:, n]‖ = ${fmt(d.down_norm, 4)}</p>`);
    out.push(this.linksHTML());
    return out.join("");
  }
}

// --- LIST: 検索の結果(header の検索から。強い順) ----------------------------------------------------------
export class SearchList {
  constructor(root, { onOpen, onClose }) {
    this.root = root;
    this.onOpen = onOpen;
    this.onClose = onClose;
  }

  loading(q) { this.root.innerHTML = `<p class="muted">${t("「{q}」を検索中…", { q: esc(q) })}</p>`; }

  error(msg) { this.root.innerHTML = `<p class="err">${esc(msg)}</p>`; }

  show(r, names) {
    const where = { "steer+": "steer+", "steer−": "steer−", "logit+": "logit+", "logit−": "logit−", name: t("名前") };
    const rows = r.hits.slice(0, 500).map((h) => {
      const b = h.best, nm = names[`${h.layer}:${h.neuron}`];
      const w = b.where.startsWith("sweep:") ? t("sweep の語") : where[b.where] || b.where;
      const sign = b.where.endsWith("−") ? "minus" : b.where.endsWith("+") ? "plus" : "";
      return `<tr data-l="${h.layer}" data-n="${h.neuron}"><td>L${h.layer}</td><td>N${h.neuron}</td>
        <td class="${sign}">${esc(w)}${b.pos ? ` #${b.pos}` : ""}</td><td class="w">${b.where === "name" ? `<b>${esc(nm || "")}</b>` : tokHTML(b.token)}</td></tr>`;
    }).join("");
    this.root.innerHTML = `
      <div class="list-head"><span>${t("「{q}」({m}){n} 件", { q: esc(r.q), m: esc(r.mode), n: r.n_hits })}${r.hits.length < r.n_hits ? t("・先頭 {n}", { n: r.hits.length }) : ""}</span>
        <button id="search-close" class="small">${t("閉じる")}</button></div>
      <p class="muted note">${t("並び: steer+ → logit+ → steer− → logit−、それぞれ lens の中の順位(#)の順。words は sweep の語の z の順")}</p>
      <div class="rank-wrap"><table class="rank mono"><tbody>${rows}</tbody></table></div>`;
    this.root.querySelector("#search-close").addEventListener("click", () => this.onClose());
    this.root.querySelector("tbody").addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr");
      if (!tr) return;
      for (const x of this.root.querySelectorAll("tr.sel")) x.classList.remove("sel");
      tr.classList.add("sel");
      this.onOpen({ layer: +tr.dataset.l, n: +tr.dataset.n });
    });
  }
}
