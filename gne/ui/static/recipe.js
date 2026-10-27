// レシピ(別冊 §2 LIST の recipe タブ、§4.8 マーク)。ユニットの追加・dose のスライダー・保存と読み込み・cvec の書き出し。
// ablate(方向の射影消去、本体設計 §6.17)の行: 方向・層の範囲・倍率。3 列チャットでは − の列で倍率を反転する。
// 編集中のレシピはこのブラウザの localStorage に下書きとして置く(使えなければ置かない)。
import { getJSON, postJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const signed = (x, d = 2) => `${x >= 0 ? "+" : "−"}${fmt(Math.abs(x), d)}`;
const isDir = (u) => u.kind === "direction";
// 方向のユニット(gne direction の方向、L:dir:NAME:dose)は dose が小さいので、スライダーを ±1・刻み 0.005 にする
const range = (u) => (isDir(u) ? { max: 1, step: 0.005, d: 3 } : { max: 3, step: 0.05, d: 2 });

// スライダーは 0(中央)から値まで塗る。+ は黄、− は赤紫。中央に 0 の目盛り
function trackBG(v, max = 3) {
  const p = ((Math.max(-max, Math.min(max, v)) + max) / (2 * max)) * 100, c = v >= 0 ? "#f2d24b" : "#c94fa7", bg = "#1f2630";
  const [a, b] = v >= 0 ? [50, p] : [p, 50];
  return `linear-gradient(to right, transparent calc(50% - 1px), #8a96a7 calc(50% - 1px), #8a96a7 calc(50% + 1px), transparent calc(50% + 1px)),` +
         `linear-gradient(to right, ${bg} ${a}%, ${c} ${a}%, ${c} ${b}%, ${bg} ${b}%)`;
}

export class Recipe {
  constructor(modelId, onChange) {
    this.key = `gne.recipe.${modelId}`;
    this.onChange = onChange;
    this.name = "";
    this.units = [];                   // [{ layer, neuron, dose }]
    this.ablate = null;                // { direction, scale, layers, from_layer } か null
    this.skipped = 0;                  // 読み込んだレシピのうち UI では扱えないユニット(gain・方向)の数
    try {
      const d = JSON.parse(localStorage.getItem(this.key) || "null");
      if (d && Array.isArray(d.units)) { this.name = d.name || ""; this.units = d.units; this.ablate = d.ablate || null; }
    } catch { /* 下書きが読めなくても空から始める */ }
  }

  _save() {
    try { localStorage.setItem(this.key, JSON.stringify({ name: this.name, units: this.units, ablate: this.ablate })); } catch { /* 置けなくても使える */ }
    this.onChange(this);
  }

  empty() { return !this.units.length && !this.ablate; }
  // ablate を倍率 s で(− の列は s < 0 = その方向を強める)。無ければ undefined
  ablateAt(s = 1) { return this.ablate ? { ...this.ablate, scale: +(this.ablate.scale * s).toFixed(4) } : undefined; }
  setAblate(a) { this.ablate = a && a.direction ? a : null; this._save(); }

  specs(scale = 1) {
    return this.units.map((u) => `${u.layer}:${isDir(u) ? `dir:${u.direction}` : u.neuron}:${u.dose * scale >= 0 ? "+" : ""}${+(u.dose * scale).toFixed(4)}`);
  }
  has(L, n) { return this.units.find((u) => !isDir(u) && u.layer === L && u.neuron === n); }
  neurons() { return this.units.filter((u) => !isDir(u)); }
  add(L, n, dose) {
    const u = this.has(L, n);
    if (u) u.dose = dose; else this.units.push({ layer: L, neuron: n, dose });
    this.units.sort((a, b) => a.layer - b.layer || (a.neuron ?? -1) - (b.neuron ?? -1));
    this._save();
  }
  set(i, dose) { this.units[i].dose = dose; this._save(); }
  remove(i) { this.units.splice(i, 1); this._save(); }
  clear() { this.units = []; this.name = ""; this.ablate = null; this.skipped = 0; this._save(); }
  load(name, units, ablate = null) {
    const ok = units.filter((u) => ((u.kind || "neuron") === "neuron" || u.kind === "direction") && (u.mode || "add") === "add");
    this.skipped = units.length - ok.length;
    this.name = name;
    this.units = ok.map((u) => (isDir(u) ? { layer: u.layer, kind: "direction", direction: u.direction, dose: u.dose }
      : { layer: u.layer, neuron: u.neuron, dose: u.dose }));
    this.ablate = ablate || null;
    this._save();
  }
}

export class RecipePanel {
  constructor(root, recipe, { onOpen, onChat, unitWords, directions }) {
    this.root = root;
    this.r = recipe;
    this.directions = directions || (() => []);   // 方向の名前(gne direction / reverse)
    this.onOpen = onOpen;
    this.onChat = onChat;
    this.unitWords = unitWords;        // (L, n) → 短い説明(sweep の語など)。無ければ ""
    this.saved = [];
  }

  async refreshSaved() {
    this.render();                     // 先に今の中身を出し、保存済みの一覧は読めてから足す
    try { this.saved = (await getJSON("/api/recipes")).filter((x) => x.this_model); } catch { this.saved = []; }
    this.render();
  }

  render() {
    const r = this.r;
    const rows = r.units.map((u, i) => {
      const g = range(u);
      const id = isDir(u)
        ? `<span class="mono" title="${t("方向(gne direction で作ったもの)。その層に足す")}">L${u.layer} dir:${esc(u.direction)}</span> <span class="muted w">${t("方向")}</span>`
        : `<a href="#" data-open="${i}">L${u.layer} N${u.neuron}</a> <span class="muted w">${esc(this.unitWords(u.layer, u.neuron))}</span>`;
      return `
      <tr data-i="${i}"><td class="id">${id}<button class="small rm" data-rm="${i}" title="${t("外す")}">×</button>
        <div class="sl"><span class="muted">−${g.max}</span><input type="range" class="dose" min="-${g.max}" max="${g.max}" step="${g.step}" value="${u.dose}" data-i="${i}" style="background:${trackBG(u.dose, g.max)}"><span class="muted">+${g.max}</span>
          <span class="sg mono ${u.dose >= 0 ? "plus" : "minus"}">${signed(u.dose, g.d)}</span>
          <input type="number" min="-10" max="10" step="${g.step}" value="${u.dose}" data-i="${i}" class="num" title="${t("値を直接入れる")}"></div></td></tr>`;
    }).join("");
    this.root.innerHTML = `
      <div class="list-tools"><input id="rc-name" placeholder="${t("レシピ名(英数字・_・-)")}" value="${esc(r.name)}" spellcheck="false">
        <button class="small" id="rc-save" title="${t("~/.gne/recipes に保存")}">${t("保存")}</button></div>
      <div class="list-tools"><select id="rc-load"><option value="">${t("保存したレシピを読む…")}</option>${
        this.saved.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}(${x.units.length})</option>`).join("")}</select></div>
      ${r.units.length ? `<table class="rc mono"><tbody>${rows}</tbody></table>`
        : `<p class="muted note">${t("ユニットがありません。NEURON の「+ レシピ」「− レシピ」で足します。")}</p>`}
      ${r.skipped ? `<p class="muted note">${t("UI では扱えないユニット(gain・方向){n} 個を外しました", { n: r.skipped })}</p>` : ""}
      ${this.ablateHTML()}
      <p class="muted note">${t("dose は rel(その層の残差ノルムの中央値に対する大きさ)。マップに + は黄、− は赤紫の輪で出ます。")}</p>
      <div class="rc-actions">
        <button id="rc-chat"${r.empty() ? " disabled" : ""}>${t("3 列チャットで試す")}</button>
        <button id="rc-export"${r.units.length ? "" : " disabled"}>${t("cvec を書き出す")}</button>
        <label class="muted"><input type="checkbox" id="rc-verify"> ${t("llama.cpp で確かめる")}</label>
      </div>
      ${r.units.length ? `<div class="rc-actions"><button class="small" id="rc-clear" title="${t("すべてのユニットを外す")}">${t("レシピを空にする")}</button></div>` : ""}
      <div id="rc-result"></div>`;
    const $ = (s) => this.root.querySelector(s);
    $("#rc-name").addEventListener("change", (ev) => { r.name = ev.target.value.trim(); r._save(); });
    $("#rc-save").addEventListener("click", () => this.save());
    const clr = $("#rc-clear");
    if (clr) clr.addEventListener("click", () => { if (confirm(t("レシピの {n} ユニットをすべて外しますか", { n: r.units.length }))) { r.clear(); this.render(); } });
    $("#rc-load").addEventListener("change", (ev) => {
      const x = this.saved.find((y) => y.name === ev.target.value);
      if (x) { r.load(x.name, x.units, x.ablate); this.render(); }
    });
    for (const el of this.root.querySelectorAll("input[data-i]")) {
      el.addEventListener("input", (ev) => {
        const i = +ev.target.dataset.i, v = +ev.target.value;
        if (!Number.isFinite(v)) return;
        r.set(i, v);
        const tr = this.root.querySelector(`tr[data-i="${i}"]`);
        for (const x of tr.querySelectorAll("input")) if (x !== ev.target) x.value = String(v);
        const g = range(r.units[i]);
        tr.querySelector("input.dose").style.background = trackBG(v, g.max);
        const sg = tr.querySelector(".sg");
        sg.textContent = signed(v, g.d); sg.className = `sg mono ${v >= 0 ? "plus" : "minus"}`;
      });
    }
    for (const el of this.root.querySelectorAll("[data-rm]")) el.addEventListener("click", () => { r.remove(+el.dataset.rm); this.render(); });
    for (const el of this.root.querySelectorAll("[data-open]")) {
      el.addEventListener("click", (ev) => { ev.preventDefault(); const u = r.units[+el.dataset.open]; this.onOpen({ layer: u.layer, n: u.neuron }); });
    }
    $("#rc-chat").addEventListener("click", () => this.onChat());
    this.wireAblate();
    $("#rc-export").addEventListener("click", () => this.export());
  }

  ablateHTML() {
    const a = this.r.ablate, dirs = this.directions();
    const opts = [`<option value="">${t("なし")}</option>`, ...dirs.map((d) => `<option${a && a.direction === d ? " selected" : ""}>${esc(d)}</option>`)];
    if (a && !dirs.includes(a.direction)) opts.push(`<option selected>${esc(a.direction)}</option>`);
    const sc = a ? a.scale : 1;
    return `<div class="rc-ablate" title="${t("方向の射影消去(§6.17): 選んだ方向を残差に書き込む重みから消す(LoRA で当てる)。gne direction で作った方向を使う")}">
        <span class="muted">ablate</span><select id="rc-ab-dir">${opts.join("")}</select>${a ? `
        <span class="muted">${t("倍率")}</span><span class="sl"><input type="range" id="rc-ab-scale" min="0" max="1.5" step="0.05" value="${sc}"><span class="mono" id="rc-ab-scale-v">×${fmt(sc, 2)}</span></span>
        <span class="muted">${t("層")}</span><input id="rc-ab-layers" value="${esc(a.layers || "all")}" spellcheck="false" title="${t("all / 10-40 / 1,5,9-12")}">
        <span class="muted">${t("測った層")}</span><input id="rc-ab-from" type="number" min="0" value="${a.from_layer ?? ""}" placeholder="${t("既定")}" title="${t("どの層で測った方向を使うか(空なら gne direction --select で選んだ層)")}">` : ""}
      </div>${dirs.length || a ? "" : `<p class="muted note">${t("方向がありません(gne direction --contrast で作れます)")}</p>`}`;
  }

  wireAblate() {
    const $ = (s) => this.root.querySelector(s);
    const cur = () => this.r.ablate || { direction: "", scale: 1, layers: "all", from_layer: null };
    $("#rc-ab-dir").addEventListener("change", (ev) => {
      const v = ev.target.value;
      this.r.setAblate(v ? { ...cur(), direction: v } : null);
      this.render();
    });
    const sl = $("#rc-ab-scale");
    if (!sl) return;
    sl.addEventListener("input", (ev) => { this.r.setAblate({ ...cur(), scale: +ev.target.value }); $("#rc-ab-scale-v").textContent = `×${fmt(+ev.target.value, 2)}`; });
    $("#rc-ab-layers").addEventListener("change", (ev) => this.r.setAblate({ ...cur(), layers: ev.target.value.trim() || "all" }));
    $("#rc-ab-from").addEventListener("change", (ev) => {
      const v = ev.target.value.trim();
      this.r.setAblate({ ...cur(), from_layer: v === "" ? null : +v });
    });
  }

  async save() {
    const res = this.root.querySelector("#rc-result");
    const name = this.root.querySelector("#rc-name").value.trim();
    if (!/^[\w.-]+$/.test(name)) { res.innerHTML = `<p class="err">${t("名前は英数字・_・-・. だけにしてください")}</p>`; return; }
    this.r.name = name; this.r._save();
    try {
      await postJSON(`/api/recipes/${encodeURIComponent(name)}`, { units: this.r.specs(), ablate: this.r.ablate });
      await this.refreshSaved();                     // 一覧を描き直してから知らせる(描き直すと #rc-result も作り直される)
      this.root.querySelector("#rc-result").innerHTML =
        `<p class="muted note">${t("保存しました({path})", { path: `~/.gne/recipes/${esc(name)}.yaml` })}</p>`;
    } catch (e) { res.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }

  async export() {
    const res = this.root.querySelector("#rc-result");
    const verify = this.root.querySelector("#rc-verify").checked;
    const name = (this.root.querySelector("#rc-name").value.trim() || "recipe").replace(/[^\w.-]/g, "_");
    res.innerHTML = `<p class="muted note">${t(verify ? "書き出し中(llama.cpp で読み直して生成を比べる、数十秒)…" : "書き出し中…")}</p>`;
    try {
      const x = await postJSON("/api/export", { units: this.r.specs(), name, verify });
      const a = document.createElement("a");                       // そのままダウンロードする
      a.href = x.url; a.download = x.file; document.body.appendChild(a); a.click(); a.remove();
      const norms = Object.entries(x.layer_norms).map(([L, v]) => `L${L} ${fmt(v, 1)}`).join(" · ");
      const v = x.verify;
      res.innerHTML = `<p class="note">${t("{file} を書き出しました", { file: `<a href="${esc(x.url)}" download="${esc(x.file)}">${esc(x.file)}</a>` })}${
          x.recipe_saved ? "" : `<br><span class="muted">${t("(レシピ自体は未保存。上の「保存」で ~/.gne/recipes に残せます)")}</span>`}</p>
        <p class="muted note mono" title="${t("層ごとに残差へ足すベクトルの長さ(残差ノルムと同じ単位)")}">${t("足すベクトルの長さ")}: ${esc(norms)}</p>
        <p class="muted note">${t("llama.cpp で使う: + 列は {plus}、− 列は {minus}", { plus: `<code>--control-vector ${esc(x.file)}</code>`, minus: `<code>--control-vector-scaled ${esc(x.file)}:-1</code>` })}</p>${v ? `
        <p class="note">${v.match ? "✓" : "✗"} ${t(v.match ? "「{p}」の {n} トークン(+1 倍、greedy)で、gne がこのファイルを読み直した生成と llama-completion --control-vector の生成が一致"
          : "「{p}」の {n} トークン(+1 倍、greedy)で、gne がこのファイルを読み直した生成と llama-completion --control-vector の生成が不一致", { p: esc(v.prompt), n: v.n_tokens })}
          ${v.differs_from_baseline ? "" : t("(無介入と同じ出力: 効きが弱い)")}</p>
        <details open><summary class="muted note">${t("出力を比べる")}</summary><p class="muted note">gne: ${esc(v.ours)}</p><p class="muted note">llama.cpp: ${esc(v.theirs)}</p></details>` : ""}
        ${x.warnings && x.warnings.length ? `<p class="muted note">${x.warnings.map(esc).join("<br>")}</p>` : ""}`;
    } catch (e) { res.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }
}

export { signed };
