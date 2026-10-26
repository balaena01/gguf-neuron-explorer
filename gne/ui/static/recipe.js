// レシピ(別冊 §2 LIST の recipe タブ、§4.8 マーク)。ユニットの追加・dose のスライダー・保存と読み込み・cvec の書き出し。
// 編集中のレシピはこのブラウザの localStorage に下書きとして置く(使えなければ置かない)。
import { getJSON, postJSON } from "./api.js";
import { esc } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const signed = (x) => `${x >= 0 ? "+" : "−"}${fmt(Math.abs(x), 2)}`;

// スライダーは 0(中央)から値まで塗る。+ は黄、− は赤紫。中央に 0 の目盛り
function trackBG(v) {
  const p = ((Math.max(-3, Math.min(3, v)) + 3) / 6) * 100, c = v >= 0 ? "#f2d24b" : "#c94fa7", bg = "#1f2630";
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
    try {
      const d = JSON.parse(localStorage.getItem(this.key) || "null");
      if (d && Array.isArray(d.units)) { this.name = d.name || ""; this.units = d.units; }
    } catch { /* 下書きが読めなくても空から始める */ }
  }

  _save() {
    try { localStorage.setItem(this.key, JSON.stringify({ name: this.name, units: this.units })); } catch { /* 置けなくても使える */ }
    this.onChange(this);
  }

  specs(scale = 1) { return this.units.map((u) => `${u.layer}:${u.neuron}:${u.dose * scale >= 0 ? "+" : ""}${+(u.dose * scale).toFixed(4)}`); }
  has(L, n) { return this.units.find((u) => u.layer === L && u.neuron === n); }
  add(L, n, dose) {
    const u = this.has(L, n);
    if (u) u.dose = dose; else this.units.push({ layer: L, neuron: n, dose });
    this.units.sort((a, b) => a.layer - b.layer || a.neuron - b.neuron);
    this._save();
  }
  set(i, dose) { this.units[i].dose = dose; this._save(); }
  remove(i) { this.units.splice(i, 1); this._save(); }
  clear() { this.units = []; this.name = ""; this._save(); }
  load(name, units) { this.name = name; this.units = units.map((u) => ({ layer: u.layer, neuron: u.neuron, dose: u.dose })); this._save(); }
}

export class RecipePanel {
  constructor(root, recipe, { onOpen, onChat, unitWords }) {
    this.root = root;
    this.r = recipe;
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
    const rows = r.units.map((u, i) => `
      <tr data-i="${i}"><td class="id"><a href="#" data-open="${i}">L${u.layer} N${u.neuron}</a>
          <span class="muted w">${esc(this.unitWords(u.layer, u.neuron))}</span><button class="small rm" data-rm="${i}" title="外す">×</button>
        <div class="sl"><span class="muted">−3</span><input type="range" class="dose" min="-3" max="3" step="0.05" value="${u.dose}" data-i="${i}" style="background:${trackBG(u.dose)}"><span class="muted">+3</span>
          <span class="sg mono ${u.dose >= 0 ? "plus" : "minus"}">${signed(u.dose)}</span>
          <input type="number" min="-10" max="10" step="0.05" value="${u.dose}" data-i="${i}" class="num" title="値を直接入れる"></div></td></tr>`).join("");
    this.root.innerHTML = `
      <div class="list-tools"><input id="rc-name" placeholder="レシピ名(英数字・_・-)" value="${esc(r.name)}" spellcheck="false">
        <button class="small" id="rc-save" title="~/.gne/recipes に保存">保存</button></div>
      <div class="list-tools"><select id="rc-load"><option value="">保存したレシピを読む…</option>${
        this.saved.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}(${x.units.length})</option>`).join("")}</select></div>
      ${r.units.length ? `<table class="rc mono"><tbody>${rows}</tbody></table>`
        : `<p class="muted note">ユニットがありません。NEURON の「+ レシピ」「− レシピ」で足します。</p>`}
      <p class="muted note">dose は rel(その層の残差ノルムの中央値に対する大きさ)。マップに + は黄、− は赤紫の輪で出ます。</p>
      <div class="rc-actions">
        <button id="rc-chat"${r.units.length ? "" : " disabled"}>3 列チャットで試す</button>
        <button id="rc-export"${r.units.length ? "" : " disabled"}>cvec を書き出す</button>
        <label class="muted"><input type="checkbox" id="rc-verify"> llama.cpp で確かめる</label>
      </div>
      ${r.units.length ? `<div class="rc-actions"><button class="small" id="rc-clear" title="すべてのユニットを外す">レシピを空にする</button></div>` : ""}
      <div id="rc-result"></div>`;
    const $ = (s) => this.root.querySelector(s);
    $("#rc-name").addEventListener("change", (ev) => { r.name = ev.target.value.trim(); r._save(); });
    $("#rc-save").addEventListener("click", () => this.save());
    const clr = $("#rc-clear");
    if (clr) clr.addEventListener("click", () => { if (confirm(`レシピの ${r.units.length} ユニットをすべて外しますか`)) { r.clear(); this.render(); } });
    $("#rc-load").addEventListener("change", (ev) => {
      const x = this.saved.find((y) => y.name === ev.target.value);
      if (x) { r.load(x.name, x.units); this.render(); }
    });
    for (const el of this.root.querySelectorAll("input[data-i]")) {
      el.addEventListener("input", (ev) => {
        const i = +ev.target.dataset.i, v = +ev.target.value;
        if (!Number.isFinite(v)) return;
        r.set(i, v);
        const tr = this.root.querySelector(`tr[data-i="${i}"]`);
        for (const x of tr.querySelectorAll("input")) if (x !== ev.target) x.value = String(v);
        tr.querySelector("input.dose").style.background = trackBG(v);
        const sg = tr.querySelector(".sg");
        sg.textContent = signed(v); sg.className = `sg mono ${v >= 0 ? "plus" : "minus"}`;
      });
    }
    for (const el of this.root.querySelectorAll("[data-rm]")) el.addEventListener("click", () => { r.remove(+el.dataset.rm); this.render(); });
    for (const el of this.root.querySelectorAll("[data-open]")) {
      el.addEventListener("click", (ev) => { ev.preventDefault(); const u = r.units[+el.dataset.open]; this.onOpen({ layer: u.layer, n: u.neuron }); });
    }
    $("#rc-chat").addEventListener("click", () => this.onChat());
    $("#rc-export").addEventListener("click", () => this.export());
  }

  async save() {
    const res = this.root.querySelector("#rc-result");
    const name = this.root.querySelector("#rc-name").value.trim();
    if (!/^[\w.-]+$/.test(name)) { res.innerHTML = `<p class="err">名前は英数字・_・-・. だけにしてください</p>`; return; }
    this.r.name = name; this.r._save();
    try {
      await postJSON(`/api/recipes/${encodeURIComponent(name)}`, { units: this.r.specs() });
      res.innerHTML = `<p class="muted note">保存しました(~/.gne/recipes/${esc(name)}.yaml)</p>`;
      this.refreshSaved();
    } catch (e) { res.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }

  async export() {
    const res = this.root.querySelector("#rc-result");
    const verify = this.root.querySelector("#rc-verify").checked;
    const name = (this.root.querySelector("#rc-name").value.trim() || "recipe").replace(/[^\w.-]/g, "_");
    res.innerHTML = `<p class="muted note">書き出し中${verify ? "(llama.cpp で読み直して生成を比べる、数十秒)" : ""}…</p>`;
    try {
      const x = await postJSON("/api/export", { units: this.r.specs(), name, verify });
      const a = document.createElement("a");                       // そのままダウンロードする
      a.href = x.url; a.download = x.file; document.body.appendChild(a); a.click(); a.remove();
      const norms = Object.entries(x.layer_norms).map(([L, v]) => `L${L} ${fmt(v, 1)}`).join(" · ");
      const v = x.verify;
      res.innerHTML = `<p class="note"><a href="${esc(x.url)}" download="${esc(x.file)}">${esc(x.file)}</a> を書き出しました${
          x.recipe_saved ? "" : `<br><span class="muted">(レシピ自体は未保存。上の「保存」で ~/.gne/recipes に残せます)</span>`}</p>
        <p class="muted note mono" title="層ごとに残差へ足すベクトルの長さ(残差ノルムと同じ単位)">足すベクトルの長さ: ${esc(norms)}</p>
        <p class="muted note">llama.cpp で使う: + 列は <code>--control-vector ${esc(x.file)}</code>、− 列は <code>--control-vector-scaled ${esc(x.file)} -1</code></p>${v ? `
        <p class="note">${v.match ? "✓" : "✗"} 「${esc(v.prompt)}」の ${v.n_tokens} トークン(+1 倍、greedy)で、gne がこのファイルを読み直した生成と llama-completion --control-vector の生成が${v.match ? "一致" : "不一致"}
          ${v.differs_from_baseline ? "" : "(無介入と同じ出力: 効きが弱い)"}</p>
        <details open><summary class="muted note">出力を比べる</summary><p class="muted note">gne: ${esc(v.ours)}</p><p class="muted note">llama.cpp: ${esc(v.theirs)}</p></details>` : ""}
        ${x.warnings && x.warnings.length ? `<p class="muted note">${x.warnings.map(esc).join("<br>")}</p>` : ""}`;
    } catch (e) { res.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }
}

export { signed };
