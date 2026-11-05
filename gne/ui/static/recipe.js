// レシピ(別冊 §2 LIST の recipe タブ、§4.8 マーク)。ユニットの追加・dose のスライダー・保存と読み込み・cvec の書き出し。
// ablate(方向の射影消去、本体設計 §6.17)の行: 方向・層の範囲・倍率。3 列チャットでは − の列で倍率を反転する。
// ヘッドの gain(CHANGE-04 D・G、L:Hh:xk): 行の倍率 k は 1 が無介入。全体に掛ける倍率 s では 1 + (k − 1)·s(eval の倍率列と同じ)。
// MoE(CHANGE-05 G): 専門家のニューロン L:Ee:Nn(add = 書き込み方向の押し、gain = 選ばれたときだけ効く)、専門家全体の gain L:Ee:xk、
// ルーターのバイアス L:Ee:route:delta(s 倍)、shared expert のニューロン L:S:Nn。
// ヘッドの方向成分の射影(CHANGE-06 G、L:Hh:sS@NAME): 行の強さ s(1 で方向成分を消す、2 で反転、0 で無介入)と方向。
// 全体に掛ける倍率でも s をそのまま倍にする(− の列は s が負 = その方向へ強める)。層ごとに、同じ方向なら LoRA は階数 1。
// 編集中のレシピはこのブラウザの localStorage に下書きとして置く(使えなければ置かない)。
import { getJSON, postJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const signed = (x, d = 2) => `${x >= 0 ? "+" : "−"}${fmt(Math.abs(x), d)}`;
const isDir = (u) => u.kind === "direction";
const isProj = (u) => u.kind === "head" && u.mode === "proj";   // ヘッドの方向成分の射影(CHANGE-06)
const isHead = (u) => u.kind === "head" && !isProj(u);          // ヘッドの gain
const isNeuron = (u) => (u.kind || "neuron") === "neuron";
export const isMoe = (u) => u.kind === "expert_neuron" || u.kind === "expert" || u.kind === "shared";
const isGain = (u) => isHead(u) || u.mode === "gain";             // 倍率 k(1 が無介入)の行
const isRoute = (u) => u.mode === "route";
const isPush = (u) => (u.kind === "expert_neuron" || u.kind === "shared") && !isGain(u);   // MoE の add = 書き込み方向の押し
const moeKey = (u) => `${u.layer}:${u.kind}:${u.expert ?? ""}:${u.neuron ?? ""}:${u.mode || "add"}`;
// 方向のユニット(gne direction の方向、L:dir:NAME:dose)は dose が小さいので、スライダーを ±1・刻み 0.005 にする
// ルーターのバイアスに足す値(route)は ±4・刻み 0.1
const range = (u) => (isDir(u) ? { max: 1, step: 0.005, d: 3 } : isRoute(u) ? { max: 4, step: 0.1, d: 1 } : { max: 3, step: 0.05, d: 2 });
const GAIN = { min: -1, max: 3, step: 0.05 };          // ヘッドの倍率 k のスライダー(0 で消す、負で反転、1 で無介入)
const PROJ = { min: 0, max: 30, step: 0.1 };           // 射影の強さ s のスライダー(0 で無介入、1 で方向成分を消す)
// 並び: 層 → ニューロン・方向・専門家のニューロン → ヘッド → 専門家全体、同じものは add → gain → route
const unitOrder = (u) => [u.layer, u.kind === "head" ? 1 : u.kind === "expert" ? 2 : 0,
  u.kind === "head" ? u.head : u.kind === "expert" ? u.expert : u.kind === "expert_neuron" ? 1e7 + u.expert * 1e5 + u.neuron
    : u.kind === "shared" ? 2e7 + u.neuron : u.neuron ?? -1, isProj(u) ? 3 : isRoute(u) ? 2 : isGain(u) ? 1 : 0];

// スライダーは 0(中央)から値まで塗る。+ は黄、− は赤紫。中央に 0 の目盛り
function trackBG(v, max = 3) {
  const p = ((Math.max(-max, Math.min(max, v)) + max) / (2 * max)) * 100, c = v >= 0 ? "#f2d24b" : "#c94fa7", bg = "#1f2630";
  const [a, b] = v >= 0 ? [50, p] : [p, 50];
  return `linear-gradient(to right, transparent calc(50% - 1px), #8a96a7 calc(50% - 1px), #8a96a7 calc(50% + 1px), transparent calc(50% + 1px)),` +
         `linear-gradient(to right, ${bg} ${a}%, ${c} ${a}%, ${c} ${b}%, ${bg} ${b}%)`;
}

// ヘッドの倍率のスライダー: 無介入の 1 から値まで塗る。強める(k > 1)は黄、弱める・消す(k < 1)は赤紫。1 に目盛り
function gainBG(k) {
  const at = (x) => ((Math.max(GAIN.min, Math.min(GAIN.max, x)) - GAIN.min) / (GAIN.max - GAIN.min)) * 100;
  const one = at(1), p = at(k), c = k >= 1 ? "#f2d24b" : "#c94fa7", bg = "#1f2630";
  const [a, b] = k >= 1 ? [one, p] : [p, one];
  return `linear-gradient(to right, transparent calc(${one}% - 1px), #8a96a7 calc(${one}% - 1px), #8a96a7 calc(${one}% + 1px), transparent calc(${one}% + 1px)),` +
         `linear-gradient(to right, ${bg} ${a}%, ${c} ${a}%, ${c} ${b}%, ${bg} ${b}%)`;
}
const kLabel = (k) => `×${fmt(k, 2)}`;
// 射影の強さのスライダー: 0(無介入)から値まで赤紫(方向成分を引く)。1(消す)と 2(反転)に目盛り
function projBG(v) {
  const at = (x) => ((Math.max(PROJ.min, Math.min(PROJ.max, x)) - PROJ.min) / (PROJ.max - PROJ.min)) * 100;
  const p = at(v), tick = (x) => `transparent calc(${at(x)}% - 1px), #8a96a7 calc(${at(x)}% - 1px), #8a96a7 calc(${at(x)}% + 1px), transparent calc(${at(x)}% + 1px)`;
  return `linear-gradient(to right, ${tick(1)}), linear-gradient(to right, ${tick(2)}),` +
         `linear-gradient(to right, #c94fa7 0%, #c94fa7 ${p}%, #1f2630 ${p}%, #1f2630 100%)`;
}
const sLabel = (s) => `s${+(+s).toFixed(2)}`;
// 射影のユニットの文字列(L:Hh:sS@NAME)→ { layer, head, s, direction }。読めなければ null
const PROJ_RE = /^(\d+):[Hh](\d+):s([+-]?[\d.]+(?:e[+-]?\d+)?)(?:@([\w.\-]+))?$/i;
export function parseProj(spec, defaultDirection = null) {
  const m = String(spec).trim().match(PROJ_RE);
  return m ? { layer: +m[1], kind: "head", mode: "proj", head: +m[2], s: +m[3], direction: m[4] || defaultDirection } : null;
}
// 全体に掛ける倍率 s でのヘッドの倍率(1 が無介入。s = −1 なら 1 の反対側: ×0 → ×2)
const gainAt = (k, s) => +(1 + (k - 1) * s).toFixed(4);

// ユニット 1 つのサーバ向けの指定(gne/steer.py の parse_unit の形)。s はレシピ全体に掛ける倍率
export function unitSpec(u, s = 1) {
  if (isProj(u)) return `${u.layer}:H${u.head}:s${+(u.s * s).toFixed(4)}${u.direction ? `@${u.direction}` : ""}`;
  const who = isHead(u) ? `H${u.head}` : isDir(u) ? `dir:${u.direction}` : u.kind === "expert_neuron" ? `E${u.expert}:N${u.neuron}`
    : u.kind === "shared" ? `S:N${u.neuron}` : u.kind === "expert" ? `E${u.expert}${isRoute(u) ? ":route" : ""}` : `${u.neuron}`;
  if (isGain(u)) return `${u.layer}:${who}:x${gainAt(u.k, s)}`;
  const d = u.dose * s;
  return `${u.layer}:${who}:${d >= 0 ? "+" : ""}${+d.toFixed(4)}`;
}

export class Recipe {
  constructor(modelId, onChange) {
    this.key = `gne.recipe.${modelId}`;
    this.onChange = onChange;
    this.name = "";
    this.units = [];                   // [{ layer, neuron, dose }]、方向 { layer, kind: "direction", direction, dose }、ヘッド { layer, kind: "head", head, k }、
                                       // MoE { layer, kind: "expert_neuron" | "shared" | "expert", expert, neuron, mode: "add" | "gain" | "route", dose | k }
    this.ablate = null;                // { direction, scale, layers, from_layer } か null
    this.directionLayer = "own";       // proj のユニットに当てる方向の層(CHANGE-06): own = そのヘッドの層の r_L、base = 基準層の r
    this.sampling = null;              // このレシピを試すときのサンプリング { temperature, top_k, top_p, min_p, presence_penalty } か null
    this.skipped = 0;                  // 読み込んだレシピのうち UI では扱えないユニット(ニューロンの gain、ヘッドの query など)の数。
                                       // ヘッドの query(CHANGE-04 E)は L:Hh の文字列の形が無く、送り返せないので外す
    try {
      const d = JSON.parse(localStorage.getItem(this.key) || "null");
      if (d && Array.isArray(d.units)) {
        this.name = d.name || ""; this.units = d.units; this.ablate = d.ablate || null; this.sampling = d.sampling || null;
        this.directionLayer = d.direction_layer === "base" ? "base" : "own";
      }
    } catch { /* 下書きが読めなくても空から始める */ }
  }

  _save() {
    try {
      localStorage.setItem(this.key, JSON.stringify({ name: this.name, units: this.units, ablate: this.ablate, sampling: this.sampling,
                                                      direction_layer: this.directionLayer }));
    } catch { /* 置けなくても使える */ }
    this.onChange(this);
  }

  empty() { return !this.units.length && !this.ablate; }
  // ablate を倍率 s で(− の列は s < 0 = その方向を強める)。無ければ undefined
  ablateAt(s = 1) { return this.ablate ? { ...this.ablate, scale: +(this.ablate.scale * s).toFixed(4) } : undefined; }
  setAblate(a) { this.ablate = a && a.direction ? a : null; this._save(); }
  setSampling(sp) { this.sampling = sp ? { ...sp } : null; this._save(); }

  // サーバに送るユニット指定。scale はレシピ全体に掛ける倍率(3 列チャットの ± 列、eval の倍率と同じ意味)
  specs(scale = 1) { return this.units.map((u) => unitSpec(u, scale)); }
  // control vector に入るもの(add)だけ。gain は LoRA、専門家全体の gain と route はスケールパッチなので cvec には入らない
  // (MoE の add は書き込み方向の押しとして入る)
  addSpecs() { return this.units.filter((u) => !isGain(u) && !isRoute(u) && !isProj(u)).map((u) => unitSpec(u)); }
  has(L, n) { return this.units.find((u) => isNeuron(u) && u.layer === L && u.neuron === n); }
  neurons() { return this.units.filter(isNeuron); }
  hasHead(L, h) { return this.units.find((u) => isHead(u) && u.layer === L && u.head === h); }
  heads() { return this.units.filter(isHead); }
  // ヘッドの方向成分の射影(CHANGE-06)。同じヘッドでも方向が違えば別の行
  projs() { return this.units.filter(isProj); }
  hasProj(L, h, dir = undefined) { return this.units.find((u) => isProj(u) && u.layer === L && u.head === h && (dir === undefined || u.direction === dir)); }
  addProj(L, h, s, direction) {
    const u = this.hasProj(L, h, direction);
    if (u) u.s = s; else this.units.push({ layer: L, kind: "head", mode: "proj", head: h, s, direction });
    this._sort();
    this._save();
  }
  removeProj(L, h, dir = undefined) { const u = this.hasProj(L, h, dir); if (u) this.remove(this.units.indexOf(u)); }
  setProjDir(i, dir) { if (isProj(this.units[i])) { this.units[i].direction = dir; this._save(); } }
  setDirectionLayer(v) { this.directionLayer = v === "base" ? "base" : "own"; this._save(); }
  // 層ごとの射影の方向の数(1 なら LoRA は階数 1 に畳める)
  projRank(L) { return new Set(this.projs().filter((u) => u.layer === L).map((u) => u.direction || "")).size; }
  // 射影のユニットの組(gne heads allocate の組など、"L:Hh:sS@NAME" の並び)でレシピを置き換える。読めないものは外して数える
  loadProj(name, specs, defaultDirection = null) {
    const us = specs.map((x) => parseProj(x, defaultDirection));
    this.load(name, us.filter(Boolean), null, null);
    this.skipped = us.filter((u) => !u).length;
    this._save();
  }
  // MoE の単位(q: { layer, kind, expert, neuron, mode })。同じ単位でも add・gain・route は別の行
  findMoe(q) { const k = moeKey(q); return this.units.find((u) => isMoe(u) && moeKey(u) === k); }
  moe() { return this.units.filter(isMoe); }
  // v は add・route なら dose、gain なら倍率 k。null で外す
  setMoe(q, v) {
    const u = this.findMoe(q);
    if (v === null) { if (u) this.remove(this.units.indexOf(u)); return; }
    const f = q.mode === "gain" ? "k" : "dose";
    if (u) u[f] = v;
    else this.units.push({ layer: q.layer, kind: q.kind, ...(q.kind !== "shared" ? { expert: q.expert } : {}),
                           ...(q.kind !== "expert" ? { neuron: q.neuron } : {}), mode: q.mode || "add", [f]: v });
    this._sort();
    this._save();
  }
  _sort() {
    this.units.sort((a, b) => { const x = unitOrder(a), y = unitOrder(b); return x[0] - y[0] || x[1] - y[1] || x[2] - y[2] || x[3] - y[3]; });
  }
  add(L, n, dose) {
    const u = this.has(L, n);
    if (u) u.dose = dose; else this.units.push({ layer: L, neuron: n, dose });
    this._sort();
    this._save();
  }
  addHead(L, h, k) {
    const u = this.hasHead(L, h);
    if (u) u.k = k; else this.units.push({ layer: L, kind: "head", head: h, k });
    this._sort();
    this._save();
  }
  removeHead(L, h) { const u = this.hasHead(L, h); if (u) this.remove(this.units.indexOf(u)); }
  // 行 i の値(ニューロン・方向・route は dose、ヘッドと gain は倍率 k)
  set(i, v) { const u = this.units[i]; if (isProj(u)) u.s = v; else if (isGain(u)) u.k = v; else u.dose = v; this._save(); }
  remove(i) { this.units.splice(i, 1); this._save(); }
  clear() { this.units = []; this.name = ""; this.ablate = null; this.sampling = null; this.skipped = 0; this.directionLayer = "own"; this._save(); }
  // directionLayer: 保存したレシピの direction_layer(proj のユニットに当てる方向の層、CHANGE-06)
  load(name, units, ablate = null, sampling = null, directionLayer = "own") {
    const moeOk = (u) => isMoe(u) && (u.kind === "shared" || (u.expert !== null && u.expert !== undefined))
      && ({ expert_neuron: ["add", "gain"], shared: ["add", "gain"], expert: ["gain", "route"] }[u.kind].includes(u.mode || "add"))
      && (u.mode === "gain" ? Number.isFinite(+u.k) && u.k !== null : true);
    const ok = units.filter((u) => (((u.kind || "neuron") === "neuron" || u.kind === "direction") && (u.mode || "add") === "add")
      || (u.kind === "head" && u.mode === "gain" && u.head !== null && u.head !== undefined && Number.isFinite(+u.k))
      || (u.kind === "head" && u.mode === "proj" && u.head !== null && u.head !== undefined && Number.isFinite(+(u.s ?? u.k)) && !!u.direction)
      || moeOk(u));
    this.skipped = units.length - ok.length;
    this.name = name;
    this.units = ok.map((u) => (isDir(u) ? { layer: u.layer, kind: "direction", direction: u.direction, dose: u.dose }
      : isProj(u) ? { layer: u.layer, kind: "head", mode: "proj", head: u.head, s: +(u.s ?? u.k), direction: u.direction }
      : isHead(u) ? { layer: u.layer, kind: "head", head: u.head, k: +u.k }
      : isMoe(u) ? { layer: u.layer, kind: u.kind, ...(u.kind !== "shared" ? { expert: +u.expert } : {}), ...(u.kind !== "expert" ? { neuron: u.neuron } : {}),
                     mode: u.mode || "add", ...(u.mode === "gain" ? { k: +u.k } : { dose: +(u.dose ?? u.delta) }) }
      : { layer: u.layer, neuron: u.neuron, dose: u.dose }));
    this.ablate = ablate || null;
    this.sampling = sampling || null;
    this.directionLayer = directionLayer === "base" ? "base" : "own";
    this._save();
  }
}

// サンプリングの値の短い表示(温度 1.0・top-k 20・…)
export const samplingLabel = (sp) => (sp ? `${t("温度")} ${+sp.temperature} · top-k ${sp.top_k} · top-p ${+sp.top_p} · min-p ${+sp.min_p} · presence ${+sp.presence_penalty}` : "");

export class RecipePanel {
  constructor(root, recipe, { onOpen, onOpenHead, onOpenExpert, onChat, unitWords, directions, projDirections, headKind, nFF, onRunJob }) {
    this.root = root;
    this.r = recipe;
    this.directions = directions || (() => []);   // 方向の名前(gne direction / reverse)
    this.projDirections = projDirections || this.directions;   // proj に使える方向(gne direction の方向セット、CHANGE-06)
    this.onOpen = onOpen;
    this.onOpenHead = onOpenHead || (() => {});   // ({ layer, head }) → HEAD パネルで開く
    this.onOpenExpert = onOpenExpert || (() => {});   // ({ layer, expert }) → EXPERT パネルで開く(CHANGE-05 G)
    this.nFF = nFF || (() => 0);                  // (L) → MoE の層の専門家の幅(通し番号 = 専門家 × 幅 + n)
    this.headKind = headKind || (() => "");       // (L) → "attention" / "線形注意" など
    this.onChat = onChat;
    this.unitWords = unitWords;        // (L, n) → 短い説明(sweep の語など)。無ければ ""
    this.saved = [];
    this.onRunJob = onRunJob || (async () => null);   // (kind, params) → ジョブ(jobs.js の start。見積もりの確認つき)
    // LoRA・焼き込みの書き出し(headproj_export のジョブ)。欄の値は描き直しても残す
    this.lxOpen = false;
    this.lx = { bake: false, requant: "q8_0", via_f16: false, llama_quantize: "", imatrix: "", verify: true, contrast: "", strip: false,
                file: "", into: "" };              // file: ファイル名(拡張子なし、空ならレシピの名前から)、into: 焼き込む先の別の GGUF
    this.lxJobId = null;
    this.lxMsg = null;                 // { html } 書き出しの結果か進み具合
  }

  async refreshSaved() {
    this.render();                     // 先に今の中身を出し、保存済みの一覧は読めてから足す
    try { this.saved = (await getJSON("/api/recipes")).filter((x) => x.this_model); } catch { this.saved = []; }
    this.render();
  }

  render() {
    const r = this.r;
    const rows = r.units.map((u, i) => {
      if (isProj(u)) return this.projRow(u, i);
      if (isGain(u)) return this.gainRow(u, i);
      const g = range(u);
      const id = isDir(u)
        ? `<span class="mono" title="${t("方向(gne direction で作ったもの)。その層に足す")}">L${u.layer} dir:${esc(u.direction)}</span> <span class="muted w">${t("方向")}</span>`
        : isMoe(u) ? this.moeId(u, i)
        : `<a href="#" data-open="${i}">L${u.layer} N${u.neuron}</a> <span class="muted w">${esc(this.unitWords(u.layer, u.neuron))}</span>`;
      return `
      <tr data-i="${i}"${isMoe(u) ? ` class="rc-moe"` : ""}><td class="id">${id}<button class="small rm" data-rm="${i}" title="${t("外す")}">×</button>
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
      ${r.skipped ? `<p class="muted note">${t("UI では扱えないユニット(ニューロンの gain・ヘッドの query・ランダム方向など){n} 個を外しました", { n: r.skipped })}</p>` : ""}
      ${r.heads().length ? `<p class="muted note">${t("ヘッドの倍率 k は 1 が無介入(0 で消す、2 で倍、負で反転)。LoRA で当てるので cvec には入りません。3 列チャットの − 列は 1 の反対側(×0 なら ×2)。")}</p>` : ""}
      ${r.projs().length ? this.projNote() : ""}
      ${r.moe().length ? `<p class="muted note">${t("MoE: 方向押し = 専門家のニューロンの add。専門家が選ばれたかに関係なく全トークンにその書き込み方向を足す(ニューロン押しではない)。選ばれたときだけ効くのは gain(×)。専門家全体の ×k と route はモデルを読み直すパッチで当てる(列ごとに 0.3〜1 秒)。3 列チャットの − 列は ×k が 2 − k、route と add は符号を反転。")}</p>` : ""}
      ${this.ablateHTML()}
      ${r.sampling ? `<div class="rc-sampling muted note" title="${t("このレシピを試すときのサンプリング。チャットの「レシピの設定」で使い、保存するとレシピのファイルに入る(OpenAI 互換 API の既定にもなる)")}">${t("サンプリング")}: <span class="mono">${esc(samplingLabel(r.sampling))}</span><button class="small" id="rc-sp-rm" title="${t("レシピから外す")}">×</button></div>` : ""}
      <p class="muted note">${t("dose は rel(その層の残差ノルムの中央値に対する大きさ)。マップに + は黄、− は赤紫の輪で出ます。")}</p>
      <div class="rc-actions">
        <button id="rc-chat"${r.empty() ? " disabled" : ""}>${t("3 列チャットで試す")}</button>
        <button id="rc-export"${r.addSpecs().length && document.body.dataset.cvec !== "off" ? "" : " disabled"}>${t("cvec を書き出す")}</button>
        <label class="muted"><input type="checkbox" id="rc-verify"> ${t("llama.cpp で確かめる")}</label>
        <button id="rc-lora"${r.heads().length || r.projs().length ? "" : " disabled"} title="${t("ヘッドの方向成分(proj)と倍率(gain)を、llama.cpp の --lora で使う LoRA アダプタか、重みに焼き込んだ GGUF に書き出す")}">${t("LoRA・焼き込み")}${this.lxOpen ? " ▴" : " ▾"}</button>
      </div>
      ${this.lxOpen ? this.lxHTML() : ""}
      ${r.units.length ? `<div class="rc-actions"><button class="small" id="rc-clear" title="${t("すべてのユニットを外す")}">${t("レシピを空にする")}</button></div>` : ""}
      <div id="rc-result"></div>`;
    const $ = (s) => this.root.querySelector(s);
    $("#rc-name").addEventListener("change", (ev) => { r.name = ev.target.value.trim(); r._save(); });
    $("#rc-save").addEventListener("click", () => this.save());
    const clr = $("#rc-clear");
    if (clr) clr.addEventListener("click", () => { if (confirm(t("レシピの {n} ユニットをすべて外しますか", { n: r.units.length }))) { r.clear(); this.render(); } });
    $("#rc-load").addEventListener("change", (ev) => {
      const x = this.saved.find((y) => y.name === ev.target.value);
      if (x) { r.load(x.name, x.units, x.ablate, x.sampling, x.direction_layer); this.render(); }
    });
    for (const el of this.root.querySelectorAll("input[data-i]")) {
      el.addEventListener("input", (ev) => {
        const i = +ev.target.dataset.i, v = +ev.target.value;
        if (!Number.isFinite(v)) return;
        r.set(i, v);
        const tr = this.root.querySelector(`tr[data-i="${i}"]`);
        for (const x of tr.querySelectorAll("input")) if (x !== ev.target) x.value = String(v);
        const sg = tr.querySelector(".sg");
        if (isProj(r.units[i])) {
          tr.querySelector("input.dose").style.background = projBG(v);
          sg.textContent = sLabel(v);
          return;
        }
        if (isGain(r.units[i])) {
          tr.querySelector("input.dose").style.background = gainBG(v);
          sg.textContent = kLabel(v); sg.className = `sg mono ${v >= 1 ? "plus" : "minus"}`;
          return;
        }
        const g = range(r.units[i]);
        tr.querySelector("input.dose").style.background = trackBG(v, g.max);
        sg.textContent = signed(v, g.d); sg.className = `sg mono ${v >= 0 ? "plus" : "minus"}`;
      });
    }
    for (const el of this.root.querySelectorAll("[data-rm]")) el.addEventListener("click", () => { r.remove(+el.dataset.rm); this.render(); });
    for (const el of this.root.querySelectorAll("select[data-pdir]")) el.addEventListener("change", () => { r.setProjDir(+el.dataset.pdir, el.value); this.render(); });
    const dl = $("#rc-dl");
    if (dl) dl.addEventListener("change", (ev) => { r.setDirectionLayer(ev.target.value); this.render(); });
    for (const el of this.root.querySelectorAll("[data-open]")) {
      el.addEventListener("click", (ev) => {
        ev.preventDefault();
        const u = r.units[+el.dataset.open];
        if (u.kind === "head") this.onOpenHead({ layer: u.layer, head: u.head });
        else if (u.kind === "expert") this.onOpenExpert({ layer: u.layer, expert: u.expert });
        else if (u.kind === "expert_neuron") this.onOpen({ layer: u.layer, n: u.expert * this.nFF(u.layer) + u.neuron });
        else this.onOpen({ layer: u.layer, n: u.neuron });
      });
    }
    $("#rc-chat").addEventListener("click", () => this.onChat());
    const sprm = $("#rc-sp-rm");
    if (sprm) sprm.addEventListener("click", () => { r.setSampling(null); this.render(); });
    this.wireAblate();
    $("#rc-export").addEventListener("click", () => this.export());
    $("#rc-lora").addEventListener("click", () => { this.lxOpen = !this.lxOpen; this.render(); });
    if (this.lxOpen) this.wireLx();
  }

  // MoE の単位の名前(CHANGE-05 G)。add には「方向押し」の印、専門家全体は EXPERT パネルへ
  moeId(u, i) {
    const push = isPush(u) ? ` <span class="badge push" title="${t("書き込み方向の押し: 専門家が選ばれたかに関係なく、全トークンでこのニューロンの書き込み方向を足す(ニューロン押しではない)")}">${t("方向押し")}</span>` : "";
    if (u.kind === "expert_neuron") return `<a href="#" data-open="${i}" title="${t("NEURON パネルで開く")}">L${u.layer} E${u.expert} N${u.neuron}</a>${push}${
      isGain(u) ? ` <span class="muted w">gain · ${t("選ばれたときだけ")}</span>` : ""}`;
    if (u.kind === "shared") return `<span class="mono">L${u.layer} S N${u.neuron}</span>${push} <span class="muted w">shared expert${isGain(u) ? " · gain" : ""}</span>`;
    return `<a href="#" data-open="${i}" title="${t("EXPERT パネルで開く")}">L${u.layer} E${u.expert}</a> <span class="muted w">${
      isRoute(u) ? t("route(ルーターのバイアスに足す)") : t("専門家全体 · gain(パッチ)")}</span>`;
  }

  // 射影の行(CHANGE-06、L33 H16 s15.4 @evil)。スライダーは強さ s(0〜30、1 で方向成分を消す、2 で反転)、方向の選択、
  // その層の射影が LoRA の階数 1 に畳めるか(層の中の方向が 1 種類なら rank 1、k 種類なら rank k)
  projRow(u, i) {
    const dirs = this.projDirections(), rank = this.r.projRank(u.layer);
    const opts = [...dirs, ...(u.direction && !dirs.includes(u.direction) ? [u.direction] : [])]
      .map((d) => `<option${d === u.direction ? " selected" : ""}>${esc(d)}</option>`).join("");
    const rk = rank === 1
      ? `<span class="badge rank1" title="${t("L{L} の射影のヘッドはどれも同じ方向なので、LoRA は階数 1 に畳める", { L: u.layer })}">rank 1</span>`
      : `<span class="badge rankk" title="${t("L{L} の射影のヘッドの方向が {n} 種類あるので、LoRA の階数は {n}(同じ方向なら 1 に畳める)", { L: u.layer, n: rank })}">rank ${rank}</span>`;
    return `
      <tr data-i="${i}" class="rc-proj"><td class="id"><a href="#" data-open="${i}" title="${t("HEAD パネルで開く")}">L${u.layer} H${u.head}</a>
        <span class="sg mono minus">${sLabel(u.s)}</span> <span class="mono">@</span><select class="rc-pdir" data-pdir="${i}" title="${t("射影する方向(gne direction の方向セット)")}">${opts}</select>
        ${rk}<button class="small rm" data-rm="${i}" title="${t("外す")}">×</button>
        <div class="sl"><span class="muted">${PROJ.min}</span><input type="range" class="dose" min="${PROJ.min}" max="${PROJ.max}" step="${PROJ.step}" value="${u.s}" data-i="${i}" style="background:${projBG(u.s)}"><span class="muted">${PROJ.max}</span>
          <input type="number" min="-300" max="300" step="${PROJ.step}" value="${+(+u.s).toFixed(3)}" data-i="${i}" class="num" title="${t("値を直接入れる")}"></div></td></tr>`;
  }

  // 射影の説明と、方向の層(own / base)の選択
  projNote() {
    const r = this.r;
    return `<div class="rc-ablate"><span class="muted" title="${t("proj のユニットに当てる方向の層。own = そのヘッドがある層の方向 r_L、base = 方向セットの基準層の r(gne direction score で記録した層)")}">${t("方向の層")}</span>
        <select id="rc-dl"><option value="own"${r.directionLayer === "own" ? " selected" : ""}>own</option><option value="base"${r.directionLayer === "base" ? " selected" : ""}>base</option></select></div>
      <p class="muted note">${t("射影 s: ヘッドが残差に書くベクトルのうち方向 r̂ に沿った成分だけを (1 − s) 倍にする(1 で消す、2 で反転、直交成分はそのまま)。LoRA で当てるので cvec には入りません。3 列チャットの − 列は s が負(その方向へ強める)。")}</p>${
        r.directionLayer === "base" ? `<p class="muted note">${t("方向の層 base は保存したレシピのファイルにだけ入ります(3 列チャットと「今のレシピで eval」はユニットの文字列を送るので own で当てます)。")}</p>` : ""}`;
  }

  // 倍率の行(ヘッド L21 H7 ×0.00、MoE の gain)。スライダーは倍率 k(−1〜3、1 が無介入)
  gainRow(u, i) {
    const id = isHead(u) ? `<a href="#" data-open="${i}" title="${t("HEAD パネルで開く")}">L${u.layer} H${u.head}</a> <span class="muted w">${esc(this.headKind(u.layer))} · gain</span>`
      : this.moeId(u, i);
    return `
      <tr data-i="${i}" class="rc-gain${isHead(u) ? " rc-head" : " rc-moe"}"><td class="id">${id}<button class="small rm" data-rm="${i}" title="${t("外す")}">×</button>
        <div class="sl"><span class="muted">${GAIN.min}</span><input type="range" class="dose" min="${GAIN.min}" max="${GAIN.max}" step="${GAIN.step}" value="${u.k}" data-i="${i}" style="background:${gainBG(u.k)}"><span class="muted">${GAIN.max}</span>
          <span class="sg mono ${u.k >= 1 ? "plus" : "minus"}">${kLabel(u.k)}</span>
          <input type="number" min="-4" max="8" step="${GAIN.step}" value="${u.k}" data-i="${i}" class="num" title="${t("値を直接入れる")}"></div></td></tr>`;
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
      await postJSON(`/api/recipes/${encodeURIComponent(name)}`, { units: this.r.specs(), ablate: this.r.ablate, sampling: this.r.sampling,
                                                                   ...(this.r.projs().length && this.r.directionLayer !== "own" ? { direction_layer: this.r.directionLayer } : {}) });
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
      const x = await postJSON("/api/export", { units: this.r.addSpecs(), name, verify });
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

  // --- LoRA・焼き込み(headproj_export のジョブ) -----------------------------------------------------------
  // ファイル名の既定(拡張子なし): レシピの名前(焼き込みは -headproj を付ける)
  lxDefaultStem() {
    const nm = (((this.root.querySelector("#rc-name") || {}).value || this.r.name || "headproj").trim().replace(/[^\w.-]/g, "_")) || "headproj";
    return this.lx.bake ? `${nm}-headproj` : nm;
  }

  lxHTML() {
    const f = this.lx, r = this.r, bake = f.bake;
    const n = r.heads().length + r.projs().length, nAdd = r.addSpecs().length;
    const dis = bake ? "" : " disabled";
    return `<div class="rc-lx">
      <div class="rc-lx-row"><span class="muted">${t("形式")}</span>
        <label title="${t("小さいファイル(同じ層・同じ方向のヘッドは階数 1 にまとまる)。llama.cpp では --lora で当てる。元のモデルはそのまま")}"><input type="radio" name="lx-kind" value="lora"${bake ? "" : " checked"}> ${t("LoRA アダプタ")}</label>
        <label title="${t("関係するテンソルだけ書き換えた GGUF を丸ごと作る。--lora なしでそのまま使える。モデルと同じくらいのディスクが要る")}"><input type="radio" name="lx-kind" value="bake"${bake ? " checked" : ""}> ${t("重みに焼き込む(GGUF)")}</label></div>
      <div class="rc-lx-row${bake ? "" : " off"}"><span class="muted" title="${t("空なら、いま読んでいるモデルに焼き込む。同じモデルの量子化していない GGUF(F16 / BF16)のパスを入れると、方向はこのモデルで作ったものを使い、そちらに焼き込む(GPU は使わない。少し誤差がある。LoRA 版との比較はできない)")}">${t("焼き込む先")}</span>
        <input id="lx-into" size="30" value="${esc(f.into)}" placeholder="${t("空ならいま読んでいるモデル")}" spellcheck="false"${dis}></div>
      <div class="rc-lx-row${bake ? "" : " off"}"><span class="muted" title="${t("書き換えたテンソルを何の型で書くか。Q8_0 は元より少し大きくなることがある、F16 はもっと大きい。下の「元の型に戻す」を使うとほぼ元の大きさ")}">${t("書き換えたテンソルの型")}</span>
        <select id="lx-requant"${dis || (f.via_f16 ? " disabled" : "")}><option value="q8_0"${f.requant === "q8_0" ? " selected" : ""}>Q8_0</option><option value="f16"${f.requant === "f16" ? " selected" : ""}>F16</option><option value="keep"${f.requant === "keep" ? " selected" : ""}>${t("元と同じ(F16・BF16 のモデル)")}</option></select>
        <label title="${t("書き換えたテンソルを F16 で書いてから、llama-quantize で元の量子化の型に戻す(ファイルの大きさを元に近く保つ。低ビットの型は imatrix があると誤差が小さい)")}"><input type="checkbox" id="lx-viaf16"${f.via_f16 ? " checked" : ""}${dis}> ${t("元の量子化の型に戻す")}</label></div>
      ${bake && f.via_f16 ? `<div class="rc-lx-row"><span class="muted">llama-quantize</span><input id="lx-lq" size="28" value="${esc(f.llama_quantize)}" placeholder="${t("空なら GNE_LLAMA_QUANTIZE か PATH から探す")}" spellcheck="false">
        <span class="muted" title="${t("低ビットの型(IQ1・IQ2・IQ3_XXS)に戻すときは必須。空なら、モデルの隣の <モデル名>.imatrix.gguf / .imatrix.dat / imatrix.gguf を探して使う")}">imatrix</span><input id="lx-imatrix" size="22" value="${esc(f.imatrix)}" placeholder="${t("空ならモデルの隣を探す")}" spellcheck="false"></div>` : ""}
      <div class="rc-lx-row${bake && !f.into.trim() ? "" : " off"}"><label title="${t("焼き込み版と LoRA 版で、方向の点数・最初のトークンの分布・最尤トークン・ppl が一致するかを確かめる(モデルを 2 回読むので時間がかかる)")}"><input type="checkbox" id="lx-verify"${f.verify ? " checked" : ""}${bake && !f.into.trim() ? "" : " disabled"}> ${t("LoRA 版と一致するか確かめる")}</label>
        ${bake && f.verify && !f.into.trim() ? `<span class="muted">${t("対照集合")}</span><input id="lx-contrast" size="24" value="${esc(f.contrast)}" placeholder="${t("空なら方向を作った対照集合")}" spellcheck="false">` : ""}</div>
      <div class="rc-lx-row"><span class="muted" title="${t("書き出すファイルの名前(拡張子なし)。空ならレシピの名前から作る。保存先は <root>/export/")}">${t("ファイル名")}</span>
        <input id="lx-file" size="26" value="${esc(f.file)}" placeholder="${esc(this.lxDefaultStem())}" spellcheck="false"><span class="muted mono">${bake ? ".gguf" : ".lora.gguf"}</span></div>
      <div class="rc-lx-row"><label title="${t("gne が書く記録(gne.* のメタデータ: レシピ・ヘッド・強さ・方向の鍵、名前の印、横の .headproj.json)を入れない。人に渡すファイル用。書いたあと、メタデータに手元のパスが残っていないかも確かめる")}"><input type="checkbox" id="lx-strip"${f.strip ? " checked" : ""}> ${t("gne の記録(メタデータ)を入れない(人に渡す用)")}</label></div>
      <div class="rc-lx-row"><button id="lx-go">${t(bake ? "焼き込む" : "LoRA を書き出す")}</button>
        <span class="muted note">${t("名前は上のレシピの名前(空なら headproj)。ヘッドのユニット {n} 個が入る", { n })}${nAdd ? ` · ${t("add のユニット {k} 個は入らない(cvec で書き出す)", { k: nAdd })}` : ""}${bake ? ` · ${t("焼き込みはモデルを丸ごともう 1 つ書く(同じくらいのディスク・時間)")}` : ""}</span></div>
      <div id="lx-result">${this.lxMsg ? this.lxMsg.html : ""}</div></div>`;
  }

  wireLx() {
    const $ = (s) => this.root.querySelector(s), f = this.lx;
    for (const el of this.root.querySelectorAll("input[name=lx-kind]")) el.addEventListener("change", () => { f.bake = el.value === "bake" && el.checked; this.render(); });
    const on = (sel, ev, fn) => { const el = $(sel); if (el) el.addEventListener(ev, fn); };
    on("#lx-requant", "change", (ev) => { f.requant = ev.target.value; });
    on("#lx-viaf16", "change", (ev) => { f.via_f16 = ev.target.checked; this.render(); });
    on("#lx-lq", "input", (ev) => { f.llama_quantize = ev.target.value; });
    on("#lx-imatrix", "input", (ev) => { f.imatrix = ev.target.value; });
    on("#lx-verify", "change", (ev) => { f.verify = ev.target.checked; this.render(); });
    on("#lx-contrast", "input", (ev) => { f.contrast = ev.target.value; });
    on("#lx-strip", "change", (ev) => { f.strip = ev.target.checked; });
    on("#lx-file", "input", (ev) => { f.file = ev.target.value; });
    on("#lx-into", "change", (ev) => {                   // 量子化していないモデルへ: 型は元のまま(keep)にし、比較はしない
      f.into = ev.target.value;
      if (f.into.trim() && f.requant !== "f16") f.requant = "keep";
      this.render();
    });
    on("#lx-go", "click", () => this.runLx());
  }

  lxSet(html) { this.lxMsg = { html }; const el = this.root.querySelector("#lx-result"); if (el) el.innerHTML = html; }

  async runLx(overwrite = false) {
    const f = this.lx, r = this.r;
    const name = ((this.root.querySelector("#rc-name") || {}).value || r.name || "headproj").trim().replace(/[^\w.-]/g, "_") || "headproj";
    const stem = (f.file.trim() || this.lxDefaultStem()).replace(/\.lora\.gguf$|\.gguf$/, "");
    if (!/^[\w.-]+$/.test(stem)) { this.lxSet(`<p class="err">${t("ファイル名は英数字・_・-・. だけにしてください")}</p>`); return; }
    const units = [...r.heads(), ...r.projs()].map((u) => r.specs()[r.units.indexOf(u)]);
    const p = { units, name, bake: f.bake, file_stem: stem, overwrite };
    if (f.strip) p.strip_meta = true;
    if (r.projs().length && r.directionLayer && r.directionLayer !== "own") p.direction_layer = r.directionLayer;
    if (f.bake) {
      if (f.via_f16) {
        p.via_f16 = true;
        if (f.llama_quantize.trim()) p.llama_quantize = f.llama_quantize.trim();
        if (f.imatrix.trim()) p.imatrix = f.imatrix.trim();
      } else p.requant = f.requant;
      if (f.into.trim()) p.into = f.into.trim();
      else if (f.verify) { p.verify = true; if (f.contrast.trim()) p.contrast = f.contrast.trim(); }
    }
    this.lxSet(`<p class="muted note">${t("ジョブを投げています…")}</p>`);
    try {
      const j = await this.onRunJob("headproj_export", p);
      if (!j) { this.lxSet(`<p class="muted note">${t("やめました")}</p>`); return; }
      this.lxJobId = j.id;
      this.lxSet(`<p class="muted note">${t(f.bake ? "焼き込んでいます(下のジョブのバーに進み具合)。終わるとここに場所が出ます" : "書き出しています。終わるとここに場所が出ます")}</p>`);
    } catch (e) { this.lxSet(`<p class="err">${esc(e.message)}</p>`); }
  }

  // app.js の onJobFinished から(headproj_export)
  lxDone(j) {
    if (j.id !== this.lxJobId) return;
    if (j.status !== "done" && j.error && j.error.code === "FILE_EXISTS") {          // 同じ名前のファイルがある: 確かめてから上書き
      this.lxSet(`<p class="err">${esc(j.error.message)}</p><button class="small" id="lx-over">${t("上書きする")}</button>
        <span class="muted note">${t("別の名前にするなら、ファイル名を変えてもう一度押す")}</span>`);
      const b = this.root.querySelector("#lx-over");
      if (b) b.addEventListener("click", () => this.runLx(true));
      return;
    }
    if (j.status !== "done") { this.lxSet(`<p class="err">${t("失敗")}: ${esc((j.error && j.error.message) || j.status)}</p>${j.error && j.error.hint ? `<p class="muted note">${esc(j.error.hint)}</p>` : ""}`); return; }
    const x = j.result || {}, mb = (x.bytes || 0) / 1e6;
    const layers = Object.entries(x.proj_layers || {}).map(([L, v]) => `L${L}: ${t("階数 {r}", { r: v.rank })}`).join(" · ");
    const v = x.verify;
    const how = x.url ? t("llama.cpp で使う: {cmd}", { cmd: `<code>llama-server -m &lt;model&gt; --lora ${esc(x.path)}</code>` })
      : t("llama.cpp で使う: {cmd}", { cmd: `<code>llama-server -m ${esc(x.path)}</code>` });
    this.lxSet(`<p class="note">${x.url ? t("{file} を書き出しました", { file: `<a href="${esc(x.url)}" download="${esc(x.file)}">${esc(x.file)}</a>` })
        : t("{file} を書き出しました", { file: `<span class="mono">${esc(x.path)}</span>` })}(${fmt(mb, mb < 10 ? 2 : 0)} MB)</p>
      ${x.url ? `<p class="muted note mono">${esc(x.path)}</p>` : ""}
      ${x.into ? `<p class="muted note">${t("焼き込む先: {f}", { f: esc(x.into) })}</p>` : ""}
      ${x.imatrix ? `<p class="muted note">${t(x.imatrix_auto ? "imatrix: {f}(モデルの隣で見つけたもの)" : "imatrix: {f}", { f: esc(x.imatrix) })}</p>` : ""}
      ${x.strip_meta ? `<p class="note">${(x.local_path_keys || []).length
        ? `⚠ ${t("gne の記録は入れていませんが、メタデータに手元のパスらしい文字列が残っています: {k}", { k: esc(x.local_path_keys.join(", ")) })}`
        : `✓ ${t("gne の記録は入れていません(手元のパスらしい文字列もメタデータに無し)")}`}</p>` : ""}
      ${layers ? `<p class="muted note mono">${esc(layers)}</p>` : ""}
      ${(x.modified || []).length ? `<p class="muted note">${t("書き換えたテンソル {n} 個", { n: x.modified.length })}: <span class="mono">${esc(x.modified.slice(0, 6).map((m) => `${m.tensor} ${m.from}→${m.to}`).join(", "))}${x.modified.length > 6 ? " …" : ""}</span></p>` : ""}
      ${v ? `<p class="note">${v.pass ? "✓" : "✗"} ${t("LoRA 版との比較: 最初のトークンの KL {kl}、最尤トークンの一致 {top}%、ppl の差 {ppl}%", { kl: Number(v.kl_first_token).toExponential(1), top: Math.round(100 * v.top1_match), ppl: fmt(v.ppl_diff_pct, 3) })}</p>` : ""}
      <p class="muted note">${how}</p>
      ${(x.warnings || []).length ? `<p class="muted note">${x.warnings.map(esc).join("<br>")}</p>` : ""}`);
  }
}

export { signed };
