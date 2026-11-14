// レシピ(右のパネル。タブは 中身・試す・書き出す)。ユニットの追加・強さのスライダー・保存と読み込み・cvec の書き出し。
// どの画面からも「レシピに足す」(1 つ足す)か「レシピを置き換える」(いまのレシピを入れ替える。空でなければ確かめる)のどちらか。
// ablate(方向の射影消去、本体設計 §6.17)の行: 方向・層の範囲・倍率。3 列チャットでは − の列で倍率を反転する。
// ヘッドの gain(CHANGE-04 D・G、L:Hh:xk): 行の倍率 k は 1 が無介入。全体に掛ける倍率 s では 1 + (k − 1)·s(eval の倍率列と同じ)。
// MoE(CHANGE-05 G): 専門家のニューロン L:Ee:Nn(add = 書き込み方向の押し、gain = 選ばれたときだけ効く)、専門家全体の gain L:Ee:xk、
// ルーターのバイアス L:Ee:route:delta(s 倍)、shared expert のニューロン L:S:Nn。
// ヘッドの方向成分の射影(CHANGE-06 G、L:Hh:sS@NAME): 行の強さ s(1 で方向成分を消す、2 で反転、0 で無介入)と方向。
// 全体に掛ける倍率でも s をそのまま倍にする(− の列は s が負 = その方向へ強める)。層ごとに、同じ方向なら LoRA は階数 1。
// 鍵つきの書き込み(CHANGE-07 G、L:ffn:keyed:aα@方向#鍵): 書き込み先(ffn・attn・S・Ee)・α(全体の倍率でそのまま倍)・方向・鍵。
// 方向・鍵を省いた行はレシピの既定(default_direction・default_key)。3 列チャット・eval には既定を埋めた形で送り、保存では
// 省いたまま既定を添える。cvec には入らない(LoRA・焼き込みは headproj_export のジョブ、焼き込みの既定は F16)。
// 編集中のレシピはこのブラウザの localStorage に下書きとして置く(使えなければ置かない)。
import { getJSON, postJSON } from "./api.js";
import { t } from "./i18n.js";
import { targetKind } from "./keys.js";
import { esc } from "./panels.js";
import { wireTabs, noteHTML } from "./ui.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const signed = (x, d = 2) => `${x >= 0 ? "+" : "−"}${fmt(Math.abs(x), d)}`;
const isDir = (u) => u.kind === "direction";
const isProj = (u) => u.kind === "head" && u.mode === "proj";   // ヘッドの方向成分の射影(CHANGE-06)
const isHead = (u) => u.kind === "head" && !isProj(u);          // ヘッドの gain
const isNeuron = (u) => (u.kind || "neuron") === "neuron" && u.mode !== "nw";
export const isKeyed = (u) => u.mode === "keyed";                // 鍵つきの書き込み(CHANGE-07)。kind は書き込み先(ffn・attn・shared・expert)
// 計算で作るニューロン(mode: nw、docs/fish.md)。ニューロンを計算した重みで上書きする。レシピの欄では表示・強さの変更・保存・
// LoRA の書き出しだけ(鍵づくりや探索の画面は無い。CLI・MCP で作る)
export const isNw = (u) => u.mode === "nw";
export function nwSpec(u, s = 1) {
  return `${u.layer}:N${u.neuron}${u.pair !== null && u.pair !== undefined ? `+N${u.pair}` : ""}:nw:a${+(u.alpha * s).toFixed(4)}`
    + `${u.direction ? `@${u.direction}` : ""}${u.key ? `#${u.key}${u.key2 ? `&${u.key2}` : ""}` : ""}`
    + `${u.s !== null && u.s !== undefined ? `!s${+u.s}` : ""}${u.cap ? `!cap${u.cap}` : ""}`;
}
export const isMoe = (u) => !isKeyed(u) && (u.kind === "expert_neuron" || u.kind === "expert" || u.kind === "shared");
const isGain = (u) => isHead(u) || u.mode === "gain";             // 倍率 k(1 が無介入)の行
const isRoute = (u) => u.mode === "route";
const isPush = (u) => isMoe(u) && (u.kind === "expert_neuron" || u.kind === "shared") && !isGain(u);   // MoE の add = 書き込み方向の押し
const moeKey = (u) => `${u.layer}:${u.kind}:${u.expert ?? ""}:${u.neuron ?? ""}:${u.mode || "add"}`;
// 方向のユニット(gne direction の方向、L:dir:NAME:dose)は dose が小さいので、スライダーを ±1・刻み 0.005 にする
// ルーターのバイアスに足す値(route)は ±4・刻み 0.1
const range = (u) => (isDir(u) ? { max: 1, step: 0.005, d: 3 } : isRoute(u) ? { max: 4, step: 0.1, d: 1 } : { max: 3, step: 0.05, d: 2 });
const GAIN = { min: -1, max: 3, step: 0.05 };          // ヘッドの倍率 k のスライダー(0 で消す、負で反転、1 で無介入)
const PROJ = { min: 0, max: 30, step: 0.1 };           // 射影の強さ s のスライダー(0 で無介入、1 で方向成分を消す)
const KEYED = { max: 3, step: 0.05 };                  // 鍵つきの書き込みの α のスライダー(±3。1 で観測した平均差をそのまま書く)
export const KEYED_KINDS = ["ffn", "attn", "shared", "expert"];
// 書き込み先の短い書き方(ユニットの文字列と同じ: ffn・attn・S・Ee)
const keyedWho = (u) => (u.kind === "shared" ? "S" : u.kind === "expert" ? `E${u.expert}` : u.kind);
const keyedTarget = (u) => (u.kind === "expert" ? `expert:${u.expert}` : u.kind);
// 並び: 層 → ニューロン・方向・専門家のニューロン → ヘッド → 専門家全体 → keyed、同じものは add → gain → route
const unitOrder = (u) => (isNw(u) ? [u.layer, 4, u.neuron ?? 0, 0]
  : isKeyed(u) ? [u.layer, 3, KEYED_KINDS.indexOf(u.kind) * 1e5 + (u.expert ?? 0), 0]
  : [u.layer, u.kind === "head" ? 1 : u.kind === "expert" ? 2 : 0,
     u.kind === "head" ? u.head : u.kind === "expert" ? u.expert : u.kind === "expert_neuron" ? 1e7 + u.expert * 1e5 + u.neuron
       : u.kind === "shared" ? 2e7 + u.neuron : u.neuron ?? -1, isProj(u) ? 3 : isRoute(u) ? 2 : isGain(u) ? 1 : 0]);

// スライダーは 0(中央)から値まで塗る。+ は黄、− は赤紫。中央に 0 の目盛り
function trackBG(v, max = 3) {
  const p = ((Math.max(-max, Math.min(max, v)) + max) / (2 * max)) * 100, c = v >= 0 ? "var(--plus)" : "var(--minus)", bg = "var(--line)";
  const [a, b] = v >= 0 ? [50, p] : [p, 50];
  return `linear-gradient(to right, transparent calc(50% - 1px), var(--text-4) calc(50% - 1px), var(--text-4) calc(50% + 1px), transparent calc(50% + 1px)),` +
         `linear-gradient(to right, ${bg} ${a}%, ${c} ${a}%, ${c} ${b}%, ${bg} ${b}%)`;
}

// ヘッドの倍率のスライダー: 無介入の 1 から値まで塗る。強める(k > 1)は黄、弱める・消す(k < 1)は赤紫。1 に目盛り
function gainBG(k) {
  const at = (x) => ((Math.max(GAIN.min, Math.min(GAIN.max, x)) - GAIN.min) / (GAIN.max - GAIN.min)) * 100;
  const one = at(1), p = at(k), c = k >= 1 ? "var(--plus)" : "var(--minus)", bg = "var(--line)";
  const [a, b] = k >= 1 ? [one, p] : [p, one];
  return `linear-gradient(to right, transparent calc(${one}% - 1px), var(--text-4) calc(${one}% - 1px), var(--text-4) calc(${one}% + 1px), transparent calc(${one}% + 1px)),` +
         `linear-gradient(to right, ${bg} ${a}%, ${c} ${a}%, ${c} ${b}%, ${bg} ${b}%)`;
}
const kLabel = (k) => `×${fmt(k, 2)}`;
// 射影の強さのスライダー: 0(無介入)から値まで赤紫(方向成分を引く)。1(消す)と 2(反転)に目盛り
function projBG(v) {
  const at = (x) => ((Math.max(PROJ.min, Math.min(PROJ.max, x)) - PROJ.min) / (PROJ.max - PROJ.min)) * 100;
  const p = at(v), tick = (x) => `transparent calc(${at(x)}% - 1px), var(--text-4) calc(${at(x)}% - 1px), var(--text-4) calc(${at(x)}% + 1px), transparent calc(${at(x)}% + 1px)`;
  return `linear-gradient(to right, ${tick(1)}), linear-gradient(to right, ${tick(2)}),` +
         `linear-gradient(to right, var(--minus) 0%, var(--minus) ${p}%, var(--line) ${p}%, var(--line) 100%)`;
}
const sLabel = (s) => `s${+(+s).toFixed(2)}`;
const aLabel = (a) => `a${a >= 0 ? "+" : "−"}${fmt(Math.abs(a), 2)}`;
// 射影のユニットの文字列(L:Hh:sS@NAME)→ { layer, head, s, direction }。読めなければ null
const PROJ_RE = /^(\d+):[Hh](\d+):s([+-]?[\d.]+(?:e[+-]?\d+)?)(?:@([\w.\-]+))?$/i;
export function parseProj(spec, defaultDirection = null) {
  const m = String(spec).trim().match(PROJ_RE);
  return m ? { layer: +m[1], kind: "head", mode: "proj", head: +m[2], s: +m[3], direction: m[4] || defaultDirection } : null;
}
// 全体に掛ける倍率 s でのヘッドの倍率(1 が無介入。s = −1 なら 1 の反対側: ×0 → ×2)
const gainAt = (k, s) => +(1 + (k - 1) * s).toFixed(4);

// 鍵つきの書き込みの指定(L:ffn:keyed:aα@方向#鍵)。方向・鍵が無ければ defaults({ direction, key })、それも無ければ省く
// (サーバでレシピの default_direction・default_key を使う)
export function keyedSpec(u, s = 1, defaults = {}) {
  const dir = u.direction || defaults.direction || "", key = u.key || defaults.key || "";
  return `${u.layer}:${keyedWho(u)}:keyed:a${+(u.alpha * s).toFixed(4)}${dir ? `@${dir}` : ""}${key ? `#${key}` : ""}`;
}

// ユニット 1 つのサーバ向けの指定(gne/steer.py の parse_unit の形)。s はレシピ全体に掛ける倍率
export function unitSpec(u, s = 1) {
  if (isNw(u)) return nwSpec(u, s);
  if (isKeyed(u)) return keyedSpec(u, s);
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
    this.directionLayer = "own";       // proj・keyed のユニットに当てる方向の層(CHANGE-06): own = その層の r_L、base = 基準層の r
    this.defaultDirection = null;      // 方向を省いた keyed のユニットの方向(レシピの default_direction、CHANGE-07)
    this.defaultKey = null;            // 鍵を省いた keyed のユニットの鍵(レシピの default_key)
    this.sampling = null;              // このレシピを試すときのサンプリング { temperature, top_k, top_p, min_p, presence_penalty } か null
    this.skipped = 0;                  // 読み込んだレシピのうち UI では扱えないユニット(ニューロンの gain、ヘッドの query など)の数。
                                       // ヘッドの query(CHANGE-04 E)は L:Hh の文字列の形が無く、送り返せないので外す
    try {
      const d = JSON.parse(localStorage.getItem(this.key) || "null");
      if (d && Array.isArray(d.units)) {
        this.name = d.name || ""; this.units = d.units; this.ablate = d.ablate || null; this.sampling = d.sampling || null;
        this.directionLayer = d.direction_layer === "base" ? "base" : "own";
        this.defaultDirection = d.default_direction || null; this.defaultKey = d.default_key || null;
      }
    } catch { /* 下書きが読めなくても空から始める */ }
  }

  // notify = false なら下書きに置くだけで知らせない(名前のように地図の印などに関わらない値)
  _save(notify = true) {
    try {
      localStorage.setItem(this.key, JSON.stringify({ name: this.name, units: this.units, ablate: this.ablate, sampling: this.sampling,
                                                      direction_layer: this.directionLayer, default_direction: this.defaultDirection,
                                                      default_key: this.defaultKey }));
    } catch { /* 置けなくても使える */ }
    if (notify) this.onChange(this);
  }
  // 名前の欄は打つたびに入れる(欄から出る前に描き直しても、入れかけの名前が消えないように)
  setName(name) { this.name = name; this._save(false); }

  empty() { return !this.units.length && !this.ablate; }
  // ablate を倍率 s で(− の列は s < 0 = その方向を強める)。無ければ undefined
  ablateAt(s = 1) { return this.ablate ? { ...this.ablate, scale: +(this.ablate.scale * s).toFixed(4) } : undefined; }
  setAblate(a) { this.ablate = a && a.direction ? a : null; this._save(); }
  setSampling(sp) { this.sampling = sp ? { ...sp } : null; this._save(); }

  // サーバに送るユニット指定。scale はレシピ全体に掛ける倍率(3 列チャットの ± 列、eval の倍率と同じ意味)。
  // fill なら keyed の省いた方向・鍵をレシピの既定で埋める(3 列チャット・eval はユニットの文字列だけを送るので)。保存は fill なし
  specs(scale = 1, fill = true) {
    const d = fill ? { direction: this.defaultDirection, key: this.defaultKey } : {};
    return this.units.map((u) => (isKeyed(u) ? keyedSpec(u, scale, d) : unitSpec(u, scale)));
  }
  // control vector に入るもの(add)だけ。gain は LoRA、専門家全体の gain と route はスケールパッチ、keyed は入力で効き分ける
  // LoRA なので cvec には入らない(MoE の add は書き込み方向の押しとして入る)
  addSpecs() { return this.units.filter((u) => !isGain(u) && !isRoute(u) && !isProj(u) && !isKeyed(u) && !isNw(u)).map((u) => unitSpec(u)); }
  // 計算で作るニューロン(mode: nw)。LoRA だけで、cvec にも「LoRA・焼き込み」にも入らない(専用の書き出し)
  nws() { return this.units.filter(isNw); }
  // FiSH のタブから: nw の行を足す。同じ層の同じニューロン(組の相手も)を使う行は置き換える(1 個のニューロンに書けるのは 1 つ)
  addNw(rows) {
    const used = (u) => [u.neuron, u.pair].filter((x) => x !== null && x !== undefined).map(Number);
    for (const q of rows) {
      const mine = new Set(used(q));
      this.units = this.units.filter((u) => !(isNw(u) && u.layer === +q.layer && used(u).some((j) => mine.has(j))));
      this.units.push({ layer: +q.layer, neuron: +q.neuron, mode: "nw", alpha: +q.alpha, direction: q.direction || null, key: q.key || null,
                        key2: q.key2 || null, pair: q.pair ?? null, s: q.s ?? null, cap: q.cap || null });
    }
    this._sort();
    this._save();
  }
  // 鍵つきの書き込み(CHANGE-07)。同じ層・書き込み先・方向・鍵なら 1 行(α を置き換える)
  keyed() { return this.units.filter(isKeyed); }
  findKeyed(q) {
    return this.units.find((u) => isKeyed(u) && u.layer === +q.layer && u.kind === q.kind && (u.expert ?? null) === (q.expert ?? null)
      && (u.direction || null) === (q.direction || null) && (u.key || null) === (q.key || null));
  }
  addKeyed(q) {
    const u = this.findKeyed(q);
    if (u) u.alpha = +q.alpha;
    else this.units.push({ layer: +q.layer, kind: q.kind, ...(q.kind === "expert" ? { expert: +q.expert } : {}), mode: "keyed",
                           alpha: +q.alpha, direction: q.direction || null, key: q.key || null });
    this._sort();
    this._save();
  }
  // 行 i の書き込み先・方向・鍵を変える(patch: { kind, expert, direction, key } の一部)
  setKeyed(i, patch) {
    const u = this.units[i];
    if (!isKeyed(u)) return;
    Object.assign(u, patch);
    if (u.kind !== "expert") delete u.expert;
    else if (!Number.isInteger(u.expert)) u.expert = 0;
    this._sort();
    this._save();
  }
  setDefaults({ direction, key } = {}) {
    if (direction !== undefined) this.defaultDirection = direction || null;
    if (key !== undefined) this.defaultKey = key || null;
    this._save();
  }
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
  set(i, v) { const u = this.units[i]; if (isKeyed(u) || isNw(u)) u.alpha = v; else if (isProj(u)) u.s = v; else if (isGain(u)) u.k = v; else u.dose = v; this._save(); }
  remove(i) { this.units.splice(i, 1); this._save(); }
  clear() {
    this.units = []; this.name = ""; this.ablate = null; this.sampling = null; this.skipped = 0; this.directionLayer = "own";
    this.defaultDirection = null; this.defaultKey = null; this._save();
  }
  // directionLayer: 保存したレシピの direction_layer(proj・keyed のユニットに当てる方向の層、CHANGE-06)。
  // defaults: { default_direction, default_key }(keyed の省いた方向・鍵、CHANGE-07)。/api/recipes が既定で埋めた行
  // (direction_default・key_default)は、既定を受け取ったときは省いた形に戻す(保存し直しても既定のまま)
  load(name, units, ablate = null, sampling = null, directionLayer = "own", defaults = {}) {
    const dd = (defaults && defaults.default_direction) || null, dk = (defaults && defaults.default_key) || null;
    const nwOk = (u) => isNw(u) && u.neuron !== null && u.neuron !== undefined && Number.isFinite(+(u.alpha ?? u.k)) && (u.alpha ?? u.k) !== null;
    const keyedOk = (u) => isKeyed(u) && KEYED_KINDS.includes(u.kind) && Number.isFinite(+(u.alpha ?? u.k)) && (u.alpha ?? u.k) !== null
      && (u.kind !== "expert" || (u.expert !== null && u.expert !== undefined));
    const moeOk = (u) => isMoe(u) && (u.kind === "shared" || (u.expert !== null && u.expert !== undefined))
      && ({ expert_neuron: ["add", "gain"], shared: ["add", "gain"], expert: ["gain", "route"] }[u.kind].includes(u.mode || "add"))
      && (u.mode === "gain" ? Number.isFinite(+u.k) && u.k !== null : true);
    const ok = units.filter((u) => (((u.kind || "neuron") === "neuron" || u.kind === "direction") && (u.mode || "add") === "add")
      || (u.kind === "head" && u.mode === "gain" && u.head !== null && u.head !== undefined && Number.isFinite(+u.k))
      || (u.kind === "head" && u.mode === "proj" && u.head !== null && u.head !== undefined && Number.isFinite(+(u.s ?? u.k)) && !!u.direction)
      || moeOk(u) || keyedOk(u) || nwOk(u));
    this.skipped = units.length - ok.length;
    this.name = name;
    this.units = ok.map((u) => (isNw(u) ? { layer: u.layer, neuron: u.neuron, mode: "nw", alpha: +(u.alpha ?? u.k), direction: u.direction || null,
                                            key: u.key || null, key2: u.key2 || null, pair: u.pair ?? null, s: u.s ?? null, cap: u.cap || null }
      : isKeyed(u) ? { layer: u.layer, kind: u.kind, ...(u.kind === "expert" ? { expert: +u.expert } : {}), mode: "keyed",
                                               alpha: +(u.alpha ?? u.k), direction: (u.direction_default && dd) ? null : u.direction || null,
                                               key: (u.key_default && dk) ? null : u.key || null }
      : isDir(u) ? { layer: u.layer, kind: "direction", direction: u.direction, dose: u.dose }
      : isProj(u) ? { layer: u.layer, kind: "head", mode: "proj", head: u.head, s: +(u.s ?? u.k), direction: u.direction }
      : isHead(u) ? { layer: u.layer, kind: "head", head: u.head, k: +u.k }
      : isMoe(u) ? { layer: u.layer, kind: u.kind, ...(u.kind !== "shared" ? { expert: +u.expert } : {}), ...(u.kind !== "expert" ? { neuron: u.neuron } : {}),
                     mode: u.mode || "add", ...(u.mode === "gain" ? { k: +u.k } : { dose: +(u.dose ?? u.delta) }) }
      : { layer: u.layer, neuron: u.neuron, dose: u.dose }));
    this.ablate = ablate || null;
    this.sampling = sampling || null;
    this.directionLayer = directionLayer === "base" ? "base" : "own";
    this.defaultDirection = dd;
    this.defaultKey = dk;
    this._save();
  }
}

// レシピを置き換える前に確かめる(どの入口でも同じ)。いまのレシピが空なら確かめない。置き換えてよければ true
export function askReplace(recipe, name) {
  if (recipe.empty()) return true;
  return window.confirm(t("いまのレシピ({n} ユニット)を {name} で置き換えますか?", { n: recipe.units.length + (recipe.ablate ? 1 : 0), name: name || "—" }));
}

// サンプリングの値の短い表示(温度 1.0・top-k 20・…)
export const samplingLabel = (sp) => (sp ? `${t("温度")} ${+sp.temperature} · top-k ${sp.top_k} · top-p ${+sp.top_p} · min-p ${+sp.min_p} · presence ${+sp.presence_penalty}` : "");

export class RecipePanel {
  constructor(root, recipe, { onOpen, onOpenHead, onOpenExpert, onChat, unitWords, directions, projDirections, headKind, nFF, onRunJob, keys, nLayer,
                               onAdded, onReplaced }) {
    this.root = root;
    this.r = recipe;
    this.tab = "units";                           // いまのタブ: units 中身 / try 試す / out 書き出す
    this.onAdded = onAdded || (() => this.render());          // このパネルの中で足したとき(app.js がパネルを開いて描き直す)
    this.onReplaced = onReplaced || (() => this.render());    // 保存したレシピで置き換えたとき
    this.directions = directions || (() => []);   // 方向の名前(gne direction / reverse)
    this.projDirections = projDirections || this.directions;   // proj・keyed に使える方向(gne direction の方向セット、CHANGE-06)
    this.keys = keys || (() => []);               // 保存した鍵の要約(/api/keys: name・target・layers、CHANGE-07)
    this.nLayer = nLayer || 0;
    this.kaOpen = false;                          // 「鍵つきの書き込みを足す」の欄を開いているか
    this.kexpDraft = null;                        // keyed の行の専門家の番号の、入れかけの値 { u: 行のユニット, v }(欄から出るまで)
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
    this.lx = { bake: false, requant: null, via_f16: false, llama_quantize: "", imatrix: "", verify: true, contrast: "", strip: false,
                file: "", into: "", adapterName: "" };              // file: ファイル名(拡張子なし、空ならレシピの名前から)、into: 焼き込む先の別の GGUF
                                                   // requant: null は既定(keyed を含むなら F16、それ以外は Q8_0)
    this.lxJobId = null;
    this.lxMsg = null;                 // { html } 書き出しの結果か進み具合
  }

  // 先に今の中身を出し、保存済みの一覧は読めてから「保存したレシピを読む…」の選択欄にだけ足す
  // (全体を描き直すと、一覧を待つ間に入れた欄の値・開いた欄が消える)
  async refreshSaved() {
    this.render();
    try { this.saved = (await getJSON("/api/recipes")).filter((x) => x.this_model); } catch { this.saved = []; }
    const el = this.root.querySelector("#rc-load");
    if (el) el.innerHTML = this.savedOptions();
  }

  setTab(tab) {
    this.tab = ["units", "try", "out"].includes(tab) ? tab : "units";
    for (const b of this.root.querySelectorAll("#recipe-tabs [data-rtab]")) {
      const on = b.dataset.rtab === this.tab;
      b.setAttribute("aria-selected", String(on));
      b.classList.toggle("on", on);
      b.tabIndex = on ? 0 : -1;
    }
    for (const el of this.root.querySelectorAll("[data-rpanel]")) el.hidden = el.dataset.rpanel !== this.tab;
  }

  savedOptions() {
    return `<option value="">${t("保存したレシピで置き換える…")}</option>${
      this.saved.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}(${x.units.length})</option>`).join("")}`;
  }

  render() {
    this.renders = (this.renders || 0) + 1;     // 描いた回数(app.js が、同じ変更で 2 度描かないように見る)
    const r = this.r;
    // 入力中の欄(id のある文字・数の欄)は、描き直した後も入力を続けられるようにフォーカスとカーソルの位置を戻す
    const fe = document.activeElement;
    const tabFocus = fe && this.root.contains(fe) && fe.matches("#recipe-tabs [role=tab]") ? fe.dataset.rtab : null;
    const focus = fe && fe.id && this.root.contains(fe) && (fe.tagName === "TEXTAREA" || (fe.tagName === "INPUT" && !["range", "checkbox", "radio"].includes(fe.type)))
      ? { id: fe.id, a: fe.selectionStart, b: fe.selectionEnd } : null;
    const rows = r.units.map((u, i) => {
      if (isNw(u)) return this.nwRow(u, i);
      if (isKeyed(u)) return this.keyedRow(u, i);
      if (isProj(u)) return this.projRow(u, i);
      if (isGain(u)) return this.gainRow(u, i);
      const g = range(u);
      const id = isDir(u)
        ? `<span class="mono" title="${t("方向(gne direction で作ったもの)。その層に足す")}">L${u.layer} dir:${esc(u.direction)}</span> <span class="muted w">${t("方向")}</span>`
        : isMoe(u) ? this.moeId(u, i)
        : `<a href="#" data-open="${i}">L${u.layer} N${u.neuron}</a> <span class="muted w">${esc(this.unitWords(u.layer, u.neuron))}</span>`;
      return `
      <tr data-i="${i}"${isMoe(u) ? ` class="rc-moe"` : ""}><td class="id">${id}<button class="small quiet rm" data-rm="${i}" title="${t("外す")}">×</button>
        <div class="sl"><span class="muted">−${g.max}</span><input type="range" class="dose" min="-${g.max}" max="${g.max}" step="${g.step}" value="${u.dose}" data-i="${i}" style="background:${trackBG(u.dose, g.max)}"><span class="muted">+${g.max}</span>
          <span class="sg mono ${u.dose >= 0 ? "plus" : "minus"}">${signed(u.dose, g.d)}</span>
          <input type="number" min="-10" max="10" step="${g.step}" value="${u.dose}" data-i="${i}" class="num" title="${t("値を直接入れる")}"></div></td></tr>`;
    }).join("");
    const tabs = [["units", t("中身")], ["try", t("試す")], ["out", t("書き出す")]];
    const panel = (k) => `role="tabpanel" data-rpanel="${k}" aria-labelledby="rtab-${k}"${this.tab === k ? "" : " hidden"}`;
    this.root.innerHTML = `
      <div class="tabs" id="recipe-tabs" role="tablist" aria-label="${t("レシピ")}">${tabs.map(([k, l]) =>
        `<button type="button" role="tab" id="rtab-${k}" data-rtab="${k}" aria-selected="${this.tab === k}" tabindex="${this.tab === k ? 0 : -1}"${this.tab === k ? ' class="on"' : ""}>${l}</button>`).join("")}</div>
      <div ${panel("units")}>
      <div class="list-tools"><input id="rc-name" placeholder="${t("レシピ名(英数字・_・-)")}" value="${esc(r.name)}" spellcheck="false">
        <button class="small" id="rc-save" title="${t("保存先の recipes フォルダに保存(既定は ~/.gne/recipes、--root を指定していればその下)")}">${t("保存")}</button></div>
      <div class="list-tools"><select id="rc-load">${this.savedOptions()}</select></div>
      ${r.units.length ? `<table class="rc mono"><tbody>${rows}</tbody></table>`
        : `<p class="muted note">${t("ユニットがありません。どの画面の「レシピに足す」でも、ここに入ります。")}</p>`}
      ${r.skipped ? `<p class="muted note">${t("UI では扱えないユニット(ニューロンの gain・ヘッドの query・ランダム方向など){n} 個を外しました", { n: r.skipped })}</p>` : ""}
      ${r.heads().length ? `${noteHTML(t("ヘッドの倍率 k は 1 が無介入(0 で消す、2 で倍、負で反転)。LoRA で当てるので cvec には入りません。3 列チャットの − 列は 1 の反対側(×0 なら ×2)。"), "muted note")}` : ""}
      ${r.projs().length || r.keyed().length ? this.projNote() : ""}
      ${r.keyed().length ? this.keyedNote() : ""}
      ${this.keyedAddHTML()}
      ${r.moe().length ? `${noteHTML(t("MoE: 方向押し = 専門家のニューロンの add。専門家が選ばれたかに関係なく全トークンにその書き込み方向を足す(ニューロン押しではない)。選ばれたときだけ効くのは gain(×)。専門家全体の ×k と route はモデルを読み直すパッチで当てる(列ごとに 0.3〜1 秒)。3 列チャットの − 列は ×k が 2 − k、route と add は符号を反転。"), "muted note")}` : ""}
      ${this.ablateHTML()}
      ${r.sampling ? `<div class="rc-sampling muted note" title="${t("このレシピを試すときのサンプリング。チャットの「レシピの設定」で使い、保存するとレシピのファイルに入る(OpenAI 互換 API の既定にもなる)")}">${t("サンプリング")}: <span class="mono">${esc(samplingLabel(r.sampling))}</span><button class="small" id="rc-sp-rm" title="${t("レシピから外す")}">×</button></div>` : ""}
      <p class="muted note">${t("ニューロンの強さは rel(その層の残差ノルムの中央値に対する大きさ)。マップに + は黄、− は赤紫の輪で出ます。")}</p>
      ${r.units.length ? `<div class="rc-actions"><button class="small quiet" id="rc-clear" title="${t("すべてのユニットを外す")}">${t("レシピを空にする")}</button></div>` : ""}
      </div>
      <div ${panel("try")}>
      ${noteHTML(t("いまのレシピを当てた返事を、何もしない返事・逆向きに当てた返事と並べて比べます。チャットは「くわしく見る」のニューロンの Chat に開きます。"), "muted note")}
      <div class="rc-actions"><button class="main" id="rc-chat"${r.empty() ? " disabled" : ""}>${t("3 列チャットで試す")}</button></div>
      </div>
      <div ${panel("out")}>
      ${noteHTML(t("書き出しはここにまとめています。足す押し方(ニューロン・方向)は cvec、ヘッド・鍵つきの書き込みは「LoRA・焼き込み」、計算で作るニューロン(FiSH)は nw の LoRA です。"), "muted note")}
      <div class="rc-actions">
        <button id="rc-export"${r.addSpecs().length && document.body.dataset.cvec !== "off" ? "" : " disabled"}>${t("cvec を書き出す")}</button>
        <label class="muted"><input type="checkbox" id="rc-verify"> ${t("llama.cpp で確かめる")}</label>
        <button id="rc-lora"${r.heads().length || r.projs().length || r.keyed().length ? "" : " disabled"} title="${t("ヘッドの方向成分(proj)と倍率(gain)、鍵つきの書き込み(keyed)を、llama.cpp の --lora で使う LoRA アダプタか、重みに焼き込んだ GGUF に書き出す")}">${t("LoRA・焼き込み")}${this.lxOpen ? " ▴" : " ▾"}</button>
      </div>
      ${r.keyed().length ? `${noteHTML(t("鍵つきの書き込み(keyed)は control vector に入りません(入力の活性で効き分けるので足し算にならない)。「LoRA・焼き込み」で書き出します。"), "muted note")}` : ""}
      ${r.nws().length ? `<div class="rc-actions"><button id="rc-nw" title="${t("計算で作るニューロン(nw)を、llama.cpp の --lora で使う LoRA アダプタに書き出す(焼き込みは無い)")}">${t("nw の LoRA を書き出す")}</button></div>
        ${noteHTML(t("計算で作るニューロン(nw)は control vector にも「LoRA・焼き込み」にも入りません。「nw の LoRA を書き出す」で別のアダプタにします。"), "muted note")}` : ""}
      ${this.lxOpen ? this.lxHTML() : ""}
      </div>
      <div id="rc-result"></div>`;
    // 中身のタブを押したら、保存したレシピの一覧も取り直して描き直す(ほかの画面で足した・保存したものを出す)
    wireTabs(this.root.querySelector("#recipe-tabs"), (k) => { this.setTab(k); if (k === "units") this.refreshSaved(); });
    const $ = (s) => this.root.querySelector(s);
    $("#rc-name").addEventListener("input", (ev) => r.setName(ev.target.value));
    $("#rc-name").addEventListener("change", (ev) => r.setName(ev.target.value.trim()));
    $("#rc-save").addEventListener("click", () => this.save());
    const clr = $("#rc-clear");
    if (clr) clr.addEventListener("click", () => { if (confirm(t("レシピの {n} ユニットをすべて外しますか", { n: r.units.length }))) { r.clear(); this.render(); } });
    $("#rc-load").addEventListener("change", (ev) => {
      const x = this.saved.find((y) => y.name === ev.target.value);
      if (!x) return;
      if (!askReplace(r, x.name)) { ev.target.value = ""; return; }      // いまのレシピが空でなければ確かめる(やめたら選択を戻す)
      r.load(x.name, x.units, x.ablate, x.sampling, x.direction_layer, { default_direction: x.default_direction, default_key: x.default_key });
      this.onReplaced();
    });
    for (const el of this.root.querySelectorAll("input[data-i]")) {
      el.addEventListener("input", (ev) => {
        const i = +ev.target.dataset.i, v = +ev.target.value;
        if (!Number.isFinite(v)) return;
        r.set(i, v);
        const tr = this.root.querySelector(`tr[data-i="${i}"]`);
        for (const x of tr.querySelectorAll("input")) if (x !== ev.target) x.value = String(v);
        const sg = tr.querySelector(".sg");
        if (isKeyed(r.units[i]) || isNw(r.units[i])) {
          tr.querySelector("input.dose").style.background = trackBG(v, KEYED.max);
          sg.textContent = aLabel(v); sg.className = `sg mono ${v >= 0 ? "plus" : "minus"}`;
          return;
        }
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
    this.wireKeyed();
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
    if ($("#rc-nw")) $("#rc-nw").addEventListener("click", () => this.exportNw());
    $("#rc-lora").addEventListener("click", () => { this.lxOpen = !this.lxOpen; this.render(); });
    if (this.lxOpen) this.wireLx();
    const el = focus && this.root.querySelector(`#${CSS.escape(focus.id)}`);
    if (el) {
      el.focus();
      try { el.setSelectionRange(focus.a, focus.b); } catch { /* 数の欄はカーソルの位置を持たない */ }
    }
    if (tabFocus) this.root.querySelector(`#recipe-tabs [data-rtab="${tabFocus}"]`)?.focus();   // タブのフォーカスも戻す(キーで操作中)
  }

  // MoE の単位の名前(CHANGE-05 G)。add には「方向押し」の印、専門家全体は EXPERT パネルへ
  moeId(u, i) {
    const push = isPush(u) ? ` <span class="badge push" title="${t("書き込み方向の押し: 専門家が選ばれたかに関係なく、全トークンでこのニューロンの書き込み方向を足す(ニューロン押しではない)")}">${t("方向押し")}</span>` : "";
    if (u.kind === "expert_neuron") return `<a href="#" data-open="${i}" title="${t("くわしく見るで開く")}">L${u.layer} E${u.expert} N${u.neuron}</a>${push}${
      isGain(u) ? ` <span class="muted w">gain · ${t("選ばれたときだけ")}</span>` : ""}`;
    if (u.kind === "shared") return `<span class="mono">L${u.layer} S N${u.neuron}</span>${push} <span class="muted w">shared expert${isGain(u) ? " · gain" : ""}</span>`;
    return `<a href="#" data-open="${i}" title="${t("くわしく見るで開く")}">L${u.layer} E${u.expert}</a> <span class="muted w">${
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
      <tr data-i="${i}" class="rc-proj"><td class="id"><a href="#" data-open="${i}" title="${t("くわしく見るで開く")}">L${u.layer} H${u.head}</a>
        <span class="sg mono minus">${sLabel(u.s)}</span> <span class="mono">@</span><select class="rc-pdir" data-pdir="${i}" title="${t("射影する方向(gne direction の方向セット)")}">${opts}</select>
        ${rk}<button class="small quiet rm" data-rm="${i}" title="${t("外す")}">×</button>
        <div class="sl"><span class="muted">${PROJ.min}</span><input type="range" class="dose" min="${PROJ.min}" max="${PROJ.max}" step="${PROJ.step}" value="${u.s}" data-i="${i}" style="background:${projBG(u.s)}"><span class="muted">${PROJ.max}</span>
          <input type="number" min="-300" max="300" step="${PROJ.step}" value="${+(+u.s).toFixed(3)}" data-i="${i}" class="num" title="${t("値を直接入れる")}"></div></td></tr>`;
  }

  // 方向の層(own / base)の選択と、射影の説明(proj があるとき)
  projNote() {
    const r = this.r;
    const tip = r.keyed().length ? t("proj・keyed のユニットに当てる方向の層。own = そのユニットがある層の方向 r_L、base = 方向セットの基準層の r(gne direction score で記録した層)")
      : t("proj のユニットに当てる方向の層。own = そのヘッドがある層の方向 r_L、base = 方向セットの基準層の r(gne direction score で記録した層)");
    return `<div class="rc-ablate"><span class="muted" title="${tip}">${t("方向の層")}</span>
        <select id="rc-dl"><option value="own"${r.directionLayer === "own" ? " selected" : ""}>own</option><option value="base"${r.directionLayer === "base" ? " selected" : ""}>base</option></select></div>
      ${r.projs().length ? `${noteHTML(t("射影 s: ヘッドが残差に書くベクトルのうち方向 r̂ に沿った成分だけを (1 − s) 倍にする(1 で消す、2 で反転、直交成分はそのまま)。LoRA で当てるので cvec には入りません。3 列チャットの − 列は s が負(その方向へ強める)。"), "muted note")}` : ""}${
        r.directionLayer === "base" ? `${noteHTML(t("方向の層 base は保存したレシピのファイルにだけ入ります(3 列チャットと「今のレシピで eval」はユニットの文字列を送るので own で当てます)。"), "muted note")}` : ""}`;
  }

  // 鍵つきの書き込みの行(CHANGE-07、L14 ffn keyed a+1.00 @方向 #鍵)。書き込み先・方向・鍵の選択(方向・鍵の「既定」は
  // レシピの default_direction・default_key)、α のスライダー(±3、1 で観測した平均差をそのまま書く、負で打ち消す)。
  // 鍵に無い層・書き込み先の食い違い・方向や鍵が無いことは行に印を出す(サーバでも BAD_RECIPE で断る)
  // 計算で作るニューロンの行: 中身(ニューロン・方向・鍵)は表示だけで、変えられるのは強さ α と「外す」
  nwRow(u, i) {
    const a = +u.alpha;
    const what = `L${u.layer} N${u.neuron}${u.pair !== null && u.pair !== undefined ? `+N${u.pair}` : ""}`;
    const form = u.pair !== null && u.pair !== undefined ? t("上限つき(2 個を組)") : u.key2 ? t("鍵 2 つ") : t("鍵 1 つ");
    return `
      <tr data-i="${i}" class="rc-nw"><td class="id"><span class="mono">${esc(what)}</span>
        <span class="badge keyed" title="${t("計算で作るニューロン: ニューロンを計算した重みで上書きし、鍵が反応する位置でだけ方向を書く(LoRA)")}">nw</span>
        <button class="small quiet rm" data-rm="${i}" title="${t("外す")}">×</button>
        <div class="rc-kx muted mono" title="${t("方向・鍵・形は direction ビューの FiSH のタブ(か CLI・MCP の gne nw …)で決める(docs/fish.md)")}">@${esc(u.direction || "—")} #${esc(u.key || "—")}${u.key2 ? `&amp;${esc(u.key2)}` : ""} · ${esc(form)}</div></td>
        <td class="sl"><input type="range" class="dose" data-i="${i}" min="${-KEYED.max}" max="${KEYED.max}" step="${KEYED.step}" value="${a}" style="background:${trackBG(a, KEYED.max)}"></td>
        <td class="nm"><input type="number" class="num" data-i="${i}" step="${KEYED.step}" value="${a}"></td>
        <td class="sg mono ${a >= 0 ? "plus" : "minus"}">${aLabel(a)}</td></tr>`;
  }

  keyedRow(u, i) {
    const r = this.r, dirs = this.projDirections(), keys = this.keys().filter((k) => !k.error);
    const kname = u.key || r.defaultKey, dname = u.direction || r.defaultDirection;
    const kinfo = kname ? keys.find((k) => k.name === String(kname).replace(/~\d+$/, "")) : null;
    const warn = [];
    if (!dname) warn.push(t("方向がありません"));
    if (!kname) warn.push(t("鍵がありません"));
    else if (!kinfo) warn.push(t("鍵 {k} が見つかりません", { k: kname }));
    else {
      if (!kinfo.layers.includes(u.layer)) warn.push(t("鍵 {k} に L{L} がありません(ある層: {ls})", { k: kname, L: u.layer, ls: kinfo.layers.join(",") }));
      if (kinfo.target !== keyedTarget(u)) warn.push(t("鍵の書き込み先は {t} です", { t: kinfo.target }));
    }
    const opts = (xs, cur, def) => [`<option value=""${cur ? "" : " selected"}>${esc(def ? t("既定: {x}", { x: def }) : t("既定(なし)"))}</option>`,
      ...[...xs, ...(cur && !xs.includes(cur) ? [cur] : [])].map((x) => `<option${x === cur ? " selected" : ""}>${esc(x)}</option>`)].join("");
    const kinds = KEYED_KINDS.map((k) => `<option value="${k}"${k === u.kind ? " selected" : ""}>${k === "shared" ? "S" : k === "expert" ? "E" : k}</option>`).join("");
    return `
      <tr data-i="${i}" class="rc-keyed"><td class="id"><span class="mono">L${u.layer}</span>
        <select class="rc-kkind" data-kkind="${i}" title="${t("書き込み先: ffn = ffn_down、attn = attn_output、S = shared expert、E = 専門家 e(鍵を作った書き込み先と同じにする)")}">${kinds}</select>${
          u.kind === "expert" ? `<input type="number" class="num rc-kexp" data-kexp="${i}" min="0" value="${esc(String(this.kexpDraft && this.kexpDraft.u === u ? this.kexpDraft.v : u.expert ?? 0))}" title="${t("専門家の番号")}">` : ""}
        <span class="badge keyed" title="${t("鍵つきの書き込み: 入力の活性に鍵が反応するときだけ方向を書く(LoRA)")}">keyed</span>
        <button class="small quiet rm" data-rm="${i}" title="${t("外す")}">×</button>
        <div class="rc-kx"><span class="mono">@</span><select data-kdir="${i}" title="${t("書き込む方向(gne direction の方向セットの r_L)")}">${opts(dirs, u.direction, r.defaultDirection)}</select>
          <span class="mono">#</span><select data-kkey="${i}" title="${t("鍵(direction ビューの「鍵」か gne direction key で作ったもの)")}">${opts(keys.map((k) => k.name), u.key, r.defaultKey)}</select>
          ${warn.map((w) => `<span class="badge warn">${esc(w)}</span>`).join(" ")}</div>
        <div class="sl"><span class="muted">−${KEYED.max}</span><input type="range" class="dose" min="-${KEYED.max}" max="${KEYED.max}" step="${KEYED.step}" value="${u.alpha}" data-i="${i}" style="background:${trackBG(u.alpha, KEYED.max)}"><span class="muted">+${KEYED.max}</span>
          <span class="sg mono ${u.alpha >= 0 ? "plus" : "minus"}">${aLabel(u.alpha)}</span>
          <input type="number" min="-100" max="100" step="${KEYED.step}" value="${+(+u.alpha).toFixed(3)}" data-i="${i}" class="num" title="${t("値を直接入れる")}"></div></td></tr>`;
  }

  // keyed の既定(方向・鍵)と説明
  keyedNote() {
    const r = this.r, dirs = this.projDirections(), keys = this.keys().filter((k) => !k.error).map((k) => k.name);
    const opt = (xs, cur) => [`<option value="">${t("なし")}</option>`,
      ...[...xs, ...(cur && !xs.includes(cur) ? [cur] : [])].map((x) => `<option${x === cur ? " selected" : ""}>${esc(x)}</option>`)].join("");
    return `<div class="rc-ablate rc-kdef" title="${t("@方向・#鍵を「既定」にした keyed の行が使う方向と鍵(保存するとレシピの default_direction・default_key)")}"><span class="muted">${t("keyed の既定")}</span>
        <span class="mono">@</span><select id="rc-kdd">${opt(dirs, r.defaultDirection)}</select>
        <span class="mono">#</span><select id="rc-kdk">${opt(keys, r.defaultKey)}</select></div>
      ${noteHTML(t("鍵つきの書き込み a: 入力の活性 x に鍵 κ が反応する(鍵の反応の強さ ρ = κᵀx が大きい)ときだけ、方向 r_L を α·ρ 倍で書く(α = 1 で、観測した A − B の平均差をそのまま)。書き込みたくない位置では ρ ≈ 0 で何も足さない。LoRA で当てるので cvec には入りません。3 列チャットの − 列は α が負(打ち消す)。"), "muted note")}`;
  }

  // 「鍵つきの書き込み(keyed)を足す」: 鍵・その鍵の層・方向・α を選んで行を足す(保存した鍵があるときだけ)
  keyedAddHTML() {
    const keys = this.keys().filter((k) => !k.error && (k.layers || []).length);
    if (!keys.length) return "";
    const dirs = this.projDirections(), f = this.ka || (this.ka = { key: "", layer: null, alpha: 1, direction: "" });
    let k = keys.find((x) => x.name === f.key);
    if (!k) { k = keys[0]; f.key = k.name; f.layer = k.layers[0]; }
    if (!k.layers.includes(f.layer)) f.layer = k.layers[0];
    if (!f.direction && !this.r.defaultDirection && dirs.length) f.direction = dirs.includes(f.key) ? f.key : dirs[0];
    const dopts = [`<option value=""${f.direction ? "" : " selected"}>${esc(this.r.defaultDirection ? t("既定: {x}", { x: this.r.defaultDirection }) : t("既定(なし)"))}</option>`,
      ...dirs.map((d) => `<option${d === f.direction ? " selected" : ""}>${esc(d)}</option>`)].join("");
    return `<details class="rc-kadd"${this.kaOpen ? " open" : ""}><summary class="muted note">${t("鍵つきの書き込み(keyed)を足す")}</summary>
      <div class="rc-ablate"><span class="mono">#</span><select id="rc-ka-key" title="${t("鍵")}">${keys.map((x) => `<option${x.name === f.key ? " selected" : ""}>${esc(x.name)}</option>`).join("")}</select>
        <span class="muted">${t("層")}</span><select id="rc-ka-L">${k.layers.map((L) => `<option value="${L}"${L === f.layer ? " selected" : ""}>L${L}</option>`).join("")}</select>
        <span class="muted mono" title="${t("鍵の書き込み先")}">${esc(k.target)}</span>
        <span class="mono">@</span><select id="rc-ka-dir" title="${t("書き込む方向(gne direction の方向セットの r_L)")}">${dopts}</select>
        <span class="muted">α</span><input id="rc-ka-a" type="number" step="${KEYED.step}" value="${f.alpha}" class="num">
        <button class="small" id="rc-ka-go">${t("レシピに足す")}</button></div></details>`;
  }

  wireKeyed() {
    const r = this.r, $ = (s) => this.root.querySelector(s);
    const each = (sel, ev, fn) => { for (const el of this.root.querySelectorAll(sel)) el.addEventListener(ev, () => fn(el)); };
    each("select[data-kkind]", "change", (el) => { r.setKeyed(+el.dataset.kkind, { kind: el.value }); this.render(); });
    // 専門家の番号は欄から出たとき(change)に入れる(打つたびに並べ替えると行の番号がずれる)。入れかけの値は描き直しても残す
    each("input[data-kexp]", "input", (el) => { this.kexpDraft = { u: r.units[+el.dataset.kexp], v: el.value }; });
    each("input[data-kexp]", "change", (el) => {
      this.kexpDraft = null;
      const v = Number(el.value);
      if (Number.isInteger(v) && v >= 0) { r.setKeyed(+el.dataset.kexp, { expert: v }); this.render(); }
    });
    each("select[data-kdir]", "change", (el) => { r.setKeyed(+el.dataset.kdir, { direction: el.value || null }); this.render(); });
    each("select[data-kkey]", "change", (el) => { r.setKeyed(+el.dataset.kkey, { key: el.value || null }); this.render(); });
    const on = (sel, ev, fn) => { const el = $(sel); if (el) el.addEventListener(ev, () => fn(el)); };
    on("#rc-kdd", "change", (el) => { r.setDefaults({ direction: el.value }); this.render(); });
    on("#rc-kdk", "change", (el) => { r.setDefaults({ key: el.value }); this.render(); });
    on(".rc-kadd", "toggle", (el) => { this.kaOpen = el.open; });
    on("#rc-ka-key", "change", (el) => { this.ka.key = el.value; this.ka.layer = null; this.render(); });
    on("#rc-ka-L", "change", (el) => { this.ka.layer = +el.value; });
    on("#rc-ka-dir", "change", (el) => { this.ka.direction = el.value; });
    on("#rc-ka-a", "input", (el) => { if (Number.isFinite(+el.value)) this.ka.alpha = +el.value; });
    on("#rc-ka-go", "click", () => {
      const k = this.keys().find((x) => x.name === this.ka.key);
      if (!k) return;
      const { kind, expert } = targetKind(k.target);
      r.addKeyed({ layer: this.ka.layer, kind, expert, alpha: this.ka.alpha, direction: this.ka.direction || null, key: k.name });
      this.onAdded();
    });
  }

  // 倍率の行(ヘッド L21 H7 ×0.00、MoE の gain)。スライダーは倍率 k(−1〜3、1 が無介入)
  gainRow(u, i) {
    const id = isHead(u) ? `<a href="#" data-open="${i}" title="${t("くわしく見るで開く")}">L${u.layer} H${u.head}</a> <span class="muted w">${esc(this.headKind(u.layer))} · gain</span>`
      : this.moeId(u, i);
    return `
      <tr data-i="${i}" class="rc-gain${isHead(u) ? " rc-head" : " rc-moe"}"><td class="id">${id}<button class="small quiet rm" data-rm="${i}" title="${t("外す")}">×</button>
        <div class="sl"><span class="muted">${GAIN.min}</span><input type="range" class="dose" min="${GAIN.min}" max="${GAIN.max}" step="${GAIN.step}" value="${u.k}" data-i="${i}" style="background:${gainBG(u.k)}"><span class="muted">${GAIN.max}</span>
          <span class="sg mono ${u.k >= 1 ? "plus" : "minus"}">${kLabel(u.k)}</span>
          <input type="number" min="-4" max="8" step="${GAIN.step}" value="${u.k}" data-i="${i}" class="num" title="${t("値を直接入れる")}"></div></td></tr>`;
  }

  ablateHTML() {
    const a = this.r.ablate, dirs = this.directions();
    const opts = [`<option value="">${t("なし")}</option>`, ...dirs.map((d) => `<option${a && a.direction === d ? " selected" : ""}>${esc(d)}</option>`)];
    if (a && !dirs.includes(a.direction)) opts.push(`<option selected>${esc(a.direction)}</option>`);
    const sc = a ? a.scale : 1;
    return `<div class="rc-ablate" title="${t("方向の射影消去: 選んだ方向を残差に書き込む重みから消す(LoRA で当てる)。gne direction で作った方向を使う")}">
        <span class="muted">ablate</span><select id="rc-ab-dir">${opts.join("")}</select>${a ? `
        <span class="muted">${t("強さ")}</span><span class="sl"><input type="range" id="rc-ab-scale" min="0" max="1.5" step="0.05" value="${sc}"><span class="mono" id="rc-ab-scale-v">×${fmt(sc, 2)}</span></span>
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
    // 層・測った層は打つたびに入れる(欄から出る前に描き直しても消えないように)
    $("#rc-ab-layers").addEventListener("input", (ev) => this.r.setAblate({ ...cur(), layers: ev.target.value.trim() || "all" }));
    $("#rc-ab-from").addEventListener("input", (ev) => {
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
      const kd = this.r.keyed().length;                // keyed は省いた方向・鍵を省いたまま、既定を添えて保存する(CHANGE-07)
      const saved = await postJSON(`/api/recipes/${encodeURIComponent(name)}`, { units: this.r.specs(1, false), ablate: this.r.ablate, sampling: this.r.sampling,
                                                                   ...((this.r.projs().length || kd) && this.r.directionLayer !== "own" ? { direction_layer: this.r.directionLayer } : {}),
                                                                   ...(kd && this.r.defaultDirection ? { default_direction: this.r.defaultDirection } : {}),
                                                                   ...(kd && this.r.defaultKey ? { default_key: this.r.defaultKey } : {}) });
      await this.refreshSaved();                     // 一覧を描き直してから知らせる(描き直すと #rc-result も作り直される)
      this.root.querySelector("#rc-result").innerHTML =
        `<p class="muted note">${t("保存しました({path})", { path: esc((saved && saved.path) || `${name}.yaml`) })}</p>`;   // 保存先はサーバが返す実際のパス(--root を指定していればその下)
    } catch (e) { res.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }

  // 計算で作るニューロン(nw)の LoRA の書き出し
  async exportNw() {
    const res = this.root.querySelector("#rc-result");
    const name = (this.root.querySelector("#rc-name").value.trim() || "recipe").replace(/[^\w.-]/g, "_");
    res.innerHTML = `<p class="muted note">${t("書き出し中…")}</p>`;
    try {
      const x = await postJSON("/api/nw/export", { units: this.r.nws().map((u) => nwSpec(u)), name });
      res.innerHTML = `<p class="note">${t("{file} を書き出しました", { file: `<a href="${esc(x.url)}" download="${esc(x.file)}">${esc(x.file)}</a>` })}</p>
        <p class="muted note">${t("llama.cpp で使う: {cmd}", { cmd: `<code>--lora ${esc(x.file)}</code>` })}</p>
        ${x.warnings && x.warnings.length ? `<p class="muted note">${x.warnings.map(esc).join("<br>")}</p>` : ""}`;
    } catch (e) { res.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }

  async export() {
    const res = this.root.querySelector("#rc-result");
    const verify = this.root.querySelector("#rc-verify").checked;
    const name = (this.root.querySelector("#rc-name").value.trim() || "recipe").replace(/[^\w.-]/g, "_");
    res.innerHTML = `<p class="muted note">${t(verify ? "書き出し中(llama.cpp で読み直して生成を比べる、数十秒)…" : "書き出し中…")}</p>`;
    try {
      const nk = this.r.keyed().length;                // keyed は cvec に入らない(サーバが警告に出す)
      const x = await postJSON("/api/export", { units: this.r.addSpecs(), name, verify, ...(nk ? { keyed: nk } : {}) });
      const a = document.createElement("a");                       // そのままダウンロードする
      a.href = x.url; a.download = x.file; document.body.appendChild(a); a.click(); a.remove();
      const norms = Object.entries(x.layer_norms).map(([L, v]) => `L${L} ${fmt(v, 1)}`).join(" · ");
      const v = x.verify;
      res.innerHTML = `<p class="note">${t("{file} を書き出しました", { file: `<a href="${esc(x.url)}" download="${esc(x.file)}">${esc(x.file)}</a>` })}${
          x.recipe_saved ? "" : `<br><span class="muted">${t("(レシピ自体は未保存。上の「保存」で保存先の recipes フォルダに残せます)")}</span>`}</p>
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
  // ファイル名の既定(拡張子なし): レシピの名前(焼き込みは -headproj を付ける)。LoRA に入るのが keyed だけなら -keyed
  // (gne headproj export の既定の名前と同じ)
  lxDefaultStem() {
    const nm = (((this.root.querySelector("#rc-name") || {}).value || this.r.name || "headproj").trim().replace(/[^\w.-]/g, "_")) || "headproj";
    if (this.lxKeyedOnly()) return `${nm}-keyed`;
    return this.lx.bake ? `${nm}-headproj` : nm;
  }
  lxKeyedOnly() { return this.r.keyed().length > 0 && !this.r.heads().length && !this.r.projs().length; }
  // 書き換えたテンソルの型の既定: keyed を含むなら auto(テンソルごとに、元が Q8_0 なら Q8_0 のまま、それ以外は F16。
  // Q8_0 のままでは編集の 1 割以上が丸めに消えるテンソルだけ F16)、それ以外は Q8_0。auto は keyed を含むときだけ選べる
  lxRequant() {
    const nk = this.r.keyed().length;
    const rq = this.lx.requant ?? (nk ? "auto" : "q8_0");
    return rq === "auto" && !nk ? "q8_0" : rq;
  }

  lxHTML() {
    const f = this.lx, r = this.r, bake = f.bake, rq = this.lxRequant(), nk = r.keyed().length;
    const n = r.heads().length + r.projs().length + nk, nAdd = r.addSpecs().length;
    const dis = bake ? "" : " disabled";
    return `<div class="rc-lx">
      <div class="rc-lx-row"><span class="muted">${t("形式")}</span>
        <label title="${t("小さいファイル(同じ層・同じ方向のヘッドは階数 1 にまとまる)。llama.cpp では --lora で当てる。元のモデルはそのまま")}"><input type="radio" name="lx-kind" value="lora"${bake ? "" : " checked"}> ${t("LoRA アダプタ")}</label>
        <label title="${t("関係するテンソルだけ書き換えた GGUF を丸ごと作る。--lora なしでそのまま使える。モデルと同じくらいのディスクが要る")}"><input type="radio" name="lx-kind" value="bake"${bake ? " checked" : ""}> ${t("重みに焼き込む(GGUF)")}</label></div>
      <div class="rc-lx-row${bake ? "" : " off"}"><span class="muted" title="${t("空なら、いま読んでいるモデルに焼き込む。同じモデルの量子化していない GGUF(F16 / BF16)のパスを入れると、方向はこのモデルで作ったものを使い、そちらに焼き込む(GPU は使わない。少し誤差がある。LoRA 版との比較はできない)")}">${t("焼き込む先")}</span>
        <input id="lx-into" size="30" value="${esc(f.into)}" placeholder="${t("空ならいま読んでいるモデル")}" spellcheck="false"${dis}></div>
      <div class="rc-lx-row${bake ? "" : " off"}"><span class="muted" title="${t("書き換えたテンソルを何の型で書くか。Q8_0 は元より少し大きくなることがある、F16 はもっと大きい。下の「元の型に戻す」を使うとほぼ元の大きさ")}">${t("書き換えたテンソルの型")}</span>
        <select id="lx-requant"${dis || (f.via_f16 ? " disabled" : "")}>${nk ? `<option value="auto"${rq === "auto" ? " selected" : ""}>${t("自動(Q8_0 は元のまま、ほかは F16)")}</option>` : ""}<option value="q8_0"${rq === "q8_0" ? " selected" : ""}>Q8_0</option><option value="f16"${rq === "f16" ? " selected" : ""}>F16</option><option value="keep"${rq === "keep" ? " selected" : ""}>${t("元と同じ(F16・BF16 のモデル)")}</option></select>
        <label title="${t("書き換えたテンソルを F16 で書いてから、llama-quantize で元の量子化の型に戻す(ファイルの大きさを元に近く保つ。低ビットの型は imatrix があると誤差が小さい)")}"><input type="checkbox" id="lx-viaf16"${f.via_f16 ? " checked" : ""}${dis}> ${t("元の量子化の型に戻す")}</label></div>
      ${bake && nk ? `<div class="rc-lx-row"><span class="muted note">${t("鍵つきの書き込みを含むので、書き換えたテンソルの型の既定は「自動」: 元が Q8_0 のテンソルは Q8_0 のまま(F16 にすると計算方法が変わり、LoRA 版とかえってずれる)、それ以外は F16。Q8_0 のままでは編集の 1 割以上が丸めに消えるテンソルだけ F16(配るなら LoRA がおすすめ)")}</span></div>` : ""}
      ${bake && f.via_f16 ? `<div class="rc-lx-row"><span class="muted">llama-quantize</span><input id="lx-lq" size="28" value="${esc(f.llama_quantize)}" placeholder="${t("空なら GNE_LLAMA_QUANTIZE か PATH から探す")}" spellcheck="false">
        <span class="muted" title="${t("低ビットの型(IQ1・IQ2・IQ3_XXS)に戻すときは必須。空なら、モデルの隣の <モデル名>.imatrix.gguf / .imatrix.dat / imatrix.gguf を探して使う")}">imatrix</span><input id="lx-imatrix" size="22" value="${esc(f.imatrix)}" placeholder="${t("空ならモデルの隣を探す")}" spellcheck="false"></div>` : ""}
      <div class="rc-lx-row${bake && !f.into.trim() ? "" : " off"}"><label title="${t("焼き込み版と LoRA 版で、方向への射影の値・最初のトークンの分布・最尤トークン・ppl が一致するかを確かめる(モデルを 2 回読むので時間がかかる)")}"><input type="checkbox" id="lx-verify"${f.verify ? " checked" : ""}${bake && !f.into.trim() ? "" : " disabled"}> ${t("LoRA 版と一致するか確かめる")}</label>
        ${bake && f.verify && !f.into.trim() ? `<span class="muted">${t("対照集合")}</span><input id="lx-contrast" size="24" value="${esc(f.contrast)}" placeholder="${t("空なら方向を作った対照集合")}" spellcheck="false">` : ""}</div>
      <div class="rc-lx-row"><span class="muted" title="${t("書き出すファイルの名前(拡張子なし)。空ならレシピの名前から作る。保存先は <root>/export/")}">${t("ファイル名")}</span>
        <input id="lx-file" size="26" value="${esc(f.file)}" placeholder="${esc(this.lxDefaultStem())}" spellcheck="false"><span class="muted mono">${bake ? ".gguf" : ".lora.gguf"}</span></div>
      <div class="rc-lx-row"><label title="${t("gne が書く記録(gne.* のメタデータ: レシピ・ヘッド・強さ・方向の鍵、名前の印、横の .headproj.json)を入れない。人に渡すファイル用。書いたあと、メタデータに手元のパスが残っていないかも確かめる")}"><input type="checkbox" id="lx-strip"${f.strip ? " checked" : ""}> ${t("gne の記録(メタデータ)を入れない(人に渡す用)")}</label></div>
      ${bake ? "" : `<div class="rc-lx-row"><span class="muted" title="${t("LoRA のファイルの中の名前の欄(general.name)。空なら「元のモデルの名前 gain レシピの名前」、記録を入れないときは adapter")}">${t("名前の欄")}</span>
        <input id="lx-adname" size="26" value="${esc(f.adapterName)}" placeholder="${f.strip ? "adapter" : ""}" spellcheck="false"></div>`}
      <div class="rc-lx-row"><button class="main" id="lx-go">${t(bake ? "焼き込む" : "LoRA を書き出す")}</button>
        <span class="muted note">${nk ? t("名前は上のレシピの名前(空なら headproj)。LoRA に入るユニット {n} 個(ヘッド {h}・鍵つきの書き込み {k})", { n, h: n - nk, k: nk })
          : t("名前は上のレシピの名前(空なら headproj)。ヘッドのユニット {n} 個が入る", { n })}${nAdd ? ` · ${t("add のユニット {k} 個は入らない(cvec で書き出す)", { k: nAdd })}` : ""}${bake ? ` · ${t("焼き込みはモデルを丸ごともう 1 つ書く(同じくらいのディスク・時間)")}` : ""}</span></div>
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
    on("#lx-strip", "change", (ev) => { const el = this.root.querySelector("#lx-adname"); if (el) el.placeholder = ev.target.checked ? "adapter" : ""; });
    on("#lx-adname", "input", (ev) => { f.adapterName = ev.target.value; });
    on("#lx-file", "input", (ev) => { f.file = ev.target.value; });
    on("#lx-into", "input", (ev) => { f.into = ev.target.value; });   // 打つたびに入れる(描き直しても消えないように)
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
    const all = r.specs();                                  // keyed の省いた方向・鍵はレシピの既定で埋める
    const units = [...r.heads(), ...r.projs(), ...r.keyed()].map((u) => all[r.units.indexOf(u)]);
    const p = { units, name, bake: f.bake, file_stem: stem, overwrite };
    if (f.strip) p.strip_meta = true;
    if (!f.bake && f.adapterName.trim()) p.adapter_name = f.adapterName.trim();
    if ((r.projs().length || r.keyed().length) && r.directionLayer && r.directionLayer !== "own") p.direction_layer = r.directionLayer;
    if (f.bake) {
      if (f.via_f16) {
        p.via_f16 = true;
        if (f.llama_quantize.trim()) p.llama_quantize = f.llama_quantize.trim();
        if (f.imatrix.trim()) p.imatrix = f.imatrix.trim();
      } else p.requant = this.lxRequant();
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
    const klayers = Object.entries(x.keyed_layers || {}).map(([L, v]) => `L${L} keyed: ${t("鍵 {k}", { k: (v.keys || []).join(", ") })} · ${t("階数 {r}", { r: v.rank })}`).join(" · ");
    const rqe = x.requant_error_max;
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
      ${klayers ? `<p class="muted note mono">${esc(klayers)}</p>` : ""}
      ${rqe !== undefined && rqe !== null ? `<p class="muted note">${t("量子化し直しで丸めに消えた編集の割合(最大){p}%", { p: fmt(100 * rqe, 2) })}</p>` : ""}
      ${(x.modified || []).length ? `<p class="muted note">${t("書き換えたテンソル {n} 個", { n: x.modified.length })}: <span class="mono">${esc(x.modified.slice(0, 6).map((m) => `${m.tensor} ${m.from}→${m.to}`).join(", "))}${x.modified.length > 6 ? " …" : ""}</span></p>` : ""}
      ${v ? `<p class="note">${v.pass ? "✓" : "✗"} ${t("LoRA 版との比較: 最初のトークンの KL {kl}、最尤トークンの一致 {top}%、ppl の差 {ppl}%", { kl: Number(v.kl_first_token).toExponential(1), top: Math.round(100 * v.top1_match), ppl: fmt(v.ppl_diff_pct, 3) })}</p>` : ""}
      <p class="muted note">${how}</p>
      ${(x.warnings || []).length ? `<p class="muted note">${x.warnings.map(esc).join("<br>")}</p>` : ""}`);
  }
}

export { signed };
