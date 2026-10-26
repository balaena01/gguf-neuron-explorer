// gne explorer: 画面全体の配線(別冊 §2)。
// U0: 構造の描画、層の選択、ジャンプ、hover の L N。
// U1: 色の重ね合わせ(overlay)、LIST の rank、NEURON(Examples / Details)、hover の語、検索、layers ビュー。
// U2: NEURON の Chat(3 列・reads you)、LIST の fires と recipe、地図のレシピの印、ジョブのバーと計算メニュー。
import { LayerCache, getBin, getJSON, loadVocab } from "./api.js";
import { ChatPanel } from "./chat.js";
import { EmbedView } from "./embed.js";
import { FiresPanel } from "./fires.js";
import { JobsBar, KIND, computeMenu } from "./jobs.js";
import { LayersView } from "./layers.js";
import { syntheticArch } from "./layout.js";
import { LayerMap } from "./map.js";
import { FIELDS, colorize, colorizeHits, emphasisFactor, emphasize, fieldLabel, globalScale, layersWith, legendHTML } from "./overlay.js";
import { NeuronPanel, RankList, SearchList, esc, toksHTML } from "./panels.js";
import { Recipe, RecipePanel } from "./recipe.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);

const state = {
  model: null, arch: null, vocab: null, map: null, view: "map", layer: 0, focus: null, note: "",
  layout: "index", web: false, webHops: 2, links: null, linkMode: "direct",
  overlay: { field: "none", threshold: null, scale: null, values: new Map(), metas: new Map(), hits: null, prev: "none" },
};
window.gne = state;                       // 自動テスト(playwright)と開発用

const MIXER = { attention: "attention", linear_attention: "線形注意", recurrent: "回帰", none: "mixer なし" };
const mixerLabel = (li) => MIXER[li.mixer];
const isMoE = () => state.arch.layers.some((li) => li.ffn.startsWith("moe"));
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));

// --- 層ごとのデータ(層単位で一括ロードし、hover のたびには通信しない、§4.6) ----------------------------------
const fieldCache = new LayerCache(async (key) => {
  const [L, name] = key.split("|");
  const r = await getBin(`/api/layer/${L}/field`, { name });
  return { L: +L, v: new Float32Array(r.buf), meta: r.meta };
});
const hoverCache = new LayerCache(async (L) => {
  const r = await getBin(`/api/layer/${L}/hover`);
  const n = state.arch.layers[L].n_ff;
  return { ids: new Int32Array(r.buf, 0, n * 24), kl: new Float32Array(r.buf, n * 24 * 4, n * 2), meta: r.meta };
});
const unitsCache = new LayerCache(async (L) => {
  const d = await getJSON(`/api/layer/${L}/units`);
  const byN = new Map();
  const main = d.runs[0];
  if (main) for (const u of main.units) if (u.kind === "neuron") byN.set(u.neuron, { ...u, n_ranked: main.n_neurons });
  return { d, byN };
});
const cov = (L) => (state.model ? state.model.coverage[L] : null);

function describeLayer(L) {
  const li = state.arch.layers[L];
  const ffn = li.ffn === "dense" ? `MLP ${li.n_ff}` : li.ffn.startsWith("moe")
    ? `MoE ${li.n_expert}×${li.n_ff}(使用 ${li.n_expert_used})${li.n_ff_shexp ? ` + shared ${li.n_ff_shexp}` : ""}` : "FFN なし";
  const heads = li.n_head ? ` ${li.n_head}h${li.n_head_kv && li.n_head_kv !== li.n_head ? `/${li.n_head_kv}kv` : ""}` : "";
  return `L${L} ${mixerLabel(li)}${heads} · ${ffn}`;
}

// --- hover の吹き出し(§4.6): L N、名前、sweep の上位語、steer ±、logit ±、今の overlay の値 ---------------------
function tooltipHTML(h) {
  const who = h.shared ? `shared N${h.neuron}` : h.expert !== undefined ? `E${h.expert} N${h.neuron}` : `N${h.neuron}`;
  const L = h.layer, n = h.neuron, lines = [];
  const name = state.model && !h.shared && h.expert === undefined ? (state.model.names || {})[`${L}:${n}`] : null;
  lines.push(`<span class="t-id">L${L} ${who}</span>${name ? `<span class="t-name">${esc(name)}</span>` : ""}`);
  if (!state.model || h.shared || h.expert !== undefined) return lines.join("<br>");
  const c = cov(L);
  let waiting = false;
  if (c.sweep_runs.length) {
    const us = unitsCache.peek(L);
    if (!us) { waiting = true; unitsCache.get(L).then(rehover); }
    else {
      const u = us.byN.get(n);
      if (u) lines.push(`<span class="t-k">sweep</span>${u.words.map(esc).join(" · ")}<span class="t-v">mass ${fmt(u.mass, 1)} · #${u.rank}/${u.n_ranked}</span>`);
    }
  }
  if (c.steer_lens || c.logit_lens) {
    const hv = hoverCache.peek(L);
    if (!hv) { waiting = true; hoverCache.get(L).then(rehover); }
    else if (state.vocab) {
      const ids = (a, b) => Array.from(hv.ids.subarray(n * 24 + a, n * 24 + b)).filter((i) => i >= 0).map((i) => state.vocab[i]);
      if (hv.meta.steer && Number.isFinite(hv.kl[2 * n])) {
        lines.push(`<span class="t-k plus">steer+</span>${toksHTML(ids(0, 6))}<span class="t-v">KL ${fmt(hv.kl[2 * n], 3)}</span>`);
        lines.push(`<span class="t-k minus">steer−</span>${toksHTML(ids(6, 12))}<span class="t-v">KL ${fmt(hv.kl[2 * n + 1], 3)}</span>`);
      }
      const lp = ids(12, 18);
      if (hv.meta.logit && lp.length) {
        lines.push(`<span class="t-k plus">logit+</span>${toksHTML(lp)}`);
        lines.push(`<span class="t-k minus">logit−</span>${toksHTML(ids(18, 24))}`);
      }
    } else waiting = true;
  }
  const f = state.overlay.field, v = state.overlay.values.get(L);
  if (v && f !== "none" && !f.startsWith("search:") && Number.isFinite(v[n])) {
    lines.push(`<span class="t-k">color</span><span class="t-v" style="margin:0">${esc(fieldLabel(f))} = ${fmt(v[n], f.startsWith("cos:") ? 4 : 3)}</span>`);
  }
  if (waiting) lines.push(`<span class="t-wait">読み込み中…</span>`);
  return lines.join("<br>");
}

function rehover() { if (state.map && state.map.hover) showTooltip(state.map.hover); }

function showTooltip(h) {
  const tip = $("#tooltip");
  if (!h) { tip.classList.add("hidden"); return; }
  tip.innerHTML = tooltipHTML(h);
  tip.classList.remove("hidden");
  const wrap = $("#map-wrap").getBoundingClientRect();
  let x = h.px + 16, y = h.py + 16;
  if (x + tip.offsetWidth > wrap.width - 6) x = Math.max(6, h.px - tip.offsetWidth - 12);
  if (y + tip.offsetHeight > wrap.height - 6) y = Math.max(6, h.py - tip.offsetHeight - 12);
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}

// --- 状態表示・凡例 -----------------------------------------------------------------------------------------
function updateStatus() {
  const m = state.model ? state.model.meta : { name: state.arch.arch };
  const map = state.map;
  const where = map && map.front !== null ? `L${map.front}` : `${state.arch.n_layer} 層`;   // 層の説明は選択欄にある
  const flash = state.flash && state.flash.until > Date.now() ? state.flash.text : "";        // ジョブの結果などの一時的な知らせ
  $("#status-main").textContent = `${m.name} · ${where}${flash ? ` · ${flash}` : ""}${state.note ? ` · ${state.note}` : ""}`;
  $("#status-main").title = $("#status-main").textContent;
  $("#perf").textContent = state.view === "embed"
    ? (embedView && embedView.data ? `${embedView.data.points.length.toLocaleString()} rollout(embed)` : "—(embed)")
    : state.view === "layers" ? `${state.arch.n_layer} 層(layers)`
    : `${(map?.nPoints ?? 0).toLocaleString()} 点 · ${map && map.fps !== null ? map.fps.toFixed(0) : "—"} fps`;
}

function legend() {
  const kinds = new Set(state.arch.layers.map((li) => li.mixer));
  const o = state.overlay;
  const it = (...xs) => `<span class="it">${xs.join("")}</span>`, sw = (c) => `<span class="sw" style="background:${c}"></span>`;
  const parts = [];
  if (kinds.has("attention")) parts.push(it(sw("#9fb3cf"), "attention(太い縁)"));
  if (kinds.has("linear_attention")) parts.push(it(sw("#3a4656"), "線形注意(細い縁)"));
  if (kinds.has("recurrent")) parts.push(it(sw("#3a4656"), "回帰(細い縁)"));
  const floors = [...o.metas.values()].map((m) => m.floor);
  const color = o.field === "none" ? it(sw("#10141b"), "データ無し")
    : `<b>${esc(fieldLabel(o.field))}</b> ${legendHTML(o.field, o.scale || {}, { threshold: o.threshold, nHits: o.hits ? o.hits.length : 0, floors })}`;
  const frame = o.field === "none" ? "" : `<div>${it("シートの枠: ", sw("#252b33"), "データ無し")}${it(sw("#5e4a26"), "→", sw("#f0b850"), "光る点の数(最多の層で最も明るい)")}` +
    (o.factor < 1 ? it(`(光る点が全体の ${Math.round(0.02 / o.factor * 100)}% あるので、点は一律に暗くし、数は枠で示す)`) : "") +
    (o.weak && o.weak.size ? it(`(${[...o.weak].map((L) => `L${L}`).join("・")} は活性統計が無く文章内 p99 で割ったので、枠の比較から外した)`) : "") + "</div>";
  const ring = (c) => `<span class="sw ring" style="border-color:${c}"></span>`;
  const marks = (state.recipe && state.recipe.units.length ? [it(ring("#f2d24b"), "レシピ +"), it(ring("#c94fa7"), "レシピ −(大きさ = |dose|)")] : [])
    .concat(state.focus ? [it(ring("#ffffff"), "開いているニューロン")] : []);
  const line = (c) => `<span class="sw line" style="background:${c}"></span>`;
  const links = state.links ? `<div>${it(state.links.web ? `web(${state.webHops} 段、${state.links.n} 本): ` : `配線(上流・下流 ${state.links.n} 本): `)}${
    it(line("#f2d24b"), "黄 = 読み手(線の先)を + に押す")}${it(line("#ff70d0"), "赤紫 = 読み手を − に押す")}${
    it(state.links.web ? `色 = 各辺の符号(経路の積ではない)、明るさ = z(表示中の最大に対する相対)、暗い線 = web の端(この先はたどっていない)${
      state.map && state.map.front !== null ? "。正面ビューの札は 1 段目だけ(全体ビューで全体を見る)" : ""}`
      : "明るさ = z(表示中の最大に対する相対)。正面ビューでは右 = 下流、左(破線)= 上流")}</div>` : "";
  const moe = isMoE() ? "、MoE は専門家ごとの小さな正方形" : "";
  const lay = state.layout && state.layout !== "index"
    ? `シート内の並び = ${state.layout} 配置(UMAP)。近くにあっても似ているのは弱い傾向だけ${state.layoutMissing && state.layoutMissing.length ? `(未計算の ${state.layoutMissing.length} 層は番号順)` : ""}`
    : "シート内の並びはニューロン番号順(近くにあっても似ているとは限らない)";
  $("#legend").innerHTML = `<div>${color}</div>${frame}<div>${parts.join("")}</div>${marks.length ? `<div>${marks.join("")}</div>` : ""}${links}` +
    `<div class="note st">各層 = 手前の板(mixer、上辺の刻みが head の数)+ 奥のシート(MLP の全ニューロン${moe})、下の線 = 残差</div>` +
    `<div class="note">${lay}</div>`;
}

// --- overlay(§4.3) --------------------------------------------------------------------------------------
let overlayToken = 0;

function fillColorSelect() {
  const sel = $("#color-select");
  const names = ["none", ...Object.keys(FIELDS).filter((k) => k !== "none"), ...(state.model ? state.model.directions.map((d) => `cos:${d}`) : [])];
  sel.innerHTML = names.map((k) => {
    const n = state.model && k !== "none" ? layersWith(k, state.model).length : 0;
    const dis = k !== "none" && !n;
    return `<option value="${esc(k)}"${dis ? " disabled" : ""}>${esc(fieldLabel(k))}${k === "none" ? "" : dis ? "(データ無し)" : `(${n} 層)`}</option>`;
  }).join("");
  const f = state.overlay.field;
  if (f.startsWith("search:") || f.startsWith("fires:")) sel.insertAdjacentHTML("beforeend", `<option value="${esc(f)}">${esc(fieldLabel(f))}</option>`);
  sel.value = state.overlay.field;
}

async function refreshModel() {
  const m = await getJSON("/api/model");
  state.model.coverage = m.coverage;
  state.model.names = m.names;
  state.model.directions = m.directions;
  state.model.layouts = m.layouts;
  state.model.connections = m.connections;
  state.model.runs = m.runs;
}

// 枠の明るさ = その層で光っている点の数(いちばん多い層を 1 とする、比例)。全体ビューでは上位 3 層に札も出す。
// 方向が強い層などが、点の大きさに頼らず一目で分かるように(F40)
// weak: 比べにくい層(fires で活性統計が無く文章内 p99 で割った層など)。最大と札からは外し、枠は最も暗くする
function showLayerCounts(lit, unit = "点", weak = new Set()) {
  const mx = Math.max(1, ...[...lit].filter(([L]) => !weak.has(L)).map(([, c]) => c));
  state.map.setFrameColors(new Map([...lit].map(([L, c]) => [L, weak.has(L) ? 0 : c / mx])));
  const top = [...lit].filter(([L, c]) => c > 0 && !weak.has(L)).sort((a, b) => b[1] - a[1]).slice(0, 3);
  state.map.setBadges(top.map(([L, c]) => ({ layer: L, text: `L${L} · ${c} ${unit}` })));
}

function recolor() {
  const o = state.overlay, map = state.map;
  map.resetColors();
  const lit = new Map(), rgbas = new Map();
  let total = 0;
  for (const [L, v] of o.values) {
    const named = new Set(Object.keys(state.model.names || {}).filter((k) => +k.split(":")[0] === L).map((k) => +k.split(":")[1]));
    const rgba = colorize(o.field, v, o.metas.get(L), o.scale, { threshold: o.threshold, named });
    rgbas.set(L, rgba);
    lit.set(L, rgba.lit);
    total += v.length;
  }
  o.factor = emphasisFactor([...lit.values()].reduce((a, b) => a + b, 0), total);   // 全層で同じ係数(F39)
  for (const [L, rgba] of rgbas) map.setLayerColors(L, emphasize(rgba, o.factor));
  o.lit = lit;
  map.setStructureDim(0.35);
  showLayerCounts(lit, "点", o.weak || new Set());
  legend();
}

async function applyOverlay(field) {
  const tok = ++overlayToken, t0 = performance.now();
  const o = state.overlay;
  if (!field.startsWith("search:")) { o.hits = null; o.factor = 1; }
  o.weak = new Set();
  if (o.field !== field && !field.startsWith("search:")) o.threshold = null;
  o.field = field;
  o.scale = null;                                  // 読み込みが終わるまで前の overlay の尺度を使わない
  $("#cos-thr-wrap").classList.toggle("hidden", !field.startsWith("cos:"));
  if (field === "none" || !state.model) {
    o.values = new Map(); o.metas = new Map(); o.scale = null;
    state.map.resetColors(); state.map.setStructureDim(1); state.map.setFrameColors(new Map()); state.map.setBadges([]);
    fillColorSelect(); legend(); rehover();
    return;
  }
  await refreshModel();
  if (tok !== overlayToken) return;
  fillColorSelect();
  const Ls = layersWith(field, state.model);
  let done = 0;
  state.note = `${fieldLabel(field)} を読み込み中 0/${Ls.length}`;
  updateStatus();
  for (const L of Ls) fieldCache.done.delete(`${L}|${field}`);   // 計算が進んでいるかもしれないので取り直す
  const res = await Promise.all(Ls.map((L) => fieldCache.get(`${L}|${field}`).then((r) => {
    done++;
    if (tok === overlayToken) { state.note = `${fieldLabel(field)} を読み込み中 ${done}/${Ls.length}`; updateStatus(); }
    return r;
  }))).catch((e) => { state.note = `エラー: ${e.message}`; updateStatus(); return null; });
  if (!res || tok !== overlayToken) return;
  const loaded = res.filter((r) => r.meta.has);
  o.values = new Map(loaded.map((r) => [r.L, r.v]));
  o.metas = new Map(loaded.map((r) => [r.L, r.meta]));
  o.scale = globalScale(field, loaded);
  if (field.startsWith("cos:")) {
    const thr = $("#cos-thr");
    if (o.threshold === null) o.threshold = o.scale.p999;           // 全層の上位 0.1%: 方向に強く沿うニューロンだけが光る
    thr.max = String(o.scale.max); thr.step = String(o.scale.max / 400); thr.value = String(o.threshold);
    $("#cos-thr-val").textContent = `|cos| ≥ ${o.threshold.toFixed(3)}`;
  }
  state.note = "";
  recolor();
  o.ms = performance.now() - t0;           // 切り替えにかかった時間(計測用)
  updateStatus();
  rehover();
}

function applyHits(q, mode, hits) {
  const o = state.overlay, map = state.map;
  ++overlayToken;
  if (!o.field.startsWith("search:")) o.prev = o.field;
  o.field = `search:${q}`;
  o.hits = hits;
  o.factor = 1;
  o.values = new Map(); o.metas = new Map(); o.scale = null;
  $("#cos-thr-wrap").classList.add("hidden");
  map.resetColors();
  const byL = new Map();
  for (const h of hits) { if (!byL.has(h.layer)) byL.set(h.layer, []); byL.get(h.layer).push(h.neuron); }
  for (const [L, ns] of byL) map.setLayerColors(L, colorizeHits(state.arch.layers[L].n_ff, ns));
  map.setStructureDim(0.35);
  showLayerCounts(new Map([...byL].map(([L, ns]) => [L, ns.length])), "件");
  fillColorSelect();
  legend();
}

// fires の値で色を付ける(全層の値を 1 回で取る)。tok = -1 は全トークンでの |最大|
async function applyFiresOverlay(fid, tok, label, thr) {
  const tokn = ++overlayToken, o = state.overlay;
  if (!o.field.startsWith("fires:") && !o.field.startsWith("search:")) o.prev = o.field;
  state.note = "fires の値を読み込み中";
  updateStatus();
  let r;
  try { r = await getBin(`/api/fires/${fid}/field`, { tok }); }
  catch (e) { state.note = `エラー: ${e.message}`; updateStatus(); return; }
  if (tokn !== overlayToken) return;
  const sizes = r.shape.split(",").map(Number), all = new Float32Array(r.buf), has = new Set(r.meta.layers);
  o.field = `fires:${fid}:${label}`;
  o.hits = null;
  o.weak = new Set(r.meta.text_p99 || []);
  o.threshold = thr;
  o.values = new Map(); o.metas = new Map();
  let off = 0;
  sizes.forEach((n, L) => { if (has.has(L)) { o.values.set(L, all.subarray(off, off + n)); o.metas.set(L, {}); } off += n; });
  o.scale = { max: r.meta.max };
  $("#cos-thr-wrap").classList.add("hidden");
  state.note = "";
  recolor();
  fillColorSelect();
  updateStatus();
  rehover();
}

// --- 配置(§4.4)と配線(§4.7) ------------------------------------------------------------------------------
const layoutCache = new LayerCache(async (key) => {
  const [L, kind] = key.split("|");
  return new Int32Array((await getBin(`/api/layer/${L}/layout`, { kind })).buf);
});

async function applyLayout(kind) {
  state.layout = kind;
  const Ls = state.arch.layers.filter((li) => li.ffn === "dense").map((li) => li.index);
  const have = new Set(kind === "index" || !state.model ? [] : (state.model.layouts || {})[kind] || []);
  if (kind !== "index") { state.note = `配置 ${kind} を読み込み中`; updateStatus(); }
  const cells = await Promise.all(Ls.map((L) => (have.has(L) ? layoutCache.get(`${L}|${kind}`).catch(() => null) : null)));
  if (state.layout !== kind) return;
  state.layoutMissing = [];
  Ls.forEach((L, i) => { state.map.setLayout(L, cells[i]); if (kind !== "index" && !cells[i]) state.layoutMissing.push(L); });
  state.note = "";
  updateStatus();
  legend();
}

// 開いたニューロンの配線。web が off なら直接の上流・下流(各 14)、on なら多段の web
async function showLinks() {
  const f = state.focus, c = state.model && state.model.connections;
  if (!f || f.kind !== "neuron" || !c || state.linkMode === "off") { state.map.setLinks(null); state.links = null; legend(); return; }
  const key = `${f.layer}:${f.n}:${state.web}`;
  state.linksKey = key;
  let edges;
  try {
    if (state.web) {
      const r = await getBin("/api/web", { layer: f.layer, n: f.n, hops: state.webHops, first: 8, k: 4, zmin: 2 });
      const a = new Int32Array(r.buf), rows = [];
      for (let i = 0; i < a.length; i += 7) rows.push(a.subarray(i, i + 7));
      const outD = new Set(), inU = new Set();                       // 続きのある辺(下流は先から、上流は元へさらに辺がある)
      for (const x of rows) { if (x[5] > 0) outD.add(`${x[0]}:${x[1]}`); else inU.add(`${x[2]}:${x[3]}`); }
      edges = rows.map((x) => ({ sl: x[0], sn: x[1], dl: x[2], dn: x[3], z: Math.abs(x[4]) / 10, sign: Math.sign(x[4]),
                                 dim: x[5] > 0 ? !outD.has(`${x[2]}:${x[3]}`) : !inU.has(`${x[0]}:${x[1]}`) }));
    } else {
      const d = await getJSON("/api/connections", { layer: f.layer, n: f.n, k: 14 });
      edges = [...d.down.map((e) => ({ sl: f.layer, sn: f.n, dl: e.layer, dn: e.neuron, z: e.z, sign: e.sign })),
               ...d.up.map((e) => ({ sl: e.layer, sn: e.neuron, dl: f.layer, dn: f.n, z: e.z, sign: e.sign }))];
    }
  } catch (e) { state.note = `エラー: ${e.message}`; updateStatus(); return; }
  if (state.linksKey !== key) return;
  state.links = { web: state.web, n: edges.length };
  state.map.setLinks(edges, { focus: { layer: f.layer, n: f.n } });
  legend();
}

// --- 検索(header、lens / words / name)。結果は LIST の「検索」タブに強い順で出し、当たりをマップで光らせる ---------------
function listTab(tab) {
  for (const b of document.querySelectorAll("#list-tabs button[data-tab]")) b.classList.toggle("on", b.dataset.tab === tab);
  for (const [t, id] of [["rank", "#rank"], ["search", "#search-list"], ["fires", "#fires-list"], ["recipe", "#recipe-list"]]) {
    $(id).classList.toggle("hidden", tab !== t);
  }
  if (tab === "recipe") recipePanel.refreshSaved();
}

async function runSearch() {
  const q = $("#search-q").value.trim(), mode = $("#search-mode").value;
  if (!q) { closeSearch(); return; }
  expand("list");
  $("#list-tabs button[data-tab=search]").classList.remove("hidden");
  listTab("search");
  searchList.loading(q);
  let r;
  try { r = await getJSON("/api/search", { q, mode, limit: 2000 }); }
  catch (e) { searchList.error(e.message); return; }
  searchList.show(r, state.model.names || {});
  applyHits(q, mode, r.hits);
}

function closeSearch() {
  $("#list-tabs button[data-tab=search]").classList.add("hidden");
  $("#search-list").innerHTML = "";
  listTab("rank");
  if (state.overlay.field.startsWith("search:")) applyOverlay(state.overlay.prev || "none");
}

// --- LIST・NEURON ----------------------------------------------------------------------------------------
let rank, neuron, layersView, searchList, chat, firesPanel, recipePanel, jobsBar, embedView;

function expand(id) {
  const p = $(`#${id}`);
  if (p.classList.contains("collapsed")) { p.classList.remove("collapsed"); state.map.resize(); }
}

function openNeuron({ layer, n, kind = "neuron", run = null, tab = null }, { focus = true } = {}) {
  const first = !state.focus;
  state.focus = { layer, n, kind };
  if (first) legend();
  expand("neuron");
  neuron.open({ layer, n, kind, run, tab });
  showLinks();
  if (kind === "neuron") {
    if (focus) state.map.focusNeuron(layer, n); else state.map.markNeuron(layer, n);
    history.replaceState(null, "", `#L${layer}/N${n}`);
    $("#jump").value = `L${layer} N${n}`;
  }
  if (rank.layer !== layer) showLayerInList(layer).then(() => rank.select(layer, n, kind));
  else rank.select(layer, n, kind);
}

async function showLayerInList(L) {
  state.layer = L;
  $("#layer-select").value = String(L);              // header の層の選択欄も LIST の層にそろえる(視点は動かさない)
  if (!state.model) return;
  const us = await unitsCache.get(L).catch(() => ({ d: { layer: L, runs: [] } }));
  if (state.layer !== L) return;
  await rank.show(L, us.d, state.model.coverage);
  for (const a of document.querySelectorAll("#rank-empty a[data-l]")) {
    a.addEventListener("click", (ev) => { ev.preventDefault(); state.map.frontView(+a.dataset.l); });
  }
}

// ジョブが終わったら、そのデータを使うものを取り直す(coverage・overlay・LIST・hover)
async function onJobFinished(j) {
  const L = j.params && j.params.layer;
  const what = `${KIND[j.kind] || j.kind}${L !== undefined ? ` L${L}` : ""}`;
  state.flash = { text: `${{ done: "完了", error: "失敗", cancelled: "中止" }[j.status]}: ${what}`, until: Date.now() + 10000 };   // 結果を先に(状態表示は狭いので)
  updateStatus();
  if (j.status !== "done" || !state.model) return;
  if (L !== undefined) { unitsCache.done.delete(L); hoverCache.done.delete(L); }
  await refreshModel();
  fillColorSelect();
  if (j.kind === "layout") { layoutCache.clear(); if (state.layout === j.params.kind) applyLayout(state.layout); }
  if (j.kind === "connections" && state.focus) { showLinks(); neuron.render(); }
  if (j.kind === "embed" && embedView) { embedView.setRuns(state.model.runs); if (embedView.run === j.params.run) embedView.load(j.params.run); }
  const f = state.overlay.field;
  if (f !== "none" && !f.startsWith("search:") && !f.startsWith("fires:")) applyOverlay(f);
  if (L !== undefined && rank.layer === L) showLayerInList(L);
}

function startJob(kind, params) {
  jobsBar.start(kind, params).catch((e) => { state.note = `エラー: ${e.message}`; updateStatus(); });
}

function recipeChanged(r) {
  state.map.setMarks(r.units.map((u) => ({ layer: u.layer, neuron: u.neuron, dose: u.dose })));
  $("#recipe-count").textContent = r.units.length ? `(${r.units.length})` : "";
  legend();
}

// マップの層正面ビューが変わったとき(ラベルのクリック・選択欄・ジャンプのどれから来ても)に呼ばれる
function onLayerChanged(L) {
  if (L !== null) {
    $("#layer-select").value = String(L);
    if (L !== state.layer || rank.layer !== L) showLayerInList(L);
  }
  $("#legend").classList.toggle("compact", L !== null);     // 正面ビューでは凡例を色の見本だけに(シートを隠さないように)
  updateStatus();
}

// --- views -----------------------------------------------------------------------------------------------
async function setView(v) {
  state.view = v;
  for (const x of document.querySelectorAll("#views button")) x.classList.toggle("on", x.dataset.view === v);
  const ph = $("#view-placeholder"), lv = $("#layers-view"), ev = $("#embed-view");
  $("#map-wrap").classList.toggle("not-map", v !== "map");
  document.body.classList.toggle("view-other", v !== "map");             // 地図専用の header の部品を薄く
  if (state.map) state.map.setActive(v === "map");
  ph.classList.add("hidden");
  lv.classList.toggle("hidden", v !== "layers");
  ev.classList.toggle("hidden", v !== "embed");
  if (v === "embed") {
    if (!state.model) { ev.innerHTML = `<p class="muted" style="padding:16px">合成の構造には rollout がありません</p>`; return; }
    if (!embedView) {
      embedView = new EmbedView(ev, {
        onOpen: (u) => { openNeuron(u, { focus: false }); embedView.draw(); },
        onCompute: (run) => startJob("embed", { run }),
        getFocus: () => state.focus,
        onChange: () => updateStatus(),
      });
      state.embedView = embedView;                                       // 自動テスト用
    }
    await refreshModel();
    embedView.setRuns(state.model.runs);
    embedView.load(embedView.run);
    return;
  }
  if (v === "layers") {
    if (!state.model) { lv.innerHTML = `<p class="muted" style="padding:16px">合成の構造には layers ビューのデータがありません</p>`; return; }
    if (!layersView) layersView = new LayersView(lv, { onPick: (L) => { setView("map"); state.map.frontView(L); } });
    state.note = "layers ビューを読み込み中(初回は逆引きの cos を全層で計算)";
    updateStatus();
    try { layersView.set(await getJSON("/api/profile"), state.arch); }
    catch (e) { state.note = `エラー: ${e.message}`; updateStatus(); return; }
    state.note = "";
    updateStatus();
  }
}

// "L14 N8949" / "14:8949" / MoE は "L10 E93 N252"
function parseJump(text) {
  const m = text.trim().match(/^L?(\d+)\s*[\s:/,]\s*(?:E(\d+)\s*[\s:/,]\s*)?N?(\d+)$/i);
  if (!m) return null;
  return { layer: +m[1], expert: m[2] !== undefined ? +m[2] : null, neuron: +m[3] };
}

function wireHeader() {
  const sel = $("#layer-select");
  sel.innerHTML = state.arch.layers.map((li) => `<option value="${li.index}">${describeLayer(li.index)}</option>`).join("");
  sel.addEventListener("change", () => state.map.frontView(+sel.value));
  const jump = $("#jump");
  jump.placeholder = isMoE() ? "L10 E0 N0" : "L14 N8949";
  jump.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter") return;
    const j = parseJump(ev.target.value);
    const li = j && state.arch.layers[j.layer];
    const moe = li && li.ffn.startsWith("moe");
    const ok = li && (moe ? j.expert !== null && j.expert < li.n_expert && j.neuron < li.n_ff : j.expert === null && j.neuron < li.n_ff);
    ev.target.style.borderColor = ok ? "" : "#c94fa7";
    if (!ok) return;
    if (moe) {
      state.map.focusNeuron(j.layer, j.expert * li.n_ff + j.neuron);
      history.replaceState(null, "", `#L${j.layer}/E${j.expert}/N${j.neuron}`);
    } else if (state.model) openNeuron({ layer: j.layer, n: j.neuron });
    else state.map.focusNeuron(j.layer, j.neuron);
  });
  $("#reset-view").addEventListener("click", () => { if (state.view !== "map") setView("map"); state.map.resetView(); });
  for (const b of document.querySelectorAll("#views button")) b.addEventListener("click", () => setView(b.dataset.view));
  for (const t of document.querySelectorAll(".panel .toggle[data-target]")) {
    t.addEventListener("click", () => { $(`#${t.dataset.target}`).classList.toggle("collapsed"); state.map.resize(); });
  }
  $("#neuron-wide").addEventListener("click", () => {
    const w = $("#neuron").classList.toggle("wide");
    $("#neuron-wide").textContent = w ? "⇔ 戻す" : "⇔ 広げる";
    state.map.resize();
  });
  for (const b of document.querySelectorAll("#list-tabs button[data-tab]")) b.addEventListener("click", () => listTab(b.dataset.tab));
  $("#color-select").addEventListener("change", (ev) => { if (ev.target.value !== state.overlay.field) applyOverlay(ev.target.value); });
  $("#layout-select").addEventListener("change", (ev) => applyLayout(ev.target.value));
  $("#web-toggle").addEventListener("click", () => {                  // 線: 直接 → web → なし → 直接
    state.linkMode = { direct: "web", web: "off", off: "direct" }[state.linkMode];
    state.web = state.linkMode === "web";
    const b = $("#web-toggle");
    b.textContent = { direct: "線: 直接", web: "線: web", off: "線: なし" }[state.linkMode];
    b.classList.toggle("on", state.linkMode === "web");
    showLinks();
  });
  $("#cos-thr").addEventListener("input", (ev) => {
    state.overlay.threshold = +ev.target.value;
    $("#cos-thr-val").textContent = `|cos| ≥ ${state.overlay.threshold.toFixed(3)}`;
    recolor();
  });
  $("#search-q").addEventListener("keydown", (ev) => { if (ev.key === "Enter") runSearch(); if (ev.key === "Escape") { ev.target.value = ""; closeSearch(); } });
  if (!state.model) { $("#search").classList.add("hidden"); $("#color-select").disabled = true; }
}

async function main() {
  const synthetic = params.get("synthetic");
  if (synthetic) {
    state.arch = syntheticArch(synthetic);
  } else {
    state.model = await getJSON("/api/model");
    state.arch = state.model.arch;
  }
  document.title = `gne · ${state.model ? state.model.meta.name : state.arch.arch}`;
  const t0 = performance.now();
  state.map = new LayerMap($("#map"), state.arch, {
    onHover: showTooltip,
    onClick: (h) => {
      if (h.expert !== undefined || h.shared || !state.model) {
        $("#jump").value = h.expert !== undefined ? `L${h.layer} E${h.expert} N${h.neuron}` : `L${h.layer} N${h.neuron}`;
        return;
      }
      openNeuron({ layer: h.layer, n: h.neuron }, { focus: false });
    },
    onLayer: onLayerChanged,
    onContext: (L, ev) => {
      if (state.model) computeMenu(ev, L, state.arch.layers[L], state.model.coverage[L], { onStart: startJob, arch: state.arch, model: state.model });
    },
    overlay: $("#overlay"),
  });
  state.initMs = performance.now() - t0;
  rank = new RankList($("#rank"), { onOpen: (u) => openNeuron(u) });
  searchList = new SearchList($("#search-list"), { onOpen: (u) => openNeuron(u), onClose: () => { $("#search-q").value = ""; closeSearch(); } });
  if (state.model) {
    state.recipe = new Recipe(state.model.meta.model_id, recipeChanged);
    chat = new ChatPanel({ recipe: state.recipe });
    recipePanel = new RecipePanel($("#recipe-list"), state.recipe, {
      onOpen: (u) => openNeuron(u),
      onChat: () => {
        const u = state.recipe.units[0];
        chat.useRecipe();
        if (!$("#neuron").classList.contains("wide")) $("#neuron-wide").click();        // 3 列は広い方が読みやすい
        const f = state.focus;
        if (f && f.kind === "neuron") openNeuron({ layer: f.layer, n: f.n, tab: "chat" }, { focus: false });
        else openNeuron({ layer: u.layer, n: u.neuron, tab: "chat" });
      },
      unitWords: (L, n) => { const us = unitsCache.peek(L); const u = us && us.byN.get(n); return u ? u.words.slice(0, 3).join(" · ") : ""; },
    });
    firesPanel = new FiresPanel($("#fires-list"), {
      nLayer: state.arch.n_layer, onOpen: (u) => openNeuron(u), onField: applyFiresOverlay,
    });
    jobsBar = new JobsBar($("#jobs-bar"), { onFinished: onJobFinished });
    jobsBar.load();
    recipeChanged(state.recipe);
    recipePanel.render();
  }
  neuron = new NeuronPanel($("#neuron-body"), {
    onRecipe: (L, n, sign) => {
      const cur = state.recipe.has(L, n);
      if (cur && Math.sign(cur.dose) === sign) state.recipe.remove(state.recipe.units.indexOf(cur));   // もう一度押すと外す
      else state.recipe.add(L, n, sign * (cur ? Math.abs(cur.dose) : 1));
      if (!$("#recipe-list").classList.contains("hidden")) recipePanel.render();
    },
    inRecipe: (L, n) => { const u = state.recipe && state.recipe.has(L, n); return u ? u.dose : null; },
    onChat: (el, d) => { chat.setNeuron(d.layer, d.neuron, d); chat.mount(el); },
    hasConnections: () => !!(state.model && state.model.connections),
    onGoto: (u) => openNeuron(u),
    onNamed: (names) => {
      state.model.names = names;
      const f = state.overlay.field;
      if (f === "named") applyOverlay("named"); else if (state.overlay.values.size) recolor();
    },
  });
  wireHeader();
  fillColorSelect();
  legend();
  updateStatus();
  setInterval(updateStatus, 1000);
  state.ready = true;                     // 構造は描けた(ここまでが U0 の初期表示)

  if (!state.model) return;
  loadVocab(`${state.model.meta.model_id}:2`).then((v) => { state.vocab = v; rehover(); });   // hover の語(IndexedDB に置く)
  const first = state.model.coverage.find((c) => c.sweep_runs.length);
  const m = location.hash.match(/^#L(\d+)(?:\/E(\d+))?\/N(\d+)/);
  if (m) {
    const li = state.arch.layers[+m[1]];
    if (li && li.ffn.startsWith("moe")) state.map.focusNeuron(+m[1], +m[2] * li.n_ff + +m[3]);
    else if (li) openNeuron({ layer: +m[1], n: +m[3] });
  } else {
    if (first) expand("list");
    showLayerInList(first ? first.layer : 0);
  }
  await applyOverlay(params.get("color") || (first ? "sweep" : "none"));
  state.overlayReady = true;
}

main().catch((e) => { $("#status").textContent = `エラー: ${e.message}`; console.error(e); state.error = String(e); });
