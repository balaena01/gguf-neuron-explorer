// gne explorer: 画面全体の配線(別冊 §2)。
// U0: 構造の描画、層の選択、ジャンプ、hover の L N。
// U1: 色の重ね合わせ(overlay)、LIST の rank、NEURON(Examples / Details)、hover の語、検索、layers ビュー。
// U2: NEURON の Chat(3 列・reads you)、LIST の fires と recipe、地図のレシピの印、ジョブのバーと計算メニュー。
// CHANGE-04 G: 板の上辺の刻み = ヘッド(hover・クリックで HEAD パネル)、overlay heads:<name> と fires のヘッドの注目、レシピのヘッド。
import { LayerCache, getBin, getJSON, loadVocab } from "./api.js";
import { AgentLink } from "./agent.js";
import { LANG, setLang, t, translateDOM } from "./i18n.js";
import { ChatPanel } from "./chat.js";
import { EmbedView } from "./embed.js";
import { EvalView } from "./eval.js";
import { FiresPanel } from "./fires.js";
import { JobsBar, KIND, computeMenu } from "./jobs.js";
import { LayersView } from "./layers.js";
import { syntheticArch } from "./layout.js";
import { LayerMap } from "./map.js";
import { FIELDS, colorize, colorizeHeads, colorizeHits, emphasisFactor, emphasize, fieldLabel, globalScale, layersWith, legendHTML } from "./overlay.js";
import { HeadPanel } from "./heads.js";
import { NeuronPanel, RankList, SearchList, esc, toksHTML } from "./panels.js";
import { Recipe, RecipePanel } from "./recipe.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);

const state = {
  model: null, arch: null, vocab: null, map: null, view: "map", layer: 0, focus: null, note: "",
  layout: "index", web: false, webHops: 2, links: null, linkMode: "direct",
  overlay: { field: "none", threshold: null, scale: null, values: new Map(), metas: new Map(), hits: null, prev: "none" },
  headFocus: null,                        // HEAD パネルで開いているヘッド { layer, head }
  heads: null,                            // ヘッドの色 { field: "heads:<name>" | "fires_heads", values: Map(層 → Float32Array(n_head)), … }
};
window.gne = state;                       // 自動テスト(playwright)と開発用

const MIXER = { attention: "attention", linear_attention: t("線形注意"), recurrent: t("回帰"), none: t("mixer なし") };
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
    ? `MoE ${li.n_expert}×${li.n_ff} (${t("使用 {n}", { n: li.n_expert_used })})${li.n_ff_shexp ? ` + shared ${li.n_ff_shexp}` : ""}` : t("FFN なし");
  const heads = li.n_head ? ` ${li.n_head}h${li.n_head_kv && li.n_head_kv !== li.n_head ? `/${li.n_head_kv}kv` : ""}` : "";
  return `L${L} ${mixerLabel(li)}${heads} · ${ffn}`;
}

// --- hover の吹き出し(§4.6): L N、名前、sweep の上位語、steer ±、logit ±、今の overlay の値 ---------------------
// ヘッド(CHANGE-04 G): L H、種類、名前、今のヘッドの色の値(対照分析の z と順位、または fires の注目)、レシピの倍率
function headTooltipHTML(h) {
  const L = h.layer, li = state.arch.layers[L], lines = [];
  const name = state.model ? (state.model.names || {})[`${L}:H${h.head}`] : null;
  const kv = li.mixer === "attention" && li.n_head_kv && li.n_head_kv !== li.n_head ? ` · kv ${Math.floor(h.head / (li.n_head / li.n_head_kv))}` : "";
  lines.push(`<span class="t-id">L${L} H${h.head}</span><span class="t-v">${h.kind === "attn" ? "attention" : t("線形注意")}${kv}</span>${name ? `<span class="t-name">${esc(name)}</span>` : ""}`);
  const ho = state.heads, v = ho && ho.values.get(L);
  if (ho && ho.field.startsWith("heads:")) {
    const r = ho.info.get(`${L}:${h.head}`);
    if (r) lines.push(`<span class="t-k">color</span><span class="t-v" style="margin:0">z ${fmt(r.z, 2)} · #${r.rank}/${ho.n} · null ${fmt(r.null_pct, 2)}${
      r.trigger_mass !== null && r.trigger_mass !== undefined ? ` · trigger ${fmt(r.trigger_mass, 3)}` : ""}${r.dir_cos !== null && r.dir_cos !== undefined ? ` · dir_cos ${fmt(r.dir_cos, 3)}` : ""}</span>`);
  } else if (ho && v && Number.isFinite(v[h.head])) {
    lines.push(`<span class="t-k">color</span><span class="t-v" style="margin:0">${t("注目 {v}({tok} へ)", { v: fmt(v[h.head], 3), tok: esc(ho.label || "") })}</span>`);
  }
  const r = state.recipe && state.recipe.hasHead(L, h.head);
  if (r) lines.push(`<span class="t-k">recipe</span><span class="t-v" style="margin:0">×${fmt(r.k, 2)}</span>`);
  if (state.model) lines.push(`<span class="t-wait">${t("クリックで HEAD パネル")}</span>`);
  return lines.join("<br>");
}

function tooltipHTML(h) {
  if (h.head !== undefined) return headTooltipHTML(h);
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
  if (waiting) lines.push(`<span class="t-wait">${t("読み込み中…")}</span>`);
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
  const where = map && map.front !== null ? `L${map.front}` : t("{n} 層", { n: state.arch.n_layer });   // 層の説明は選択欄にある
  const flash = state.flash && state.flash.until > Date.now() ? state.flash.text : "";        // ジョブの結果などの一時的な知らせ
  const sp = state.model && state.model.speed;                  // GPU / CPU の層と推定の速さ(CHANGE-03 C)。CPU の層があるときだけ出す
  const off = sp && sp.cpu_layers > 0 ? ` · ${t("GPU {g} 層 / CPU {c} 層、約 {v} tok/s", { g: sp.gpu_layers, c: sp.cpu_layers, v: Math.round(sp.decode_tok_s) })}` : "";
  $("#status-main").textContent = `${m.name} · ${where}${off}${flash ? ` · ${flash}` : ""}${state.note ? ` · ${state.note}` : ""}`;
  $("#status-main").title = $("#status-main").textContent + (sp ? ` · ${t("GPU {g} 層 / CPU {c} 層、約 {v} tok/s", { g: sp.gpu_layers, c: sp.cpu_layers, v: Math.round(sp.decode_tok_s) })}` : "");
  $("#perf").textContent = state.view === "embed"
    ? (embedView && embedView.data ? `${embedView.data.points.length.toLocaleString()} rollouts (embed)` : "— (embed)")
    : state.view === "layers" ? "(layers)"                                // 層の数は左にある
    : state.view === "eval" ? "(eval)"
    : `${t("{n} 点", { n: (map?.nPoints ?? 0).toLocaleString() })} · ${map && map.fps !== null ? map.fps.toFixed(0) : "—"} fps`;
}

function legend() {
  const kinds = new Set(state.arch.layers.map((li) => li.mixer));
  const o = state.overlay;
  const it = (...xs) => `<span class="it">${xs.join("")}</span>`, sw = (c) => `<span class="sw" style="background:${c}"></span>`;
  const itl = (x, cls = "") => `<span class="it long ${cls}">${x}</span>`;    // 長い説明は折り返してよい(地図の幅を越えない)
  const parts = [];
  if (kinds.has("attention")) parts.push(it(sw("#9fb3cf"), t("attention(太い縁)")));
  if (kinds.has("linear_attention")) parts.push(it(sw("#3a4656"), t("線形注意(細い縁)")));
  if (kinds.has("recurrent")) parts.push(it(sw("#3a4656"), t("回帰(細い縁)")));
  const floors = [...o.metas.values()].map((m) => m.floor);
  const color = o.field === "none" ? it(sw("#10141b"), t("データ無し"))
    : `<b>${esc(fieldLabel(o.field))}</b> ${legendHTML(o.field, o.scale || {}, { threshold: o.threshold, nHits: o.hits ? o.hits.length : 0, floors })}`;
  const fh = state.heads && state.heads.field === "fires_heads"                                  // fires のヘッドの注目(点の色と一緒に出す)
    ? `<div><b>${esc(fieldLabel("fires_heads"))}</b> ${legendHTML("fires_heads", state.heads.scale)}</div>` : "";
  const litWhat = o.field.startsWith("heads:") ? t("光るヘッドの数(最多の層で最も明るい)") : t("光る点の数(最多の層で最も明るい)");
  const frame = o.field === "none" ? "" : `<div>${it(`${t("シートの枠")}: `, sw("#252b33"), t("データ無し"))}${it(sw("#5e4a26"), "→", sw("#f0b850"), litWhat)}` +
    (o.factor < 1 ? itl(t("(光る点が全体の {p}% あるので、点は一律に暗くし、数は枠で示す)", { p: Math.round(0.02 / o.factor * 100) })) : "") +
    (o.weak && o.weak.size ? itl(t("({layers} は活性統計が無く文章内 p99 で割ったので、枠の比較から外した)", { layers: [...o.weak].map((L) => `L${L}`).join(", ") })) : "") + "</div>";
  const ring = (c) => `<span class="sw ring" style="border-color:${c}"></span>`;
  const ag = agent ? agent.state : null;
  const hbox = (c) => `<span class="sw hbox" style="border-color:${c}"></span>`;
  const marks = (state.recipe && state.recipe.neurons().length ? [it(ring("#f2d24b"), t("レシピ +")), it(ring("#c94fa7"), t("レシピ −(大きさ = |dose|)"))] : [])
    .concat(state.focus ? [it(ring("#ffffff"), t("開いているニューロン"))] : [])
    .concat(state.recipe && state.recipe.heads().length ? [it(hbox("#f2d24b"), hbox("#c94fa7"), t("レシピのヘッド(黄 = 強める ×k > 1、赤紫 = 弱める・消す ×k < 1)"))] : [])
    .concat(state.headFocus ? [it(hbox("#ffffff"), t("開いているヘッド"))] : [])
    .concat(ag && ag.highlights.length ? [it(`<span class="sw ring dash" style="border-color:#6fe3ff"></span>`, `${t("エージェントの強調")}${ag.label ? ` (${esc(ag.label)})` : ""}`)] : [])
    .concat(ag && ag.notes.length ? [itl(`<span class="sw ring" style="border-color:#6fe3ff;width:9px;height:9px"></span><span class="ag-note-sw">agent ✎</span>` +
      t("エージェントの注釈(小さな輪。札は 28 字まで、マウスを当てると全文、クリックで開く)"))] : [])
    .concat(ag && ag.highlights.length + ag.notes.length > 30 ? [itl(t("(札は 30 枚まで。残りは輪だけで、全部は header の agent の一覧)"))] : []);
  const line = (c) => `<span class="sw line" style="background:${c}"></span>`;
  const links = state.links ? `<div>${it(state.links.web ? t("web({h} 段、{n} 本): ", { h: state.webHops, n: state.links.n }) : t("配線(上流・下流 {n} 本): ", { n: state.links.n }))}${
    it(line("#f2d24b"), t("黄 = 読み手(線の先)を + に押す"))}${it(line("#ff70d0"), t("赤紫 = 読み手を − に押す"))}${
    itl(state.links.web ? t("色 = 各辺の符号(経路の積ではない)、明るさ = z(表示中の最大に対する相対)、暗い線 = web の端(この先はたどっていない)") +
      (state.map && state.map.front !== null ? t("。正面ビューの札は 1 段目だけ(全体ビューで全体を見る)") : "")
      : t("明るさ = z(表示中の最大に対する相対)。正面ビューでは右 = 下流、左(破線)= 上流"), "lk")}</div>` : "";
  const moe = isMoE() ? t("、MoE は専門家ごとの小さな正方形") : "";
  const lay = state.layout && state.layout !== "index"
    ? t("シート内の並び = {k} 配置(UMAP)。近くにあっても似ているのは弱い傾向だけ", { k: state.layout }) +
      (state.layoutMissing && state.layoutMissing.length ? t("(未計算の {n} 層は番号順)", { n: state.layoutMissing.length }) : "")
    : t("シート内の並びはニューロン番号順(近くにあっても似ているとは限らない)");
  const st = state.model && state.arch.layers.some((li) => li.head_dim)
    ? t("各層 = 手前の板(mixer、上辺の刻み = ヘッド。マウスを当てると番号、クリックで HEAD パネル)+ 奥のシート(MLP の全ニューロン{moe})、下の線 = 残差", { moe })
    : t("各層 = 手前の板(mixer、上辺の刻みが head の数)+ 奥のシート(MLP の全ニューロン{moe})、下の線 = 残差", { moe });
  $("#legend").innerHTML = `<div>${color}</div>${fh}${frame}<div>${parts.join("")}</div>${marks.length ? `<div>${marks.join("")}</div>` : ""}${links}` +
    `<div class="note st">${st}</div>` +
    `<div class="note">${lay}</div>`;
}

// --- overlay(§4.3) --------------------------------------------------------------------------------------
let overlayToken = 0;

function fillColorSelect() {
  const sel = $("#color-select");
  const names = ["none", ...Object.keys(FIELDS).filter((k) => k !== "none"), ...(state.model ? state.model.directions.map((d) => `cos:${d}`) : [])];
  const heads = state.model ? state.model.heads || [] : [];                // 保存済みのヘッドの対照分析(CHANGE-04 G)
  sel.innerHTML = names.map((k) => {
    const n = state.model && k !== "none" ? layersWith(k, state.model).length : 0;
    const dis = k !== "none" && !n;
    return `<option value="${esc(k)}"${dis ? " disabled" : ""}>${esc(fieldLabel(k))}${k === "none" ? "" : dis ? ` (${t("データ無し")})` : ` (${t("{n} 層", { n })})`}</option>`;
  }).join("") + heads.map((h) => `<option value="${esc(`heads:${h.name}`)}">${esc(fieldLabel(`heads:${h.name}`))} (${t("{n} ヘッド", { n: h.n_heads })})</option>`).join("");
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
  state.model.heads = m.heads;
}

// --- ヘッドの色(CHANGE-04 G): 板の上辺の帯。heads:<name> は対照分析の z、fires_heads は fires で選んだトークンへの注目 ----------
const headsCache = new LayerCache((name) => getJSON(`/api/heads/${encodeURIComponent(name)}`));
const firesHeadsCache = new LayerCache(async (fid) => {
  const r = await getBin(`/api/fires/${fid}/heads`);
  const [n, T] = r.shape.split(",").map(Number);
  return { v: new Float32Array(r.buf), n, T, meta: r.meta };
});

// 今の state.heads で帯を塗る。層ごとの光るヘッドの数を返す
function colorHeads() {
  const ho = state.heads, cols = new Map(), lit = new Map();
  if (!ho) { state.map.setHeadColors(null); return lit; }
  for (const [L, v] of ho.values) {
    const c = colorizeHeads(ho.field === "fires_heads" ? "fires_heads" : "heads", v,
                            { ...ho.scale, nullPct: ho.nulls ? ho.nulls.get(L) : null });
    cols.set(L, c);
    lit.set(L, c.lit);
  }
  state.map.setHeadColors(cols);
  return lit;
}

function clearHeads() {
  if (!state.heads) return;
  state.heads = null;
  state.map.setHeadColors(null);
}

// overlay heads:<name>: 対照分析の全ヘッドの z。点(ニューロン)は塗らない
async function applyHeadsOverlay(field, tok) {
  const name = field.slice(6);
  headsCache.done.delete(name);                                   // 作り直したかもしれないので取り直す
  const d = await headsCache.get(name);
  if (tok !== overlayToken) return null;
  const values = new Map(), nulls = new Map(), info = new Map(), lit = [];
  d.rows.forEach((r, i) => {
    const li = state.arch.layers[r.layer];
    if (!li || !(r.head < li.n_head)) return;
    if (!values.has(r.layer)) { values.set(r.layer, new Float32Array(li.n_head).fill(NaN)); nulls.set(r.layer, new Float32Array(li.n_head).fill(NaN)); }
    values.get(r.layer)[r.head] = r.z ?? NaN;
    nulls.get(r.layer)[r.head] = r.null_pct ?? NaN;
    info.set(`${r.layer}:${r.head}`, { ...r, rank: i + 1 });
    if ((r.null_pct ?? 0) >= 0.95 && Number.isFinite(r.z)) lit.push(r.z);
  });
  state.heads = { field, values, nulls, info, n: d.rows.length,
                  scale: { lo: lit.length ? Math.min(...lit) : 0, hi: lit.length ? Math.max(...lit) : 1 } };
  return colorHeads();
}

// fires のヘッドの注目(fires_heads): 選んだトークンへの、後ろの位置からの注目の平均。全トークン(-1)は各ヘッドの最大
async function applyFiresHeads(fid, tok, label) {
  const tokn = overlayToken;
  let x;
  try { x = await firesHeadsCache.get(fid); }
  catch (e) { state.note = `${t("エラー")}: ${e.message}`; updateStatus(); clearHeads(); legend(); return; }
  if (tokn !== overlayToken) return;
  const values = new Map();
  let mx = 0;
  x.meta.heads.forEach(([L, h], i) => {
    if (!values.has(L)) values.set(L, new Float32Array(state.arch.layers[L].n_head).fill(NaN));
    const row = x.v.subarray(i * x.T, (i + 1) * x.T);
    let v = NaN;
    if (tok === -1) { for (let p = 1; p < x.T; p++) if (Number.isFinite(row[p]) && !(row[p] <= v)) v = row[p]; }
    else if (tok < x.T) v = row[tok];
    values.get(L)[h] = v;
    if (Number.isFinite(v)) mx = Math.max(mx, v);
  });
  state.heads = { field: "fires_heads", values, scale: { max: mx }, label };
  colorHeads();
  legend();
  rehover();
}

// 枠の明るさ = その層で光っている点の数(いちばん多い層を 1 とする、比例)。全体ビューでは上位 3 層に札も出す。
// 方向が強い層などが、点の大きさに頼らず一目で分かるように(F40)
// weak: 比べにくい層(fires で活性統計が無く文章内 p99 で割った層など)。最大と札からは外し、枠は最も暗くする
function showLayerCounts(lit, unit = "点", weak = new Set()) {           // unit: 点 / 件
  const mx = Math.max(1, ...[...lit].filter(([L]) => !weak.has(L)).map(([, c]) => c));
  state.map.setFrameColors(new Map([...lit].map(([L, c]) => [L, weak.has(L) ? 0 : c / mx])));
  const top = [...lit].filter(([L, c]) => c > 0 && !weak.has(L)).sort((a, b) => b[1] - a[1]).slice(0, 3);
  state.map.setBadges(top.map(([L, c]) => ({ layer: L, text: `L${L} · ${t(`{c} ${unit}`, { c })}` })));
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
  clearHeads();
  if (field.startsWith("heads:") && state.model) {                       // ヘッドの対照分析: 板の上辺の帯だけを塗る
    o.values = new Map(); o.metas = new Map(); o.factor = 1;
    state.map.resetColors(); state.map.setStructureDim(0.35);
    state.note = t("{f} を読み込み中 {a}/{b}", { f: fieldLabel(field), a: 0, b: 1 });
    updateStatus();
    await refreshModel();
    if (tok !== overlayToken) return;
    fillColorSelect();
    let lit;
    try { lit = await applyHeadsOverlay(field, tok); }
    catch (e) { state.note = `${t("エラー")}: ${e.message}`; updateStatus(); return; }
    if (!lit) return;
    o.scale = state.heads.scale;
    state.note = "";
    showLayerCounts(lit, "ヘッド");
    legend();
    o.ms = performance.now() - t0;
    updateStatus();
    rehover();
    return;
  }
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
  state.note = t("{f} を読み込み中 {a}/{b}", { f: fieldLabel(field), a: 0, b: Ls.length });
  updateStatus();
  for (const L of Ls) fieldCache.done.delete(`${L}|${field}`);   // 計算が進んでいるかもしれないので取り直す
  const res = await Promise.all(Ls.map((L) => fieldCache.get(`${L}|${field}`).then((r) => {
    done++;
    if (tok === overlayToken) { state.note = t("{f} を読み込み中 {a}/{b}", { f: fieldLabel(field), a: done, b: Ls.length }); updateStatus(); }
    return r;
  }))).catch((e) => { state.note = `${t("エラー")}: ${e.message}`; updateStatus(); return null; });
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
  clearHeads();
  map.resetColors();
  const byL = new Map();
  for (const h of hits) { if (!byL.has(h.layer)) byL.set(h.layer, []); byL.get(h.layer).push(h.neuron); }
  for (const [L, ns] of byL) map.setLayerColors(L, colorizeHits(state.arch.layers[L].n_ff, ns));
  map.setStructureDim(0.35);
  showLayerCounts(new Map([...byL].map(([L, ns]) => [L, ns.length])), "件");
  fillColorSelect();
  legend();
}

// fires の値で色を付ける(全層の値を 1 回で取る)。tok = -1 は全トークンでの |最大|。
// heads なら板の上辺のヘッドも、選んだトークンへの注目で塗る(fires_heads)
async function applyFiresOverlay(fid, tok, label, thr, heads = false) {
  const tokn = ++overlayToken, o = state.overlay;
  if (!o.field.startsWith("fires:") && !o.field.startsWith("search:")) o.prev = o.field;
  state.note = t("fires の値を読み込み中");
  updateStatus();
  let r;
  try { r = await getBin(`/api/fires/${fid}/field`, { tok }); }
  catch (e) { state.note = `${t("エラー")}: ${e.message}`; updateStatus(); return; }
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
  if (heads) applyFiresHeads(fid, tok, label); else { clearHeads(); legend(); }
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
  if (kind !== "index") { state.note = t("配置 {k} を読み込み中", { k: kind }); updateStatus(); }
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
  } catch (e) { state.note = `${t("エラー")}: ${e.message}`; updateStatus(); return; }
  if (state.linksKey !== key) return;
  state.links = { web: state.web, n: edges.length };
  state.map.setLinks(edges, { focus: { layer: f.layer, n: f.n } });
  legend();
}

// --- 検索(header、lens / words / name)。結果は LIST の「検索」タブに強い順で出し、当たりをマップで光らせる ---------------
function listTab(tab) {
  for (const b of document.querySelectorAll("#list-tabs button[data-tab]")) b.classList.toggle("on", b.dataset.tab === tab);
  for (const [name, id] of [["rank", "#rank"], ["search", "#search-list"], ["fires", "#fires-list"], ["recipe", "#recipe-list"]]) {
    $(id).classList.toggle("hidden", tab !== name);
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
let rank, neuron, layersView, searchList, chat, firesPanel, recipePanel, jobsBar, embedView, evalView, agent, headPanel;

function expand(id) {
  const p = $(`#${id}`);
  if (p.classList.contains("collapsed")) { p.classList.remove("collapsed"); state.map.resize(); }
}

// NEURON と HEAD は同列の右パネル。画面が狭いときは、開く方でないもう一方をたたむ(地図の幅を残す)
function expandRight(id) {
  const other = id === "neuron" ? "head" : "neuron";
  if (window.innerWidth < 1700 && !$(`#${other}`).classList.contains("collapsed")) $(`#${other}`).classList.add("collapsed");
  expand(id);
  state.map.resize();
}

function openNeuron({ layer, n, kind = "neuron", run = null, tab = null }, { focus = true } = {}) {
  const first = !state.focus;
  state.shownNote = agent ? agent.noteOf(layer, n) : null;
  state.focus = { layer, n, kind };
  if (first) legend();
  expandRight("neuron");
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

// HEAD パネルでヘッドを開く(CHANGE-04 G)。視点は動かさず、板の上辺の刻みに白い印
function openHead({ layer, head, tab = null }) {
  const li = state.arch.layers[layer];
  if (!li || !li.head_dim || !(head >= 0 && head < li.n_head)) return false;
  state.headFocus = { layer, head };
  expandRight("head");
  headPanel.open({ layer, head, tab });
  state.map.markHead(layer, head);
  history.replaceState(null, "", `#L${layer}/H${head}`);
  $("#jump").value = `L${layer} H${head}`;
  legend();
  return true;
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
  state.flash = { text: `${t({ done: "完了", error: "失敗", cancelled: "中止" }[j.status])}: ${what}`, until: Date.now() + 10000 };   // 結果を先に(状態表示は狭いので)
  updateStatus();
  if (j.status !== "done" || !state.model) return;
  if (L !== undefined) { unitsCache.done.delete(L); hoverCache.done.delete(L); }
  await refreshModel();
  fillColorSelect();
  if (j.kind === "layout") { layoutCache.clear(); if (state.layout === j.params.kind) applyLayout(state.layout); }
  if (j.kind === "connections" && state.focus) { showLinks(); neuron.render(); }
  if (j.kind === "eval" && evalView) evalView.refresh(j.result && j.result.name);
  if (j.kind === "heads_contrast") {                                     // ヘッドの対照分析: overlay の選択肢と、開いているヘッドの Contrast
    headsCache.clear();
    if (headPanel && state.headFocus) { headPanel.sums.clear(); headPanel.open(state.headFocus); }
  }
  if (j.kind === "embed" && embedView) { embedView.setRuns(state.model.runs); if (embedView.run === j.params.run) embedView.load(j.params.run); }
  const f = state.overlay.field;
  if (f !== "none" && !f.startsWith("search:") && !f.startsWith("fires:")) applyOverlay(f);
  if (L !== undefined && rank.layer === L) showLayerInList(L);
}

function startJob(kind, params) {
  jobsBar.start(kind, params).catch((e) => { state.note = `${t("エラー")}: ${e.message}`; updateStatus(); });
}

function recipeChanged(r) {
  state.map.setMarks(r.neurons().map((u) => ({ layer: u.layer, neuron: u.neuron, dose: u.dose })));   // 方向のユニットは印なし
  state.map.setHeadMarks(r.heads().map((u) => ({ layer: u.layer, head: u.head, k: u.k })));          // ヘッドは板の上辺の刻みに
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
  const ph = $("#view-placeholder"), lv = $("#layers-view"), ev = $("#embed-view"), xv = $("#eval-view");
  $("#map-wrap").classList.toggle("not-map", v !== "map");
  document.body.classList.toggle("view-other", v !== "map");             // 地図専用の header の部品を薄く
  if (state.map) state.map.setActive(v === "map");
  ph.classList.add("hidden");
  lv.classList.toggle("hidden", v !== "layers");
  ev.classList.toggle("hidden", v !== "embed");
  xv.classList.toggle("hidden", v !== "eval");
  if (v === "eval") {
    if (!state.model) { xv.innerHTML = `<p class="muted" style="padding:16px">${t("合成の構造には eval がありません")}</p>`; return; }
    if (!evalView) {
      evalView = new EvalView(xv, { recipe: state.recipe, onRun: (p) => startJob("eval", p) });
      state.evalView = evalView;                                         // 自動テスト用
    }
    await evalView.refresh();
    return;
  }
  if (v === "embed") {
    if (!state.model) { ev.innerHTML = `<p class="muted" style="padding:16px">${t("合成の構造には rollout がありません")}</p>`; return; }
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
    if (!state.model) { lv.innerHTML = `<p class="muted" style="padding:16px">${t("合成の構造には layers ビューのデータがありません")}</p>`; return; }
    if (!layersView) layersView = new LayersView(lv, { onPick: (L) => { setView("map"); state.map.frontView(L); } });
    state.note = t("layers ビューを読み込み中(初回は逆引きの cos を全層で計算)");
    updateStatus();
    try { layersView.set(await getJSON("/api/profile"), state.arch); }
    catch (e) { state.note = `${t("エラー")}: ${e.message}`; updateStatus(); return; }
    state.note = "";
    updateStatus();
  }
}

// "L14 N8949" / "14:8949" / MoE は "L10 E93 N252" / ヘッドは "L21 H7"・"21:H7"
function parseJump(text) {
  const h = text.trim().match(/^L?(\d+)\s*[\s:/,]\s*H(\d+)$/i);
  if (h) return { layer: +h[1], head: +h[2] };
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
    if (j && j.head !== undefined) {                                     // ヘッドは HEAD パネルで(視点は動かさない)
      const ok = !!state.model && openHead({ layer: j.layer, head: j.head });
      ev.target.style.borderColor = ok ? "" : "#c94fa7";
      return;
    }
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
  for (const tg of document.querySelectorAll(".panel .toggle[data-target]")) {
    tg.addEventListener("click", () => { $(`#${tg.dataset.target}`).classList.toggle("collapsed"); state.map.resize(); });
  }
  $("#neuron-wide").addEventListener("click", () => {
    const w = $("#neuron").classList.toggle("wide");
    $("#neuron-wide").textContent = t(w ? "⇔ 戻す" : "⇔ 広げる");
    state.map.resize();
  });
  $("#head-wide").addEventListener("click", () => {
    const w = $("#head").classList.toggle("wide");
    $("#head-wide").textContent = t(w ? "⇔ 戻す" : "⇔ 広げる");
    state.map.resize();
    if (headPanel && headPanel.res && headPanel.tab === "pattern") headPanel.drawOut();   // ヒートマップを新しい幅で描き直す
  });
  for (const b of document.querySelectorAll("#list-tabs button[data-tab]")) b.addEventListener("click", () => listTab(b.dataset.tab));
  $("#color-select").addEventListener("change", (ev) => { if (ev.target.value !== state.overlay.field) applyOverlay(ev.target.value); });
  $("#layout-select").addEventListener("change", (ev) => applyLayout(ev.target.value));
  $("#web-toggle").addEventListener("click", () => {                  // 線: 直接 → web → なし → 直接
    state.linkMode = { direct: "web", web: "off", off: "direct" }[state.linkMode];
    state.web = state.linkMode === "web";
    const b = $("#web-toggle");
    b.textContent = t({ direct: "線: 直接", web: "線: web", off: "線: なし" }[state.linkMode]);
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
  translateDOM();                                                        // index.html の文言を今の言語に
  for (const b of document.querySelectorAll("#lang button")) {         // 右上の JP / EN
    b.classList.toggle("on", b.dataset.lang === LANG);
    b.addEventListener("click", () => { if (b.dataset.lang !== LANG) setLang(b.dataset.lang); });
  }
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
      if (h.head !== undefined) {                                        // 板の上辺の刻み = ヘッド
        if (state.model) openHead({ layer: h.layer, head: h.head }); else $("#jump").value = `L${h.layer} H${h.head}`;
        return;
      }
      if (h.expert !== undefined || h.shared || !state.model) {
        $("#jump").value = h.expert !== undefined ? `L${h.layer} E${h.expert} N${h.neuron}` : `L${h.layer} N${h.neuron}`;
        return;
      }
      openNeuron({ layer: h.layer, n: h.neuron }, { focus: false });
    },
    onLayer: onLayerChanged,
    onOpen: (u) => openNeuron(u),
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
    state.chat = chat;                                                   // 自動テスト用
    recipePanel = new RecipePanel($("#recipe-list"), state.recipe, {
      onOpen: (u) => openNeuron(u),
      onChat: () => {
        const u = state.recipe.neurons()[0];
        chat.useRecipe();
        if (!$("#neuron").classList.contains("wide")) $("#neuron-wide").click();        // 3 列は広い方が読みやすい
        const f = state.focus;
        if (f && f.kind === "neuron") openNeuron({ layer: f.layer, n: f.n, tab: "chat" }, { focus: false });
        else if (u) openNeuron({ layer: u.layer, n: u.neuron, tab: "chat" });
        else openNeuron({ layer: state.layer, n: 0, tab: "chat" });                  // ablate だけのレシピ: チャットの置き場所に

      },
      unitWords: (L, n) => { const us = unitsCache.peek(L); const u = us && us.byN.get(n); return u ? u.words.slice(0, 3).join(" · ") : ""; },
      directions: () => (state.model ? state.model.directions : []),
      onOpenHead: (u) => openHead(u),
      headKind: (L) => mixerLabel(state.arch.layers[L]),
    });
    headPanel = new HeadPanel($("#head-body"), {
      onRecipe: (L, h, k) => {
        const cur = state.recipe.hasHead(L, h);
        if (cur && cur.k === k) state.recipe.removeHead(L, h); else state.recipe.addHead(L, h, k);   // 同じ倍率でもう一度押すと外す
        if (!$("#recipe-list").classList.contains("hidden")) recipePanel.render();
      },
      inRecipe: (L, h) => { const u = state.recipe.hasHead(L, h); return u ? u.k : null; },
      onNamed: (names) => { state.model.names = names; },
      onOverlay: (field) => { if (state.view !== "map") setView("map"); applyOverlay(field); },
      onGoto: (u) => openHead(u),
    });
    state.headPanel = headPanel;                                         // 自動テスト用
    firesPanel = new FiresPanel($("#fires-list"), {
      nLayer: state.arch.n_layer, onOpen: (u) => openNeuron(u), onField: applyFiresOverlay,
    });
    jobsBar = new JobsBar($("#jobs-bar"), { onFinished: onJobFinished });
    $("#agent-btn").classList.remove("hidden");
    agent = new AgentLink({
      button: $("#agent-btn"), pop: $("#agent-pop"),
      onState: (s, ev) => {
        state.map.setAgent(s);
        legend();
        // 知らせはエージェントの新しい操作で必ず上書きする。開いたときに届く「今の状態」は知らせにしない
        if (ev) state.flash = { text: agent.lastText(), until: Date.now() + 10000 };
        updateStatus();
        const f = state.focus, nt = f ? agent.noteOf(f.layer, f.n) : null;   // 開いているニューロンの注釈が変わったら描き直す
        if (f && nt !== state.shownNote) { state.shownNote = nt; neuron.render(); }
      },
      onOpen: (u) => { if (state.view !== "map") setView("map"); openNeuron(u); },
    });
    state.agent = agent;
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
    agentNote: (L, n) => (agent ? agent.noteOf(L, n) : null),
    onEval: async (name) => { await setView("eval"); if (evalView) await evalView.refresh(name); },
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
  const m = location.hash.match(/^#L(\d+)(?:\/E(\d+))?\/N(\d+)/), mh = location.hash.match(/^#L(\d+)\/H(\d+)/);
  if (mh) {
    if (first) expand("list");
    showLayerInList(+mh[1]);
    openHead({ layer: +mh[1], head: +mh[2] });
  } else if (m) {
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

main().catch((e) => { $("#status").textContent = `${t("エラー")}: ${e.message}`; console.error(e); state.error = String(e); });
