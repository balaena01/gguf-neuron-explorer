// direction ビューの「鍵(keyed)」(CHANGE-07 G): 鍵つきの書き込み(mode: keyed)の鍵を対照集合から作り(direction_key のジョブ、
// gne direction key)、作成に使っていない例で状態を分けているかを確かめる(direction_key_check のジョブ、gne direction key-check)。
// 1. 鍵を作る: 対照集合は「ファイルを選ぶ」(configs/contrast/ と <root>/contrasts/ の YAML)か「JSON を書く」(ひな形 configs/keyed/ を
//    選ぶと入力欄に入り、手で直せる。head_pipeline と同じく、表を spec としてジョブに渡し、ジョブが <root>/contrasts/<name>.yaml に書く)。
//    層・名前・positions(before:<正規表現> なら推論時と同じ文脈の返事の「書き込みたい位置」= 作り方 (b))・q_mode・subset・
//    比較用の鍵の本数。ボタンは 3 つ:
//    - 「方向から key-check まで」(direction_key_pipeline のジョブ): 方向(gne direction --positions response と同じ。同じ名前の方向が
//      あれば使い回し、「方向を作り直す」で作り直す)→ 鍵 → 作成と反対側の subset での key-check を続けて回す。
//    - 「鍵を作る」(direction_key のジョブ): 鍵だけ。
//    - 「専門家を選ぶ材料を出す」(MoE の層があるモデルだけ、direction_key_experts のジョブ): 層ごとの専門家の選ばれ方の表。行の
//      「この専門家で鍵を作る」で、書き込み先をその専門家・層をその層にした鍵づくりの欄にする。
//    終わると層ごとの kn・gain・q_rank・作成に使った例での AUC と警告。行の「+ レシピ」で keyed のユニット(L:ffn:keyed:aα@方向#鍵)を
//    レシピに足し、「key-verify」で LoRA が式どおりに書くかを確かめる(direction_key_verify のジョブ、V30)。
// 2. key-check: 層ごとの例単位・トークン単位の AUC、key-check の選択性(例ごとの ρ の平均の差、書き込みたい位置 − 書き込みたくない位置)と
//    比較用の鍵の p95、○/×(比較用の鍵を超えるか)、保留集合か(held_out)と重なりの警告。層ごとの鍵の反応の強さ ρ = κᵀx の分布の
//    ヒストグラム(書き込みたい位置・書き込みたくない位置・dont_touch の群。群ごとの割合で重ねる SVG)。
// 3. 保存した鍵の一覧(<root>/keys/<model_id>/)。「見る」で作成の記録、「確かめる」で key-check の欄に入れる。
import { getJSON } from "./api.js";
import { LANG, t } from "./i18n.js";
import { esc } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const short = (s, n = 60) => { const x = String(s ?? ""); return x.length > n ? `${x.slice(0, n - 1)}…` : x; };
const NAME_RE = /^[\w.][\w.-]*$/;
const SPEC_NAME_RE = /^[\w.-]+$/;
const OPP = { even: "odd", odd: "even", all: "all" };            // key-check は作成と反対側の subset で
const mark = (b) => (b === true ? `<span class="kx-ok">○</span>` : b === false ? `<span class="kx-ng">×</span>` : "—");
const DRAFT = "gne.keyed.draft";                                 // 入力欄の下書き(このブラウザだけ)
const SRC = "gne.keyed.src";                                     // 対照集合の入れ方(file / json)
const SETTINGS = ["layers", "target", "q_mode", "subset", "pseudo", "max_tokens", "alpha"];   // 表の keyed に書ける設定
const DT_KEYS = ["name", "system", "prompts", "replies", "positions"];
// 表の keyed → 欄(target は専門家の番号と分けて入れる、alpha は「+ レシピ」の α の既定)
const FIELDS = [["layers", "#kx-layers"], ["q_mode", "#kx-qmode"], ["subset", "#kx-subset"], ["pseudo", "#kx-pseudo"], ["max_tokens", "#kx-mt"]];

// 鍵の target(ffn / attn / shared / expert:e)→ レシピのユニットの kind と専門家
export function targetKind(target) {
  const m = String(target || "ffn").match(/^expert:(\d+)$/);
  return m ? { kind: "expert", expert: +m[1] } : { kind: String(target || "ffn"), expert: null };
}

// 層の指定("8,14,20"・"10-16"・"8, 14")→ 数のリスト。読めなければ null
export function parseLayers(s) {
  const out = [];
  for (const part of String(s || "").split(",").map((x) => x.trim()).filter(Boolean)) {
    const m = part.match(/^(\d+)(?:\s*-\s*(\d+))?$/);
    if (!m) return null;
    const a = +m[1], b = m[2] === undefined ? a : +m[2];
    if (b < a) return null;
    for (let L = a; L <= b; L++) out.push(L);
  }
  return out.length ? [...new Set(out)].sort((x, y) => x - y) : null;
}

// positions の書き方を確かめる(gne/keyed_build.py の parse_positions と同じ決まり)。エラーの文か null と、警告の文か null
function checkPositions(v, key) {
  const s = String(v);
  const m = s.trimStart().match(/^(not-before|before):([\s\S]*)$/);
  if (!m) {
    return /^(response|assistant|last|last:[1-9]\d*)$/.test(s.trim()) ? [null, null]
      : [t("{k} は response / assistant / last / last:K / before:<正規表現> / not-before:<正規表現> です({v})", { k: key, v: s }), null];
  }
  if (!m[2]) return [t("{k} に正規表現がありません", { k: key }), null];
  try { new RegExp(m[2], "u"); } catch (e) {
    return [null, t("{k} の正規表現をブラウザでは読めませんでした(サーバの Python の正規表現で読みます): {m}", { k: key, m: e.message })];
  }
  return [null, null];
}

// 片側(方向の A / B)の作り方の短い説明
function sideText(d, s) {
  const sys = d[`system_${s}`], rec = d[`recipe_${s}`];
  const parts = [];
  if (rec !== undefined && rec !== null) parts.push(Array.isArray(rec) ? t("レシピ(ユニット {n} 個: {u})", { n: rec.length, u: rec.slice(0, 3).join(", ") }) : t("レシピ {r}", { r: typeof rec === "string" ? rec : "…" }));
  parts.push(sys ? `system「${short(sys)}」` : t("system なし"));
  return parts.join(" + ");
}

// 入力欄の文字列(鍵の対照集合の JSON)を確かめる。{ spec, errors, warns, summary }。サーバ(keyed_build.read_key_spec と
// keyed_pipeline.clean_spec)と同じ決まり
export function checkKeySpec(text) {
  const out = { spec: null, errors: [], warns: [], summary: null };
  if (!String(text || "").trim()) return out;
  let d;
  try { d = JSON.parse(text); } catch (e) { out.errors.push(t("JSON として読めません: {m}", { m: e.message })); return out; }
  if (!d || typeof d !== "object" || Array.isArray(d)) { out.errors.push(t("中身は表(オブジェクト)にしてください")); return out; }
  out.spec = d;
  if (!SPEC_NAME_RE.test(String(d.name ?? ""))) out.errors.push(t("name は英数字・_・-・. だけの名前にしてください(対照集合・方向・鍵のファイル名になります)"));
  // 質問: 両側で同じなら user_prompts、側ごとに違うなら prompts_a / prompts_b。1 問は文字列か会話の履歴
  const isTurns = (q) => Array.isArray(q) && q.length > 0 && q.length % 2 === 1
    && q.every((m, i) => m && typeof m === "object" && m.role === (i % 2 ? "assistant" : "user") && typeof m.content === "string");
  const isList = (v) => Array.isArray(v) && v.every((u) => typeof u === "string" || isTurns(u));
  let listOk = true;
  for (const k of ["user_prompts", "prompts_a", "prompts_b"]) {
    if (d[k] !== undefined && d[k] !== null && !isList(d[k])) {
      out.errors.push(t("{k} は質問のリストです。1 問は文字列か、会話の履歴([{\"role\": \"user\", \"content\": …}, {\"role\": \"assistant\", …}, …] を user で始めて user で終える)", { k }));
      listOk = false;
    }
  }
  const pa = isList(d.prompts_a) ? d.prompts_a : isList(d.user_prompts) ? d.user_prompts : null;
  const pb = isList(d.prompts_b) ? d.prompts_b : isList(d.user_prompts) ? d.user_prompts : null;
  let n = null;
  if (listOk && (!pa || !pb)) out.errors.push(t("質問がありません: 両側で同じなら user_prompts、側ごとに違うなら prompts_a と prompts_b を書いてください"));
  else if (pa && pb) {
    n = Math.min(pa.length, pb.length);
    if (n < 4) out.errors.push(t("質問は 4 個以上にしてください(いま {n} 個)", { n }));
    else if (n < 16) out.warns.push(t("質問が {n} 個と少ないので、鍵の作成(偶数番目)と key-check(奇数番目)がそれぞれ {h} 個ほどになり、key-check の値がぶれます(数十個を勧めます)", { n, h: Math.ceil(n / 2) }));
  }
  const pos = d.positions === undefined || d.positions === null ? "response" : String(d.positions);
  for (const [k, v] of [["positions", d.positions], ["positions_b", d.positions_b]]) {
    if (v === undefined || v === null) continue;
    const [err, warn] = checkPositions(v, k);
    if (err) out.errors.push(err);
    if (warn) out.warns.push(warn);
  }
  const before = /^\s*before:/.test(pos);
  const samePrompts = !!(pa && pb) && pa.length === pb.length && pa.every((u, i) => JSON.stringify(u) === JSON.stringify(pb[i]));
  const hasRecipe = (d.recipe_a !== undefined && d.recipe_a !== null) || (d.recipe_b !== undefined && d.recipe_b !== null);
  const sysA = d.system_a !== undefined && d.system_a !== null, sysB = d.system_b !== undefined && d.system_b !== null;
  const dirOk = hasRecipe || !samePrompts || ((sysA || sysB) && d.system_a !== d.system_b);   // 方向(A − B)を作れるか
  if (!before && !dirOk) out.errors.push(t("作り方 (a) では A と B の違いが要ります: system_a・system_b(片方だけでも可)・recipe_a / recipe_b のどれか"));
  else if (before && !dirOk) out.warns.push(t("system_a が無いので、この対照集合からは方向を作れません(同じ名前の方向がもうあれば「方向から key-check まで」で使えます)"));
  const dts = d.dont_touch === undefined || d.dont_touch === null ? [] : Array.isArray(d.dont_touch) ? d.dont_touch : [d.dont_touch];
  const dtSum = [];
  dts.forEach((g, i) => {
    const w = t("dont_touch の {i} 番目", { i: i + 1 });
    if (!g || typeof g !== "object" || Array.isArray(g)) { out.errors.push(t("{w} は {system?, prompts, replies?, positions?} の表です", { w })); return; }
    const bad = Object.keys(g).filter((k) => !DT_KEYS.includes(k));
    if (bad.length) out.errors.push(t("{w} に知らないキー {k} があります(name・system・prompts・replies・positions)", { w, k: bad.join(", ") }));
    const pr = g.prompts;
    if (!(typeof pr === "string" && pr) && !(isList(pr) && pr.length)) out.errors.push(t("{w}: prompts は質問のリストか、プロンプトの組の YAML のパスです", { w }));
    if (g.positions !== undefined && g.positions !== null) {
      const [err, warn] = checkPositions(g.positions, `${w}: positions`);
      if (err) out.errors.push(err);
      if (warn) out.warns.push(warn);
    }
    dtSum.push({ name: String(g.name || `dt${i}`), n: Array.isArray(pr) ? pr.length : null, positions: g.positions ? String(g.positions) : null });
  });
  if (before && !dts.length) out.warns.push(t("dont_touch がありません。返事の最初で一致する言い方では、鍵がどの返事の冒頭にも反応しやすくなります(ふつうの質問を dont_touch に入れる)"));
  if (d.dont_touch_start !== undefined && d.dont_touch_start !== null && typeof d.dont_touch_start !== "boolean") out.errors.push(t("dont_touch_start は true か false です"));
  if (d.keyed !== undefined && d.keyed !== null) {
    if (typeof d.keyed !== "object" || Array.isArray(d.keyed)) out.errors.push(t("keyed は設定の表です({k})", { k: SETTINGS.join(", ") }));
    else {
      const bad = Object.keys(d.keyed).filter((k) => !SETTINGS.includes(k));
      if (bad.length) out.errors.push(t("keyed に {k} は使えません({ok})", { k: bad.join(", "), ok: SETTINGS.join(", ") }));
    }
  }
  if (n !== null) out.summary = { n: samePrompts ? n : `${pa.length} / ${pb.length}`, before, positions: pos, dt: dtSum, A: sideText(d, "a"), B: sideText(d, "b"), dirOk };
  return out;
}

// key-check の start_rho の 1 群({median: [位置 0, 1, 2], max, n})→ 中央値が最大の位置 { j, m, hot }。例 2 つ以上の位置から
// 選び(無ければ全部の位置から)、hot は例 2 つ以上で 0.4 以上(サーバの警告 keyed_build._start_hot と同じ決まり)。値が無ければ null
export const START_RHO_WARN = 0.4;
export function startPeak(v) {
  if (!v || !Array.isArray(v.median)) return null;
  const all = v.median.map((m, j) => ({ j, m, n: Array.isArray(v.n) ? +v.n[j] || 0 : 0 }))
    .filter((x) => x.m !== null && x.m !== undefined && Number.isFinite(+x.m));
  if (!all.length) return null;
  const pool = all.some((x) => x.n >= 2) ? all.filter((x) => x.n >= 2) : all;
  const p = pool.reduce((a, b) => (+b.m > +a.m ? b : a));
  return { j: p.j, m: +p.m, hot: p.n >= 2 && +p.m >= START_RHO_WARN };
}

// 群の名前(A・B・dt:<名前>)の表示と色の組
const groupLabel = (g) => (g === "A" ? t("書き込みたい位置") : g === "B" ? t("書き込みたくない位置") : `dont_touch ${g.replace(/^dt:/, "")}`);
const groupCls = (g, i) => (g === "A" ? "ga" : g === "B" ? "gb" : `gd${i % 3}`);

// 層 1 つの ρ の分布(hist: { edges, A, B, "dt:<名前>": [...] })。群ごとの割合(その群の数で割る)で重ねる。
// 書き込みたくない位置を先に塗り、書き込みたい位置を上に、dont_touch は線(段)で。0 と 1(κᵀμ = 1、書き込みたい位置の平均が来る所)に縦の点線。棒にマウスで数と割合
export function histSVG(h, { title = "" } = {}) {
  if (!h || !Array.isArray(h.edges) || h.edges.length < 2) return `<p class="muted note">${t("ヒストグラムがありません")}</p>`;
  const W = 320, H = 140, PL = 34, PR = 8, PT = 18, PB = 24;
  const edges = h.edges, nb = edges.length - 1, lo = edges[0], hi = edges[nb];
  const groups = Object.keys(h).filter((k) => k !== "edges" && Array.isArray(h[k]));
  const order = [...groups.filter((g) => g === "B"), ...groups.filter((g) => g === "A"), ...groups.filter((g) => g.startsWith("dt:"))];
  const share = {};
  let top = 0;
  for (const g of order) {
    const n = h[g].reduce((a, b) => a + b, 0);
    share[g] = { n, v: h[g].map((c) => (n ? c / n : 0)) };
    top = Math.max(top, ...share[g].v);
  }
  top = top || 1;
  const X = (v) => PL + ((v - lo) / (hi - lo || 1)) * (W - PL - PR), Y = (f) => PT + (1 - f / top) * (H - PT - PB);
  const parts = [];
  for (let i = 0; i <= 2; i++) {                                         // 横の目盛り(割合)
    const f = (top * i) / 2;
    parts.push(`<line class="grid" x1="${PL}" x2="${W - PR}" y1="${Y(f)}" y2="${Y(f)}"/><text class="ax" x="${PL - 4}" y="${Y(f) + 3}" text-anchor="end">${Math.round(f * 100)}%</text>`);
  }
  for (const v of [0, 1]) {                                              // ρ = 0(何も足さない)と ρ = 1(書き込みたい位置の平均)
    if (v <= lo || v >= hi) continue;
    parts.push(`<line class="ref" x1="${X(v)}" x2="${X(v)}" y1="${PT}" y2="${H - PB}"/><text class="ax" x="${X(v)}" y="${H - PB + 11}" text-anchor="middle">${v}</text>`);
  }
  parts.push(`<text class="ax" x="${PL}" y="${H - 4}" text-anchor="start">${fmt(lo, 2)}</text><text class="ax" x="${W - PR}" y="${H - 4}" text-anchor="end">${fmt(hi, 2)}</text>`);
  let di = 0;
  for (const g of order) {
    const cls = groupCls(g, g.startsWith("dt:") ? di++ : 0), s = share[g], lab = esc(groupLabel(g));
    const tip = (i) => `${lab}: ${h[g][i]} (${Math.round(s.v[i] * 1000) / 10}%) · ρ ${fmt(edges[i], 2)}〜${fmt(edges[i + 1], 2)}`;
    if (g.startsWith("dt:")) {
      let d = "";
      for (let i = 0; i < nb; i++) d += `${i ? "L" : "M"}${X(edges[i])},${Y(s.v[i])}L${X(edges[i + 1])},${Y(s.v[i])}`;
      parts.push(`<path class="line ${cls}" d="${d}"><title>${lab} (n=${s.n})</title></path>`);
      continue;
    }
    for (let i = 0; i < nb; i++) {
      if (!h[g][i]) continue;
      const x0 = X(edges[i]), x1 = X(edges[i + 1]);
      parts.push(`<rect class="bar ${cls}" x="${x0 + 0.5}" y="${Y(s.v[i])}" width="${Math.max(0.5, x1 - x0 - 1)}" height="${Y(0) - Y(s.v[i])}"><title>${tip(i)}</title></rect>`);
    }
  }
  return `<svg class="kx-hist" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title || "ρ")}"><text class="tt" x="${PL}" y="11">${esc(title)}</text>${parts.join("")}</svg>`;
}

export class KeyPanel {
  constructor(root, { onRun, onAddKeyed, directions, nLayer, moeLayers } = {}) {
    this.root = root;
    this.onRun = onRun || (async () => null);       // (kind, params) → Promise<ジョブ | null>(jobs.js の start。見積もりの確認つき)
    this.onAddKeyed = onAddKeyed || (() => {});     // ({ layer, kind, expert, alpha, direction, key }) → レシピに足す
    this.directions = directions || (() => []);     // 方向セットの名前(gne direction。/api/model の direction_sets)
    this.nLayer = nLayer || 0;
    this.moe = new Set(moeLayers || []);            // MoE の層(専門家を選ぶ材料のボタンはこれがあるときだけ)
    this.contrasts = [];
    this.keys = [];
    this.dirs = [];                                 // /api/directions(作ったばかりの方向も入るように、ここでも取り直す)
    this.templates = [];                            // configs/keyed/ のひな形(/api/keyed-templates)
    this.tplText = null;                            // 最後に入れたひな形の文字列(書き換えていなければ確かめずに差し替える)
    this.lastSet = null;                            // 最後に欄に入れた表の keyed(変わったときだけ欄を書き換える)
    this.alphaNext = null;                          // 次に結果を描くとき「+ レシピ」の α の欄に入れる値(表の keyed.alpha、無ければ 1)
    this.pipeResult = null;                         // 最後の「方向から key-check まで」の結果
    this.preferDir = null;                          // 次に結果を描くとき「書き込む方向」で選ぶ方向(一括で作ったばかりの方向)
    this.jobs = { build: null, check: null, pipe: null, exp: null, verify: null };   // 投げたジョブの id
    this.handled = new Set();                       // 結果を出し終えたジョブ(SSE と onJobFinished の両方から来る)
    this.built = null;                              // 最後に作った鍵の結果(direction_key)か、一覧で開いた鍵(/api/keys/<名前>)
    this.checked = null;                            // 最後の key-check の結果
    this.experts = null;                            // 最後の専門家の選ばれ方(direction_key_experts)
    this.verified = null;                           // 最後の key-verify(direction_key_verify)
    this.jspec = { spec: null, errors: [], warns: [], summary: null };   // 入力欄を確かめた結果
    this.loaded = false;
    const n = this.nLayer, sug = n ? [0.3, 0.5, 0.7].map((f) => Math.round((n - 1) * f)).join(",") : "";
    root.innerHTML = `
      <p class="muted note pp-help">${t("鍵つきの書き込み(keyed): 残差に書き込む重み行列(ffn_down など)にランク 1 の ΔW = α·d·κᵀ を足し、入力の活性 x に鍵 κ が反応する(鍵の反応の強さ ρ = κᵀx が大きい)ときだけ方向 d(方向セットの r_L)を書く。κ は書き込みたい位置の活性の平均 μ から、書き込みたくない位置と dont_touch の活性が張る部分空間を除いたもの(κᵀμ = 1)。鍵を作ったら、作成に使っていない例(subset odd か別の対照集合)で key-check を回し、比較用の鍵(ラベルをシャッフルして同じ手順で作った鍵)より選択性が高いことを確かめてから、レシピに足す。")}</p>
      <section class="kx-sec"><h4>${t("鍵を作る")}</h4>
        <div class="pp-in"><span class="muted">${t("対照集合")}</span>
          <span class="seg" id="kx-src"><button data-src="file" title="${t("configs/contrast/ か <root>/contrasts/ の YAML を選ぶ")}">${t("ファイルを選ぶ")}</button><button data-src="json" title="${t("用途別のひな形(configs/keyed/)から始めて、入力欄の JSON を手で直す")}">${t("JSON を書く(ひな形から)")}</button></span>
          <label class="muted" title="${t("用途別のひな形(configs/keyed/)。選ぶと入力欄に入る")}">${t("ひな形")}
            <select id="kx-tpl-sel"><option value="">${t("(ひな形を選ぶ)")}</option></select></label>
          <button class="small" id="kx-tpl" title="${t("選んだひな形を入力欄に入れる")}">${t("入れる")}</button></div>
        <div id="kx-src-file">
          <div class="pp-in">
            <select id="kx-contrast" aria-label="${t("対照集合")}"></select>
            <input id="kx-cpath" size="34" placeholder="${t("または YAML のパス(入れるとこちらを使う)")}" spellcheck="false">
          </div>
          <div id="kx-cinfo" class="muted note"></div>
        </div>
        <div id="kx-src-json" class="hidden">
          <div class="pp-in">
            <label class="muted">${t("JSON ファイル")} <input type="file" id="kx-file" accept=".json,application/json"></label>
            <span class="muted note">${t("または下に貼り付け")}</span></div>
          <div id="kx-tpl-about" class="muted note"></div>
          <textarea id="kx-text" class="mono" rows="9" spellcheck="false" placeholder='{"name": "…", "system_a": "…", "positions": "before:(です|ます)。", "user_prompts": ["…", "…", "…", "…"], "dont_touch": [{"name": "english", "prompts": ["…"]}]}'></textarea>
          <div id="kx-jcheck" class="note"></div>
        </div>
        <div class="pp-set">
          <label class="muted" title="${t("鍵の名前(<root>/keys/<model_id>/<名前>.npz)。空なら対照集合の名前")}">${t("名前")} <input id="kx-name" size="16" spellcheck="false"></label>
          <label class="muted" title="${t("鍵を作る層(カンマ区切りか 10-16)。層ごとに κ を作る")}">${t("層")} <input id="kx-layers" size="12" value="${sug}" placeholder="8,14,20" spellcheck="false"></label>
          <label class="muted" title="${t("鍵が入力を読む、残差に書き込む重み行列。ffn = ffn_down(ニューロンの活性)、attn = attn_output(ヘッドの出力の連結)、shared = shared expert、expert = 専門家 e(選ばれた位置だけ)")}">${t("書き込み先")}
            <select id="kx-target"><option value="ffn">ffn</option><option value="attn">attn</option><option value="shared">shared</option><option value="expert">expert</option></select>
            <input id="kx-expert" type="number" min="0" class="hidden" placeholder="e" style="width:4em"></label>
          <label class="muted" title="${t("書き込みたい位置。空なら対照集合の値(無ければ response)。before:<正規表現> は一致の直前のトークン(推論時と同じ文脈の返事で、そのトークンを決める位置 = 作り方 (b))。response・last・assistant は system_a の文脈(作り方 (a))")}">positions <input id="kx-pos" size="22" list="kx-pos-list" spellcheck="false"></label>
          <datalist id="kx-pos-list"><option value="response"><option value="last"><option value="assistant"><option value="before:(です|ます)。"></datalist>
          <label class="muted" title="${t("Q(書き込みたくない位置の活性が張る部分空間の基底)の作り方。examples = 例ごとの平均、means = 群ごとの平均(よく効くが選択性は粗い)、samples = トークンの標本(選択性が高いが、標本が多いと鍵が消える)")}">q_mode
            <select id="kx-qmode"><option>examples</option><option>means</option><option>samples</option></select></label>
          <label class="muted" title="${t("鍵を作るのに使う質問。even = 偶数番目(key-check は odd で確かめる)、all = 全部(保留集合が残らない)")}">subset
            <select id="kx-subset"><option>even</option><option>odd</option><option>all</option></select></label>
          <label class="muted" title="${t("比較用の鍵の本数。書き込みたい位置・書き込みたくない位置のラベルを、偏らないように例ごとにシャッフルして、同じ手順で作るランダムな対照。key-check の p95 に使う")}">${t("比較用の鍵")} <input id="kx-pseudo" type="number" min="0" max="256" value="16" style="width:4.5em"></label>
          <label class="muted" title="${t("返事を生成するときの長さ(返事を与えるなら関係ない)")}">max_tokens <input id="kx-mt" type="number" min="1" max="2048" value="96" style="width:5em"></label>
        </div>
        <details class="kx-adv"><summary class="muted note">${t("細かい設定")}</summary><div class="pp-set">
          <label class="muted" title="${t("書き込みたくない位置。空なら before:<re> のとき not-before:<re>、それ以外は positions と同じ")}">positions_b <input id="kx-posb" size="22" spellcheck="false"></label>
          <label class="muted" title="${t("返事の最初の何トークンを飛ばすか。空なら対照集合の値(無ければ 0)")}">skip <input id="kx-skip" type="number" min="0" style="width:4em"></label>
          <label class="muted" title="${t("モデル自身の返事のファイル(YAML / JSON のリストか {プロンプト: 返事}、サーバの作業場所からの相対)。空なら対照集合の replies か、無介入で生成")}">${t("返事のファイル")} <input id="kx-replies" size="26" spellcheck="false"></label>
          <label class="muted" title="${t("他の介入を当てた状態で鍵を作る(保存したレシピの名前か、ユニット 14:8949:+1 などを空白区切りで)。空なら無介入")}">${t("介入(iv)")} <input id="kx-iv" size="18" spellcheck="false"></label>
          <label class="muted" title="${t("比較用の鍵のシャッフルの種")}">seed <input id="kx-seed" type="number" value="0" style="width:4.5em"></label>
          <label class="muted" title="${t("「方向から key-check まで」で方向を作るとき、A・B の返事を生成する長さ")}">${t("方向の返事の長さ")} <input id="kx-rt" type="number" min="1" max="2048" value="96" style="width:5em"></label>
        </div></details>
        <div class="pp-go">
          <button id="kx-pipe-go" title="${t("方向(gne direction --positions response と同じ)→ 鍵 → 作成と反対側の subset での key-check を 1 回のジョブで回す")}">${t("方向から key-check まで")}</button>
          <label class="muted" title="${t("同じ名前の方向があっても使い回さず、返事を作って作り直す")}"><input type="checkbox" id="kx-rebuild"> ${t("方向を作り直す")}</label>
          <button id="kx-go" title="${t("鍵だけを作る(書き込む方向は別に用意する)")}">${t("鍵を作る")}</button>
          <button id="kx-exp-go" class="hidden" title="${t("MoE の層で、書き込みたい位置・書き込みたくない位置で各専門家が選ばれた割合を出す(専門家 e を書き込み先にする鍵の専門家を選ぶ)")}">${t("専門家を選ぶ材料を出す")}</button>
          <span class="muted note" id="kx-status"></span></div>
        <div id="kx-dirinfo" class="muted note"></div>
        <div id="kx-prog" class="note mono"></div>
        <div id="kx-exp" class="hidden"><h4 class="note">${t("専門家の選ばれ方(MoE)")} <span class="muted note" id="kx-estatus"></span></h4><div id="kx-eprog" class="note mono"></div><div id="kx-eres"></div></div>
        <div id="kx-pipe"></div>
        <div id="kx-build"></div>
        <div id="kx-verify" class="hidden"><h4 class="note">${t("LoRA が式どおりに書くか(key-verify)")} <span class="muted note" id="kx-vstatus"></span></h4>
          <details><summary class="muted note">${t("流す文")}</summary><textarea id="kx-vtexts" class="mono" rows="3" spellcheck="false" placeholder="${t("1 行に 1 文。空なら既定の 2 文(語尾の違う日本語と英語)。専門家の鍵は、その専門家が選ばれる文を")}"></textarea></details>
          <div id="kx-vprog" class="note mono"></div><div id="kx-vres"></div></div>
      </section>
      <section class="kx-sec"><h4>${t("鍵の分離を確かめる(key-check)")}</h4>
        <div class="pp-set">
          <label class="muted">${t("鍵")} <select id="kc-key"></select></label>
          <label class="muted" title="${t("確かめる対照集合。鍵を作ったものと同じなら、作成と反対側の subset で(重なると AUC が甘くなる)。なしなら鍵に保存した作成時の例で測る(甘い)")}">${t("対照集合")} <select id="kc-contrast"></select></label>
          <label class="muted">subset <select id="kc-subset"><option>odd</option><option>even</option><option>all</option></select></label>
          <label class="muted" title="${t("空なら鍵の全層")}">${t("層")} <input id="kc-layers" size="10" spellcheck="false"></label>
          <label class="muted" title="${t("空なら鍵の作成と同じ")}">max_tokens <input id="kc-mt" type="number" min="1" max="2048" style="width:5em"></label>
          <label class="muted" title="${t("鍵を作ったときに返事を与えたなら、同じファイル")}">${t("返事のファイル")} <input id="kc-replies" size="22" spellcheck="false"></label>
          <label class="muted" title="${t("作成に使った例と重なっていたら測らずに止める")}"><input type="checkbox" id="kc-strict"> strict</label>
        </div>
        <div class="pp-go"><button id="kc-go">${t("key-check を回す")}</button> <span class="muted note" id="kc-status"></span></div>
        <div id="kc-prog" class="note mono"></div>
        <div id="kc-res"></div>
      </section>
      <section class="kx-sec"><h4>${t("保存した鍵")}</h4><div id="kx-keys"><p class="muted note">${t("読み込み中…")}</p></div></section>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#kx-contrast").addEventListener("change", () => this.contrastChanged());
    this.$("#kx-cpath").addEventListener("input", () => this.contrastChanged());
    this.$("#kx-target").addEventListener("change", (ev) => this.$("#kx-expert").classList.toggle("hidden", ev.target.value !== "expert"));
    this.$("#kx-rebuild").addEventListener("change", () => this.dirInfo());
    this.$("#kx-go").addEventListener("click", () => this.runBuild());
    this.$("#kx-pipe-go").addEventListener("click", () => this.runPipeline());
    this.$("#kx-exp-go").addEventListener("click", () => this.runExperts());
    this.$("#kc-go").addEventListener("click", () => this.runCheck());
    this.$("#kx-exp-go").classList.toggle("hidden", !this.moe.size);
    for (const b of root.querySelectorAll("#kx-src button")) b.addEventListener("click", () => this.setSource(b.dataset.src));
    // 入力欄(JSON): 下書き・ファイル・ひな形
    const ta = this.$("#kx-text");
    try { ta.value = localStorage.getItem(DRAFT) || ""; } catch { /* 下書きが無くても使える */ }
    ta.addEventListener("input", () => this.checkJSON());
    this.$("#kx-file").addEventListener("change", async (ev) => {
      const f = ev.target.files && ev.target.files[0];
      if (!f) return;
      ta.value = await f.text();
      this.$("#kx-tpl-about").textContent = "";
      this.checkJSON();
    });
    const sel = this.$("#kx-tpl-sel");
    this.$("#kx-tpl").addEventListener("click", () => this.insertTemplate(sel.value, false));
    sel.addEventListener("change", () => this.insertTemplate(sel.value, true));
    getJSON("/api/keyed-templates").then((list) => {
      this.templates = Array.isArray(list) ? list : [];
      for (const x of this.templates) {
        const o = document.createElement("option");
        o.value = x.file;
        o.textContent = (x[LANG] && x[LANG].title) || (x.ja && x.ja.title) || x.file;
        sel.appendChild(o);
      }
    }).catch(() => { /* ひな形が無くてもファイルと貼り付けで使える */ });
    let src = "file";
    try { src = localStorage.getItem(SRC) === "json" ? "json" : "file"; } catch { /* 既定はファイル */ }
    this.setSource(src, false);
    root.addEventListener("click", (ev) => this.onClick(ev));
  }

  // direction ビューで「鍵」を開いたとき・ジョブが終わったとき: 対照集合の候補・保存した鍵・方向を取り直す
  async refresh() {
    const [cs, ks, ds] = await Promise.all([getJSON("/api/contrasts").catch(() => []), getJSON("/api/keys").catch(() => []),
                                            getJSON("/api/directions").catch(() => null)]);
    this.contrasts = Array.isArray(cs) ? cs : [];
    this.keys = Array.isArray(ks) ? ks : [];
    if (Array.isArray(ds)) this.dirs = ds;
    this.fillContrasts();
    this.fillKeys();
    this.renderKeys();
    this.loaded = true;
  }

  // 書き込む方向の候補(この画面で取った /api/directions と、/api/model の方向セット)
  dirNames() {
    return [...new Set([...this.dirs.map((d) => d.name), ...(this.directions() || [])])].sort();
  }

  fillContrasts() {
    const opts = (withNone) => {
      const grp = (src, label) => {
        const xs = this.contrasts.filter((c) => c.src === src);
        return xs.length ? `<optgroup label="${esc(label)}">${xs.map((c) => `<option value="${esc(c.path)}">${esc(c.file)}${c.mode === "same" ? " · before" : ""}</option>`).join("")}</optgroup>` : "";
      };
      return (withNone ? `<option value="">${t("(なし: 作成時の例で測る、甘い)")}</option>` : "") + grp("configs", "configs/contrast") + grp("root", "<root>/contrasts");
    };
    for (const [sel, none] of [["#kx-contrast", false], ["#kc-contrast", true]]) {
      const el = this.$(sel), cur = el.value;
      el.innerHTML = opts(none);
      if (cur && [...el.options].some((o) => o.value === cur)) el.value = cur;
      else if (!none) {                                       // 最初は before:<re> の対照集合(作り方 (b) の例)を選んでおく
        const first = this.contrasts.find((c) => c.mode === "same") || this.contrasts[0];
        if (first) el.value = first.path;
      }
    }
    this.contrastChanged();
  }

  // パスか名前 → 一覧の対照集合(作った後の <root>/contrasts/<名前>.yaml も名前で見つける)
  findContrast(path, name = null) {
    return this.contrasts.find((c) => path && c.path === path)
      || (path ? this.contrasts.find((c) => c.src === "root" && String(path).endsWith(`/${c.file}.yaml`)) : null)
      || (name ? this.contrasts.find((c) => c.src === "root" && c.file === name) : null) || null;
  }

  fillKeys(select = null) {
    const el = this.$("#kc-key"), cur = select || el.value;
    const ok = this.keys.filter((k) => !k.error);
    el.innerHTML = ok.length ? ok.map((k) => `<option value="${esc(k.name)}">${esc(k.name)}</option>`).join("") : `<option value="">${t("(なし)")}</option>`;
    if (cur && ok.some((k) => k.name === cur)) el.value = cur;
    this.$("#kc-go").disabled = !ok.length;
  }

  // 対照集合の入れ方: ファイルを選ぶ(file)/ JSON を書く(json)
  setSource(src, remember = true) {
    this.src = src === "json" ? "json" : "file";
    for (const b of this.root.querySelectorAll("#kx-src button")) b.classList.toggle("on", b.dataset.src === this.src);
    this.$("#kx-src-file").classList.toggle("hidden", this.src !== "file");
    this.$("#kx-src-json").classList.toggle("hidden", this.src !== "json");
    if (remember) { try { localStorage.setItem(SRC, this.src); } catch { /* 覚えられなくても使える */ } }
    if (this.src === "json") this.checkJSON();
    else this.contrastChanged();
  }

  // 選んだ対照集合の要約(質問の数・positions・mode・dont_touch)。名前と positions の欄の既定もそれに
  contrastChanged() {
    if (this.src === "json") { this.dirInfo(); return; }
    const path = this.$("#kx-cpath").value.trim() || this.$("#kx-contrast").value;
    const c = this.contrasts.find((x) => x.path === path);
    const info = this.$("#kx-cinfo");
    this.$("#kx-name").placeholder = c ? c.name : t("(対照集合の名前)");
    this.$("#kx-pos").placeholder = c ? c.positions || "response" : t("(対照集合の値)");
    this.dirInfo();
    if (!c) { info.textContent = path ? t("(一覧に無いファイル: そのままジョブに渡します)") : t("対照集合がありません(configs/contrast/ か <root>/contrasts/ の YAML)"); return; }
    const mode = c.mode === "same" ? t("作り方 (b): 推論時と同じ文脈(system なし)の返事の、正規表現の直前の位置を書き込みたい位置、同じ返事のそれ以外を書き込みたくない位置にする")
      : t("作り方 (a): system_a(や recipe_a)の文脈の活性を書き込みたい位置、system_b の文脈を書き込みたくない位置にする");
    info.innerHTML = `${esc(c.label)} · ${c.n_prompts ?? "—"} ${t("問")} · positions <span class="mono">${esc(c.positions || "response")}</span> · ${esc(mode)}${
      c.dont_touch.length ? ` · dont_touch: ${esc(c.dont_touch.join(", "))}` : ""}${c.replies ? ` · ${t("返事つき")}` : ""}`;
  }

  // ひな形を入力欄に入れる(configs/keyed/ のファイル名)。選び直したときは、書きかけの JSON を消す前に確かめる
  insertTemplate(file, ask) {
    const x = this.templates.find((y) => y.file === file);
    if (!x) return;
    const ta = this.$("#kx-text"), about = this.$("#kx-tpl-about");
    const cur = ta.value.trim();
    if (ask && cur && cur !== this.tplText && !window.confirm(t("入力欄の JSON をひな形で置き換えます。よいですか?"))) {
      this.$("#kx-tpl-sel").value = "";
      return;
    }
    ta.value = JSON.stringify(x.spec, null, 2);
    this.tplText = ta.value.trim();
    about.textContent = (x[LANG] && x[LANG].text) || (x.ja && x.ja.text) || "";
    about.title = `configs/keyed/${x.file}`;
    if (this.src !== "json") this.setSource("json");
    else this.checkJSON();
  }

  // 入力欄を確かめて要約を出す。表の keyed が変わったら、その値を欄に入れる
  checkJSON() {
    const text = this.$("#kx-text").value;
    try { localStorage.setItem(DRAFT, text); } catch { /* 置けなくても使える */ }
    const c = checkKeySpec(text);
    this.jspec = c;
    const el = this.$("#kx-jcheck");
    if (!c.spec && !c.errors.length) el.innerHTML = `<span class="muted">${t("JSON を読み込むか貼り付けてください(「ひな形」で用途別の例を選べます)")}</span>`;
    else {
      const s = c.summary;
      const mode = s ? (s.before ? t("作り方 (b): 推論時と同じ文脈(system なし)の返事の、正規表現の直前の位置を書き込みたい位置、同じ返事のそれ以外を書き込みたくない位置にする")
        : t("作り方 (a): system_a(や recipe_a)の文脈の活性を書き込みたい位置、system_b の文脈を書き込みたくない位置にする")) : "";
      const dt = s && s.dt.length ? s.dt.map((g) => `${esc(g.name)}${g.n !== null ? ` ${g.n} ${t("問")}` : ""}${g.positions ? ` (${esc(g.positions)})` : ""}`).join("、") : "";
      el.innerHTML = `${s ? `<div class="pp-sum"><b>${esc(String(c.spec.name ?? ""))}</b> · ${t("質問 {n} 個", { n: s.n })} · positions <span class="mono">${esc(s.positions)}</span><br>${esc(mode)}${
        dt ? `<br>dont_touch: ${dt}` : ""}<br><span class="muted">${t("方向: A = {a} · B = {b}", { a: esc(s.A), b: esc(s.B) })}</span></div>` : ""}${
        c.errors.map((x) => `<div class="err">${esc(x)}</div>`).join("")}${c.warns.map((x) => `<div class="pp-warn">${esc(x)}</div>`).join("")}`;
    }
    if (c.spec) {
      this.$("#kx-name").placeholder = SPEC_NAME_RE.test(String(c.spec.name ?? "")) ? String(c.spec.name) : t("(対照集合の名前)");
      this.$("#kx-pos").placeholder = c.spec.positions ? String(c.spec.positions) : "response";
      const kd = c.spec.keyed && typeof c.spec.keyed === "object" && !Array.isArray(c.spec.keyed) ? c.spec.keyed : {};
      const key = JSON.stringify(kd);
      if (key !== this.lastSet) {                              // 表の keyed にある値だけを欄に入れる(無い欄はそのまま)
        this.lastSet = key;
        for (const [k, sel] of FIELDS) if (kd[k] !== undefined && kd[k] !== null) this.$(sel).value = Array.isArray(kd[k]) ? kd[k].join(",") : String(kd[k]);
        if (kd.target) {
          const { kind, expert } = targetKind(kd.target);
          this.$("#kx-target").value = kind;
          this.$("#kx-expert").classList.toggle("hidden", kind !== "expert");
          if (expert !== null) this.$("#kx-expert").value = String(expert);
        }
        this.alphaNext = kd.alpha !== null && kd.alpha !== undefined && kd.alpha !== "" && Number.isFinite(+kd.alpha) ? +kd.alpha : 1;
      }
    }
    this.dirInfo();
  }

  // いまの対照集合の名前(方向・鍵の名前の既定)
  contrastName() {
    if (this.src === "json") return this.jspec.spec && SPEC_NAME_RE.test(String(this.jspec.spec.name ?? "")) ? String(this.jspec.spec.name) : null;
    const path = this.$("#kx-cpath").value.trim() || this.$("#kx-contrast").value;
    const c = this.contrasts.find((x) => x.path === path);
    return c ? c.name : path ? String(path).split("/").pop().replace(/\.(ya?ml|json)$/, "") : null;
  }

  // 「方向から key-check まで」で使う方向が、もうあるか
  dirInfo() {
    const el = this.$("#kx-dirinfo");
    if (!el) return;
    const name = this.contrastName();
    if (!name) { el.textContent = ""; return; }
    const d = this.dirs.find((x) => x.name === name) || (this.dirNames().includes(name) ? { name } : null);
    const rebuild = this.$("#kx-rebuild").checked;
    el.innerHTML = !d ? t("書き込む方向 {d} はまだありません。「方向から key-check まで」は A・B の返事を作って方向を作ります", { d: `<b class="mono">${esc(name)}</b>` })
      : rebuild ? t("書き込む方向 {d} を作り直します(A・B の返事を作ります)", { d: `<b class="mono">${esc(name)}</b>` })
      : `${t("書き込む方向 {d} はもうあるので使い回します(作り直すなら「方向を作り直す」)", { d: `<b class="mono">${esc(name)}</b>` })}${
        d.positions && d.positions !== "response" ? ` <span class="pp-warn">${t("この方向は positions {p} で作られています(一括は response で作ります)", { p: esc(d.positions) })}</span>` : ""}`;
  }

  status(which, s, err = false) {
    const el = this.$({ build: "#kx-status", pipe: "#kx-status", check: "#kc-status", exp: "#kx-estatus", verify: "#kx-vstatus" }[which] || "#kx-status");
    el.textContent = s;
    el.classList.toggle("err", err);
  }

  // --- 欄 → ジョブの引数 -----------------------------------------------------------------------------------
  // 対照集合: ファイル(contrast)か入力欄の表(spec、head_pipeline と同じく表のまま渡す)
  sourceParams(which) {
    if (this.src === "json") {
      const c = this.jspec;
      if (!c.spec) { this.status(which, t("入力欄に対照集合の JSON を入れてください(「ひな形」で例を選べます)"), true); return null; }
      if (c.errors.length) { this.status(which, t("入力欄の JSON に直すところがあります: {e}", { e: c.errors[0] }), true); return null; }
      return { spec: c.spec };
    }
    const contrast = this.$("#kx-cpath").value.trim() || this.$("#kx-contrast").value;
    if (!contrast) { this.status(which, t("対照集合を選んでください"), true); return null; }
    return { contrast };
  }

  // 鍵の作成・一括・専門家の選ばれ方に共通の欄(層・位置・返事など)。expertsOnly なら専門家の選ばれ方に要るものだけ
  formParams(which, { expertsOnly = false } = {}) {
    const v = (s) => this.$(s).value.trim();
    const layers = parseLayers(v("#kx-layers"));
    if (!layers || (this.nLayer && layers.some((L) => L >= this.nLayer))) {
      this.status(which, t("層は 0〜{m} の数をカンマ区切りで(例 8,14,20 か 10-16)", { m: this.nLayer ? this.nLayer - 1 : "?" }), true);
      return null;
    }
    const p = { layers: layers.join(","), subset: v("#kx-subset"), max_tokens: +v("#kx-mt") || 96 };
    for (const [k, sel] of [["positions", "#kx-pos"], ["positions_b", "#kx-posb"], ["replies_a", "#kx-replies"]]) if (v(sel)) p[k] = v(sel);
    const iv = v("#kx-iv");                                   // ユニットの書き方(: を含む)ならリストに、それ以外は保存したレシピの名前かパス
    if (iv) p.iv = iv.includes(":") ? iv.split(/[\s,]+/).filter(Boolean) : iv;
    if (v("#kx-skip") !== "") p.skip = +v("#kx-skip");
    if (expertsOnly) return p;
    const name = v("#kx-name");
    if (name && !NAME_RE.test(name)) { this.status(which, t("名前は英数字・_・-・. だけにしてください"), true); return null; }
    const tg = v("#kx-target"), e = v("#kx-expert");
    if (tg === "expert" && !/^\d+$/.test(e)) { this.status(which, t("専門家の番号を入れてください"), true); return null; }
    Object.assign(p, { q_mode: v("#kx-qmode"), pseudo: +v("#kx-pseudo") || 0 });
    if (tg !== "ffn") p.target = tg === "expert" ? `expert:${e}` : tg;
    if (name) p.name = name;
    if (v("#kx-seed") !== "" && +v("#kx-seed") !== 0) p.seed = +v("#kx-seed");
    return p;
  }

  buildParams(which = "build") {
    const s = this.sourceParams(which);
    if (!s) return null;
    const f = this.formParams(which);
    return f ? { ...s, ...f } : null;
  }

  // --- 鍵を作る(direction_key のジョブ) ---------------------------------------------------------------------
  async runBuild() {
    const p = this.buildParams("build");
    if (!p) return;
    await this.launch("build", "direction_key", p, "#kx-prog", t("鍵を作っています(モデルを流します。GPU を待つことがあります)…"));
  }

  // --- 方向から key-check まで(direction_key_pipeline のジョブ) -------------------------------------------------
  async runPipeline() {
    const p = this.buildParams("pipe");
    if (!p) return;
    if (this.$("#kx-rebuild").checked) p.rebuild_direction = true;
    const rt = +this.$("#kx-rt").value;
    if (Number.isInteger(rt) && rt > 0 && rt !== 96) p.response_tokens = rt;
    await this.launch("pipe", "direction_key_pipeline", p, "#kx-prog", t("方向 → 鍵 → key-check を回しています(モデルを流します。GPU を待つことがあります)…"));
  }

  // --- 専門家を選ぶ材料(direction_key_experts のジョブ。MoE の層だけ) -------------------------------------------
  async runExperts() {
    const s = this.sourceParams("exp");
    if (!s) { this.$("#kx-exp").classList.remove("hidden"); return; }
    const f = this.formParams("exp", { expertsOnly: true });
    this.$("#kx-exp").classList.remove("hidden");
    if (!f) return;
    const Ls = f.layers.split(",").map(Number), moe = Ls.filter((L) => this.moe.has(L));
    const all = [...this.moe].sort((a, b) => a - b);
    if (!moe.length) {
      this.status("exp", t("層の欄に MoE の層がありません(MoE の層: {ls})", { ls: all.length > 12 ? `${all.slice(0, 12).join(",")}…` : all.join(",") }), true);
      return;
    }
    f.layers = moe.join(",");
    await this.launch("exp", "direction_key_experts", { ...s, ...f, top: 8 }, "#kx-eprog",
                      moe.length < Ls.length ? t("MoE の層 {ls} だけで測っています(ほかの層は MoE ではありません)…", { ls: f.layers }) : t("測っています(モデルを流します)…"));
  }

  // --- key-verify(direction_key_verify のジョブ、数秒) ----------------------------------------------------------
  async runVerify(layer) {
    const b = this.built;
    const dir = this.$("#kx-dir") ? this.$("#kx-dir").value : "";
    if (!b || !dir) return;
    this.$("#kx-verify").classList.remove("hidden");
    const p = { layer, key: b.r.name, direction: dir, target: b.r.target || "ffn" };
    const texts = this.$("#kx-vtexts").value.split("\n").map((x) => x.trim()).filter(Boolean);
    if (texts.length) p.texts = texts;
    await this.launch("verify", "direction_key_verify", p, "#kx-vprog", t("L{L} で確かめています…", { L: layer }));
  }

  // ジョブを投げて、進み具合をこの欄にも出す
  async launch(which, kind, p, progSel, running) {
    this.status(which, t("ジョブを投げています…"));
    try {
      const j = await this.onRun(kind, p);
      if (!j) { this.status(which, t("やめました")); return; }
      this.jobs[which] = j.id;
      this.status(which, running);
      this.follow(j.id, progSel);
    } catch (e) { this.status(which, `${t("エラー")}: ${e.message}`, true); }
  }

  // --- key-check(direction_key_check のジョブ、ヒストグラムに使うので値は落とさない) -----------------------------------
  checkParams() {
    const v = (s) => this.$(s).value.trim();
    const key = v("#kc-key");
    if (!key) { this.status("check", t("鍵を選んでください"), true); return null; }
    const p = { key, subset: v("#kc-subset") };
    if (v("#kc-contrast")) p.contrast = v("#kc-contrast");
    if (v("#kc-layers")) {
      const Ls = parseLayers(v("#kc-layers"));
      if (!Ls) { this.status("check", t("層は数をカンマ区切りで(空なら鍵の全層)"), true); return null; }
      p.layers = Ls.join(",");
    }
    if (v("#kc-mt")) p.max_tokens = +v("#kc-mt");
    if (v("#kc-replies")) p.replies_a = v("#kc-replies");
    if (this.$("#kc-strict").checked) p.strict = true;
    return p;
  }

  async runCheck() {
    const p = this.checkParams();
    if (!p) return;
    await this.launch("check", "direction_key_check", p, "#kc-prog", p.contrast ? t("測っています(モデルを流します)…") : t("測っています…"));
  }

  // ジョブの進み具合をここにも出す(下のジョブのバーと同じ SSE)。終われば結果も出す
  follow(id, sel) {
    const el = this.$(sel);
    const show = (j) => {
      if (!["queued", "running"].includes(j.status)) { el.innerHTML = ""; return; }
      const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
      el.innerHTML = j.status === "queued" ? `<span class="muted">${t("待機中")}</span>`
        : `<span class="jb-bar"><span style="width:${pct}%"></span></span> ${pct}% <span class="muted">${esc(j.stage || "")}</span>`;
    };
    show({ status: "queued" });
    let es;
    try { es = new EventSource(`/api/jobs/${id}/events`); } catch { return; }
    es.onmessage = (m) => {
      const j = JSON.parse(m.data);
      show(j);
      if (["done", "error", "cancelled"].includes(j.status)) { es.close(); this.jobEnded(j); }
    };
    es.onerror = () => { es.close(); el.innerHTML = ""; };
  }

  // app.js の onJobFinished(と上の SSE)から。同じジョブは 1 回だけ
  async jobEnded(j) {
    const which = Object.keys(this.jobs).find((k) => this.jobs[k] === j.id) || null;
    if (!which) {
      if ((j.kind === "direction_key" || j.kind === "direction_key_pipeline") && j.status === "done" && this.loaded) await this.refresh();   // 別のところで作った鍵も一覧に
      return;
    }
    if (this.handled.has(j.id)) return;
    this.handled.add(j.id);
    this.$({ build: "#kx-prog", pipe: "#kx-prog", check: "#kc-prog", exp: "#kx-eprog", verify: "#kx-vprog" }[which]).innerHTML = "";
    if (j.status !== "done") {
      const e = j.error || {};
      this.status(which, j.status === "error" ? `${t("失敗")}: ${e.message || ""}${e.hint ? `(${e.hint})` : ""}` : t("止めました"), j.status === "error");
      return;
    }
    const r = j.result || {};
    const done = () => this.status(which, t("できました({s} 秒)", { s: fmt(r.seconds, 1) }));
    if (which === "build") {
      done();
      this.built = { kind: "build", params: j.params || {}, r };
      await this.refresh();
      this.fillKeys(r.name);
      this.renderBuilt();
    } else if (which === "pipe") {
      done();
      const params = { ...(j.params || {}), contrast: r.contrast_path };
      delete params.spec;
      this.built = { kind: "build", params, r: r.key || {} };
      this.preferDir = r.direction ? r.direction.name : null;
      if (r.alpha_hint !== null && r.alpha_hint !== undefined) this.alphaNext = +r.alpha_hint;
      this.pipeResult = r;
      await this.refresh();
      this.fillKeys(r.name);
      this.renderPipe();
      this.renderBuilt();
      if (r.check) {
        this.checked = r.check;
        this.prepareCheck(r.name, false);
        this.renderCheck();
      }
    } else if (which === "check") {
      done();
      this.checked = r;
      this.renderCheck();
    } else if (which === "exp") {
      done();
      this.experts = r;
      this.renderExperts();
    } else if (which === "verify") {
      this.status("verify", r.v30 ? t("合格(V30)") : t("不合格(V30)"), !r.v30);
      this.verified = r;
      this.renderVerify();
    }
  }

  // --- 結果: 方向から key-check まで(各段の時間と方向を使い回したか) ----------------------------------------------
  renderPipe() {
    const el = this.$("#kx-pipe"), r = this.pipeResult;
    if (!r) { el.innerHTML = ""; return; }
    const d = r.direction || {};
    const how = d.reused ? t("使い回しました") : d.rebuilt ? t("作り直しました") : t("作りました");
    const sub = r.check ? r.check.subset : null;
    el.innerHTML = `<div class="kx-res kx-pipe-sum"><p class="note">${t("方向から key-check まで")}: ${t("方向")} <b class="mono">${esc(d.name || "—")}</b>(${esc(how)}${d.reused ? "" : ` · ${fmt(d.seconds, 1)} ${t("秒")}`})
      → ${t("鍵")} <b class="mono">${esc(r.name)}</b>(${fmt(r.key_seconds, 1)} ${t("秒")})${r.check ? ` → key-check(subset ${esc(sub || "—")} · ${fmt(r.check_seconds, 1)} ${t("秒")})` : ""} · ${t("合計 {s} 秒", { s: fmt(r.seconds, 1) })}</p>
      ${(r.warnings || []).length ? `<ul class="evx-notes">${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}</div>`;
  }

  // --- 結果: 鍵の層ごとの表(作った直後か、一覧で開いた鍵) -------------------------------------------------------
  // 層の行に揃える: [{ layer, kn, gain, q_rank, build_auc, weak }]
  _rows() {
    const b = this.built;
    if (!b) return [];
    if (b.kind === "build") return (b.r.layers || []).map((x) => ({ layer: x.layer, kn: x.kn, gain: x.gain, q_rank: x.q_rank, build_auc: x.build_auc, weak: !!x.weak }));
    const k = b.r, info = k.info || {}, weak = new Set(info.weak || []);
    return (k.layers || []).map((L) => ({ layer: L, kn: (k.kn || {})[L], gain: (k.gain || {})[L], q_rank: (k.q_rank || {})[L],
                                          build_auc: (info.build_auc || {})[L], weak: weak.has(L) }));
  }

  renderBuilt() {
    const el = this.$("#kx-build"), b = this.built;
    if (!b) { el.innerHTML = ""; this.$("#kx-verify").classList.add("hidden"); return; }
    const r = b.r, isBuild = b.kind === "build", info = isBuild ? r : (r.info || {});
    const name = r.name, target = r.target || "ffn", mode = isBuild ? r.mode : info.mode;
    const dirs = this.dirNames(), want = this.$("#kx-dir") ? this.$("#kx-dir").value : null;
    const cn = isBuild ? r.contrast : info.contrast;
    const pick = this.preferDir && dirs.includes(this.preferDir) ? this.preferDir : want && dirs.includes(want) ? want : dirs.includes(cn) ? cn : dirs[0] || "";
    this.preferDir = null;
    let alpha = this.$("#kx-alpha") && this.$("#kx-alpha").value !== "" ? this.$("#kx-alpha").value : 1;   // 描き直しても手で入れた α は残す
    if (this.alphaNext !== null) { alpha = this.alphaNext; this.alphaNext = null; }
    const counts = r.counts || {};
    const head = `<p class="note">${t("鍵")} <b class="mono">${esc(name)}</b> · ${t("書き込み先")} <span class="mono">${esc(target)}</span> · mode <span class="mono">${esc(mode || "—")}</span>${
      mode === "same" ? ` <span class="muted">(${t("推論時と同じ文脈の返事")})</span>` : mode === "system" ? ` <span class="muted">(${t("system_a の文脈")})</span>` : ""} · positions <span class="mono">${esc(r.positions || "—")}</span>${
      (isBuild ? r.positions_b : info.positions_b) ? ` · positions_b <span class="mono">${esc(isBuild ? r.positions_b : info.positions_b)}</span>` : ""} · q_mode ${esc(r.q_mode || "—")} · subset ${esc(r.subset || "—")} · ${
      t("書き込みたい側 {a} 例 · 書き込みたくない側 {b} 例", { a: counts.a ?? "—", b: counts.b ?? "—" })}${counts.dt ? ` · dont_touch ${counts.dt}` : ""} · ${t("比較用の鍵 {n} 本", { n: r.n_pseudo ?? "—" })}${
      isBuild && r.matched && mode === "same" ? ` · ${t("正規表現が一致した例 {a} / {n}", { a: r.matched.A, n: r.matched.of })}` : ""}${isBuild && Number.isFinite(+r.seconds) ? ` · ${fmt(r.seconds, 1)} ${t("秒")}` : ""}</p>`;
    const add = `<div class="kx-add muted note">${t("書き込む方向")} <select id="kx-dir">${dirs.length ? dirs.map((d) => `<option${d === pick ? " selected" : ""}>${esc(d)}</option>`).join("") : `<option value="">${t("(なし)")}</option>`}</select>
      α <input id="kx-alpha" type="number" step="0.05" value="${esc(String(alpha))}" style="width:5em" title="${t("α = 1 で、観測した A − B の平均差 r_L をそのまま書く。負で打ち消す")}">
      <span>${dirs.length ? t("行の「+ レシピ」で、その層の keyed のユニットをレシピに足す。「key-verify」で LoRA が式どおりに書くかを確かめる") : t("方向がありません(「方向から key-check まで」で作れます)")}</span></div>`;
    const rows = this._rows().map((x) => `<tr${x.weak ? ` class="rnd"` : ""}><td>L${x.layer}</td><td title="${t("|k| / |μ|: μ のうち、書き込みたくない位置の部分空間に直交する割合")}">${fmt(x.kn, 3)}</td><td title="${t("1 / kn: 書き込みたくない位置への漏れが最大これだけ増える")}">${fmt(x.gain, 1)}</td><td>${x.q_rank ?? "—"}</td><td>${fmt(x.build_auc, 3)}</td>
      <td class="l">${x.weak ? `<span class="pp-warn">${t("弱い(kn < 0.05)")}</span>` : ""}</td>
      <td><button class="small" data-kadd="${x.layer}"${dirs.length ? "" : " disabled"} title="${t("この層の keyed のユニットをレシピに足す")}">${t("+ レシピ")}</button>
        <button class="small" data-kverify="${x.layer}"${dirs.length ? "" : " disabled"} title="${t("この層で、実行時の LoRA が式 α(κᵀx)d どおりに書くかを確かめる(V30、α = 1・2、数秒)")}">key-verify</button></td></tr>`).join("");
    const dropped = isBuild ? (r.dropped || []).map((d) => `L${d.layer}: ${d.error}`) : Object.entries(info.dropped || {}).map(([L, m]) => `L${L}: ${m}`);
    const warns = isBuild ? r.warnings || [] : [];
    const notes = isBuild ? r.notes || [] : [];
    el.innerHTML = `<div class="kx-res">${head}${add}
      <div class="evx-scroll"><table class="evx mono kx-tab"><thead><tr><th>${t("層")}</th><th>kn</th><th>gain</th><th>q_rank</th><th title="${t("作成に使った例での、書き込みたい位置 対 書き込みたくない位置の AUC(甘い)。key-check で保留した例を測る")}">${t("作成時の AUC")}</th><th class="l">${t("注意")}</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      ${dropped.length ? `<p class="err note">${t("鍵にならず外した層")}: ${dropped.map(esc).join(" / ")}</p>` : ""}
      ${warns.length ? `<ul class="evx-notes">${warns.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
      ${notes.length ? `<p class="muted note">${notes.map(esc).join("<br>")}</p>` : ""}
      ${isBuild && (r.samples || []).length ? `<details><summary class="muted note">${t("読んだ返事の例")}</summary>${r.samples.map((s) => `<p class="muted note kx-sample">${esc(s)}</p>`).join("")}</details>` : ""}
      <div class="pp-actions"><button class="small" data-kcheck="${esc(name)}" data-krun="1" title="${t("key-check の欄に入れて回す(作ったときの対照集合と返事、作成と反対側の subset)")}">${t("この鍵で key-check(subset {s})", { s: OPP[r.subset] || "odd" })}</button></div></div>`;
    this.$("#kx-verify").classList.remove("hidden");
    this.$("#kx-dir").addEventListener("change", () => { this.$("#kx-vres").innerHTML = ""; });
  }

  // --- 結果: 専門家の選ばれ方(direction_key_experts) --------------------------------------------------------------
  renderExperts() {
    const el = this.$("#kx-eres"), r = this.experts;
    if (!r) { el.innerHTML = ""; return; }
    const pct = (x) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${Math.round(+x * 1000) / 10}%`);
    const layers = (r.layers || []).map((ly) => {
      const rows = (ly.top || []).map((x) => {
        const dt = Object.entries(x.rate_dt || {}).map(([g, v]) => `${esc(g)} ${pct(v)}`).join(" · ");
        return `<tr><td>E${x.expert}</td><td>${pct(x.rate_a)}</td><td>${pct(x.rate_b)}</td><td>${x.diff >= 0 ? "+" : "−"}${pct(Math.abs(x.diff))}</td>
          <td>${x.lift === null || x.lift === undefined ? "—" : `${fmt(x.lift, 2)}×`}</td><td>${fmt(x.weight_a, 3)}</td><td>${fmt(x.rows_a, 1)}</td><td class="l muted">${dt || "—"}</td>
          <td><button class="small" data-kexp="${ly.layer}:${x.expert}" title="${t("書き込み先をこの専門家、層をこの層にして、鍵づくりの欄に入れる")}">${t("この専門家で鍵を作る")}</button></td></tr>`;
      }).join("");
      return `<p class="note">L${ly.layer} · ${t("専門家 {n} 個から {k} 個を選ぶ(一様なら {u})", { n: ly.n_expert, k: ly.n_used, u: pct(ly.base_rate) })} · ${
        t("書き込みたい位置 {a} · 書き込みたくない位置 {b}", { a: ly.positions_a, b: ly.positions_b })}</p>
        <div class="evx-scroll"><table class="evx mono kx-tab kx-exps"><thead><tr><th>${t("専門家")}</th>
          <th title="${t("書き込みたい位置でこの専門家が選ばれた割合(例ごとの割合の平均)")}">rate_a</th>
          <th title="${t("書き込みたくない位置でこの専門家が選ばれた割合")}">rate_b</th>
          <th title="${t("rate_a − rate_b。大きいほど、書き込みたくない位置ではあまり選ばれない")}">${t("差")}</th>
          <th title="${t("rate_a / rate_b")}">lift</th>
          <th title="${t("書き込みたい位置で選ばれたときのルーターの重みの平均(鍵の gate_a の見込み)")}">${t("重み")}</th>
          <th title="${t("書き込みたい位置でこの専門家が選ばれる行の数の見込み(rate_a × 書き込みたい位置の数)。少ないと鍵が例の本文に反応しやすい")}">${t("行の見込み")}</th>
          <th class="l" title="${t("dont_touch の群でこの専門家が選ばれた割合")}">dont_touch</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }).join("");
    el.innerHTML = `<div class="kx-res"><p class="muted note">${t("専門家の鍵は、その専門家が選ばれた位置の行だけで作る。rate_a が高いだけでなく、rate_b との差が大きく、行の見込みが多い専門家を選ぶ")} · ${
      t("対照集合")} ${esc(r.contrast || "—")} · positions <span class="mono">${esc(r.positions || "—")}</span> · subset ${esc(r.subset || "—")} · ${fmt(r.seconds, 1)} ${t("秒")}</p>${layers}
      ${(r.notes || []).length ? `<p class="muted note">${r.notes.map(esc).join("<br>")}</p>` : ""}</div>`;
  }

  // 専門家を選ぶ → 書き込み先 expert:e・層 L の鍵づくりの欄にする
  pickExpert(L, e) {
    this.$("#kx-target").value = "expert";
    this.$("#kx-expert").classList.remove("hidden");
    this.$("#kx-expert").value = String(e);
    this.$("#kx-layers").value = String(L);
    const base = this.$("#kx-name").value.trim() || this.contrastName() || "key";
    this.$("#kx-name").value = `${base.replace(/_L\d+E\d+$/, "")}_L${L}E${e}`;
    this.status("build", t("書き込み先を L{L} の専門家 {e} にしました。「方向から key-check まで」か「鍵を作る」で進めます", { L, e }));
    this.$("#kx-pipe-go").scrollIntoView({ block: "nearest" });
    this.$("#kx-pipe-go").focus();
  }

  // --- 結果: key-verify(V30) --------------------------------------------------------------------------------------
  renderVerify() {
    const el = this.$("#kx-vres"), r = this.verified;
    if (!r) { el.innerHTML = ""; return; }
    const ex = r.kind === "expert";
    const rows = (r.rows || []).map((x) => `<tr><td>${x.alpha >= 0 ? "+" : "−"}${fmt(Math.abs(x.alpha), 2)}</td><td>${x.rel_err_median == null ? "—" : (+x.rel_err_median).toExponential(1)}</td>
      <td>${x.n_big}/${x.n_tokens}</td><td>${fmt(x.cos_median, 4)}</td><td>${fmt(x.leak_b, 3)}</td>${
      ex ? `<td>${x.other_max === null || x.other_max === undefined ? "—" : (+x.other_max).toExponential(1)}</td><td>${x.selection_changed ?? "—"}</td>` : ""}<td>${mark(x.ok)}</td></tr>`).join("");
    el.innerHTML = `<div class="kx-res"><p class="note">L${r.layer} <span class="mono">${esc(r.tensor)}</span> · ${t("鍵")} <b class="mono">${esc(r.key)}</b> · ${t("方向")} <span class="mono">${esc(r.direction)}</span> · ‖d‖ ${fmt(r.d_norm, 3)} · kn ${fmt(r.kn, 3)}${
      r.post_norm ? ` · ${t("post-norm の補正あり(一次近似)")}` : ""}${r.gate !== null && r.gate !== undefined ? ` · gate_a ${fmt(r.gate, 3)}` : ""}</p>
      <div class="evx-scroll"><table class="evx mono kx-tab kx-verify"><thead><tr><th>α</th>
        <th title="${t("|κᵀx| の大きいトークン(上位 25%)での ‖Δ − α(κᵀx)d‖ / ‖α(κᵀx)d‖ の中央値。合格 < 3%")}">${t("相対誤差の中央値")}</th>
        <th title="${t("相対誤差を測った行 / 全部の行(専門家の鍵は、その専門家が選ばれた (トークン, 枠))")}">${t("測った行")}</th>
        <th title="${t("出力の差 Δ と書く向き d の cos の中央値(1 に近いほど式どおり)")}">cos</th>
        <th title="${t("鍵の作成に使った書き込みたくない側の例での、例ごとの Δ の平均の大きさ ÷ (|α|‖d‖)。合格 < 0.1")}">${t("書き込みたくない側への漏れ")}</th>${
        ex ? `<th title="${t("ほかの専門家の枠の差(0 のはず、合格 < 1e-3)")}">${t("ほかの専門家の枠の差")}</th><th title="${t("LoRA を当てて選ばれる専門家が変わったトークンの数(0 のはず)")}">${t("選ばれ方の変化")}</th>` : ""}<th></th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note">${r.v30 ? `<span class="kx-ok">○</span> ${t("合格(V30)")}` : `<span class="kx-ng">×</span> ${t("不合格(V30)")}`} <span class="muted">${
        t("相対誤差の中央値 < 3%")}${r.judged_leak ? ` · ${t("書き込みたくない側への漏れ < 0.1")}` : ` · ${t("書き込みたくない側への漏れは判定なし")}`}${ex ? ` · ${t("ほかの専門家の枠の差 < 1e-3、選ばれ方の変化 0")}` : ""}</span></p>
      ${r.note ? `<p class="pp-warn note">${esc(r.note)}</p>` : ""}</div>`;
  }

  // --- 結果: key-check の表とヒストグラム ----------------------------------------------------------------------------
  renderCheck() {
    const el = this.$("#kc-res"), r = this.checked;
    if (!r) { el.innerHTML = ""; return; }
    const ov = r.overlap || {};
    const why = [];
    if (ov.subset_overlap) why.push(t("同じ対照集合で subset が重なっています(作成 {a}・確認 {b})", { a: ov.built_subset ?? "—", b: ov.subset ?? "—" }));
    if (ov.hash_overlap) why.push(t("作成に使った例と同じ (system, プロンプト) が {n} 個あります", { n: ov.hash_overlap }));
    const held = r.held_out ? `<p class="note kx-held"><span class="kx-ok">✓</span> ${t("保留した例で測りました(作成 {a}・確認 {b}、重なり 0)", { a: ov.built_subset ?? "—", b: ov.subset ?? "—" })}</p>`
      : `<p class="note err kx-held">⚠ ${t("作成に使った例と重なっています。AUC は甘い値です")}${why.length ? `: ${why.map(esc).join("、")}` : r.contrast ? "" : `: ${t("対照集合を渡していないので、作成時の例ごとの平均で測りました")}`}</p>`;
    const counts = r.counts || {};
    const cnt = Object.entries(counts).map(([g, c]) => `${esc(g === "A" || g === "B" ? groupLabel(g) : g.replace(/^dt:/, "dt "))} ${c.examples}/${c.of}`).join(" · ");
    const beat = r.all_beat_pseudo === true ? `<span class="kx-ok">○</span> ${t("どの層も比較用の鍵を超えました")}`
      : r.all_beat_pseudo === false ? `<span class="kx-ng">×</span> ${t("比較用の鍵を超えない層があります(鍵が本文を拾っているかもしれません。対照集合の本文をそろえる)")}` : "";
    const rows = (r.layers || []).map((x) => {
      const dt = Object.entries(x.dont_touch_median || {}).map(([g, v]) => `${esc(g)} ${fmt(v, 2)}`).join(" · ");
      // 返事の冒頭(返事の最初の 3 トークンを決める位置 0〜2)での ρ: 書き込みたくない位置と dont_touch の群ごとに、位置ごとの
      // 中央値が最大の位置。サーバの key-check の警告(keyed_build._start_hot)と同じ選び方で、0.4 以上(例 2 つ以上)は赤
      const sr = x.start_rho || {};
      const st = [...(sr.B ? [["B", sr.B]] : []), ...Object.entries(sr.dont_touch || {})]
        .map(([g, v]) => [g, startPeak(v)]).filter(([, p]) => p)
        .map(([g, p]) => `<span class="${p.hot ? "kx-ng" : ""}">${esc(g === "B" ? groupLabel(g) : g)} ${fmt(p.m, 2)}${t("(位置 {j})", { j: p.j })}</span>`).join(" · ");
      return `<tr${x.layer === r.best_layer ? ` class="best"` : ""} data-kl="${x.layer}"><td>L${x.layer}${x.layer === r.best_layer ? " ★" : ""}</td>
        <td>${fmt(x.auc, 3)}</td><td>${fmt(x.auc_token, 3)}</td><td class="muted">${fmt((x.pseudo_auc_token || {}).p95 ?? (x.pseudo_auc || {}).p95, 3)}</td><td>${mark(x.beats_pseudo_auc)}</td>
        <td>${fmt(x.selectivity, 3)}</td><td class="muted">${fmt((x.pseudo_selectivity || {}).p95, 3)}</td><td>${mark(x.beats_pseudo)}</td>
        <td class="muted">${fmt(x.b_split_auc, 3)}</td><td class="l">${dt || "—"}${st ? `<br><span class="muted">${t("返事の冒頭")}:</span> ${st}` : ""}</td><td class="muted">${fmt(x.kn, 3)}</td></tr>`;
    }).join("");
    const hists = (r.layers || []).map((x) => `<figure class="kx-fig" data-kl="${x.layer}">${histSVG(x.hist, { title: `L${x.layer} · AUC ${fmt(x.auc_token ?? x.auc, 3)} · ${t("選択性")} ${fmt(x.selectivity, 2)}` })}</figure>`).join("");
    const groups = new Set();
    for (const x of r.layers || []) for (const g of Object.keys(x.hist || {})) if (g !== "edges") groups.add(g);
    let di = 0;
    const legend = [...groups].sort((a, b) => (a === "A" ? -1 : b === "A" ? 1 : a === "B" ? -1 : b === "B" ? 1 : a < b ? -1 : 1))
      .map((g) => `<span><i class="kx-sw ${groupCls(g, g.startsWith("dt:") ? di++ : 0)}"></i>${esc(groupLabel(g))}</span>`).join("");
    el.innerHTML = `<div class="kx-res">
      <p class="note">${t("鍵")} <b class="mono">${esc(r.key)}</b> · ${t("対照集合")} ${esc(r.contrast || t("(なし)"))}${r.subset ? ` (subset ${esc(r.subset)})` : ""} · mode ${esc(r.mode || "—")} · positions <span class="mono">${esc(r.positions || "—")}</span>${
        r.positions_b ? ` · positions_b <span class="mono">${esc(r.positions_b)}</span>` : ""}${cnt ? ` · ${cnt}` : ""} · ${t("比較用の鍵 {n} 本", { n: r.n_pseudo ?? "—" })} · ${fmt(r.seconds, 1)} ${t("秒")}</p>
      ${held}${beat ? `<p class="note">${beat}</p>` : ""}
      <div class="evx-scroll"><table class="evx mono kx-tab kx-check"><thead><tr><th>${t("層")}</th>
        <th title="${t("例ごとの ρ の平均での、書き込みたい位置 対 書き込みたくない位置の AUC")}">${t("AUC(例)")}</th><th title="${t("トークンごとの ρ での AUC")}">${t("AUC(トークン)")}</th>
        <th title="${t("比較用の鍵のトークン単位の AUC の 95 パーセンタイル")}">${t("比較用の鍵の p95")}</th><th title="${t("トークン単位の AUC が比較用の鍵の p95 を超えるか")}">${t("AUC で超える")}</th>
        <th title="${t("key-check の選択性: 例ごとの鍵の反応の強さ ρ の平均の差(書き込みたい位置 − 書き込みたくない位置)。書き込む量 α(κᵀx)d の両者の差にあたる。eval の選択性(残差の変化)とは別の量")}">${t("選択性")}</th><th title="${t("比較用の鍵の選択性の 95 パーセンタイル")}">${t("比較用の鍵の p95")}</th>
        <th title="${t("選択性が比較用の鍵の p95 を超え、例の AUC が 0.5 を超えるか(beats_pseudo)")}">${t("比較用の鍵を超える")}</th>
        <th title="${t("参考: 書き込みたくない位置を 2 つに分け、その差から作った鍵の AUC(0.5 付近なら、鍵が書き込みたくない位置の中の差を拾っていない)")}">${t("書き込みたくない位置の 2 分割")}</th>
        <th class="l" title="${t("dont_touch の群の ρ の中央値(0 に近いほど書き込まない)。下の段は、書き込みたくない位置と dont_touch の群の、返事の冒頭(返事の最初の 3 トークンを決める位置 0〜2。位置 0 は返事を書き始める直前の位置)での ρ の、位置ごとの中央値のうち最大のものとその位置(0.4 以上は赤: 返事の冒頭で鍵が反応している。下の警告と同じ決まり)")}">${t("dont_touch 中央値")}</th><th>kn</th></tr></thead><tbody>${rows}</tbody></table></div>
      <h4 class="note">${t("層ごとの鍵の反応の強さ ρ = κᵀx の分布(群ごとの割合。縦の点線: 0 = 何も足さない、1 = 書き込みたい位置の平均)")}</h4>
      <div class="evx-legend note kx-legend">${legend}</div>
      <div class="kx-hists">${hists}</div>
      ${(r.warnings || []).length ? `<ul class="evx-notes">${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
      ${(r.notes || []).length ? `<p class="muted note">${r.notes.map(esc).join("<br>")}</p>` : ""}</div>`;
  }

  // --- 保存した鍵の一覧 ------------------------------------------------------------------------------------------
  renderKeys() {
    const el = this.$("#kx-keys");
    if (!this.keys.length) { el.innerHTML = `<p class="muted note">${t("鍵はまだありません(上の「鍵を作る」か gne direction key)")}</p>`; return; }
    const rows = this.keys.map((k) => (k.error
      ? `<tr><td class="l">${esc(k.name)}</td><td class="l err" colspan="8">${esc(k.error)}</td></tr>`
      : `<tr><td class="l">${esc(k.name)}</td><td>${esc(k.target)}</td><td class="l">${k.layers.map((L) => `L${L}`).join(" ")}</td>
        <td class="l">${k.layers.map((L) => fmt((k.kn || {})[L], 2)).join(" ")}</td><td>${esc(k.q_mode)}</td><td class="l">${esc(k.positions)}</td><td>${esc(k.subset)}</td>
        <td>${k.n_pseudo}</td><td>${(k.counts || {}).a ?? "—"} / ${(k.counts || {}).b ?? "—"}</td>
        <td><button class="small" data-kview="${esc(k.name)}">${t("見る")}</button> <button class="small" data-kcheck="${esc(k.name)}">${t("確かめる")}</button></td></tr>`)).join("");
    el.innerHTML = `<div class="evx-scroll"><table class="evx mono kx-keys"><thead><tr><th class="l">${t("名前")}</th><th>${t("書き込み先")}</th><th class="l">${t("層")}</th><th class="l">kn</th><th>q_mode</th><th class="l">positions</th><th>subset</th><th>${t("比較用の鍵")}</th><th>${t("例(書き込みたい側 / 書き込みたくない側)")}</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  async view(name) {
    try {
      const k = await getJSON(`/api/keys/${encodeURIComponent(name)}`);
      this.built = { kind: "saved", r: { ...k, n_pseudo: k.n_pseudo } };
      this.renderBuilt();
      this.$("#kx-build").scrollIntoView({ block: "nearest" });
    } catch (e) { this.status("build", `${t("エラー")}: ${e.message}`, true); }
  }

  // 鍵を key-check の欄に入れる: 作ったときの対照集合(作った直後なら分かる)と反対側の subset、同じ返事のファイル
  prepareCheck(name, scroll = true) {
    this.fillKeys(name);
    const b = this.built && this.built.r && this.built.r.name === name ? this.built : null;
    const k = this.keys.find((x) => x.name === name);
    const subset = (b && b.r.subset) || (k && k.subset) || "even";
    this.$("#kc-subset").value = OPP[subset] || "odd";
    if (b && b.kind === "build") {
      const p = b.params || {}, sel = this.$("#kc-contrast");
      const c = this.findContrast(p.contrast || null, p.spec && p.spec.name ? String(p.spec.name) : null);
      const v = c ? c.path : p.contrast;
      if (v && [...sel.options].some((o) => o.value === v)) sel.value = v;
      this.$("#kc-replies").value = typeof p.replies_a === "string" ? p.replies_a : "";
    }
    if (scroll) this.$("#kc-go").scrollIntoView({ block: "nearest" });
  }

  onClick(ev) {
    const a = ev.target.closest("[data-kadd]");
    if (a) {
      const b = this.built;
      if (!b) return;
      const dir = this.$("#kx-dir") ? this.$("#kx-dir").value : "", alpha = +(this.$("#kx-alpha") || {}).value;
      if (!dir) return;
      const { kind, expert } = targetKind(b.r.target);
      this.onAddKeyed({ layer: +a.dataset.kadd, kind, expert, alpha: Number.isFinite(alpha) ? alpha : 1, direction: dir, key: b.r.name });
      a.textContent = t("足しました");
      return;
    }
    const vb = ev.target.closest("[data-kverify]");
    if (vb) { this.runVerify(+vb.dataset.kverify); return; }
    const x = ev.target.closest("[data-kexp]");
    if (x) { const [L, e] = x.dataset.kexp.split(":").map(Number); this.pickExpert(L, e); return; }
    const c = ev.target.closest("[data-kcheck]");
    if (c) { this.prepareCheck(c.dataset.kcheck); if (c.dataset.krun) this.runCheck(); return; }
    const v = ev.target.closest("[data-kview]");
    if (v) this.view(v.dataset.kview);
  }
}
