// 層の幾何ビュー(gne heads geometry): heads writes の材料の、層ごとの更新ベクトル d_l(A 側の平均の更新 − B 側の平均の更新)を比べる。
// 層どうしのコサイン類似度(ヒートマップ)、特異値分解(寄与率と基準、モードごとの層のパターン)、方向ベクトルとの cos、
// 質問ごとの更新があれば同じ層の中の質問に沿った分解、heads refine の方向との比べ、方向ベクトルとして保存。
// 計算は保存した配列だけで、サーバがその場で返す(GPU を使わない)。図は Canvas 2D。値はマウス・キーボードで読める行と、表で読める。
import { getJSON, postJSON } from "./api.js";
import { t, tl } from "./i18n.js";
import { esc } from "./panels.js";
import { noteHTML } from "./ui.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
// 整数の並びを 0-3,8 の形に(細かい設定の欄に戻すとき)
const compactList = (xs) => {
  const out = [];
  for (let i = 0; i < xs.length; i++) {
    let j = i;
    while (j + 1 < xs.length && xs[j + 1] === xs[j] + 1) j++;
    out.push(j > i ? `${xs[i]}-${xs[j]}` : `${xs[i]}`);
    i = j;
  }
  return out.join(",");
};
const sgn = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
// 短い札は、辞書に別の場面の訳がある語(注意 = notes など)とぶつからないよう tl で英語を直に書く
const kindName = (k) => ({ attention: tl("注意", "attention"), linear_attention: tl("線形注意", "linear attention"), recurrent: tl("再帰", "recurrent"),
  conv: tl("畳み込み", "conv"), mlp: "MLP", none: tl("なし", "none") }[k] || k);
const partName = (p) => ({ layer: tl("層全体", "whole layer"), attn: tl("注意", "attention"), mlp: "MLP" }[p] || p);
const visitName = (v) => tl(`${v} 回目`, `visit ${v}`);

// --- 色(:root の変数から。符号のある値は 2 色 + 灰の真ん中、OKLab で混ぜる) ---------------------------------------
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function hex2rgb(h) { const n = parseInt(h.replace("#", ""), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const gam = (c) => Math.round(255 * Math.min(1, Math.max(0, c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055)));
function toLab([r, g, b]) {
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function fromLab([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [gam(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s), gam(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    gam(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)];
}
function diverging() {
  const neg = toLab(hex2rgb(cssVar("--div-neg"))), mid = toLab(hex2rgb(cssVar("--div-mid"))), pos = toLab(hex2rgb(cssVar("--div-pos")));
  const steps = 64, tab = [];
  for (let i = -steps; i <= steps; i++) {
    const f = Math.abs(i) / steps, end = i < 0 ? neg : pos;
    const c = fromLab(mid.map((v, k) => v + (end[k] - v) * f));
    tab.push(`rgb(${c[0]},${c[1]},${c[2]})`);
  }
  return (v, range) => tab[Math.round(Math.max(-1, Math.min(1, v / range)) * steps) + steps];
}

export class GeometryView {
  constructor(root, { onRun } = {}) {
    this.root = root;
    this.onRun = onRun;            // (kind, params) → ジョブを投げる(質問ごとの更新を作る heads_writes)
    this.list = [];
    this.data = null;
    this.loadSeq = 0;
    this.mode = "rows";            // 特異値分解の版(rows = 行の大きさをそろえる / none)
    this.range = "one";            // 色の範囲(one = ±1 / max = 対角以外の最大)
    this.loopLine = true;          // 同じ重みの 1 回目と 2 回目の線を枠で示す
    this.cursor = null;            // ヒートマップの選んだセル [i, j](キーボードとマウス)
    this.wsel = null;              // 質問に沿った分解で選んだ行
    this.block = { r: "all", c: "all" };   // ヒートマップで見る部品(行・列)。行が多いときは部品ごとに切り出して見る
    this.zoom = 1;                 // ヒートマップのセルの倍率
    root.innerHTML = `<div class="evx-tools geo-tools">
        <label class="muted">${t("材料(heads writes)")} <select id="geo-pick"></select></label>
        <label class="muted" title="${t("行の単位。注意 = その層の注意の出力全体(線形注意の層では線形注意の出力)")}">${t("行")} <select id="geo-parts">
          <option value="layer">${t("層全体")}</option><option value="attn,mlp">${t("注意と MLP")}</option><option value="layer,attn,mlp">${t("層全体・注意・MLP")}</option></select></label>
        <label class="muted" title="${t("寄与率・層ごとの説明率を何番目まで出すか")}">${t("上位")} <input id="geo-k" type="number" min="1" max="12" value="6" size="3"></label>
        <label class="muted" title="${t("A と B のラベルを入れ替えて作る基準の回数(質問ごとの更新があるときだけ。0 で切る)")}">${t("入れ替え")} <input id="geo-perm" type="number" min="0" max="2000" value="200" size="4"></label>
        <button class="main" id="geo-go">${t("計算する")}</button><span id="geo-status" class="muted note" role="status"></span></div>
      <details class="geo-more"><summary class="muted note">${t("細かい設定(CLI の指定と同じ。空なら既定)")}</summary>
        <div class="evx-tools">
          <label class="muted" title="${t("使う層(例 10-30、3,7,12)。空なら材料の全部の層(--layers)")}">${t("層")} <input id="geo-layers" size="10" spellcheck="false" placeholder="${t("全部")}"></label>
          <label class="muted" title="${t("計算から除く残差の次元(例 1234,567)。表の「上位 5 次元」を見て手で決める(--exclude-dims)")}">${t("除く次元")} <input id="geo-exdims" size="12" spellcheck="false" placeholder="${t("なし")}"></label>
          <label class="muted" title="${t("比べる方向ベクトル(カンマ区切り)。空なら heads refine の方向(<名前>_r1 など)があればそれ(--compare)")}">${t("比べる方向")} <input id="geo-compare" size="18" spellcheck="false" placeholder="${t("refine の方向")}"></label>
          <label class="muted"><input type="checkbox" id="geo-nocompare"> ${t("比べない")}</label>
          <label class="muted" title="${t("半分どうしの cos がこれ未満の行に信用しない印を付ける(--min-half-cos、既定 0.5)")}">${t("信用しない印の基準")} <input id="geo-minhalf" type="number" min="-1" max="1" step="0.05" value="0.5" size="4"></label>
          <label class="muted" title="${t("特異値分解の前に行の大きさをそろえるか(--normalize)")}">${t("特異値分解")} <select id="geo-normalize"><option value="both">${t("両方")}</option><option value="rows">${t("行の大きさをそろえる")}</option><option value="none">${t("そろえない")}</option></select></label>
          <label class="muted" title="${t("ラベルの入れ替えの乱数(--perm-seed)")}">${t("乱数")} <input id="geo-seed" type="number" min="0" value="0" size="4"></label>
        </div></details>
      <div id="geo-make" class="note"></div>
      ${noteHTML(t("heads writes の材料の、層ごとの更新ベクトル(A 側の平均の更新 − B 側の平均の更新。その層が残差に足した量)を、層どうしのコサイン類似度と特異値分解で比べる。モデルは流さない。質問ごとの更新(heads writes --save-q)があれば、ラベル入れ替えの基準と、同じ層の中の質問に沿った分解も出る。"), "muted note evx-help")}
      <div id="geo-body"></div>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#geo-pick").addEventListener("change", (ev) => this.load(ev.target.value));
    this.$("#geo-go").addEventListener("click", () => this.run());
    this.tip = document.createElement("div");
    this.tip.className = "geo-tip mono hidden";
    root.appendChild(this.tip);
    new ResizeObserver(() => this.draw()).observe(root);
  }

  status(msg, err = false) {
    const el = this.$("#geo-status");
    el.textContent = msg || "";
    el.classList.toggle("err", !!err);
  }

  async refresh(select) {
    let list;
    try { list = await getJSON("/api/geometry"); } catch (e) { list = []; this.status(e.message, true); }
    this.list = list;
    const pick = this.$("#geo-pick");
    pick.innerHTML = list.length
      ? list.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}${x.has_q ? ` · ${t("質問ごと")}` : ""}${x.has_result ? "" : ` · ${t("未計算")}`}</option>`).join("")
      : `<option value="">${t("(なし)")}</option>`;
    this.$("#geo-go").disabled = !list.length;
    const name = select || (this.data && this.data.material && this.data.material.name) || (list[0] && list[0].name);
    if (name && list.some((x) => x.name === name)) { pick.value = name; await this.load(name); }
    else this.render();
  }

  meta(name) { return this.list.find((x) => x.name === name) || {}; }

  async load(name) {
    const seq = ++this.loadSeq;
    const m = this.meta(name);
    this.$("#geo-perm").disabled = !m.has_q;
    this.makeHTML(m);
    if (!name || !m.has_result) {
      this.data = null;
      this.render(name ? t("{n} はまだ計算していません。「計算する」で作ります(数秒、GPU を使わない)。", { n: name }) : null);
      return;
    }
    let d;
    try { d = await getJSON(`/api/geometry/${encodeURIComponent(name)}`); }
    catch (e) { if (seq === this.loadSeq) { this.data = null; this.render(); this.status(e.message, true); } return; }
    if (seq !== this.loadSeq) return;
    this.data = d;
    this.cursor = null;
    this.wsel = null;
    this.syncControls();
    this.render();
  }

  syncControls() {
    const st = (this.data && this.data.settings) || {};
    if (st.parts) this.$("#geo-parts").value = st.parts.join(",");
    if (st.top_k) this.$("#geo-k").value = st.top_k;
    if (st.permutations !== undefined) this.$("#geo-perm").value = st.permutations;
    this.$("#geo-layers").value = st.layers ? compactList(st.layers) : "";
    this.$("#geo-exdims").value = (st.exclude_dims || []).join(",");
    this.$("#geo-minhalf").value = st.min_half_cos ?? 0.5;
    this.$("#geo-normalize").value = st.normalize && st.normalize.length === 1 ? st.normalize[0] : "both";
    this.$("#geo-seed").value = st.seed ?? 0;
    this.block = { r: "all", c: "all" };
  }

  // 質問ごとの更新が無い材料: heads writes --save-q をジョブで流して作り直す(材料を作ったときの設定で。対照集合は方向を作ったもの)
  makeHTML(m) {
    const el = this.$("#geo-make");
    if (!el) return;
    if (!m || !m.name || m.has_q || !this.onRun || !m.direction) { el.innerHTML = ""; return; }
    el.innerHTML = `<span class="muted">${t("この材料には質問ごとの更新がありません(ラベル入れ替えの基準と質問に沿った分解に要る)。")}</span>
      <button class="small" id="geo-make-q" title="${t("heads writes --save-q を、材料を作ったときの設定(方向・選ぶ層・測る位置・質問の数)でジョブとして流し、同じ名前の材料を作り直す。上位の部品を 1 個ずつ流し直す確かめ(verify)は省くので、元の材料にあった確かめの結果は残らない。GPU を使う(1〜2 分)")}">${t("質問ごとの更新を作る")}</button>`;
    el.querySelector("#geo-make-q").addEventListener("click", () => {
      const p = { direction: m.direction, name: m.name, save_q: true, verify_top: 0 };
      for (const k of ["select_layer", "positions", "subset", "max_examples", "loop_direction"]) if (m.settings && m.settings[k] !== undefined && m.settings[k] !== null) p[k] = m.settings[k];
      if (m.settings && m.settings.kinds) p.kinds = m.settings.kinds.join(",");
      this.onRun("heads_writes", p);
      this.status(t("質問ごとの更新を作るジョブを投げました。終わると一覧が更新されます"));
    });
  }

  async run() {
    const name = this.$("#geo-pick").value;
    if (!name) return;
    const p = { parts: this.$("#geo-parts").value, top_k: +this.$("#geo-k").value || 6,
      normalize: this.$("#geo-normalize").value, seed: Math.max(0, +this.$("#geo-seed").value || 0) };
    if (this.meta(name).has_q) p.permutations = Math.max(0, +this.$("#geo-perm").value || 0);
    const mh = this.$("#geo-minhalf").value.trim();
    if (mh !== "" && Number.isFinite(+mh)) p.min_half_cos = +mh;
    const lay = this.$("#geo-layers").value.trim(), ex = this.$("#geo-exdims").value.trim(), cmp = this.$("#geo-compare").value.trim();
    if (lay) p.layers = lay;
    if (ex) p.exclude_dims = ex;
    if (this.$("#geo-nocompare").checked) p.compare = "";
    else if (cmp) p.compare = cmp;
    this.status(t("計算しています…"));
    this.$("#geo-go").disabled = true;
    try {
      const d = await postJSON(`/api/geometry/${encodeURIComponent(name)}`, p);
      this.data = d;
      this.cursor = null;
      this.wsel = null;
      const m = this.meta(name);
      m.has_result = true;
      this.status(t("{s} 秒で計算しました", { s: fmt(d.seconds, 1) }));
      this.render();
    } catch (e) { this.status(e.message, true); }
    finally { this.$("#geo-go").disabled = false; }
  }

  // --- 描く ---------------------------------------------------------------------------------------------
  render(msg) {
    const body = this.$("#geo-body"), d = this.data;
    if (!d) {
      body.innerHTML = noteHTML(msg || t("材料がありません。gne heads writes --name <方向>(質問ごとの分解も見るなら --save-q)で作ります。"), "muted note");
      return;
    }
    const m = d.material, rows = d.rows, sm = d.similarity;
    const ck = [t("半分ずつの平均と全体の差 最大 {v}", { v: (+m.halves_mean_err_max).toExponential(1) })];
    if (m.cumsum_cos_select !== undefined && m.cumsum_cos_select !== null) ck.push(t("更新の累積と方向ベクトルの cos: 選ぶ層 {a}(最小 {b})", { a: fmt(m.cumsum_cos_select, 3), b: fmt(m.cumsum_cos_min, 3) }));
    if (m.check_layer_rel_max !== undefined) ck.push(t("部品から組み立てた更新と残差の差 最大 {v}%", { v: fmt(100 * m.check_layer_rel_max, 2) }));
    const reading = d.reading || [];
    const loop = d.loop;
    body.innerHTML = `
      <div class="geo-reading" role="note"><b>${t("読み取りの注意")}</b>: ${esc(reading[0] || "")}
        <details><summary class="muted">${t("ほかの注意({n})", { n: Math.max(0, reading.length - 1) })}</summary><ul>${reading.slice(1).map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details></div>
      <p class="note mono">${esc(m.name)} <span class="muted">· ${t("方向")} ${esc(m.direction || "—")} · ${t("選ぶ層")} L${m.select_layer} · A ${m.n_a} / B ${m.n_b} ${t("問")}
        · ${t("行")} ${sm.n}(${(d.settings.parts || []).map(partName).join("・")})${m.q ? ` · ${t("質問ごとの更新あり")}(${m.q.paired ? t("対") : t("側ごとに別の質問")})` : ""}
        · ${esc(d.arch.arch)}${d.arch.checked ? "" : ` (${t("確かめていない構造")})`}</span></p>
      <p class="note muted">${t("材料の確かめ")}: ${ck.join(" / ")}</p>
      ${(d.notes || []).length ? `<ul class="geo-notes">${d.notes.map((n) => `<li>${esc(n.text)}</li>`).join("")}</ul>` : ""}

      <h4 class="note">${t("1. 層 × 層のコサイン類似度")}</h4>
      <div class="geo-ctl">
        <span class="seg" role="group" aria-label="${t("色の範囲")}"><button data-range="one" class="${this.range === "one" ? "on" : ""}">±1</button><button data-range="max" class="${this.range === "max" ? "on" : ""}">${t("±対角以外の最大")}</button></span>
        ${loop ? `<label class="muted"><input type="checkbox" id="geo-loopline" ${this.loopLine ? "checked" : ""}> ${t("同じ重みの 1 回目と 2 回目を枠で示す")}</label>` : ""}
        ${(d.settings.parts || []).length > 1 ? ["r", "c"].map((k) => `<label class="muted">${k === "r" ? t("行の部品") : t("列の部品")} <select data-block="${k}">
          <option value="all">${t("全部")}</option>${d.settings.parts.map((p) => `<option value="${p}" ${this.block[k] === p ? "selected" : ""}>${partName(p)}</option>`).join("")}</select></label>`).join("") : ""}
        <span class="seg" role="group" aria-label="${t("セルの大きさ")}">${[1, 2, 4].map((z) => `<button data-zoom="${z}" class="${this.zoom === z ? "on" : ""}">${z === 1 ? t("自動") : `×${z}`}</button>`).join("")}</span>
        <span class="muted note">${t("対角以外の |cos| 平均 {a}・最大 {b}", { a: fmt(sm.offdiag_abs_mean, 3), b: fmt(sm.offdiag_abs_max, 3) })}${sm.above_null95 ? ` · ${t("ラベル入れ替えの 95% を超える組 {n} / {of}", sm.above_null95)}` : ""}</span></div>
      <div class="geo-legend note" id="geo-legend"></div>
      <p class="note warn" id="geo-heat-hint" hidden>${t("行が多くてセルが小さいので、行・列の部品を選んで切り出すか、セルを大きくするか、下の表で読みます")}</p>
      <div class="geo-heat-wrap"><canvas id="geo-heat" tabindex="0" role="img" aria-describedby="geo-readout"
        aria-label="${t("層どうしのコサイン類似度のヒートマップ。矢印キーでセルを選ぶと、下の行に値が出る")}"></canvas></div>
      <p class="note mono geo-readout" id="geo-readout" aria-live="polite">${t("セルにマウスを当てるか、図を選んで矢印キーで動かすと、ここに値が出ます")}</p>
      <details id="geo-heat-table"><summary class="muted note">${t("表で見る(行ごとの値と、類似度の行列)")}</summary><div class="geo-tables"></div></details>
      ${loop ? this.loopHTML(loop) : ""}

      <h4 class="note">${t("2. 特異値分解(層に沿った分解)")}</h4>
      <div class="geo-ctl"><span class="seg" role="group" aria-label="${t("特異値分解の版")}">${Object.keys(d.svd).map((k) => `<button data-mode="${k}" class="${this.mode === k ? "on" : ""}">${k === "rows" ? t("行の大きさをそろえる") : t("そろえない")}</button>`).join("")}</span></div>
      <div class="geo-legend note" id="geo-svd-legend"></div>
      <canvas id="geo-svd" role="img" aria-label="${t("モードごとの寄与率と基準の棒グラフ(数は下の表)")}"></canvas>
      <div id="geo-svd-table"></div>
      <p class="note muted">${t("モードごとの層のパターン(U の列。横は行の並び、縦はその行がその向きを使う重み)")}</p>
      <canvas id="geo-u" tabindex="0" role="img" aria-describedby="geo-u-readout" aria-label="${t("モードごとの層のパターン")}"></canvas>
      <p class="note mono geo-readout" id="geo-u-readout"></p>

      <h4 class="note">${t("3. 方向ベクトルとの関係")}</h4>
      <p class="note muted">${t("cos(d_l, r̂_l) = その層の更新の差と、CHiPS がその層で使う方向との cos。cos(d_l, r̂_S) = 選ぶ層の方向との cos。2 乗がその方向に沿う割合")}</p>
      <canvas id="geo-dir" tabindex="0" role="img" aria-describedby="geo-dir-readout" aria-label="${t("行ごとの方向ベクトルとの cos")}"></canvas>
      <p class="note mono geo-readout" id="geo-dir-readout"></p>

      ${d.within ? this.withinHTML(d) : `<h4 class="note">${t("4. 同じ層の中の質問ごとの差")}</h4>${noteHTML(t("質問ごとの更新が無いので出ていません。gne heads writes --name <方向> --save-q で作ってから計算します。"), "muted note")}`}
      ${(d.compare || []).length ? this.compareHTML(d) : ""}
      ${this.exportHTML(d)}`;
    this.$$ = (s) => [...body.querySelectorAll(s)];
    for (const b of this.$$("[data-range]")) b.addEventListener("click", () => { this.range = b.dataset.range; for (const x of this.$$("[data-range]")) x.classList.toggle("on", x === b); this.drawHeat(); });
    for (const b of this.$$("[data-mode]")) b.addEventListener("click", () => { this.mode = b.dataset.mode; for (const x of this.$$("[data-mode]")) x.classList.toggle("on", x === b); this.drawSvd(); this.drawU(); });
    for (const sel of this.$$("[data-block]")) sel.addEventListener("change", () => {
      this.block[sel.dataset.block] = sel.value;
      this.cursor = null;
      const det = body.querySelector("#geo-heat-table");                // 表も選んだ部品で作り直す
      delete det.dataset.done;
      if (det.open) { det.querySelector(".geo-tables").innerHTML = this.tablesHTML(this.data); det.dataset.done = "1"; }
      this.drawHeat();
    });
    for (const b of this.$$("[data-zoom]")) b.addEventListener("click", () => { this.zoom = +b.dataset.zoom; for (const x of this.$$("[data-zoom]")) x.classList.toggle("on", x === b); this.drawHeat(); });
    const ll = body.querySelector("#geo-loopline");
    if (ll) ll.addEventListener("change", () => { this.loopLine = ll.checked; this.drawHeat(); });
    const det = body.querySelector("#geo-heat-table");
    det.addEventListener("toggle", () => { if (det.open && !det.dataset.done) { det.querySelector(".geo-tables").innerHTML = this.tablesHTML(d); det.dataset.done = "1"; } });
    this.wireHeat();
    this.wireBars("#geo-u", "#geo-u-readout", () => this.uHits);
    this.wireBars("#geo-dir", "#geo-dir-readout", () => this.dirHits);
    for (const tr of this.$$("tr[data-wi]")) tr.addEventListener("click", () => this.pickWithin(+tr.dataset.wi));
    const ex = body.querySelector("#geo-export-go");
    if (ex) ex.addEventListener("click", () => this.doExport());
    this.draw();
    if (d.within && this.wsel !== null) this.pickWithin(this.wsel);
  }

  loopHTML(loop) {
    const c = (x) => (x ? `${sgn(x.cos)} <span class="muted">(${sgn(x.cos_halves[0])}, ${sgn(x.cos_halves[1])})</span>` : "—");
    return `<details class="geo-loop" open><summary class="note">${t("同じ重みの 1 回目と 2 回目(段のずれ {s})。cos の右のかっこは、質問を半分ずつにして測った値", { s: loop.shift })}</summary>
      <table class="tbl mono"><thead><tr><th>${t("重みの層")}</th><th>${visitName(1)}</th><th>${visitName(2)}</th><th>${t("層全体 cos")}</th><th>${t("注意 cos")}</th><th>${t("MLP cos")}</th><th>${t("大きさ 1 回目 / 2 回目")}</th></tr></thead>
      <tbody>${loop.pairs.map((p) => `<tr><td>${p.stored}</td><td>L${p.first}</td><td>L${p.second}</td><td>${c(p.layer)}</td><td>${c(p.attn)}</td><td>${c(p.mlp)}</td>
        <td>${p.layer ? `${fmt(p.layer.norm_first, 3)} / ${fmt(p.layer.norm_second, 3)}` : "—"}</td></tr>`).join("")}</tbody></table></details>`;
  }

  withinHTML(d) {
    const ws = d.within_summary, w = d.within;
    const bar = (v) => (v === null || v === undefined ? "—" : `<span class="geo-cellbar" style="--v:${Math.max(0, Math.min(1, v))}"></span>${fmt(v)}`);
    return `<h4 class="note">${t("4. 同じ層の中の質問ごとの差")}</h4>
      <p class="note muted">${ws.paired ? t("対になった対照集合: 質問ごとの差 e_q = u_q^A − u_q^B") : t("側ごとに別の質問: 質問ごとの差 e_q = u_q^A − (B 側の平均)")}。
        ${t("割合は Σ‖e_q‖² に対して。取り残し = 平均の向き d̂ に直交する分(1 本の方向で消しても残る、質問ごとのばらつき)。安定度 = 偶数番目・奇数番目の質問で別々に求めた取り残しの向き(1 番目)どうしの |cos|。")}
        ${ws.paired ? "" : t("A/B = A 側の片方の半分で求めた向きに沿った、A 側のもう半分の広がり ÷ B 側の質問どうしの広がり。B/A は役割を入れ替えた同じ量。両方とも 1 より大きければ、それぞれの側の話題の違いの向き。")}</p>
      <div class="tbl-wrap" tabindex="0" role="region" aria-label="${t("同じ層の中の質問ごとの差")}"><table class="tbl mono geo-within"><thead><tr><th class="l">${t("行")}</th><th>${t("d̂ で説明")}</th><th>${t("r̂_l で説明")}</th><th>${t("r̂_S で説明")}</th>
        <th>${t("取り残し")}</th><th>${t("取り残し 1 番目")}</th><th>${t("安定度")}</th>${ws.paired ? "" : `<th>A/B</th><th>B/A</th>`}</tr></thead>
      <tbody>${w.map((x, i) => `<tr data-wi="${i}" class="pick${d.rows[i].reliable ? "" : " geo-unrel"}"><td class="l">${esc(x.label)}${d.rows[i].reliable ? "" : " ×"}</td><td>${bar(x.share_mean)}</td><td>${bar(x.share_own)}</td><td>${bar(x.share_select)}</td>
        <td>${bar(x.leftover_total)}</td><td>${fmt((x.leftover_explained || [])[0], 3)}</td><td>${fmt((x.leftover_stability || [])[0])}</td>
        ${ws.paired ? "" : `<td>${fmt((x.leftover_ab_ratio || [])[0])}</td><td>${fmt((x.leftover_ba_ratio || [])[0])}</td>`}</tr>`).join("")}</tbody></table></div>
      <p class="note">${t("平均の向き d̂ との cos が低い A 側の質問(信用できる層全体の行 {n} 行の平均)", { n: ws.rows_used })}:</p>
      <ol class="geo-qlist">${ws.lowest.map((x) => `<li><span class="mono">${sgn(x.mean_cos)}</span> ${esc(x.text || `#${x.q}`)}</li>`).join("")}</ol>
      <div id="geo-wq"><p class="note muted">${t("表の行を選ぶと、その行の質問ごとの cos が出ます")}</p></div>`;
  }

  pickWithin(i) {
    const d = this.data;
    if (!d || !d.within || !d.within[i]) return;
    this.wsel = i;
    for (const tr of this.$$("tr[data-wi]")) tr.classList.toggle("on", +tr.dataset.wi === i);
    const qs = (d.within_summary.questions || []);
    const order = d.within[i].q_cos.map((c, j) => [c, j]).sort((a, b) => a[0] - b[0]);
    this.$("#geo-wq").innerHTML = `<p class="note">${t("{r} の質問ごとの cos(e_q, d̂)(低い順)", { r: esc(d.within[i].label) })}</p>
      <ol class="geo-qlist">${order.map(([c, j]) => `<li><span class="mono">${sgn(c)}</span> ${esc((qs[j] && qs[j].text) || `#${qs[j] ? qs[j].q : j}`)}</li>`).join("")}</ol>`;
  }

  compareHTML(d) {
    const K = Math.min(3, d.svd[this.mode] ? d.svd[this.mode].k : 3);
    return `<h4 class="note">${t("5. 2 本目の方向の候補との比べ")}</h4>
      <p class="note muted">${t("heads refine の方向は、組を両側に当てた状態で残りの質問だけを A にした残差の平均の差。特異値分解の向きは層ごとの更新の差。同じ残差の空間の向きとして cos を比べているだけで、意味は違う量")}</p>
      ${d.compare.map((c) => `<details><summary class="note mono">${esc(c.name)} · ${t("選ぶ層での方向ベクトルどうしの cos")} ${sgn(c.select.cos_direction_select, 3)}</summary>
        <div class="tbl-wrap" tabindex="0" role="region" aria-label="${esc(c.name)}"><table class="tbl mono"><thead><tr><th class="l">${t("行")}</th><th>cos(r̂ᴰ, r̂_l)</th><th>cos(r̂ᴰ, d_l)</th>${c.material ? `<th>${t("材料どうし")}</th>` : ""}
          ${Array.from({ length: K }, (_, k) => `<th>V${k + 1}(${t("そろえる")})</th>`).join("")}</tr></thead>
        <tbody>${c.rows.map((x) => `<tr><td class="l">${esc(d.rows[x.i].label)}</td><td>${sgn(x.cos_direction)}</td><td>${sgn(x.cos_update)}</td>${c.material ? `<td>${sgn(x.cos_material)}</td>` : ""}
          ${Array.from({ length: K }, (_, k) => `<td>${sgn((x.cos_v_rows || [])[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>`).join("")}`;
  }

  exportHTML(d) {
    const opts = [];
    for (const [mode, s] of Object.entries(d.svd)) for (let k = 1; k <= s.k; k++) opts.push([`svd-${mode}:${k}`, `${mode === "rows" ? t("特異値分解(そろえる)") : t("特異値分解(そろえない)")} V${k}`]);
    if (d.within) for (let k = 1; k <= (d.within_summary.k || 1); k++) opts.push([`within:${k}`, t("各層の取り残しの向き {k} 番目", { k })]);
    opts.push(["update", t("その層が足した分の向き(層全体の更新の差 d̂_l)")], ["update-attn", t("その層が足した分の向き(注意の出力全体)")],
      ["update-mlp", t("その層が足した分の向き(MLP の出力全体)")]);
    return `<h4 class="note">${t("6. 方向ベクトルとして保存する")}</h4>
      <p class="note warn">${t("確かめのための向きです。全部の層から消すと、応答が同じ文の繰り返しになることがあります(各側 64 問の対照集合で、特異値分解の 1 番目を全部の層から消すと 30 問中 12〜15 問)。消した応答を読んで確かめます。層ごとの向き(取り残し・その層が足した分)には最後の層の向きが無く(更新がほぼ 0 の層・信用しない層も保存しない)、その層では手前で一番近い層の向きが使われます。")}</p>
      ${noteHTML(t("選んだ向きを、今の方向ベクトルの形で保存する(単位ベクトル。対照集合・測る位置は元の方向から写す)。direction score・heads writes --name・headproj ablate で使える。2 本目の向きが応答に効くかを確かめるためのもので、1 つの部品に 2 本以上の向きを当てる射影は作らない。"), "muted note")}
      <div class="evx-tools"><label class="muted">${tl("向き", "direction")} <select id="geo-export-src">${opts.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join("")}</select></label>
        <label class="muted">${t("名前")} <input id="geo-export-name" size="20" spellcheck="false" value="${esc(`${d.material.name}_v`)}"></label>
        <label class="muted"><input type="checkbox" id="geo-export-force"> ${t("同じ名前の方向があれば置き換える")}</label>
        <button id="geo-export-go">${t("保存する")}</button></div><div id="geo-export-out"></div>`;
  }

  async doExport() {
    const d = this.data, out = this.$("#geo-export-out");
    const name = this.$("#geo-export-name").value.trim(), src = this.$("#geo-export-src").value;
    if (!name) { this.$("#geo-export-name").focus(); return; }
    try {
      const e = await postJSON(`/api/geometry/${encodeURIComponent(d.name)}/export`, { out: name, source: src, force: this.$("#geo-export-force").checked });
      out.innerHTML = `<p class="note ok">${t("保存しました")}: <span class="mono">${esc(e.path)}</span>(${esc(e.source)}、${t("層")} ${e.layers}、${t("選ぶ層の残差の平均の差との cos")} ${sgn(e.cos_select, 3)})</p>
        <ul class="geo-notes">${(e.notes || []).map((n) => `<li>${esc(n.text)}</li>`).join("")}</ul>`;
    } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }

  tablesHTML(d) {
    const rows = d.rows, C = d.similarity.matrix;
    const hasNull = rows.some((r) => r.norm_null95 !== undefined);
    const rt = `<div class="tbl-wrap" tabindex="0" role="region" aria-label="${t("行ごとの値")}"><table class="tbl mono"><caption class="muted note">${t("行ごとの値")}</caption>
      <thead><tr><th class="l">${t("行")}</th><th class="l">${t("種類")}</th><th>${tl("回", "visit")}</th><th>${tl("大きさ", "norm")}</th>${hasNull ? `<th>${t("大きさ÷基準")}</th>` : ""}<th>${t("半分 cos")}</th><th>${t("雑音比")}</th>
        <th>cos r̂_l</th><th>cos r̂_S</th><th>${t("上位 5 次元")}</th><th class="l">${t("印")}</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><th class="l" scope="row">${esc(r.label)}</th><td class="l">${kindName(r.mixer)}</td><td>${r.visit || ""}</td><td>${fmt(r.norm, 3)}</td>
        ${hasNull ? `<td>${r.norm_null95 ? fmt(r.norm / r.norm_null95) : "—"}</td>` : ""}<td>${fmt(r.half_cos)}</td><td>${fmt(r.noise_ratio)}</td><td>${sgn(r.cos_own)}</td><td>${sgn(r.cos_select)}</td>
        <td>${fmt(r.top_dims && r.top_dims.share)}</td><td class="l">${r.reliable ? "" : `× ${t("信用しない")}`}</td></tr>`).join("")}</tbody></table></div>`;
    const { ri, ci } = this.blockIdx();                          // 行列はヒートマップで選んだ部品の行・列だけ
    const mt = `<div class="tbl-wrap geo-matrix" tabindex="0" role="region" aria-label="${t("類似度の行列")}"><table class="tbl mono"><caption class="muted note">${t("類似度の行列(cos)")}</caption>
      <thead><tr><th></th>${ci.map((j) => `<th scope="col">${esc(rows[j].label)}</th>`).join("")}</tr></thead>
      <tbody>${ri.map((i) => `<tr><th scope="row" class="l">${esc(rows[i].label)}</th>${ci.map((j) => `<td>${sgn(C[i][j])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    return rt + mt;
  }

  // --- Canvas ---------------------------------------------------------------------------------------------
  draw() {
    if (!this.data || !this.root.isConnected) return;
    this.drawHeat();
    this.drawSvd();
    this.drawU();
    this.drawDir();
  }

  canvas(sel, w, h) {
    const cv = this.root.querySelector(sel);
    if (!cv) return null;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.width = `${w}px`; cv.style.height = `${h}px`;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.font = `${cssVar("--fs-s") || "11px"} ${cssVar("--mono")}`;
    return g;
  }

  width() { return Math.max(320, Math.min(this.root.clientWidth - 32, 1100)); }

  // ヒートマップで見る行・列(部品で切り出す)。戻り値は全体の行の番号の並び
  blockIdx() {
    const rows = this.data.rows, pick = (p) => rows.map((r, i) => i).filter((i) => p === "all" || rows[i].part === p);
    const ri = pick(this.block.r), ci = pick(this.block.c);
    return { ri: ri.length ? ri : rows.map((_, i) => i), ci: ci.length ? ci : rows.map((_, i) => i) };
  }

  heatGeom() {
    const { ri, ci } = this.blockIdx(), W = this.width();
    const L0 = 64, strip = 8, R0 = 176, T0 = 8;
    const n = Math.max(ri.length, ci.length);
    const auto = Math.max(2, Math.min(18, Math.floor((Math.min(W - L0 - 2 * strip - R0 - 8, 760)) / n)));
    const cell = Math.min(36, auto * this.zoom);
    return { ri, ci, nr: ri.length, nc: ci.length, auto, L0, strip, R0, T0, cell, x0: L0 + 2 * strip + 4, w: cell * ci.length, h: cell * ri.length };
  }

  drawHeat() {
    const d = this.data;
    if (!d) return;
    const G = this.heatGeom(), rows = d.rows, C = d.similarity.matrix;
    const W = G.x0 + G.w + G.R0, H = G.T0 + G.h + 28;
    const g = this.canvas("#geo-heat", W, H);
    if (!g) return;
    const col = diverging(), range = this.range === "one" ? 1 : Math.max(0.05, d.similarity.offdiag_abs_max || 1);
    this.heatRange = range;
    G.ri.forEach((i, a) => G.ci.forEach((j, b) => {
      g.fillStyle = col(C[i][j], range);
      g.fillRect(G.x0 + b * G.cell, G.T0 + a * G.cell, G.cell, G.cell);
    }));
    // 縁: 層の種類・回、信用しない行
    const kc = { attention: cssVar("--link"), linear_attention: cssVar("--d1"), mlp: cssVar("--accent") };
    const gap = G.cell > 3 ? 1 : 0;
    G.ri.forEach((i, a) => {
      const r = rows[i], y = G.T0 + a * G.cell;
      g.fillStyle = kc[r.part === "mlp" ? "mlp" : r.mixer] || cssVar("--text-4");
      g.fillRect(G.L0, y, G.strip - 1, Math.max(1, G.cell - gap));
      if (r.visit) { g.fillStyle = r.visit === 1 ? cssVar("--text-3") : cssVar("--dim"); g.fillRect(G.L0 + G.strip, y, G.strip - 1, Math.max(1, G.cell - gap)); }
      if (!r.reliable) {                                          // 信用しない行: 斜線(行列の幅の中だけ)
        g.save();
        g.beginPath(); g.rect(G.x0, y, G.w, G.cell); g.clip();
        g.strokeStyle = cssVar("--muted"); g.lineWidth = 1;
        g.beginPath();
        for (let k = -G.cell; k < G.w; k += 6) { g.moveTo(G.x0 + k, y + G.cell); g.lineTo(G.x0 + k + G.cell, y); }
        g.stroke();
        g.restore();
      }
    });
    // 行の名前(左)と、列の名前(下に、間引いて)
    g.fillStyle = cssVar("--text-4"); g.textAlign = "right"; g.textBaseline = "middle";
    const every = Math.max(1, Math.ceil(12 / G.cell));
    const lastOk = (G.nr - 1) % every === 0 || (G.nr - 1) % every >= every / 2;   // 最後の行の名前は、手前の名前と重ならないときだけ
    G.ri.forEach((i, a) => { if (a % every === 0 || (a === G.nr - 1 && lastOk)) g.fillText(rows[i].label, G.L0 - 4, G.T0 + a * G.cell + G.cell / 2); });
    // 部品の区切り
    g.strokeStyle = cssVar("--line-strong"); g.lineWidth = 1;
    for (let a = 1; a < G.nr; a++) if (rows[G.ri[a]].part !== rows[G.ri[a - 1]].part) { g.beginPath(); g.moveTo(G.x0, G.T0 + a * G.cell); g.lineTo(G.x0 + G.w, G.T0 + a * G.cell); g.stroke(); }
    for (let b = 1; b < G.nc; b++) if (rows[G.ci[b]].part !== rows[G.ci[b - 1]].part) { g.beginPath(); g.moveTo(G.x0 + b * G.cell, G.T0); g.lineTo(G.x0 + b * G.cell, G.T0 + G.h); g.stroke(); }
    // 同じ重みの 1 回目と 2 回目(行と列で同じ部品の組)
    if (d.loop && this.loopLine) {
      const rpos = new Map(G.ri.map((i, a) => [`${rows[i].part}:${rows[i].layer}`, a])), cpos = new Map(G.ci.map((j, b) => [`${rows[j].part}:${rows[j].layer}`, b]));
      g.strokeStyle = cssVar("--sel"); g.lineWidth = 1;
      for (const p of d.loop.pairs) for (const part of d.settings.parts) for (const [x, y] of [[p.first, p.second], [p.second, p.first]]) {
        const a = rpos.get(`${part}:${x}`), b = cpos.get(`${part}:${y}`);
        if (a !== undefined && b !== undefined) g.strokeRect(G.x0 + b * G.cell + 0.5, G.T0 + a * G.cell + 0.5, G.cell - 1, G.cell - 1);
      }
    }
    // 右: 大きさの棒(行の。別の軸)
    const bx = G.x0 + G.w + 14, bw = G.R0 - 40, nmax = Math.max(...G.ri.map((i) => rows[i].norm)) || 1;
    G.ri.forEach((i, a) => {
      const r = rows[i];
      g.fillStyle = r.reliable ? cssVar("--side-b") : cssVar("--dim");
      g.fillRect(bx, G.T0 + a * G.cell + (gap ? 0.5 : 0), Math.max(1, (r.norm / nmax) * bw), Math.max(1, G.cell - gap));
      if (r.norm_null95) {                                        // ラベル入れ替えの 95%: 細い縦線
        g.fillStyle = cssVar("--warn");
        g.fillRect(bx + (r.norm_null95 / nmax) * bw, G.T0 + a * G.cell, 1, Math.max(1, G.cell));
      }
    });
    g.fillStyle = cssVar("--text-4"); g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.fillText(t("大きさ ‖d‖(最大 {v})", { v: fmt(nmax, 3) }), bx, G.T0 + G.h + 16);
    const blk = this.block.r === "all" && this.block.c === "all" ? t("行と列は同じ並び(部品ごとにまとめて層の順)")
      : t("行 = {r}、列 = {c}", { r: this.block.r === "all" ? t("全部") : partName(this.block.r), c: this.block.c === "all" ? t("全部") : partName(this.block.c) });
    g.fillText(blk, G.x0, G.T0 + G.h + 16);
    // 選んだセル
    if (this.cursor) {
      const [a, b] = this.cursor;
      g.strokeStyle = cssVar("--accent"); g.lineWidth = 2;
      g.strokeRect(G.x0 + b * G.cell - 1, G.T0 + a * G.cell - 1, G.cell + 2, G.cell + 2);
    }
    this.heat = G;
    const hint = this.$("#geo-heat-hint");
    if (hint) hint.hidden = G.cell >= 5;
    this.legend();
  }

  legend() {
    const d = this.data, el = this.$("#geo-legend");
    if (!el || !d) return;
    const col = diverging(), r = this.heatRange || 1;
    const stops = [-1, -0.5, 0, 0.5, 1].map((f) => `${col(f * r, r)} ${50 + f * 50}%`).join(", ");
    el.innerHTML = `<span class="geo-scale"><span class="muted">cos</span> <span class="mono">${sgn(-r)}</span><i class="geo-ramp" style="background: linear-gradient(to right, ${stops})"></i><span class="mono">${sgn(r)}</span></span>
      <span><i style="background:var(--link)"></i>${kindName("attention")}</span><span><i style="background:var(--d1)"></i>${kindName("linear_attention")}</span><span><i style="background:var(--accent)"></i>MLP</span>
      ${d.loop ? `<span><i style="background:var(--text-3)"></i>${visitName(1)}</span><span><i style="background:var(--dim)"></i>${visitName(2)}</span>` : ""}
      <span><i class="geo-hatch"></i>${t("信用しない行(半分どうしの cos が {v} 未満)", { v: d.settings.min_half_cos })}</span>
      <span><i style="background:var(--side-b)"></i>${tl("大きさ", "norm")}</span>${d.rows.some((x) => x.norm_null95) ? `<span><i class="geo-tick"></i>${t("入れ替えの 95%")}</span>` : ""}`;
  }

  heatCell(ev) {
    const G = this.heat;
    if (!G) return null;
    const cv = this.root.querySelector("#geo-heat"), rect = cv.getBoundingClientRect();
    const x = ev.clientX - rect.left - G.x0, y = ev.clientY - rect.top - G.T0;
    const b = Math.floor(x / G.cell), a = Math.floor(y / G.cell);
    return a >= 0 && b >= 0 && a < G.nr && b < G.nc ? [a, b] : null;
  }

  cellText([a, b]) {
    const d = this.data, G = this.heat, i = G.ri[a], j = G.ci[b];
    const ra = d.rows[i], rb = d.rows[j], c = d.similarity.matrix[i][j];
    const nl = d.similarity.null95 ? d.similarity.null95[i][j] : null;
    const kind = (r) => (r.part === "mlp" ? "MLP" : kindName(r.mixer)) + (r.visit ? `・${visitName(r.visit)}` : "");
    return `cos ${sgn(c, 3)} · ${ra.label} (${kind(ra)}、‖d‖ ${fmt(ra.norm, 3)}、${t("半分 cos")} ${fmt(ra.half_cos)}${ra.reliable ? "" : ` ×`}) × ${rb.label} (${kind(rb)}、‖d‖ ${fmt(rb.norm, 3)}、${t("半分 cos")} ${fmt(rb.half_cos)}${rb.reliable ? "" : ` ×`})`
      + (nl !== null ? ` · ${t("入れ替えの 95%")} |cos| ${fmt(nl, 3)}` : "");
  }

  wireHeat() {
    const cv = this.root.querySelector("#geo-heat"), out = this.root.querySelector("#geo-readout");
    if (!cv) return;
    const show = (cell, ev) => {
      if (!cell) { this.tip.classList.add("hidden"); return; }
      const s = this.cellText(cell);
      out.textContent = s;
      if (ev) {
        this.tip.textContent = s;
        const r0 = this.root.getBoundingClientRect();
        this.tip.style.left = `${Math.min(ev.clientX - r0.left + 14, this.root.clientWidth - 360)}px`;
        this.tip.style.top = `${ev.clientY - r0.top + this.root.scrollTop + 14}px`;
        this.tip.classList.remove("hidden");
      }
    };
    cv.addEventListener("pointermove", (ev) => show(this.heatCell(ev), ev));
    cv.addEventListener("pointerleave", () => this.tip.classList.add("hidden"));
    cv.addEventListener("click", (ev) => { const c = this.heatCell(ev); if (c) { this.cursor = c; this.drawHeat(); show(c); } });
    cv.addEventListener("keydown", (ev) => {
      const G = this.heat;
      const mv = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0] }[ev.key];
      if (!mv || !G) return;
      ev.preventDefault();
      const [a, b] = this.cursor || [0, 0];
      this.cursor = this.cursor ? [Math.max(0, Math.min(G.nr - 1, a + mv[0])), Math.max(0, Math.min(G.nc - 1, b + mv[1]))] : [0, 0];
      this.drawHeat();
      show(this.cursor);
    });
  }

  drawSvd() {
    const d = this.data, s = d && d.svd[this.mode];
    if (!s) return;
    const W = Math.min(this.width(), 560), H = 200, L0 = 46, B0 = 24, T0 = 10;
    const g = this.canvas("#geo-svd", W, H);
    if (!g) return;
    const vals = [...s.explained, ...s.baseline_orthogonal, ...s.baseline_noise, ...(s.null95 || [])];
    const hi = Math.max(0.05, ...vals) * 1.1, Y = (v) => T0 + (1 - v / hi) * (H - T0 - B0);
    const bw = (W - L0 - 10) / s.k;
    g.strokeStyle = cssVar("--line-soft"); g.fillStyle = cssVar("--text-4"); g.textAlign = "right"; g.textBaseline = "middle";
    for (const v of [0, hi / 2, hi]) { g.beginPath(); g.moveTo(L0, Y(v)); g.lineTo(W - 10, Y(v)); g.stroke(); g.fillText(fmt(v, 2), L0 - 4, Y(v)); }
    for (let k = 0; k < s.k; k++) {
      const x = L0 + k * bw;
      g.fillStyle = cssVar("--link");
      g.fillRect(x + bw * 0.2, Y(s.explained[k]), bw * 0.6, Math.max(1, Y(0) - Y(s.explained[k])));
      const tick = (v, color, dash) => { g.strokeStyle = color; g.lineWidth = 2; g.setLineDash(dash); g.beginPath(); g.moveTo(x + bw * 0.1, Y(v)); g.lineTo(x + bw * 0.9, Y(v)); g.stroke(); g.setLineDash([]); };
      tick(s.baseline_orthogonal[k], cssVar("--text-2"), [6, 3]);
      tick(s.baseline_noise[k], cssVar("--ok"), [2, 2]);
      if (s.null95) { g.fillStyle = cssVar("--warn"); const cx = x + bw / 2, cy = Y(s.null95[k]); g.beginPath(); g.moveTo(cx, cy - 5); g.lineTo(cx + 5, cy); g.lineTo(cx, cy + 5); g.lineTo(cx - 5, cy); g.closePath(); g.fill(); }
      g.fillStyle = cssVar("--text-4"); g.textAlign = "center"; g.textBaseline = "alphabetic";
      g.fillText(`V${k + 1}`, x + bw / 2, H - 6);
    }
    this.$("#geo-svd-legend").innerHTML = `<span><i style="background:var(--link)"></i>${t("寄与率")}</span><span><i class="geo-dash"></i>${t("行が直交していたときの値")}</span>
      <span><i class="geo-dot"></i>${t("雑音の目安(半分どうしの差)")}</span>${s.null95 ? `<span><i class="geo-diamond"></i>${t("ラベル入れ替えの 95%(その差自身の全体に対する割合)")}</span>` : ""}`;
    this.$("#geo-svd-table").innerHTML = `<div class="tbl-wrap" tabindex="0" role="region" aria-label="${t("寄与率と基準")}"><table class="tbl mono"><thead><tr><th>k</th><th>${t("寄与率")}</th><th>${t("累積")}</th>
      <th>${t("直交の基準")}</th><th>${t("雑音の基準")}</th>${s.null95 ? `<th>${t("入れ替え 95%")}</th>` : ""}<th>${t("半分で求めた向きの説明率")}</th><th>${t("半分どうしの重なり")}</th><th>cos r̂_S</th></tr></thead>
      <tbody>${s.explained.map((e, k) => `<tr><td>V${k + 1}</td><td>${fmt(e, 3)}</td><td>${fmt(s.cumulative[k], 3)}</td><td>${fmt(s.baseline_orthogonal[k], 3)}</td><td>${fmt(s.baseline_noise[k], 3)}</td>
        ${s.null95 ? `<td>${fmt(s.null95[k], 3)}</td>` : ""}<td>${fmt(s.half_cross[k], 3)}</td><td>${fmt(s.half_overlap[k], 3)}</td><td>${sgn((s.cos_select || [])[k])}</td></tr>`).join("")}</tbody></table></div>
      <p class="note muted">${t("半分で求めた向きの説明率と重なりは累積。そろえない版の上位は、どの層の更新が大きいかを主に映す(直交の基準と並べて読む)")}</p>
      ${s.n_unreliable ? `<p class="note warn">${t("この特異値分解には、信用しない行が {n} 行入っています(同じ重さで数えています)", { n: s.n_unreliable })}</p>` : ""}`;
  }

  drawU() {
    const d = this.data, s = d && d.svd[this.mode];
    if (!s) return;
    const rows = d.rows, K = s.k, W = this.width(), ph = 54, L0 = 46, H = K * (ph + 8) + 18;
    const g = this.canvas("#geo-u", W, H);
    if (!g) return;
    const bw = (W - L0 - 10) / rows.length;
    this.uHits = { L0, bw, ph, n: rows.length, m: K, text: (i, k) => `V${k + 1} · ${rows[i].label}: U ${sgn(s.U[i][k], 3)}${rows[i].reliable ? "" : ` (× ${t("信用しない")})`} · ${t("この行の上位 {k} 本での説明率", { k: k + 1 })} ${fmt(s.row_explained[i][k])}` };
    for (let k = 0; k < K; k++) {
      const y0 = 4 + k * (ph + 8), mid = y0 + ph / 2;
      const amp = Math.max(1e-9, ...rows.map((_, i) => Math.abs(s.U[i][k])));
      g.strokeStyle = cssVar("--line-mid"); g.beginPath(); g.moveTo(L0, mid); g.lineTo(W - 10, mid); g.stroke();
      g.fillStyle = cssVar("--text-4"); g.textAlign = "right"; g.textBaseline = "middle"; g.fillText(`V${k + 1}`, L0 - 6, mid);
      rows.forEach((r, i) => {
        const v = s.U[i][k], h = (v / amp) * (ph / 2 - 2);
        g.fillStyle = r.reliable ? (v >= 0 ? cssVar("--div-pos") : cssVar("--div-neg")) : cssVar("--dim");
        g.fillRect(L0 + i * bw + bw * 0.15, Math.min(mid, mid - h), Math.max(1, bw * 0.7), Math.max(1, Math.abs(h)));
      });
    }
    g.fillStyle = cssVar("--text-4"); g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.fillText(`${rows[0].label} … ${rows[rows.length - 1].label}`, L0, H - 4);
  }

  drawDir() {
    const d = this.data;
    if (!d) return;
    const rows = d.rows, W = this.width(), ph = 64, L0 = 70, H = 2 * (ph + 10) + 18;
    const g = this.canvas("#geo-dir", W, H);
    if (!g) return;
    const bw = (W - L0 - 10) / rows.length, keys = [["cos_own", "cos(d_l, r̂_l)"], ["cos_select", "cos(d_l, r̂_S)"]];
    this.dirHits = { L0, bw, ph, n: rows.length, m: 2, gap: 10, text: (i, k) => `${rows[i].label}: ${keys[k][1]} ${sgn(rows[i][keys[k][0]], 3)}(${t("2 乗 {v}", { v: fmt((rows[i][keys[k][0]] || 0) ** 2) })})` };
    keys.forEach(([key, label], k) => {
      const y0 = 4 + k * (ph + 10), mid = y0 + ph / 2;
      g.strokeStyle = cssVar("--line-mid"); g.beginPath(); g.moveTo(L0, mid); g.lineTo(W - 10, mid); g.stroke();
      g.fillStyle = cssVar("--text-4"); g.textAlign = "right"; g.textBaseline = "middle"; g.fillText(label, L0 - 6, mid);
      g.fillText("+1", L0 - 6, y0 + 4); g.fillText("−1", L0 - 6, y0 + ph - 4);
      rows.forEach((r, i) => {
        const v = r[key];
        if (v === null || v === undefined) return;
        const h = v * (ph / 2 - 2);
        g.fillStyle = r.reliable ? (v >= 0 ? cssVar("--div-pos") : cssVar("--div-neg")) : cssVar("--dim");
        g.fillRect(L0 + i * bw + bw * 0.15, Math.min(mid, mid - h), Math.max(1, bw * 0.7), Math.max(1, Math.abs(h)));
      });
    });
  }

  wireBars(sel, outSel, hits) {
    const cv = this.root.querySelector(sel), out = this.root.querySelector(outSel);
    if (!cv) return;
    let cur = null;
    const pick = (ev) => {
      const h = hits();
      if (!h) return null;
      const r = cv.getBoundingClientRect(), x = ev.clientX - r.left - h.L0, y = ev.clientY - r.top - 4;
      const i = Math.floor(x / h.bw), k = Math.floor(y / (h.ph + (h.gap ?? 8)));
      return i >= 0 && i < h.n && k >= 0 && k < h.m ? [i, k] : null;
    };
    const show = (c) => { const h = hits(); if (c && h) { try { out.textContent = h.text(c[0], c[1]); } catch { /* 範囲の外 */ } } };
    cv.addEventListener("pointermove", (ev) => show(pick(ev)));
    cv.addEventListener("keydown", (ev) => {
      const h = hits();
      if (!h) return;
      const mv = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowDown: [0, 1], ArrowUp: [0, -1] }[ev.key];
      if (!mv) return;
      ev.preventDefault();
      cur = cur ? [Math.max(0, Math.min(h.n - 1, cur[0] + mv[0])), Math.max(0, Math.min(h.m - 1, cur[1] + mv[1]))] : [0, 0];
      show(cur);
    });
  }
}
