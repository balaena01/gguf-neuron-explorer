// writers ビュー(書き手の強さ、gne writers): A の状態(例 ヤンデレの system)で、各ニューロンが残差に実際に書き足した量。
// 層ごとに、参照の層の方向 r* を mixer(attention・畳み込み・線形注意)と MLP がそれぞれ何割書いたかを棒で、
// 累積を線で描く(埋め込みから足していくと参照の層で 100% になる)。下に、方向へ書き足したニューロンの上位。Canvas 2D。
import { getJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const C = { text: "#aeb9c8", muted: "#6b7686", grid: "#1a212b", mlp: "#e8b04a", mixer: "#9fb3cf", cum: "#6fcf8f", zero: "#39424f" };
const L0 = 52, R0 = 16, T0 = 14, B0 = 30, H = 240;
const pct = (x, d = 1) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(d)}%`);
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));

export class WritersView {
  constructor(root, { onRun, onOpen, onOverlay }) {
    this.root = root;
    this.onRun = onRun;                // (params) → ジョブ writers を投げる
    this.onOpen = onOpen;              // ({layer, n}) → ニューロンを開く
    this.onOverlay = onOverlay;        // (name) → 地図に writers:<name> を重ねる
    this.list = [];
    this.data = null;
    root.innerHTML = `<div class="evx-tools">
        <label class="muted">${t("保存した分析")} <select id="wrx-pick"></select></label>
        <button id="wrx-map" title="${t("地図の色を、この分析の寄与にする")}">${t("地図に重ねる")}</button>
        <span class="evx-run">
          <label class="muted">${t("対照集合")} <input id="wrx-contrast" placeholder="configs/contrast/yandere.yaml" size="30" spellcheck="false"></label>
          <label class="muted" title="${t("A の返事を生成せずに読ませる(別のモデルの返事を借りるとき)。空なら生成する")}">${t("A の返事")} <input id="wrx-replies" placeholder="${t("(生成する)")}" size="16" spellcheck="false"></label>
          <label class="muted">${t("参照の層")} <input id="wrx-ref" placeholder="${t("既定")}" size="4"></label>
          <button id="wrx-go">${t("測る")}</button></span></div>
      <p class="muted note evx-help">${t("A(見たい状態、例 ヤンデレの system)と B(中立)の返事のトークンで、各ニューロンが残差に書き足した量(活性の差 × 書く向き)を測る。寄与は、参照の層の A − B の方向 r* への直接の成分。活性の差だけを見る「読み手」と違い、方向に実際に書いているものが上に来る。")}</p>
      <div id="wrx-body"></div>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#wrx-pick").addEventListener("change", (ev) => this.load(ev.target.value));
    this.$("#wrx-go").addEventListener("click", () => this.run());
    this.$("#wrx-map").addEventListener("click", () => { if (this.data) this.onOverlay(this.data.name); });
    new ResizeObserver(() => this.draw()).observe(root);
  }

  async refresh(select) {
    try { this.list = await getJSON("/api/writers"); } catch { this.list = []; }
    const pick = this.$("#wrx-pick");
    pick.innerHTML = this.list.length
      ? this.list.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}(L${x.ref_layer}、MLP ${pct(x.totals.mlp, 0)})</option>`).join("")
      : `<option value="">${t("(なし)")}</option>`;
    this.$("#wrx-map").disabled = !this.list.length;
    const name = select || (this.data && this.data.name) || (this.list[0] && this.list[0].name);
    if (name && this.list.some((x) => x.name === name)) { pick.value = name; await this.load(name); }
    else this.render();
  }

  async load(name) {
    if (!name) { this.data = null; this.render(); return; }
    try { this.data = await getJSON(`/api/writers/${encodeURIComponent(name)}`); }
    catch (e) { this.data = null; this.$("#wrx-body").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    this.render();
  }

  run() {
    const contrast = this.$("#wrx-contrast").value.trim();
    if (!contrast) { this.$("#wrx-contrast").focus(); return; }
    const p = { contrast };
    const rep = this.$("#wrx-replies").value.trim(), ref = this.$("#wrx-ref").value.trim();
    if (rep) p.replies_a = rep;
    if (ref !== "" && Number.isFinite(+ref)) p.ref_layer = +ref;
    this.onRun(p);
  }

  render() {
    const body = this.$("#wrx-body"), d = this.data;
    if (!d) { body.innerHTML = `<p class="muted note">${t("分析がまだありません。上の「測る」か gne writers --contrast で作ります(MCP の writers でも)。")}</p>`; return; }
    const tt = d.totals;
    const rows = (xs, start = 1) => xs.map((x, i) => `<tr data-layer="${x.layer}" data-n="${x.neuron}">
        <td>${i + start}</td><td>L${x.layer} N${x.neuron}</td><td class="${x.share >= 0 ? "ok" : "ng"}">${pct(x.share, 2)}</td>
        <td>${fmt(x.act_a)}</td><td>${fmt(x.act_b)}</td><td>${fmt(x.act_diff)}</td><td>${fmt(x.align)}</td></tr>`).join("");
    const head = `<thead><tr><th>#</th><th>${t("住所")}</th><th title="${t("参照の方向 r* への直接の寄与(‖r*‖ に対する割合)")}">${t("寄与")}</th>
      <th title="${t("A の返事のトークンでの平均の活性")}">${t("活性 A")}</th><th title="${t("B の返事のトークンでの平均の活性")}">${t("活性 B")}</th>
      <th title="${t("活性の差(読み手の指標)")}">${t("差")}</th><th title="${t("書く向き(post norm 込み)と r* の cos")}">${t("向き")}</th></tr></thead>`;
    body.innerHTML = `
      <p class="note mono">${esc(d.contrast.split("/").pop())} <span class="muted">· A ${d.n_tokens.A} / B ${d.n_tokens.B} ${t("トークン")} · A=${esc(d.replies.A === "given" ? t("読ませた返事") : t("生成"))} · ${fmt(d.seconds, 0)} s</span></p>
      <p class="note">${t("L{L} の方向(‖r*‖ = {n})を書いた割合", { L: d.ref_layer, n: fmt(d.ref_norm) })}:
        <b>${t("埋め込み")} ${pct(tt.embed, 0)} · mixer ${pct(tt.mixer, 0)} · MLP ${pct(tt.mlp, 0)}</b>
        <span class="muted">· ${t("MLP の上位 {a} 個 {x}、{b} 個 {y}、{c} 個 {z}", { a: 10, x: pct(tt.top10, 0), b: 100, y: pct(tt.top100, 0), c: 1000, z: pct(tt.top1000, 0) })}
        ${d.mlp_check_max !== null && d.mlp_check_max !== undefined ? ` · ${t("確かめ(式と捕捉の差)最大 {v}", { v: pct(d.mlp_check_max, 1) })}` : ""}</span></p>
      <div class="evx-legend note"><span><i style="background:${C.mixer}"></i>${t("mixer(attention・畳み込み)が書いた分")}</span>
        <span><i style="background:${C.mlp}"></i>${t("MLP が書いた分")}</span><span><i style="background:${C.cum}"></i>${t("累積(埋め込みから)")}</span></div>
      <canvas id="wrx-canvas"></canvas>
      <h4 class="note">${t("方向へ書き足したニューロン(クリックで開く)")}</h4>
      <table class="evx mono wrx">${head}<tbody>${rows(d.top)}</tbody></table>
      <details><summary class="muted note">${t("方向から引いたニューロン({n})", { n: d.bottom.length })}</summary>
        <table class="evx mono wrx">${head}<tbody>${rows(d.bottom)}</tbody></table></details>
      <details><summary class="muted note">${t("返事の例(A / B)")}</summary>${d.samples.map((s) => `<p class="note"><span class="muted">${esc(s.prompt)}</span><br>
        A: ${esc(s.A)}<br><span class="muted">B: ${esc(s.B)}</span></p>`).join("")}</details>`;
    for (const tr of body.querySelectorAll("tr[data-layer]")) tr.addEventListener("click", () => this.onOpen({ layer: +tr.dataset.layer, n: +tr.dataset.n }));
    this.cv = body.querySelector("#wrx-canvas");
    this.draw();
  }

  draw() {
    const cv = this.cv, d = this.data;
    if (!cv || !d || !cv.isConnected) return;
    const rows = d.layers.filter((x) => x.mixer_to_ref !== undefined);
    const nr = d.ref_norm || 1;
    const w = Math.max(320, this.root.clientWidth - 24), dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = H * dpr; cv.style.width = `${w}px`; cv.style.height = `${H}px`;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, H);
    let cum = d.totals.embed;
    const cums = rows.map((x) => (cum += (x.mixer_to_ref + (x.mlp_to_ref || 0)) / nr));
    const vals = rows.flatMap((x) => [x.mixer_to_ref / nr, (x.mlp_to_ref || 0) / nr]);
    const hi = Math.max(0.05, ...vals, ...cums), lo = Math.min(0, ...vals, ...cums);
    const Y = (v) => T0 + ((hi - v) / (hi - lo || 1)) * (H - T0 - B0);
    const bw = (w - L0 - R0) / Math.max(1, rows.length), X = (i) => L0 + i * bw;
    g.font = "11px ui-monospace, monospace";
    for (const v of [lo, 0, hi / 2, hi]) {
      g.strokeStyle = v === 0 ? C.zero : C.grid; g.beginPath(); g.moveTo(L0, Y(v)); g.lineTo(w - R0, Y(v)); g.stroke();
      g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(`${Math.round(v * 100)}%`, L0 - 6, Y(v) + 4);
    }
    rows.forEach((x, i) => {
      const parts = [[x.mixer_to_ref / nr, C.mixer, 0], [(x.mlp_to_ref || 0) / nr, C.mlp, 1]];
      for (const [v, col, k] of parts) {
        g.fillStyle = col;
        const y0 = Y(0), y1 = Y(v), x0 = X(i) + bw * (0.12 + 0.38 * k);
        g.fillRect(x0, Math.min(y0, y1), bw * 0.36, Math.max(1, Math.abs(y1 - y0)));
      }
      if (rows.length <= 40 || i % 2 === 0) { g.fillStyle = C.muted; g.textAlign = "center"; g.fillText(String(x.layer), X(i) + bw / 2, H - B0 + 14); }
    });
    g.strokeStyle = C.cum; g.lineWidth = 2; g.beginPath();
    cums.forEach((v, i) => (i ? g.lineTo(X(i) + bw / 2, Y(v)) : g.moveTo(X(i) + bw / 2, Y(v))));
    g.stroke(); g.lineWidth = 1;
    g.fillStyle = C.muted; g.textAlign = "left"; g.fillText(t("層"), L0, H - 4);
  }
}
