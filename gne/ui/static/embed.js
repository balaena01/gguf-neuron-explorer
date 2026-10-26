// embed ビュー(別冊 §7): 1 つの run の全 rollout を文埋め込み → UMAP 2D にした散布図。Canvas 2D。
// 色: kind(ニューロン / ランダム対照 / ベースライン)・prompt・dose・neuron。hover で本文、クリックでそのニューロンを開く。
// 描く順: 点 → ベースライン(白い縁の ◇)→ プロンプトの札(かたまりの上)→ 開いているニューロンの白い輪 → hover の輪(水色)。
// 凡例は図の外(下)に置き、点を隠さない。
import { getJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const PAL = ["#5fb3b3", "#e8b04a", "#9b8bd9", "#7fc97f", "#f08a5d", "#6fa8dc", "#d4a5c9", "#c9c96f"];
const TOP = ["#ff8c42", "#4fd1c5", "#b388ff", "#9ccc65", "#ff6e9c", "#64b5f6", "#ffd54f", "#e57373"];   // neuron の色: 上位 8 本
// dose の色は |dose| の区切りで固定(run によらず同じ値は同じ色)。強いほど濃い色、+ は黄 → 橙、− は淡い赤紫 → 濃い赤紫
const DOSE_BINS = [0.6, 1, 2, Infinity];
const DOSE_POS = ["#fff2a8", "#f2d24b", "#ffb347", "#ff7f2a"], DOSE_NEG = ["#ffc2ec", "#ff70d0", "#e040b0", "#b0308a"];
const doseColor = (d) => (d >= 0 ? DOSE_POS : DOSE_NEG)[DOSE_BINS.findIndex((b) => Math.abs(d) < b)];
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const sgn = (x) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x)}`;

export class EmbedView {
  constructor(root, { onOpen, onCompute, getFocus, onChange }) {
    this.onChange = onChange || (() => {});
    this.root = root;
    this.onOpen = onOpen;
    this.onCompute = onCompute;
    this.getFocus = getFocus;
    this.runs = [];
    this.run = null;
    this.data = null;
    this.color = "kind";
    this.view = { s: 1, x: 0, y: 0 };            // 0〜1 の座標 → 画面: 拡大 s、平行移動 (x, y)
    this.hover = -1;
    this.top = [];                               // その run の順位の上位(neuron の色)
    root.innerHTML = `
      <div class="ev-tools"><label class="muted">run <select id="ev-run"></select></label>
        <label class="muted">${t("色")} <select id="ev-color"><option value="kind">kind</option><option value="prompt">prompt</option>
          <option value="dose">dose</option><option value="neuron">neuron</option></select></label>
        <button id="ev-compute" class="hidden">${t("埋め込みを計算")}</button><button id="ev-reset" class="small hidden">${t("表示を戻す")}</button>
        <span class="muted note" id="ev-status"></span></div>
      <div id="ev-info" class="mono"></div>
      <div class="ev-wrap"><canvas id="ev-canvas"></canvas><div id="ev-empty" class="hidden"><div id="ev-empty-msg"></div></div></div>
      <div id="ev-legend"></div>`;
    this.cv = root.querySelector("#ev-canvas");
    root.querySelector("#ev-run").addEventListener("change", (ev) => this.load(ev.target.value));
    root.querySelector("#ev-color").addEventListener("change", (ev) => { this.color = ev.target.value; this.draw(); });
    root.querySelector("#ev-compute").addEventListener("click", () => this.onCompute(this.run));
    root.querySelector("#ev-reset").addEventListener("click", () => { this.view = { s: 1, x: 0, y: 0 }; this.draw(); });
    this.cv.addEventListener("wheel", (ev) => this._zoom(ev), { passive: false });
    this.cv.addEventListener("pointerdown", (ev) => { this._drag = { x: ev.clientX, y: ev.clientY, vx: this.view.x, vy: this.view.y, moved: false }; });
    this.cv.addEventListener("pointermove", (ev) => this._move(ev));
    this.cv.addEventListener("pointerup", (ev) => {
      const d = this._drag; this._drag = null;
      if (d && !d.moved) this._click(ev);
    });
    this.cv.addEventListener("pointerleave", () => { this._drag = null; this.hover = -1; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(root);
    this._guide();
  }

  _guide(missing = false) {
    this.root.querySelector("#ev-info").innerHTML = `<span class="muted">${this.run ? `${esc(this.run)}: ` : ""}${missing
      ? t("埋め込みが未計算です。下の案内から計算できます")
      : t("点にマウスを当てると本文、クリックでそのニューロンを開く(ベースラインは開けない)。ホイールで拡大、ドラッグで移動")}</span>`;
  }

  setRuns(runs) {
    this.runs = runs || [];
    const sel = this.root.querySelector("#ev-run");
    sel.innerHTML = this.runs.map((r) => `<option value="${esc(r.run)}">${esc(r.run)}${r.embedded ? "" : t("(未計算)")}</option>`).join("");
    if (!this.runs.some((r) => r.run === this.run)) {
      const first = this.runs.find((r) => r.embedded) || this.runs[0];
      this.run = first ? first.run : null;
    }
    sel.value = this.run || "";
  }

  async load(run) {
    this.run = run;
    this.data = null;
    this.hover = -1;
    this.top = [];
    this.view = { s: 1, x: 0, y: 0 };
    this._guide();                                                          // 前の run の本文を残さない
    const st = this.root.querySelector("#ev-status"), btn = this.root.querySelector("#ev-compute");
    const empty = this.root.querySelector("#ev-empty"), msg = this.root.querySelector("#ev-empty-msg"), reset = this.root.querySelector("#ev-reset");
    reset.classList.add("hidden");
    empty.classList.add("hidden");
    if (!run) { st.textContent = ""; msg.textContent = t("rollout のある run がありません(sweep を回すとできます)"); empty.classList.remove("hidden"); this.draw(); this.onChange(); return; }
    const info = this.runs.find((r) => r.run === run);
    if (info && !info.embedded) {                                          // 未計算なら取りに行かない(404 を出さない)
      st.textContent = "";
      btn.classList.remove("hidden");
      const mb = info.size / 1e6;                                         // rollout の本数の目安(L22-main_rel は 88 MB で 4494 本)
      const n = Math.max(1, Math.round((info.size / 88e6) * 4494));
      msg.innerHTML = `${t("{run} の埋め込みはまだありません(rollout はおよそ {n} 本、{mb} MB)。", { run: `<b>${esc(run)}</b>`, n: n.toLocaleString(), mb: mb.toFixed(1) })}<br>
        ${t("上の「埋め込みを計算」で作れます(GPU を使う。目安: {t})。", { t: n > 1000 ? t("約 {n} 秒", { n: Math.round((n / 4494) * 80) }) : t("数秒〜十数秒") })}`;
      empty.classList.remove("hidden");
      this._guide(true);
      this.draw();
      this.onChange();
      return;
    }
    st.textContent = t("読み込み中…");
    try {
      this.data = await getJSON("/api/embeddings", { run });
      const m = this.data.meta;
      st.textContent = t("{n} rollout · {model} · {dim} 次元 → UMAP", { n: m.n.toLocaleString(), model: m.model, dim: m.dim });
      btn.classList.add("hidden");
      reset.classList.remove("hidden");
      this._prepare();
      getJSON(`/api/layer/${m.layer}/units`).then((u) => {               // neuron の色: この run の順位の上位 8 本
        const r = u.runs.find((x) => x.run === run);
        if (r && this.run === run) { this.top = r.units.filter((x) => x.kind === "neuron").slice(0, 8).map((x) => x.neuron); this.draw(); }
      }).catch(() => {});
    } catch {
      st.textContent = "";
      btn.classList.remove("hidden");
      msg.textContent = t("この run の埋め込みはまだありません");
      empty.classList.remove("hidden");
    }
    this.draw();
    this.onChange();
  }

  // プロンプトごとの dose の一覧と、dose の値ごとの本数(凡例・札・色に使う)
  _prepare() {
    const pts = this.data.points;
    this.doses = [...new Set(pts.filter((p) => p[2] !== 2).map((p) => +p[5]))].sort((a, b) => a - b);
    this.doseCount = Object.fromEntries(this.doses.map((d) => [d, pts.filter((p) => p[2] !== 2 && +p[5] === d).length]));
    this.promptDoses = {};
    for (const pr of this.data.meta.prompts) {
      this.promptDoses[pr] = [...new Set(pts.filter((p) => p[4] === pr && p[2] !== 2).map((p) => +p[5]))].sort((a, b) => a - b);
    }
    this.doseColor = Object.fromEntries(this.doses.map((d) => [d, doseColor(d)]));
  }

  // --- 座標 --------------------------------------------------------------------------------------------
  _box() {
    const W = this.cv.clientWidth, H = this.cv.clientHeight, pad = 28, side = Math.max(10, Math.min(W, H) - 2 * pad);
    return { W, H, ox: (W - side) / 2, oy: (H - side) / 2, side };
  }

  _toScreen(x, y) {
    const b = this._box(), v = this.view;
    return [b.ox + (x * v.s + v.x) * b.side, b.oy + ((1 - y) * v.s + v.y) * b.side];
  }

  _zoom(ev) {
    ev.preventDefault();
    const b = this._box(), r = this.cv.getBoundingClientRect();
    const mx = (ev.clientX - r.left - b.ox) / b.side, my = (ev.clientY - r.top - b.oy) / b.side;
    const k = Math.exp(-ev.deltaY * 0.0015), v = this.view;
    const s = Math.max(0.5, Math.min(40, v.s * k)), kk = s / v.s;
    this.view = { s, x: mx - (mx - v.x) * kk, y: my - (my - v.y) * kk };
    this.draw();
  }

  _nearest(ev) {
    if (!this.data) return -1;
    const r = this.cv.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
    let best = -1, bd = 64;                                                // 8 px 以内
    this.data.points.forEach((p, i) => {
      const [x, y] = this._toScreen(p[0], p[1]);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  _move(ev) {
    const d = this._drag;
    if (d) {
      const b = this._box(), dx = (ev.clientX - d.x) / b.side, dy = (ev.clientY - d.y) / b.side;
      if (Math.abs(ev.clientX - d.x) + Math.abs(ev.clientY - d.y) > 3) d.moved = true;
      if (d.moved) { this.view = { ...this.view, x: d.vx + dx, y: d.vy + dy }; this.draw(); return; }
    }
    const i = this._nearest(ev);
    if (i !== this.hover) { this.hover = i; this.draw(); this._info(i); }
  }

  _click(ev) {
    const i = this._nearest(ev);
    if (i < 0) return;
    const p = this.data.points[i];
    if (p[2] === 0) this.onOpen({ layer: this.data.meta.layer, n: p[3], run: this.run });
    else if (p[2] === 1) this.onOpen({ layer: this.data.meta.layer, n: p[3], kind: "random", run: this.run });
  }

  _info(i) {
    const el = this.root.querySelector("#ev-info");
    if (i < 0 || !this.data) return;
    const p = this.data.points[i], txt = this.data.texts[p[6]];
    const who = p[2] === 0 ? `L${this.data.meta.layer} N${p[3]}` : p[2] === 1 ? `random ${p[3]}` : t("ベースライン(無介入)");
    el.innerHTML = `<b>${esc(who)}</b> <span class="muted">prompt ${esc(p[4])} · dose ${p[2] === 2 ? "0" : sgn(+p[5])}${p[2] === 2 ? "" : ` · ${t("クリックで開く")}`}</span><div class="ev-text">${esc(txt)}</div>`;
  }

  // --- 色 ----------------------------------------------------------------------------------------------
  _color(p) {
    const m = this.data.meta;
    if (this.color === "kind") return ["rgba(232,176,74,0.55)", "rgba(120,140,170,0.7)", null][p[2]];
    if (this.color === "prompt") return PAL[m.prompts.indexOf(p[4]) % PAL.length];
    if (this.color === "dose") return p[2] === 2 ? null : this.doseColor[+p[5]];
    if (p[2] !== 0) return p[2] === 1 ? "rgba(70,80,95,0.5)" : null;       // neuron: 上位 8 本だけ色、ほかは灰
    const k = this.top.indexOf(p[3]);
    return k >= 0 ? TOP[k] : "rgba(110,118,130,0.35)";
  }

  _legend() {
    const el = this.root.querySelector("#ev-legend");
    if (!this.data) { el.innerHTML = ""; return; }
    const sw = (c) => `<span class="sw" style="background:${c}"></span>`;
    const it = (...x) => `<span class="it">${x.join("")}</span>`;
    const base = it(`<span class="sw dia"></span>`, t("ベースライン(無介入)"));
    let row = "";
    if (this.color === "kind") row = it(sw("#e8b04a"), t("ニューロンで押した生成")) + it(sw("#788caa"), t("ランダム方向の対照")) + base;
    if (this.color === "prompt") row = this.data.meta.prompts.map((p, i) => it(sw(PAL[i % PAL.length]), esc(p))).join("") + base;
    if (this.color === "dose") {
      row = this.doses.map((d) => it(sw(this.doseColor[d]), t("{d}({n} 本)", { d: sgn(d), n: this.doseCount[d] }))).join("") + base +
        it(t("(色は |dose| の区切り 0.6 / 1 / 2 で固定、強いほど濃い)"));
      const only = Object.entries(this.promptDoses).filter(([, ds]) => ds.length === 1).map(([p, ds]) => t("{d} は {p} だけ", { d: sgn(ds[0]), p }));
      if (only.length) row += it(`(${only.join(", ")})`);
    }
    if (this.color === "neuron") {
      row = this.top.map((n, i) => it(sw(TOP[i]), `N${n}`)).join("") + it(sw("#6e7682"), t("ほかのニューロン")) + it(sw("#465060"), t("ランダム対照")) + base;
      row = it(t("この run の順位の上位 8 本: ")) + row;
    }
    const f = this.getFocus(), nf = f ? this.data.points.filter((p) => this._isFocus(p, f)).length : 0;
    const ring = nf ? it('<span class="sw ring" style="border-color:#fff"></span>', t("開いている {who} の rollout({n} 本)", { who: f.kind === "random" ? `random ${f.n}` : `L${f.layer} N${f.n}`, n: nf })) : "";
    el.innerHTML = `<div>${row}${ring}${it('<span class="sw ring" style="border-color:#6fe3ff"></span>', t("マウスの下の点"))}</div>
      <div class="note">${t("すぐ近くの点どうしは文が似ている。かたまり同士の距離・大きさ・向き、座標軸には意味がない(UMAP)")}</div>`;
  }

  _isFocus(p, f) {
    return f && f.layer === this.data.meta.layer && p[2] !== 2 && ((f.kind === "random") === (p[2] === 1)) && p[3] === f.n;
  }

  draw() {
    if (!this.root.offsetParent) return;
    const W = this.cv.parentElement.clientWidth, H = this.cv.parentElement.clientHeight, dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cv.style.width = `${W}px`; this.cv.style.height = `${H}px`;
    this.cv.width = Math.round(W * dpr); this.cv.height = Math.round(H * dpr);
    const g = this.cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    this._legend();
    if (!this.data) return;
    const pts = this.data.points, f = this.getFocus(), r0 = pts.length < 200 ? 4 : 2.2;
    const inView = (x, y) => x > -10 && y > -10 && x < W + 10 && y < H + 10;
    // 1. 点(ベースライン以外)。neuron の色のときは灰色を先に、上位 8 本を最後に少し大きく(灰の下に埋もれないように)
    const isTop = (p) => this.color === "neuron" && p[2] === 0 && this.top.includes(p[3]);
    for (const pass of [0, 1]) {
      for (const p of pts) {
        if (p[2] === 2 || isTop(p) !== (pass === 1)) continue;
        const [x, y] = this._toScreen(p[0], p[1]);
        if (!inView(x, y)) continue;
        g.fillStyle = this._color(p);
        g.beginPath(); g.arc(x, y, pass ? Math.max(r0, 3.4) : r0, 0, 7); g.fill();
      }
    }
    // 2. ベースライン: 白い縁の ◇(白は選択の輪と紛れないよう縁だけ)
    for (const p of pts) {
      if (p[2] !== 2) continue;
      const [x, y] = this._toScreen(p[0], p[1]), s = 6;
      g.fillStyle = this.color === "prompt" ? PAL[this.data.meta.prompts.indexOf(p[4]) % PAL.length] : "#11161d";   // prompt の色なら中を塗る
      g.strokeStyle = "#ffffff"; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(x, y - s); g.lineTo(x + s, y); g.lineTo(x, y + s); g.lineTo(x - s, y); g.closePath(); g.fill(); g.stroke();
    }
    // 3. プロンプトの札: かたまりの上辺の少し上(点を隠さない)。1 つの dose しか無いプロンプトはそれを添える
    g.font = "12px system-ui, sans-serif"; g.textAlign = "center";
    for (const pr of this.data.meta.prompts) {
      const ps = pts.filter((p) => p[4] === pr && p[2] !== 2).map((p) => this._toScreen(p[0], p[1]));
      if (!ps.length) continue;
      // 中央値の近く(図の 20% 以内)の点だけで位置を決める(崩れた出力などが別のかたまりに混ざっても引っ張られない)
      const med = (a) => [...a].sort((u, v) => u - v)[Math.floor(a.length / 2)];
      const mx = med(ps.map((q) => q[0])), my = med(ps.map((q) => q[1])), R = this._box().side * 0.2 * this.view.s;
      const core = ps.filter((q) => Math.hypot(q[0] - mx, q[1] - my) < R);
      const use = core.length >= 3 ? core : ps;
      const cx = med(use.map((q) => q[0]));
      const ys = use.map((q) => q[1]).sort((a, b) => a - b);
      let top = ys[Math.floor(ys.length * 0.01)];                       // 外れ点を除いた最上端
      for (const b of pts.filter((p) => p[2] === 2 && p[4] === pr)) {     // そのプロンプトの ◇ がかたまりの上にあれば、さらに上へ
        const [bx, by] = this._toScreen(b[0], b[1]);
        if (Math.abs(bx - cx) < 80) top = Math.min(top, by - 6);
      }
      const ds = this.promptDoses[pr] || [];
      const text = ds.length === 1 ? t("{p}({d} のみ)", { p: pr, d: sgn(ds[0]) }) : pr;
      const w = g.measureText(text).width + 10, y = Math.max(12, top - 14);
      g.fillStyle = "rgba(8,10,14,0.6)"; g.fillRect(cx - w / 2, y - 9, w, 18);
      g.fillStyle = "#dfe6ee"; g.fillText(text, cx, y + 4);
    }
    // 4. 開いているニューロンの rollout: 白い輪
    g.lineWidth = 1.8; g.strokeStyle = "#ffffff";
    for (const p of pts) {
      if (!this._isFocus(p, f)) continue;
      const [x, y] = this._toScreen(p[0], p[1]);
      g.beginPath(); g.arc(x, y, 7, 0, 7); g.stroke();
    }
    // 5. マウスの下の点: 黒い縁の水色の太い輪(どの色の上でも見える)
    if (this.hover >= 0) {
      const [x, y] = this._toScreen(pts[this.hover][0], pts[this.hover][1]);
      g.lineWidth = 5; g.strokeStyle = "#000"; g.beginPath(); g.arc(x, y, 10, 0, 7); g.stroke();
      g.lineWidth = 2.5; g.strokeStyle = "#6fe3ff"; g.beginPath(); g.arc(x, y, 10, 0, 7); g.stroke();
    }
  }
}
