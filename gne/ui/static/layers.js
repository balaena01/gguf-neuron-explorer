// layers ビュー(別冊 §6): 層ごとの要約を 2D の深さプロファイルで一望する。Canvas 2D、x = 層。
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const C = {
  text: "#aeb9c8", muted: "#6b7686", grid: "#1a212b", band: "rgba(120,140,170,0.28)", line: "#c9d6e6",
  amber: "#e8b04a", plus: "#f2d24b", minus: "#c94fa7", attn: "#9fb3cf", lin: "#3a4656", rnd: "#7c8796", hl: "rgba(255,255,255,0.07)",
};
const LEFT = 200, RIGHT = 24, TOP = 14;
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));

export class LayersView {
  constructor(root, { onPick }) {
    this.root = root;
    this.onPick = onPick;
    this.data = null;
    this.arch = null;
    this.dir = null;
    this.hoverL = null;
    root.innerHTML = `<div class="lv-tools"><label class="muted">${t("逆引きの方向")} <select id="lv-dir"></select></label>
      <span class="muted note">${t("層にマウスを当てると下に値、クリックでその層をマップで正面から見る")}</span></div>
      <div id="lv-tip" class="mono"><span class="muted">${t("(層にマウスを当てると、ここにその層の値が出ます)")}</span></div>
      <canvas id="lv-canvas"></canvas>`;
    this.cv = root.querySelector("#lv-canvas");
    this.tip = root.querySelector("#lv-tip");
    root.querySelector("#lv-dir").addEventListener("change", (ev) => { this.dir = ev.target.value; this.draw(); });
    this.cv.addEventListener("mousemove", (ev) => this._move(ev));
    this.cv.addEventListener("mouseleave", () => { this.hoverL = null; this.draw(); });
    this.cv.addEventListener("click", (ev) => { const L = this._layerAt(ev); if (L !== null) this.onPick(L); });
    new ResizeObserver(() => this.draw()).observe(root);
  }

  set(profile, arch) {
    this.data = profile;
    this.arch = arch;
    const sel = this.root.querySelector("#lv-dir");
    sel.innerHTML = profile.directions.length ? profile.directions.map((d) => `<option>${esc(d)}</option>`).join("") : `<option value="">${t("(なし)")}</option>`;
    if (!profile.directions.includes(this.dir)) this.dir = profile.directions[0] || null;
    sel.value = this.dir || "";
    this.draw();
  }

  _rows() {
    return [
      { key: "arch", label: t("構成"), sub: [t("塗り = attention"), t("枠 = 線形注意")], h: 56 },
      { key: "rho", label: t("残差ノルム ρ_L"), sub: [t("帯 = p10〜p90"), t("線 = 中央値")], h: 86 },
      { key: "sweep", label: t("対照群超え"), sub: ["null_pct ≥ 0.95", t("の割合(数/母数)")], h: 92 },
      { key: "kl", label: "steer KL", sub: [t("● 中央値 ○ p90"), t("黄 + / 赤紫 −")], h: 86 },
      { key: "repeat", label: t("反復(+0.8)"), sub: [t("distinct-4 の中央値"), t("琥珀 ニューロン / 灰 random")], h: 80 },
      { key: "dir", label: t("逆引き"), sub: [t("マップと同じ閾値を"), t("超えるニューロンの数")], h: 86 },
      { key: "cov", label: t("データの有無"), h: 92 },
    ];
  }

  _x(L) {
    const n = this.data.layers.length, w = this.cv.clientWidth - LEFT - RIGHT;
    return LEFT + (n <= 1 ? w / 2 : (L / (n - 1)) * w);
  }

  _layerAt(ev) {
    if (!this.data) return null;
    const r = this.cv.getBoundingClientRect(), x = ev.clientX - r.left, n = this.data.layers.length;
    if (x < LEFT - 10 || x > this.cv.clientWidth - RIGHT + 10) return null;
    const w = this.cv.clientWidth - LEFT - RIGHT;
    return Math.max(0, Math.min(n - 1, Math.round(((x - LEFT) / w) * (n - 1))));
  }

  _move(ev) {
    const L = this._layerAt(ev);
    if (L !== this.hoverL) { this.hoverL = L; this.draw(); }
    if (L === null) return;
    const d = this.data.layers[L], li = this.arch.layers[L];
    const kind = { attention: "attention", linear_attention: t("線形注意"), recurrent: t("回帰"), none: t("mixer なし") }[li.mixer];
    const nd = t("—(データ無し)");
    const dd = this.dir ? d.directions[this.dir] : null;
    const lines = [
      `<b>L${L}</b> ${kind}${li.ffn.startsWith("moe") ? " · MoE" : ""}`,
      `ρ_L ${d.rho ? `${fmt(d.rho[1], 1)} (${fmt(d.rho[0], 1)}–${fmt(d.rho[2], 1)})` : nd}`,
      `${t("対照群超え")} ${d.sweep ? `${fmt(d.sweep.frac_null95 * 100, 0)}% (${d.sweep.n_null95}/${d.sweep.n}, ${esc(d.sweep.run)})` : nd}`,
      d.steer_kl ? `${t("steer KL(中央値 / p90)")} + ${fmt(d.steer_kl.pos[0], 3)} / ${fmt(d.steer_kl.pos[1], 3)} · − ${fmt(d.steer_kl.neg[0], 3)} / ${fmt(d.steer_kl.neg[1], 3)}`
        : `steer KL ${nd}`,
      `distinct-4 ${d.repeat ? `${t("ニューロン")} ${fmt(d.repeat.neuron, 3)} · random ${fmt(d.repeat.random, 3)}` : nd}`,
      `${t("逆引き")} ${this.dir ? `${esc(this.dir)}: ${dd ? t("|cos| ≥ {thr} が {n} 個 · 上位 1% 平均 {m} · 最大 {mx}", { thr: fmt(dd.thr, 3), n: dd.n_above, m: fmt(dd.top1_mean, 3), mx: fmt(dd.max, 3) }) : nd}` : "—"}`,
    ];
    this.tip.innerHTML = `${lines[0]}<br>${lines.slice(1, 4).join(" · ")}<br>${lines.slice(4).join(" · ")}`;   // 図の上の情報行(データを隠さない)
  }

  draw() {
    if (!this.data || !this.root.offsetParent) return;
    const rows = this._rows(), H = TOP + rows.reduce((a, r) => a + r.h, 0) + 36, W = this.root.clientWidth;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cv.style.width = `${W}px`; this.cv.style.height = `${H}px`;
    this.cv.width = Math.round(W * dpr); this.cv.height = Math.round(H * dpr);
    const g = this.cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.font = "12px ui-monospace, Menlo, Consolas, monospace";
    const Ls = this.data.layers, n = Ls.length;
    if (this.hoverL !== null) {
      const step = n > 1 ? (this._x(1) - this._x(0)) : 40;
      g.fillStyle = C.hl;
      g.fillRect(this._x(this.hoverL) - step / 2, TOP - 6, step, H - TOP - 20);
    }
    let y = TOP;
    for (const row of rows) {
      g.fillStyle = C.text; g.textAlign = "left";
      g.font = "13px system-ui, sans-serif";
      g.fillText(row.label, 12, y + 16);
      if (row.sub) {                                                  // 見出しの下に説明(行ごと。軸の数字とぶつからない幅で)
        g.fillStyle = C.muted; g.font = "11px system-ui, sans-serif";
        row.sub.forEach((t, i) => g.fillText(t, 12, y + 32 + i * 14));
      }
      g.font = "12px ui-monospace, Menlo, Consolas, monospace";
      g.strokeStyle = C.grid; g.beginPath(); g.moveTo(LEFT - 8, y + row.h - 4); g.lineTo(W - RIGHT + 8, y + row.h - 4); g.stroke();
      this[`_row_${row.key}`](g, y + 6, row.h - 16);
      y += row.h;
    }
    // x 軸の層番号(重ならないように間引く)
    const minGap = 30, step = n > 1 ? this._x(1) - this._x(0) : minGap;
    const k = [1, 2, 4, 5, 10, 20, 25, 50].find((s) => s * step >= minGap) || 100;
    g.fillStyle = C.muted; g.textAlign = "center";
    const shown = [];
    for (let L = 0; L < n; L += k) shown.push(L);
    if (shown[shown.length - 1] !== n - 1) {                     // 最後の層は必ず出す(近すぎる直前の番号は外す)
      if ((n - 1 - shown[shown.length - 1]) * step < minGap) shown.pop();
      shown.push(n - 1);
    }
    for (const L of shown) g.fillText(`L${L}`, this._x(L), y + 16);
  }

  _scaleY(top, h, lo, hi) { return (v) => top + h - ((v - lo) / Math.max(hi - lo, 1e-9)) * h; }

  _none(g, x, top, h) { g.fillStyle = "#2a323d"; g.fillRect(x - 1.5, top + h / 2 - 1.5, 3, 3); }

  _row_arch(g, top, h) {
    this.arch.layers.forEach((li, L) => {
      const x = this._x(L), w = Math.max(3, Math.min(12, (this._x(1) - this._x(0)) * 0.6 || 12));
      if (li.mixer === "attention") { g.fillStyle = C.attn; g.fillRect(x - w / 2, top + 2, w, h - 4); }
      else { g.strokeStyle = li.mixer === "none" ? "#222" : "#5d6b7e"; g.strokeRect(x - w / 2 + 0.5, top + 2.5, w - 1, h - 5); }
      if (li.ffn.startsWith("moe")) { g.fillStyle = C.amber; g.fillRect(x - 1.5, top + h + 1, 3, 3); }
    });
  }

  _row_rho(g, top, h) {
    const rs = this.data.layers.map((d) => d.rho);
    const hi = Math.max(...rs.filter(Boolean).map((r) => r[2]), 1);
    const Y = this._scaleY(top, h, 0, hi * 1.05);
    g.fillStyle = C.band;
    this.data.layers.forEach((d, L) => {
      if (!d.rho) { this._none(g, this._x(L), top, h); return; }
      const w = Math.max(2, Math.min(10, (this._x(1) - this._x(0)) * 0.5 || 10));
      g.fillRect(this._x(L) - w / 2, Y(d.rho[2]), w, Y(d.rho[0]) - Y(d.rho[2]));
    });
    g.strokeStyle = C.line; g.lineWidth = 1.5; g.beginPath();
    let started = false;
    this.data.layers.forEach((d, L) => {
      if (!d.rho) { started = false; return; }
      const x = this._x(L), yy = Y(d.rho[1]);
      if (started) g.lineTo(x, yy); else { g.moveTo(x, yy); started = true; }
    });
    g.stroke(); g.lineWidth = 1;
    g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(fmt(hi, 0), LEFT - 12, top + 10);
  }

  _row_sweep(g, top, h) {
    // 文字は棒の上の 2 段(近い層とは段を交互に)。母数が 20 未満の層の棒は薄く(割合が当てにならない)
    const Y = this._scaleY(top + 40, h - 40, 0, 1);
    let lastL = -10, lane = 0;
    this.data.layers.forEach((d, L) => {
      const x = this._x(L);
      if (!d.sweep) { this._none(g, x, top, h); return; }
      const w = Math.max(3, Math.min(14, (this._x(1) - this._x(0)) * 0.6 || 14));
      g.globalAlpha = d.sweep.n < 20 ? 0.35 : 1;
      g.fillStyle = C.amber; g.fillRect(x - w / 2, Y(d.sweep.frac_null95), w, Y(0) - Y(d.sweep.frac_null95));
      g.globalAlpha = 1;
      lane = L - lastL <= 3 ? 1 - lane : 0;
      lastL = L;
      const ty = top + 10;                                          // 文字は棒の上の段。近い層とは左右に振り分ける
      g.textAlign = lane ? "left" : "right";
      const tx = lane ? x + 2 : x - 2;
      g.fillStyle = C.text; g.fillText(`${Math.round(d.sweep.frac_null95 * 100)}%`, tx, ty + 2);
      g.fillStyle = C.muted; g.font = "10px ui-monospace, Menlo, Consolas, monospace";
      g.fillText(`${d.sweep.n_null95}/${d.sweep.n}`, tx, ty + 15);
      g.font = "12px ui-monospace, Menlo, Consolas, monospace";
    });
  }

  _row_kl(g, top, h) {
    const hi = Math.max(...this.data.layers.filter((d) => d.steer_kl).flatMap((d) => [d.steer_kl.pos[1], d.steer_kl.neg[1]]), 0.01);
    const Y = this._scaleY(top, h, 0, hi * 1.08);
    this.data.layers.forEach((d, L) => {
      const x = this._x(L);
      if (!d.steer_kl) { this._none(g, x, top, h); return; }
      for (const [k, col, dx] of [["pos", C.plus, -3], ["neg", C.minus, 3]]) {
        const [p50, p90] = d.steer_kl[k];
        g.fillStyle = col; g.beginPath(); g.arc(x + dx, Y(p50), 3, 0, 7); g.fill();
        g.strokeStyle = col; g.beginPath(); g.arc(x + dx, Y(p90), 3, 0, 7); g.stroke();
      }
    });
    g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(fmt(hi, 2), LEFT - 12, top + 10);
  }

  _row_repeat(g, top, h) {
    const vals = this.data.layers.filter((d) => d.repeat).flatMap((d) => [d.repeat.neuron, d.repeat.random]).filter((v) => v !== null);
    const lo = Math.min(...vals, 0.9) - 0.01;
    const Y = this._scaleY(top, h, lo, 1);
    this.data.layers.forEach((d, L) => {
      const x = this._x(L);
      if (!d.repeat) { this._none(g, x, top, h); return; }
      if (d.repeat.random !== null) { g.fillStyle = C.rnd; g.beginPath(); g.arc(x + 3, Y(d.repeat.random), 3, 0, 7); g.fill(); }
      if (d.repeat.neuron !== null) { g.fillStyle = C.amber; g.beginPath(); g.arc(x - 3, Y(d.repeat.neuron), 3, 0, 7); g.fill(); }
    });
    g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(fmt(lo, 2), LEFT - 12, top + h); g.fillText("1.00", LEFT - 12, top + 10);
  }

  _row_dir(g, top, h) {
    if (!this.dir) { g.fillStyle = C.muted; g.textAlign = "left"; g.fillText(t("逆引きの方向がありません(gne reverse --contrast)"), LEFT, top + h / 2); return; }
    const ds = this.data.layers.map((d) => d.directions[this.dir]);
    const hi = Math.max(...ds.filter(Boolean).map((x) => x.n_above), 1);
    const Y = this._scaleY(top + 12, h - 12, 0, hi);
    const ranked = ds.map((x, L) => [L, x ? x.n_above : -1]).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([L]) => L);
    ds.forEach((x, L) => {
      const px = this._x(L);
      if (!x) { this._none(g, px, top, h); return; }
      const w = Math.max(3, Math.min(14, (this._x(1) - this._x(0)) * 0.6 || 14));
      g.fillStyle = C.amber; g.fillRect(px - w / 2, Y(x.n_above), w, Math.max(1, Y(0) - Y(x.n_above)));   // 0 件も細い線で(データ無しの点と分ける)
      if (ranked.includes(L) && x.n_above > 0) { g.fillStyle = C.text; g.textAlign = "center"; g.fillText(String(x.n_above), px, Y(x.n_above) - 3); }
    });
    g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(String(hi), LEFT - 12, top + 22);
  }

  _row_cov(g, top, h) {
    const keys = [["sweep", (c) => c.sweep], ["steer", (c) => c.steer_lens], ["logit", (c) => c.logit_lens],
                  ["act", (c) => c.act_stats], [t("逆引き"), (c) => c.directions.length > 0]];
    const rh = h / keys.length;
    keys.forEach(([name, f], i) => {
      const yy = top + rh * (i + 0.5);
      g.fillStyle = C.muted; g.textAlign = "right"; g.font = "12px ui-monospace, Menlo, Consolas, monospace";
      g.fillText(name, LEFT - 12, yy + 4);
      this.data.layers.forEach((d, L) => {
        const on = f(d.coverage);
        g.fillStyle = on ? C.amber : "#232a34";
        g.fillRect(this._x(L) - 2.5, yy - 2.5, 5, 5);
      });
    });
    g.font = "12px ui-monospace, Menlo, Consolas, monospace";
  }
}
