// direction ビュー(CHANGE-06 G): gne direction の方向セットの速い指標(gne direction score)。Canvas 2D。
// 上: 層ごとの A / B の射影の値(残差 · r̂_L)の cohen_d と、基準層の方向との cos。基準層(記録したもの)と cohen_d 最大の層に印。
// 下: 選んだ層(グラフをクリック、既定は基準層)での例ごとの A / B の射影の値の分布と、例ごとの表。
// 基準層は「L を基準層にする」で手で上書きできる(direction_score の set_base)。初めて測るときに基準層がまだ無ければ、
// cohen_d が最大の層が記録される(gne direction score と同じ)。基準層があれば測り直しても上書きしない。
// 上の切り替えで「JSON から方向とヘッドを作る」(pipeline.js、head_pipeline のジョブ)と「強さの候補」(candidates.js、
// heads_candidates のジョブの結果と名前を伏せた採点)に替わる。方向が 1 つも無ければ「JSON から作る」を開く。
// 「鍵(keyed)」(keys.js、CHANGE-07 G)は鍵つきの書き込みの鍵の作成(direction_key)・key-check(direction_key_check)・
// 方向から key-check までの一括(direction_key_pipeline)・専門家の選ばれ方(direction_key_experts)・key-verify(direction_key_verify)。
import { getJSON, postJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc, qText } from "./panels.js";
import { CandidatesPanel } from "./candidates.js";
import { KeyPanel } from "./keys.js";
import { PipelinePanel } from "./pipeline.js";

const C = { text: "#aeb9c8", muted: "#6b7686", grid: "#1a212b", amber: "#e8b04a", b: "#9fb3cf", base: "#f2f5f8", best: "#6fcf8f",
            sel: "rgba(255,255,255,0.07)" };
const L0 = 52, R0 = 52, T0 = 16, B0 = 30, LH = 250, HH = 170;
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));

export class DirectionView {
  constructor(root, { onChanged, pipeline, candidates, keyed } = {}) {
    this.root = root;
    this.onChanged = onChanged || (() => {});   // 基準層を記録・上書きした → /api/model を取り直す
    this.mode = null;                  // "dir"(方向を見る)か "pipe"(JSON から作る)。null は未選択(方向が無ければ pipe)
    this.list = [];
    this.name = null;
    this.meta = null;                  // /api/directions/<name>(層・基準層・層どうしの cos)
    this.scores = new Map();           // 名前 → /api/directions/<name>/score の結果(このページの中で覚える)
    this.sel = null;                   // 分布を見る層
    this.hoverL = null;
    this.busy = false;
    root.innerHTML = `<div class="dv-mode"><span class="seg" id="dv-mode"><button data-m="dir">${t("方向を見る")}</button><button data-m="pipe">${t("JSON から方向とヘッドを作る")}</button><button data-m="cand">${t("強さの候補")}</button><button data-m="key" title="${t("鍵つきの書き込み(keyed)の鍵を対照集合から作り、作成に使っていない例で状態を分けているかを確かめる")}">${t("鍵(keyed)")}</button></span></div>
      <div id="dv-dir"><div class="dv-tools">
        <label class="muted">${t("方向")} <select id="dv-pick"></select></label>
        <button id="dv-run" title="${t("対照集合の A / B のプロンプトを流して、全層の射影の値を測り直す(モデルを使う。GPU を待つことがあります)")}">${t("測り直す")}</button>
        <button id="dv-base" disabled></button>
        <span class="muted note" id="dv-status"></span></div>
      <div id="dv-info" class="note mono"></div>
      <div id="dv-tip" class="mono"><span class="muted">${t("(層にマウスを当てると、ここにその層の値が出ます。クリックでその層の分布)")}</span></div>
      <canvas id="dv-line"></canvas>
      <div class="evx-legend note">
        <span><i style="background:${C.amber}"></i>${t("cohen_d(A と B の射影の値の離れ方、左の目盛り)")}</span>
        <span><i style="background:${C.b}"></i>${t("基準層の方向との cos(右の目盛り)")}</span>
        <span><i style="background:${C.base};width:2px;height:12px"></i>${t("基準層")}</span>
        <span><i style="background:${C.best};width:2px;height:12px"></i>${t("cohen_d が最大の層")}</span></div>
      <h4 class="note" id="dv-hist-title"></h4>
      <canvas id="dv-hist"></canvas>
      <div id="dv-ex"></div></div>
      <div id="dv-pipe" class="hidden"></div>
      <div id="dv-cand" class="hidden"></div>
      <div id="dv-key" class="hidden"></div>`;
    this.$ = (s) => root.querySelector(s);
    this.pipe = new PipelinePanel(this.$("#dv-pipe"), { ...(pipeline || {}), onShowDirection: (name) => this.showDirection(name),
                                                        onOpenCandidates: (name) => this.openCandidates(name) });
    this.cand = new CandidatesPanel(this.$("#dv-cand"), candidates || {});
    this.key = new KeyPanel(this.$("#dv-key"), keyed || {});
    for (const b of root.querySelectorAll("#dv-mode button")) b.addEventListener("click", () => this.setMode(b.dataset.m));
    this.cv = this.$("#dv-line");
    this.hv = this.$("#dv-hist");
    this.$("#dv-pick").addEventListener("change", (ev) => this.select(ev.target.value));
    this.$("#dv-run").addEventListener("click", () => this.measure());
    this.$("#dv-base").addEventListener("click", () => { if (this.sel !== null) this.measure({ setBase: this.sel }); });
    this.cv.addEventListener("mousemove", (ev) => { const L = this._layerAt(ev); if (L !== this.hoverL) { this.hoverL = L; this.draw(); } });
    this.cv.addEventListener("mouseleave", () => { this.hoverL = null; this.draw(); });
    this.cv.addEventListener("click", (ev) => { const L = this._layerAt(ev); if (L !== null) { this.sel = L; this.draw(); } });
    new ResizeObserver(() => this.draw()).observe(root);
  }

  // 方向を見る / JSON から作る の切り替え。方向を見るに替えたとき、まだ測っていなければ測る
  setMode(m) {
    this._modeUI(m);
    if (m === "pipe") this.pipe.refreshList();
    else if (m === "cand") this.cand.refreshList();
    else if (m === "key") this.key.refresh();
    else if (this.name && this.meta && this.meta.name === this.name && !this.scores.has(this.name) && this.meta.contrast_ok) this.measure();
  }

  _modeUI(m) {
    this.mode = m;
    for (const b of this.root.querySelectorAll("#dv-mode button")) b.classList.toggle("on", b.dataset.m === m);
    this.$("#dv-dir").classList.toggle("hidden", m !== "dir");
    this.$("#dv-pipe").classList.toggle("hidden", m !== "pipe");
    this.$("#dv-cand").classList.toggle("hidden", m !== "cand");
    this.$("#dv-key").classList.toggle("hidden", m !== "key");
    if (m === "dir") this.draw();
  }

  // 「強さの候補」でその結果を開く
  async openCandidates(name, mode = "summary") {
    this.setMode("cand");
    await this.cand.open(name, mode);
  }

  // app.js の onJobFinished から: 一括(head_pipeline)と候補(heads_candidates)
  async jobEnded(j) {
    if (String(j.kind).startsWith("direction_key")) return this.key.jobEnded(j);   // CHANGE-07: 鍵(作成・確認・専門家・key-verify・一括)
    if (j.kind === "head_pipeline" || j.kind === "heads_refine") return this.pipe.jobEnded(j);
    if (j.kind !== "heads_candidates") return;
    const mine = j.id === this.pipe.candJobId;
    this.pipe.candJobEnded(j);
    if (j.status === "done" && mine) this.setMode("cand");
    await this.cand.jobEnded(j, mine);
  }

  // 一括の結果から「方向を見る」: 方向を見るに替え、一覧を取り直してその方向を選ぶ(まだ測っていなければ測る)
  async showDirection(name) {
    this.name = name;
    this._modeUI("dir");
    await this.refresh();
  }

  async refresh() {
    try { this.list = await getJSON("/api/directions"); } catch { this.list = []; }
    if (this.mode === null) this.setMode(this.list.length ? "dir" : "pipe");
    else if (this.mode === "pipe") this.pipe.refreshList();
    else if (this.mode === "cand") this.cand.refreshList();
    else if (this.mode === "key") this.key.refresh();
    const pick = this.$("#dv-pick");
    pick.innerHTML = this.list.length ? this.list.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}</option>`).join("")
      : `<option value="">${t("(なし)")}</option>`;
    if (!this.list.length) {
      this.name = null; this.meta = null;
      this.$("#dv-info").innerHTML = `<p class="muted">${t("方向がありません(gne direction --contrast で作れます)")}</p>`;
      this.$("#dv-run").disabled = true;
      this.draw();
      return;
    }
    const name = this.list.some((x) => x.name === this.name) ? this.name : this.list[0].name;
    pick.value = name;
    await this.select(name);
  }

  async select(name) {
    this.name = name;
    try { this.meta = await getJSON(`/api/directions/${encodeURIComponent(name)}`); }
    catch (e) { this.meta = null; this.status(`${t("エラー")}: ${e.message}`); this.draw(); return; }
    if (this.name !== name) return;
    this.sel = this.meta.base_layer;
    this.status("");
    this.draw();
    if (!this.scores.has(name) && this.meta.contrast_ok && this.mode === "dir") await this.measure();
  }

  status(s) { this.$("#dv-status").textContent = s; }

  // 全層を測る。setBase を渡すと基準層を手で上書きしてから測る。基準層がまだ無ければ cohen_d 最大の層が記録される
  async measure({ setBase = null } = {}) {
    const name = this.name;
    if (!name || this.busy || !this.meta) return;
    this.busy = true;
    this.draw();
    this.status(t("流しています(GPU を待つことがあります)…"));
    const body = setBase !== null ? { set_base: setBase } : { record_base: this.meta.base_layer === null ? null : false };
    try {
      const r = await postJSON(`/api/directions/${encodeURIComponent(name)}/score`, body);
      this.scores.set(name, r);
      if (this.name === name) {
        const had = this.meta.base_layer;
        this.meta = r.meta;
        if (setBase !== null || this.sel === null) this.sel = this.meta.base_layer;
        this.status(t("{s} 秒 · A {a} / B {b} 例", { s: fmt(r.seconds, 1), a: r.n.A ?? "—", b: r.n.B ?? "—" }) +
          (setBase !== null ? ` · ${t("基準層を L{L} にしました", { L: setBase })}` : r.base_recorded && had === null ? ` · ${t("基準層 L{L} を記録しました", { L: r.meta.base_layer })}` : "") +
          (r.warnings && r.warnings.length ? ` · ${r.warnings.join(" / ")}` : ""));
        if (setBase !== null || (r.base_recorded && had === null)) this.onChanged();
      }
    } catch (e) { if (this.name === name) this.status(`${t("エラー")}: ${e.message}`); }
    finally { this.busy = false; }
    this.draw();
    // 測っている間に別の方向を選んでいたら、そちらも測る
    if (this.name !== name && this.meta && this.meta.name === this.name && !this.scores.has(this.name) && this.meta.contrast_ok) await this.measure();
  }

  // グラフの層(測った層。まだ測っていなければ方向セットの層)
  _layers() {
    const r = this.scores.get(this.name);
    return r ? r.layers.map((x) => x.layer) : this.meta ? this.meta.layers : [];
  }

  _x(L) {
    const Ls = this._layers(), w = this.cv.clientWidth - L0 - R0, a = Ls[0], b = Ls[Ls.length - 1];
    return L0 + (b === a ? w / 2 : ((L - a) / (b - a)) * w);
  }

  _layerAt(ev) {
    const Ls = this._layers();
    if (!Ls.length) return null;
    const r = this.cv.getBoundingClientRect(), x = ev.clientX - r.left;
    if (x < L0 - 12 || x > this.cv.clientWidth - R0 + 12) return null;
    let best = null, bd = Infinity;
    for (const L of Ls) { const d = Math.abs(this._x(L) - x); if (d < bd) { bd = d; best = L; } }
    return best;
  }

  // 基準層の方向との cos(層どうしの cos の行列の、基準層の行)
  _cosBase(L) {
    const m = this.meta;
    if (!m || m.base_layer === null || m.base_layer === undefined) return null;
    const i = m.layers.indexOf(m.base_layer), j = m.layers.indexOf(L);
    return i < 0 || j < 0 ? null : m.cos[i][j];
  }

  draw() {
    const m = this.meta, r = m ? this.scores.get(this.name) : null;
    const base = this.$("#dv-base");
    base.disabled = !r || this.busy || this.sel === null || this.sel === m.base_layer;
    base.textContent = this.sel !== null ? t("L{L} を基準層にする", { L: this.sel }) : t("基準層にする");
    base.title = t("選んだ層を、この方向の基準層(proj の direction_layer: base と、因果スクリーニングで射影の値を測る主の層の既定)として記録する");
    this.$("#dv-run").disabled = this.busy || !m || !m.contrast_ok;
    if (m) {
      const best = r ? r.best_layer : null;
      this.$("#dv-info").innerHTML = `${t("対照集合")} ${esc(m.source || "—")}${m.contrast_ok ? "" : ` <span class="err">${t("(見つかりません: 測れません)")}</span>`} · ${esc(m.method || "")} · positions ${esc(m.positions || "—")} · A ${m.n.A ?? "—"} / B ${m.n.B ?? "—"} · ${
        t("基準層")} <b>${m.base_layer === null || m.base_layer === undefined ? t("未記録") : `L${m.base_layer}`}</b>${best !== null && best !== undefined ? ` · ${t("cohen_d 最大")} L${best}` : ""}`;
    }
    this._line(m, r);
    this._tip(m, r);
    this._hist(m, r);
  }

  _line(m, r) {
    const cv = this.cv, w = Math.max(320, this.root.clientWidth - 24), dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = LH * dpr; cv.style.width = `${w}px`; cv.style.height = `${LH}px`;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, LH);
    if (!m) return;
    const Ls = this._layers();
    if (!Ls.length) return;
    const ds = r ? r.layers.map((x) => x.cohen_d).filter((x) => Number.isFinite(x)) : [];
    const dMax = Math.max(1, ...ds) * 1.08, dMin = Math.min(0, ...ds);
    const cs = Ls.map((L) => this._cosBase(L)).filter((x) => x !== null), cMin = Math.min(0, ...cs);
    const Yd = (v) => T0 + (1 - (v - dMin) / (dMax - dMin || 1)) * (LH - T0 - B0);
    const Yc = (v) => T0 + (1 - (v - cMin) / (1 - cMin || 1)) * (LH - T0 - B0);
    g.font = "11px ui-monospace, monospace";
    g.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {                                         // 目盛り: 左 = cohen_d、右 = cos
      const v = dMin + ((dMax - dMin) * i) / 4, y = Yd(v);
      g.strokeStyle = C.grid; g.beginPath(); g.moveTo(L0, y); g.lineTo(w - R0, y); g.stroke();
      g.fillStyle = C.amber; g.textAlign = "right"; g.fillText(fmt(v, 1), L0 - 6, y + 4);
      g.fillStyle = C.b; g.textAlign = "left"; g.fillText(fmt(cMin + ((1 - cMin) * i) / 4, 2), w - R0 + 6, Yc(cMin + ((1 - cMin) * i) / 4) + 4);
    }
    const sel = this.sel, hv = this.hoverL;
    for (const [L, col] of [[sel, C.sel], [hv, "rgba(255,255,255,0.05)"]]) {
      if (L === null || !Ls.includes(L)) continue;
      const step = Ls.length > 1 ? (this._x(Ls[Ls.length - 1]) - this._x(Ls[0])) / (Ls.length - 1) : 20;
      g.fillStyle = col; g.fillRect(this._x(L) - step / 2, T0, step, LH - T0 - B0);
    }
    g.textAlign = "center";
    const every = Math.max(1, Math.ceil(Ls.length / Math.max(1, Math.floor((w - L0 - R0) / 28))));
    Ls.forEach((L, i) => {
      if (i % every && L !== sel && L !== m.base_layer) return;
      g.fillStyle = L === sel ? C.text : C.muted;
      g.fillText(`${L}`, this._x(L), LH - B0 + 15);
    });
    const vline = (L, col, label, dy) => {
      if (L === null || L === undefined || !Ls.includes(L)) return;
      g.strokeStyle = col; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(this._x(L), T0); g.lineTo(this._x(L), LH - B0); g.stroke(); g.setLineDash([]);
      g.fillStyle = col; g.textAlign = "left"; g.fillText(label, this._x(L) + 4, T0 + 10 + dy);
    };
    vline(m.base_layer, C.base, t("基準 L{L}", { L: m.base_layer }), 0);
    if (r && r.best_layer !== m.base_layer) vline(r.best_layer, C.best, t("最大 L{L}", { L: r.best_layer }), 14);
    const series = [[Ls.map((L) => [L, this._cosBase(L)]), C.b, Yc, [4, 3]]];
    if (r) series.unshift([r.layers.map((x) => [x.layer, x.cohen_d]), C.amber, Yd, []]);
    for (const [pts0, col, Y, dash] of series) {
      const pts = pts0.filter(([, v]) => v !== null && Number.isFinite(v));
      g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2; g.setLineDash(dash);
      g.beginPath();
      pts.forEach(([L, v], i) => (i ? g.lineTo(this._x(L), Y(v)) : g.moveTo(this._x(L), Y(v))));
      g.stroke();
      g.setLineDash([]);
      if (!dash.length) for (const [L, v] of pts) { g.beginPath(); g.arc(this._x(L), Y(v), L === sel ? 4 : 2.5, 0, 2 * Math.PI); g.fill(); }
    }
    g.lineWidth = 1;
    if (!r) {
      g.fillStyle = C.muted; g.textAlign = "center";
      g.fillText(this.busy ? t("測っています…") : m.contrast_ok ? t("まだ測っていません(「測り直す」)") : t("対照集合が見つからないので cohen_d は測れません"), w / 2, T0 + 24);
    }
  }

  _tip(m, r) {
    const tip = this.$("#dv-tip"), L = this.hoverL ?? this.sel;
    if (!m || L === null || L === undefined) return;
    const row = r ? r.layers.find((x) => x.layer === L) : null, i = m.layers.indexOf(L);
    const st = (x) => (x ? `${fmt(x.mean, 2)} ± ${fmt(x.sd, 2)} (${fmt(x.min, 1)}〜${fmt(x.max, 1)})` : "—");
    tip.innerHTML = `<b>L${L}</b>${L === m.base_layer ? ` <span class="muted">${t("基準層")}</span>` : ""} · cohen_d ${row ? fmt(row.cohen_d, 2) : "—"} · ${
      t("基準層との cos")} ${fmt(this._cosBase(L), 3)} · |mean_A − mean_B| ${i >= 0 ? fmt(m.norm[i], 2) : "—"}<br>A ${st(row && row.A)} · B ${st(row && row.B)}`;
  }

  // 選んだ層での例ごとの射影の値の分布(A = 琥珀、B = 青灰)と、例ごとの表
  _hist(m, r) {
    const cv = this.hv, w = Math.max(320, this.root.clientWidth - 24), dpr = window.devicePixelRatio || 1, L = this.sel;
    const A = r && L !== null ? (r.values.A || {})[String(L)] : null, B = r && L !== null ? (r.values.B || {})[String(L)] : null;
    this.$("#dv-hist-title").textContent = A || B ? t("L{L} の例ごとの射影の値(残差 · r̂_L、最後のトークン)", { L }) : "";
    cv.style.display = A || B ? "block" : "none";
    if (!(A || B)) { this.$("#dv-ex").innerHTML = ""; return; }
    cv.width = w * dpr; cv.height = HH * dpr; cv.style.width = `${w}px`; cv.style.height = `${HH}px`;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, HH);
    const all = [...(A || []), ...(B || [])];
    let lo = Math.min(...all), hi = Math.max(...all);
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const nb = Math.max(8, Math.min(30, Math.round(Math.sqrt(all.length) * 3)));
    const bin = (v) => Math.min(nb - 1, Math.floor(((v - lo) / (hi - lo)) * nb));
    const cnt = (xs) => { const c = new Array(nb).fill(0); for (const v of xs || []) c[bin(v)]++; return c; };
    const ca = cnt(A), cb = cnt(B), top = Math.max(1, ...ca, ...cb);
    const T1 = T0 + 12;                                                   // 平均の札の分だけ下げる
    const X = (v) => L0 + ((v - lo) / (hi - lo)) * (w - L0 - R0), bw = (w - L0 - R0) / nb, Y = (n) => T1 + (1 - n / top) * (HH - T1 - B0);
    g.font = "11px ui-monospace, monospace";
    for (let n = 0; n <= top; n += Math.max(1, Math.ceil(top / 4))) {
      g.strokeStyle = C.grid; g.beginPath(); g.moveTo(L0, Y(n)); g.lineTo(w - R0, Y(n)); g.stroke();
      g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(`${n}`, L0 - 6, Y(n) + 4);
    }
    for (const [c, col] of [[cb, "rgba(159,179,207,0.55)"], [ca, "rgba(232,176,74,0.6)"]]) {
      g.fillStyle = col;
      c.forEach((n, i) => { if (n) g.fillRect(L0 + i * bw + 1, Y(n), Math.max(1, bw - 2), Y(0) - Y(n)); });
    }
    const mean = (xs) => (xs && xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
    for (const [xs, col, lab] of [[A, C.amber, "A"], [B, C.b, "B"]]) {
      const mu = mean(xs);
      if (mu === null) continue;
      g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(X(mu), T1 - 2); g.lineTo(X(mu), HH - B0); g.stroke(); g.lineWidth = 1;
      g.fillStyle = col; g.textAlign = "center"; g.fillText(`${lab} ${t("平均")} ${fmt(mu, 1)}`, X(mu), T0 + 4);
    }
    g.fillStyle = C.muted; g.textAlign = "center";
    for (const v of [lo, hi, ...(lo < 0 && hi > 0 ? [0] : [])]) g.fillText(fmt(v, 1), X(v), HH - B0 + 15);
    const ps = r.prompts || [], n = Math.max((A || []).length, (B || []).length);
    const rows = Array.from({ length: n }, (_, i) => {
      const a = A ? A[i] : null, b = B ? B[i] : null;
      return `<tr><td class="w">${esc(ps.length === n ? qText(ps[i]) : `#${i}`)}</td><td>${fmt(a, 2)}</td><td>${fmt(b, 2)}</td><td>${a !== null && b !== null ? fmt(a - b, 2) : "—"}</td></tr>`;
    }).join("");
    this.$("#dv-ex").innerHTML = `<details><summary class="muted note">${t("例ごとの射影の値(L{L}、{n} 件)", { L, n })}</summary>
      <table class="evx mono dv-ex"><thead><tr><th>${t("プロンプト")}</th><th>A</th><th>B</th><th>A − B</th></tr></thead><tbody>${rows}</tbody></table></details>`;
  }
}
