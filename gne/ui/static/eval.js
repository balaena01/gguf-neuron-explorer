// eval ビュー(CHANGE-01 A、本体設計 §6.14): レシピの倍率ごとの副作用。x = 倍率、y = 性格の変化量と能力の低下など(%)。
// 保存済みの eval(<root>/evals/<model_id>/)を選んで見るか、今のレシピでジョブ eval を投げる。Canvas 2D。
// 反応性(CHANGE-04 F)を測った eval では、トリガーあり・なし・差の列(ヘッドの対照があればその p95 と null)と線を足す。
import { getJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const C = { text: "#aeb9c8", muted: "#6b7686", grid: "#1a212b", amber: "#e8b04a", minus: "#c94fa7", ppl: "#7c8796", judge: "#9fb3cf", ok: "#6fcf8f",
            react: "#5fd0c0" };
const L0 = 46, R0 = 16, T0 = 12, B0 = 34, H = 260;
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const pct = (x, d = 1) => (x === null || x === undefined ? "—" : `${fmt(x * 100, d)}%`);

function series(d) {
  const s = [
    { key: "novelty", label: t("性格の変化量(無介入に無い語の割合)"), color: C.amber, v: (r) => r.persona.novelty * 100 },
    { key: "drop", label: t("能力の低下(pt)"), color: C.minus, v: (r) => r.capability_drop_pt },
    { key: "ppl", label: t("ppl の悪化(%、100 で頭打ち)"), color: C.ppl, dash: [4, 4], v: (r) => Math.min(100, Math.max(0, r.ppl_worse_pct)) },
  ];
  if (d.judge) s.push({ key: "judge", label: t("judge(1〜5 を 0〜100 に)"), color: C.judge, v: (r) => (r.judge === null ? null : ((r.judge - 1) / 4) * 100) });
  if (d.reactivity) {                                                 // 反応性(CHANGE-04 F): トリガーあり − なしの判定の一致率(負は 0 に)
    s.push({ key: "reactivity", label: t("反応性(トリガーあり − なし、%)"), color: C.react, v: (r) => (r.reactivity ? Math.max(0, r.reactivity.reactivity * 100) : null) });
    if (d.reactivity.head_controls) {
      s.push({ key: "ctrl_react", label: t("ヘッドの対照の反応性 p95"), color: C.react, thin: true,
               v: (r) => (r.dose === 0 ? (r.reactivity ? Math.max(0, r.reactivity.reactivity * 100) : null)
                 : r.reactivity && r.reactivity.ctrl && r.reactivity.ctrl.p95 !== null ? Math.max(0, r.reactivity.ctrl.p95 * 100) : null) });
    }
  }
  if (d.random_controls) {                                            // 対照(CHANGE-03 A): 同じ大きさのランダムな向きのレシピ。薄い線
    const c = (f) => (r) => (r.dose === 0 ? 0 : r.ctrl ? f(r.ctrl) : null);
    s.push({ key: "ctrl_novelty", label: t("対照の新語 p95"), color: C.amber, thin: true, v: c((x) => (x.novelty.p95 === null ? null : x.novelty.p95 * 100)) },
      { key: "ctrl_drop", label: t("対照の能力の低下 p50"), color: C.minus, thin: true, v: c((x) => x.capability_drop_pt.p50) },
      { key: "ctrl_ppl", label: t("対照の ppl の悪化 p50"), color: C.ppl, thin: true, dash: [4, 4], v: c((x) => (x.ppl_worse_pct.p50 === null ? null : Math.min(100, Math.max(0, x.ppl_worse_pct.p50)))) });
  }
  return s;
}

export class EvalView {
  constructor(root, { recipe, onRun }) {
    this.root = root;
    this.recipe = recipe;              // UI のレシピ(recipe.js の Recipe)
    this.onRun = onRun;                // (params) → ジョブ eval を投げる
    this.list = [];
    this.data = null;
    this.dose = null;                  // 下に出力の例を出す倍率
    root.innerHTML = `<div class="evx-tools">
        <label class="muted">${t("保存した eval")} <select id="evx-pick"></select></label>
        <span class="evx-run"><span class="muted" id="evx-rc"></span>
          <label class="muted">${t("倍率")} <input id="evx-doses" value="0,0.25,0.5,0.75,1" size="16" spellcheck="false"></label>
          <label class="muted">when <select id="evx-when"><option>all</option><option>gen</option><option>assistant</option></select></label>
          <label class="muted" title="${t("各倍率で同じ大きさのランダムな向きのレシピも測る(時間はおよそ本数倍)")}">${t("対照")} <input id="evx-rc-n" type="number" min="0" max="64" value="0" style="width:4em"></label>
          <label class="muted" title="${t("反応性(CHANGE-04 F): reactivity の対(トリガーあり/なし)と validate.match を持つ対照集合の YAML。空なら測らない")}">${t("反応性")} <input id="evx-react" placeholder="configs/contrast/….yaml" size="22" spellcheck="false"></label>
          <label class="muted" title="${t("レシピのヘッドを同じ k のランダムなヘッドに替えた対照で反応性も測る本数(反応性があるときだけ)")}">${t("ヘッドの対照")} <input id="evx-hc-n" type="number" min="0" max="64" value="0" style="width:4em"></label>
          <button id="evx-go">${t("今のレシピで eval")}</button></span></div>
      <p class="muted note evx-help">${t("倍率はレシピ全体に掛ける係数(0 = 無介入)。能力 = exact と format の正答率、ppl は参照テキスト、混入は日本語の回答に出た簡体字・ハングルなどの割合。4B で数分。")}</p>
      <div id="evx-body"></div>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#evx-pick").addEventListener("change", (ev) => this.load(ev.target.value));
    this.$("#evx-go").addEventListener("click", () => this.run());
    new ResizeObserver(() => this.draw()).observe(root);
  }

  async refresh(select) {
    const r = this.recipe;
    this.$("#evx-rc").textContent = r && r.units.length ? t("レシピ {name}({n} ユニット)", { name: r.name || "—", n: r.units.length }) : t("レシピが空です");
    this.$("#evx-go").disabled = !(r && r.units.length);
    try { this.list = await getJSON("/api/evals"); } catch { this.list = []; }
    const pick = this.$("#evx-pick");
    pick.innerHTML = this.list.length
      ? this.list.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}(${esc((x.created || "").replace("T", " ").slice(0, 16))})</option>`).join("")
      : `<option value="">${t("(なし)")}</option>`;
    const name = select || (this.data && this.data.name) || (this.list[0] && this.list[0].name);
    if (name && this.list.some((x) => x.name === name)) { pick.value = name; await this.load(name); }
    else this.render();
  }

  async load(name) {
    if (!name) { this.data = null; this.render(); return; }
    try { this.data = await getJSON(`/api/evals/${encodeURIComponent(name)}`); }
    catch (e) { this.data = null; this.$("#evx-body").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    const ds = this.data.rows.map((r) => r.dose);
    if (!ds.includes(this.dose)) this.dose = ds[ds.length - 1];
    this.render();
  }

  run() {
    const doses = this.$("#evx-doses").value.split(",").map((x) => +x.trim()).filter((x) => Number.isFinite(x));
    if (!doses.length) return;
    const p = { units: this.recipe.specs(), doses, when: this.$("#evx-when").value };
    const nc = Math.max(0, Math.floor(+this.$("#evx-rc-n").value || 0));
    if (nc) p.random_controls = nc;
    const react = this.$("#evx-react").value.trim(), hc = Math.max(0, Math.floor(+this.$("#evx-hc-n").value || 0));
    if (react) { p.reactivity = react; if (hc) p.head_controls = hc; }
    if (this.recipe.name) p.name = this.recipe.name;
    this.onRun(p);
  }

  render() {
    const body = this.$("#evx-body");
    const d = this.data;
    if (!d) { body.innerHTML = `<p class="muted note">${t("eval がまだありません。上の「今のレシピで eval」で作ります(gne eval / MCP の eval でも)。")}</p>`; return; }
    const th = d.suite.thresholds;
    const units = d.recipe.units.map((u) => (u.mode === "query" ? `L${u.layer} H${u.head} query ${u.direction}→${String(u.seek || "").split("/").pop().replace(/\.ya?ml$/, "")} ×${fmt(u.k, 2)}`
      : u.mode === "gain" ? `L${u.layer} ${u.kind === "head" ? `H${u.head}` : `N${u.neuron}`} ×${fmt(u.k, 2)}`
      : `L${u.layer} ${u.kind === "random" ? `random${u.seed}` : u.kind === "direction" ? `dir:${u.direction}` : `N${u.neuron}`} ${u.dose >= 0 ? "+" : "−"}${fmt(Math.abs(u.dose), 2)}`));
    const rx = d.reactivity, hc = rx && rx.head_controls > 0;
    // 反応性(CHANGE-04 F)の列: トリガーあり・なしの判定の一致率と差。ヘッドの対照があればその p95 と null_pct
    const rxCells = (r) => {
      if (!rx) return "";
      const x = r.reactivity;
      if (!x) return `<td></td><td></td><td></td>${hc ? "<td></td><td></td>" : ""}`;
      return `<td>${pct(x.with, 0)}</td><td>${pct(x.without, 0)}</td><td class="${x.reactivity > 0 ? "plus" : ""}">${x.reactivity >= 0 ? "+" : "−"}${pct(Math.abs(x.reactivity), 0)}</td>${
        hc ? (x.ctrl ? `<td class="muted">${x.ctrl.p95 === null ? "—" : `${x.ctrl.p95 >= 0 ? "+" : "−"}${pct(Math.abs(x.ctrl.p95), 0)}`}</td><td class="${x.null_pct >= 0.95 ? "ok" : "ng"}">${fmt(x.null_pct, 2)}</td>`
          : "<td></td><td></td>") : ""}`;
    };
    const ok = d.rows.filter((r) => r.shippable && r.dose > 0).map((r) => r.dose);
    const rows = d.rows.map((r) => `<tr class="${r.dose === this.dose ? "on" : ""}" data-dose="${r.dose}">
        <td>${fmt(r.dose, 2)}</td><td>${pct(r.capability)}</td><td>${fmt(r.capability_drop_pt, 1)}</td>
        <td>${fmt(r.ppl_mean, 2)}</td><td>${r.ppl_worse_pct >= 0 ? "+" : ""}${fmt(r.ppl_worse_pct, 1)}%</td>
        <td>${pct(r.mix_zh + r.mix_other, 2)}</td><td>${fmt(r.persona.novelty, 2)}</td><td>${fmt(r.persona.distinct4, 2)}</td>
        <td>${pct(r.persona.capped, 0)}</td>${d.judge ? `<td>${fmt(r.judge, 2)}</td>` : ""}${d.random_controls ? (r.ctrl
          ? `<td class="muted">${fmt(r.ctrl_novelty_p95, 2)}</td><td class="${r.novelty_null_pct >= 0.95 ? "ok" : "ng"}">${fmt(r.novelty_null_pct, 2)}</td><td class="muted">${fmt(r.ctrl_capability_drop_p50, 1)}</td><td class="muted">${fmt(r.ctrl_ppl_worse_p50, 1)}%</td>`
          : "<td></td><td></td><td></td><td></td>") : ""}${rxCells(r)}
        <td class="${r.shippable ? "ok" : "ng"}">${r.shippable ? t("配布可") : Object.entries(r.ok).filter(([, v]) => !v).map(([k]) => ({ capability: t("能力"), ppl: "ppl", mix: t("混入") }[k])).join("・") + " ×"}</td></tr>`).join("");
    body.innerHTML = `
      <p class="note mono">${esc(units.join(" · "))} <span class="muted">· when=${esc(d.when)} · ${t("{n} タスク", { n: d.suite.n_tasks })} · ${fmt(d.seconds, 0)} s${d.random_controls ? ` · ${t("対照 {n} 本", { n: d.random_controls })}` : ""}${
        rx ? ` · ${t("反応性 {name}({n} 対)", { name: esc(rx.name), n: rx.n_pairs })}${hc ? ` · ${t("ヘッドの対照 {n} 本", { n: rx.head_controls })}` : ""}` : ""}</span></p>
      <p class="note">${ok.length ? t("配布可の最大倍率: {d}", { d: fmt(Math.max(...ok), 2) }) : t("配布可の倍率はありません(0 を除く)")}
        <span class="muted">${t("基準: 能力 −{c}pt 以内・ppl +{p}% 以内・混入が増えない", { c: th.capability_drop_pt, p: th.ppl_worse_pct })}</span></p>
      <div class="evx-legend note">${series(d).map((s) => `<span><i style="background:${s.color}${s.thin ? ";opacity:.45;height:1px" : ""}"></i>${esc(s.label)}</span>`).join("")}
        <span><i class="dot" style="background:${C.ok}"></i>${t("配布可")}</span></div>
      <canvas id="evx-canvas"></canvas>
      <table class="evx mono"><thead><tr><th>${t("倍率")}</th><th>${t("能力")}</th><th>${t("低下pt")}</th><th>ppl</th><th>${t("悪化")}</th><th>${t("混入")}</th>
        <th title="${t("無介入の出力に無い語の割合(プロンプトごとの平均)")}">${t("新語")}</th><th title="distinct-4">d4</th><th title="${t("max_tokens で打ち切られた割合")}">${t("打切")}</th>${d.judge ? "<th>judge</th>" : ""}${d.random_controls ? `<th title="${t("同じ大きさのランダムな向きの対照レシピの新語の 95% 点")}">${t("対照p95")}</th><th title="${t("新語が対照の何割より多いか(0.95 未満は対照と区別できない)")}">null</th><th title="${t("対照の能力の低下(中央値、pt)")}">${t("対照低下")}</th><th title="${t("対照の ppl の悪化(中央値)")}">${t("対照悪化")}</th>` : ""}${rx ? `<th title="${t("トリガーありのプロンプトで性格の判定(validate.match)に当たった割合")}">${t("あり")}</th><th title="${t("トリガーなしのプロンプトで判定に当たった割合")}">${t("なし")}</th><th title="${t("反応性 = あり − なし({n} 対)", { n: rx.n_pairs })}">${t("反応性")}</th>${
          hc ? `<th title="${t("レシピのヘッドを同じ k のランダムなヘッドに替えた対照の反応性の 95% 点")}">${t("対照p95")}</th><th title="${t("反応性が対照の何割より大きいか(0.95 未満は対照と区別できない)")}">null</th>` : ""}` : ""}<th></th></tr></thead>
        <tbody>${rows}</tbody></table>
      <div id="evx-sample"></div>`;
    for (const tr of body.querySelectorAll("tr[data-dose]")) tr.addEventListener("click", () => { this.dose = +tr.dataset.dose; this.render(); });
    this.cv = body.querySelector("#evx-canvas");
    this.draw();
    this.sample();
  }

  sample() {
    const d = this.data, el = this.$("#evx-sample");
    const r = d.rows.find((x) => x.dose === this.dose);
    const s = d.samples[String(this.dose)];                        // キーは Python の f"{dose:g}"
    if (!r || !s) { el.innerHTML = ""; return; }
    const words = r.persona.top_words.map(([w, n]) => `${esc(w)}(${n})`).join(" · ");
    el.innerHTML = `<h4 class="note">${t("倍率 {d} の出力の例", { d: fmt(this.dose, 2) })}</h4>
      ${words ? `<p class="muted note">${t("複数のプロンプトに新しく出た語")}: <span class="mono">${words}</span></p>` : ""}
      ${s.persona.map((x) => `<p class="note"><span class="muted mono">${esc(x.id)}</span> ${esc(x.text)}</p>`).join("")}
      ${s.failed.length ? `<details><summary class="muted note">${t("落ちたタスク({n})", { n: r.failed.exact.length + r.failed.format.length })}</summary>${
        s.failed.map((x) => `<p class="note"><span class="muted mono">${esc(x.id)}</span> ${esc(x.text)}</p>`).join("")}</details>` : ""}
      ${r.reactivity && r.reactivity.samples && r.reactivity.samples.length ? `<details><summary class="muted note">${t("反応性の例(トリガーあり / なし)")}</summary>${
        r.reactivity.samples.map((x) => `<p class="note"><span class="muted mono">${t("あり")}</span> ${esc(x.with)}<br><span class="muted mono">${t("なし")}</span> ${esc(x.without)}</p>`).join("")}</details>` : ""}`;
  }

  draw() {
    const cv = this.cv, d = this.data;
    if (!cv || !d || !cv.isConnected) return;
    const w = Math.max(320, this.root.clientWidth - 24), dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = H * dpr; cv.style.width = `${w}px`; cv.style.height = `${H}px`;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, H);
    const xs = d.rows.map((r) => r.dose), x0 = Math.min(0, ...xs), x1 = Math.max(...xs) || 1;
    const X = (v) => L0 + ((v - x0) / (x1 - x0 || 1)) * (w - L0 - R0), Y = (v) => T0 + (1 - Math.max(0, Math.min(100, v)) / 100) * (H - T0 - B0);
    g.font = "11px ui-monospace, monospace";
    g.lineWidth = 1;
    for (const v of [0, 25, 50, 75, 100]) {
      g.strokeStyle = C.grid; g.beginPath(); g.moveTo(L0, Y(v)); g.lineTo(w - R0, Y(v)); g.stroke();
      g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(`${v}`, L0 - 6, Y(v) + 4);
    }
    g.textAlign = "center";
    for (const r of d.rows) {
      g.fillStyle = r.dose === this.dose ? C.text : C.muted;
      g.fillText(fmt(r.dose, 2), X(r.dose), H - B0 + 16);
      if (r.shippable) { g.fillStyle = C.ok; g.beginPath(); g.arc(X(r.dose), H - B0 + 26, 3, 0, 2 * Math.PI); g.fill(); }
    }
    g.strokeStyle = C.minus; g.setLineDash([2, 3]);                       // 能力の低下の基準
    g.beginPath(); g.moveTo(L0, Y(d.suite.thresholds.capability_drop_pt)); g.lineTo(w - R0, Y(d.suite.thresholds.capability_drop_pt)); g.stroke();
    g.setLineDash([]);
    for (const s of series(d)) {
      const pts = d.rows.map((r) => [r.dose, s.v(r)]).filter(([, v]) => v !== null && Number.isFinite(v));
      g.strokeStyle = s.color; g.fillStyle = s.color; g.lineWidth = s.thin ? 1 : 2; g.globalAlpha = s.thin ? 0.45 : 1;
      g.setLineDash(s.dash || []);
      g.beginPath();
      pts.forEach(([x, v], i) => (i ? g.lineTo(X(x), Y(v)) : g.moveTo(X(x), Y(v))));
      g.stroke();
      g.setLineDash([]);
      if (!s.thin) for (const [x, v] of pts) { g.beginPath(); g.arc(X(x), Y(v), 3, 0, 2 * Math.PI); g.fill(); }
      g.globalAlpha = 1;
    }
    g.lineWidth = 1;
  }
}
