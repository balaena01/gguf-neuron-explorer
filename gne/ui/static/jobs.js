// ジョブ(別冊 §8.3)。下のバーに実行中・待機中のジョブと進捗、層の右クリックで計算メニュー。
// GPU を使うジョブはユニットの切れ目で、チャットや fires に GPU を譲る(§8.1)。
import { deleteJSON, getJSON, postJSON } from "./api.js";
import { esc } from "./panels.js";

export const KIND = { lens_logit: "logit lens", lens_steer: "steer lens", sweep_rank: "sweep", sweep: "sweep", rank: "rank",
               stats_resid: "残差の統計", stats_act: "活性の統計", reverse: "逆引き" };

function eta(s) {
  if (s === null || s === undefined || !Number.isFinite(s)) return "";
  if (s < 60) return `残り ${Math.round(s)} 秒`;
  if (s < 3600) return `残り ${Math.round(s / 60)} 分`;
  return `残り ${(s / 3600).toFixed(1)} 時間`;
}

function what(j) {
  const p = j.params || {};
  const bits = [KIND[j.kind] || j.kind];
  if (p.layer !== undefined) bits.push(`L${p.layer}`);
  if (p.layers !== undefined) bits.push(`L${[].concat(p.layers).join(",")}`);
  if (p.design) bits.push(p.design);
  if (p.neurons) bits.push(/^\d/.test(p.neurons) ? `N${p.neurons.replace("-", "–")}` : `ニューロン ${p.neurons}`);
  if (p.random_controls !== undefined) bits.push(`+ 対照 ${p.random_controls}`);
  return bits.join(" ");
}

export class JobsBar {
  constructor(root, { onFinished }) {
    this.root = root;
    this.onFinished = onFinished;      // (job) → coverage の取り直しなど
    this.jobs = new Map();
    this.sources = new Map();
    this.root.addEventListener("click", (ev) => {
      const x = ev.target.closest("[data-cancel]");
      if (x) deleteJSON(`/api/jobs/${x.dataset.cancel}`).then((v) => this.update(v));
      const y = ev.target.closest("[data-clear]");
      if (y) { for (const [id, j] of this.jobs) if (!["queued", "running"].includes(j.status)) this.jobs.delete(id); this.render(); }
    });
  }

  // 起動時と、そのあと 3 秒ごとに一覧を取り直す(MCP や別のタブから投げたジョブも出すため)
  async load() {
    const once = async (quiet) => {
      try { for (const j of await getJSON("/api/jobs")) if (!this.sources.has(j.id)) this.update(j, quiet); } catch { /* つながらない間は飛ばす */ }
    };
    await once(true);
    this.render();
    setInterval(() => once(false), 3000);
  }

  async start(kind, params) {
    const j = await postJSON("/api/jobs", { kind, params });
    this.update(j);
    return j;
  }

  update(j, quiet = false) {
    const prev = this.jobs.get(j.id);
    this.jobs.set(j.id, j);
    if (["queued", "running"].includes(j.status) && !this.sources.has(j.id)) this.follow(j.id);
    const ended = ["done", "error", "cancelled"].includes(j.status);
    if (ended && prev && !["done", "error", "cancelled"].includes(prev.status) && !quiet) this.onFinished(j);
    this.render();
  }

  follow(id) {
    const es = new EventSource(`/api/jobs/${id}/events`);
    this.sources.set(id, es);
    es.onmessage = (m) => {
      const j = JSON.parse(m.data);
      this.update(j);
      if (["done", "error", "cancelled"].includes(j.status)) { es.close(); this.sources.delete(id); }
    };
    es.onerror = () => { es.close(); this.sources.delete(id); };
  }

  render() {
    const js = [...this.jobs.values()].reverse();
    if (!js.length) { this.root.innerHTML = `<span class="muted">jobs: なし(層の番号やラベルを右クリックすると計算を投げられます)</span>`; return; }
    const one = (j) => {
      const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
      const st = { queued: "待機中", running: "", done: "完了", error: "エラー", cancelled: "中止済み" }[j.status];
      const bar = j.status === "running" ? `<span class="jb-bar"><span style="width:${pct}%"></span></span> ${pct}%` : "";
      const tip = j.status === "error" ? ` title="${esc(j.error && j.error.message)}"` : "";
      const p = j.params || {};                                                    // 段階の名前が「lens steer L14」のように名前と重なるなら出さない
      const stage = j.status === "running" && j.stage && !(p.layer !== undefined && j.stage.includes(`L${p.layer}`)) ? ` ${esc(j.stage)}` : "";
      const secs = j.status === "done" && j.seconds ? ` ${Math.round(j.seconds)} 秒` : "";
      return `<span class="jb ${j.status}"${tip}><span class="what">${esc(what(j))}</span> ${bar} <span class="st muted">${esc(st)}${secs}${stage} ${eta(j.eta_s)}${
        j.yields ? ` · GPU を譲った ${j.yields} 回` : ""}</span>${["queued", "running"].includes(j.status) ? ` <button class="small" data-cancel="${j.id}" title="このジョブを止める">止める</button>` : ""}</span>`;
    };
    // 走っている・待っているものを先に。終わったものは最新の 2 件まで
    const live = js.filter((j) => ["queued", "running"].includes(j.status)), ended = js.filter((j) => !live.includes(j));
    const shown = [...live, ...ended.slice(0, 2)];
    this.root.innerHTML = `<span class="jb-list">${shown.map(one).join(`<span class="muted"> │ </span>`)}${
      ended.length > 2 ? `<span class="muted"> ほか ${ended.length - 2} 件</span>` : ""}</span>${
      ended.length ? ` <button class="small" data-clear="1" title="終わったジョブを一覧から消す(ジョブの結果は消えない)">終わった分を消す</button>` : ""}`;
  }
}

// 層の計算メニュー。目安の時間は Qwen3.5-4B・RTX 3080 の実測(docs/measurements.md)から n_ff で比例させたもの
export function computeMenu(ev, L, li, cov, { onStart }) {
  document.querySelector("#compute-menu")?.remove();
  const n = li.n_ff, k = n / 9216;
  const mins = (x) => (x < 1 ? `${Math.round(x * 60)} 秒` : x < 90 ? `${Math.round(x)} 分` : `${(x / 60).toFixed(1)} 時間`);
  const done = (x) => (x ? ` <span class="cm-done">済み${typeof x === "number" ? `(${x})` : ""}・やり直す</span>` : "");
  const runs = (d) => (cov && cov.sweep_runs.some((r) => r.endsWith(d)) ? true : 0);
  const items = [
    ["lens_logit", { layer: L }, `logit lens(重みだけ、CPU)`, `約 ${mins(0.6 * k)}`, cov && cov.logit_lens],
    ["lens_steer", { layer: L }, `steer lens(全 ${n} ニューロン + 対照 64)`, `約 ${mins((0.145 * (n + 64)) / 60)}`, cov && cov.steer_lens],
    ["sweep_rank", { layer: L, design: "quick", neurons: "random:64", random_controls: 16 }, "sweep quick(ランダムに 64 個 + 対照 16)", "約 5 分", runs("quick")],
    ["sweep_rank", { layer: L, design: "main_rel", neurons: "random:384" }, "sweep main_rel(ランダムに 384 個 + 対照 64)", `約 ${mins((6.4 * 448) / 60)}`, runs("main_rel")],
  ];
  const m = document.createElement("div");
  m.id = "compute-menu";
  m.innerHTML = `<div class="cm-head">L${L} を計算する <span class="muted">(目安は 4B の実測から)</span></div>` +
    items.map(([, , t, tm, x], i) => `<div class="cm-item" data-i="${i}">${esc(t)} <span class="muted">${esc(tm)}</span>${done(x)}</div>`).join("");
  m.style.left = `${Math.min(ev.clientX + 14, window.innerWidth - 400)}px`;               // 右クリックしたラベルを覆わないように
  m.style.top = `${Math.min(ev.clientY + 10, window.innerHeight - 170)}px`;
  document.body.appendChild(m);
  const close = () => { m.remove(); document.removeEventListener("pointerdown", away, true); };
  const away = (e) => { if (!m.contains(e.target)) close(); };
  document.addEventListener("pointerdown", away, true);
  for (const el of m.querySelectorAll(".cm-item")) {
    el.addEventListener("click", () => { const [kind, p] = items[+el.dataset.i]; close(); onStart(kind, p); });
  }
}
