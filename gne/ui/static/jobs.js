// ジョブ(別冊 §8.3)。下のバーに実行中・待機中のジョブと進捗、層の右クリックで計算メニュー。
// GPU を使うジョブはユニットの切れ目で、チャットや fires に GPU を譲る(§8.1)。
import { deleteJSON, getJSON, postJSON } from "./api.js";
import { t as tr } from "./i18n.js";
import { esc } from "./panels.js";

export const KIND = { embed: tr("埋め込み"), connections: "connections", layout: tr("配置"), lens_logit: "logit lens", lens_steer: "steer lens", sweep_rank: "sweep", sweep: "sweep", rank: "rank",
               stats_resid: tr("残差の統計"), stats_act: tr("活性の統計"), reverse: tr("逆引き"), eval: "eval",
               heads_contrast: tr("ヘッドの対照分析") };

function eta(s) {
  if (s === null || s === undefined || !Number.isFinite(s)) return "";
  if (s < 60) return tr("残り {n} 秒", { n: Math.round(s) });
  if (s < 3600) return tr("残り {n} 分", { n: Math.round(s / 60) });
  return tr("残り {n} 時間", { n: (s / 3600).toFixed(1) });
}

function what(j) {
  const p = j.params || {};
  const bits = [KIND[j.kind] || j.kind];
  if (p.layer !== undefined) bits.push(`L${p.layer}`);
  if (p.kind) bits.push(p.kind);
  if (p.run) bits.push(p.run);
  if (p.layers !== undefined) bits.push(p.layers === "all" ? tr("全層") : `L${[].concat(p.layers).join(",")}`);
  if (p.design) bits.push(p.design);
  if (p.contrast) bits.push(String(p.contrast).split("/").pop());
  if (j.kind === "eval") bits.push(p.name || p.recipe || tr("{n} ユニット", { n: (p.units || []).length }), ...(p.when && p.when !== "all" ? [`@${p.when}`] : []));
  if (p.neurons) bits.push(/^\d/.test(p.neurons) ? `N${p.neurons.replace("-", "–")}` : tr("ニューロン {s}", { s: p.neurons }));
  if (p.random_controls !== undefined) bits.push(tr("+ 対照 {n}", { n: p.random_controls }));
  if (j.kind === "eval" && p.reactivity) bits.push(tr("+ 反応性"));
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
      try {                                                      // gne worker のジョブ(CHANGE-03 B)。名前の前にワーカー名を付ける
        for (const f of (await getJSON("/api/workers")).jobs) {
          if (!["queued", "running"].includes(f.status) && !this.jobs.has(`w:${f.id}`)) continue;
          this.update({ id: `w:${f.id}`, kind: f.kind, params: f.params || {}, status: f.status, done: f.done || 0,
                        total: f.total || 0, stage: f.stage || "", worker: f.worker, eta_s: null, yields: 0,
                        seconds: f.finished && f.started ? f.finished - f.started : null, error: f.error }, quiet);
        }
      } catch { /* ワーカーの置き場所が無い */ }
    };
    await once(true);
    this.render();
    setInterval(() => once(false), 3000);
  }

  async start(kind, params) {
    let j;
    try { j = await postJSON("/api/jobs", { kind, params }); }
    catch (e) {                                                  // 見積もりが 30 分を超えたら確かめてから投げ直す(CHANGE-03 C)
      if (e.code !== "NEEDS_CONFIRM") throw e;
      const est = (e.body && e.body.estimate) || {};
      const m = est.seconds ? Math.round(est.seconds / 60) : "?";
      if (!window.confirm(tr("{kind} の見積もりは約 {m} 分です。走らせますか?", { kind, m }))) return null;
      j = await postJSON("/api/jobs", { kind, params: { ...params, yes: true } });
    }
    this.update(j);
    return j;
  }

  update(j, quiet = false) {
    const prev = this.jobs.get(j.id);
    this.jobs.set(j.id, j);
    if (["queued", "running"].includes(j.status) && !this.sources.has(j.id) && !j.worker) this.follow(j.id);
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
    if (!js.length) { this.root.innerHTML = `<span class="muted">${tr("jobs: なし(層の番号やラベルを右クリックすると計算を投げられます)")}</span>`; return; }
    const one = (j) => {
      const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
      const st = { queued: tr("待機中"), running: "", done: tr("完了"), error: tr("エラー"), cancelled: tr("中止済み") }[j.status];
      const bar = j.status === "running" ? `<span class="jb-bar"><span style="width:${pct}%"></span></span> ${pct}%` : "";
      const tip = j.status === "error" ? ` title="${esc(j.error && j.error.message)}"` : "";
      const p = j.params || {};                                                    // 段階の名前が「lens steer L14」のように名前と重なるなら出さない
      const stage = j.status === "running" && j.stage && !(p.layer !== undefined && j.stage.includes(`L${p.layer}`)) ? ` ${esc(j.stage)}` : "";
      const secs = j.status === "done" && j.seconds ? ` ${tr("{n} 秒", { n: Math.round(j.seconds) })}` : "";
      return `<span class="jb ${j.status}"${tip}><span class="what">${j.worker ? `<span class="muted">[${esc(j.worker)}]</span> ` : ""}${esc(what(j))}</span> ${bar} <span class="st muted">${esc(st)}${secs}${stage} ${eta(j.eta_s)}${
        j.yields ? ` · ${tr("GPU を譲った {n} 回", { n: j.yields })}` : ""}</span>${["queued", "running"].includes(j.status) && !j.worker ? ` <button class="small" data-cancel="${j.id}" title="${tr("このジョブを止める")}">${tr("止める")}</button>` : ""}</span>`;
    };
    // 走っている・待っているものを先に。終わったものは最新の 2 件まで
    const live = js.filter((j) => ["queued", "running"].includes(j.status)), ended = js.filter((j) => !live.includes(j));
    const shown = [...live, ...ended.slice(0, 2)];
    this.root.innerHTML = `<span class="jb-list">${shown.map(one).join(`<span class="muted"> │ </span>`)}${
      ended.length > 2 ? `<span class="muted"> ${tr("ほか {n} 件", { n: ended.length - 2 })}</span>` : ""}</span>${
      ended.length ? ` <button class="small" data-clear="1" title="${tr("終わったジョブを一覧から消す(ジョブの結果は消えない)")}">${tr("終わった分を消す")}</button>` : ""}`;
  }
}

// 層の計算メニュー。目安の時間は Qwen3.5-4B・RTX 3080 の実測(docs/measurements.md)から n_ff で比例させたもの
export function computeMenu(ev, L, li, cov, { onStart, arch, model }) {
  document.querySelector("#compute-menu")?.remove();
  const n = li.n_ff, k = n / 9216;
  const mins = (x) => (x < 1 ? tr("{n} 秒", { n: Math.round(x * 60) }) : x < 90 ? tr("{n} 分", { n: Math.round(x) }) : tr("{n} 時間", { n: (x / 60).toFixed(1) }));
  const about = (x) => tr("約 {t}", { t: x });
  const done = (x) => (x ? ` <span class="cm-done">${tr("済み")}${typeof x === "number" ? ` (${x})` : ""} · ${tr("やり直す")}</span>` : "");
  const runs = (d) => (cov && cov.sweep_runs.some((r) => r.endsWith(d)) ? true : 0);
  const nL = arch.n_layer, pairs = (nL * (nL - 1)) / 2, d = arch.n_embd || 2560;
  const lay = (kind) => ((model.layouts || {})[kind] || []);
  const items = [
    ["lens_logit", { layer: L }, tr("logit lens(重みだけ、CPU)"), about(mins(0.6 * k)), cov && cov.logit_lens],
    ["lens_steer", { layer: L }, tr("steer lens(全 {n} ニューロン + 対照 64)", { n }), about(mins((0.145 * (n + 64)) / 60)), cov && cov.steer_lens],
    ["sweep_rank", { layer: L, design: "quick", neurons: "random:64", random_controls: 16 }, tr("sweep quick(ランダムに 64 個 + 対照 16)"), about(mins(5)), runs("quick")],
    ["sweep_rank", { layer: L, design: "main_rel", neurons: "random:384" }, tr("sweep main_rel(ランダムに 384 個 + 対照 64)"), about(mins((6.4 * 448) / 60)), runs("main_rel")],
    ["layout", { kind: "writes", layers: [L] }, tr("配置 writes(この層、W_down → UMAP)"), about(mins((25 * k) / 60)), lay("writes").includes(L)],
    ["layout", { kind: "writes", layers: "all" }, tr("配置 writes(全層)"), about(mins((25 * k * nL) / 60)), lay("writes").length === nL && lay("writes").length],
    ["layout", { kind: "lens", layers: [L] }, tr("配置 lens(この層、lens の語 → UMAP)"), about(mins((10 * k) / 60)), lay("lens").includes(L)],
    ["connections", {}, tr("connections(全 {p} 層ペア、重みだけ、CPU)", { p: pairs }), about(mins((4.5 * k * k * (d / 2560) * pairs) / 60)), !!model.connections],
  ];
  const m = document.createElement("div");
  m.id = "compute-menu";
  m.innerHTML = `<div class="cm-head">${tr("L{L} を計算する", { L })} <span class="muted">${tr("(目安は 4B の実測から)")}</span></div>` +
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
