// eval ビュー(CHANGE-01 A、本体設計 §6.14): レシピの倍率ごとの副作用。x = 倍率、y = 性格の変化量と能力の低下など(%)。
// 保存済みの eval(<root>/evals/<model_id>/)を選んで見るか、今のレシピでジョブ eval を投げる。Canvas 2D。
// 反応性(CHANGE-04 F)を測った eval では、トリガーあり・なし・差の列(ヘッドの対照があればその p95 と null)と線を足す。
// CHANGE-06 G: 方向を使うレシピの eval では方向の指標の列(A 側の射影の値の下がり方・直交成分のずれ・判定の一致率、対照があれば null)。
// 方式の比較(eval --compare-methods)は方式ごとの表、ヘッド数の曲線(eval --alloc)は x = ヘッド数の小さな折れ線 3 つと表。
// CHANGE-07 G: keyed のユニットを持つレシピの eval では、行に eval の選択性(書き込みたい位置 − 書き込みたくない位置の r̂ 方向の変化)の列と、
// 結果の keyed(方式の比較: keyed・同じ α の add・書き込みたい位置をそろえた add / gain / proj の、書き込みたくない位置・dont_touch の乱れ、比、
// 比較用の鍵での eval の選択性)の表。
import { getJSON } from "./api.js";
import { t } from "./i18n.js";
import { FieldtestPanel } from "./fieldtest.js";
import { esc } from "./panels.js";
import { Steps } from "./steps.js";
import { noteHTML } from "./ui.js";

const C = { text: "#aeb9c8", muted: "#6b7686", grid: "#1a212b", amber: "#e8b04a", minus: "#c94fa7", ppl: "#7c8796", judge: "#9fb3cf", ok: "#6fcf8f",
            react: "#5fd0c0" };
const L0 = 46, R0 = 16, T0 = 12, B0 = 34, H = 260;
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const pct = (x, d = 1) => (x === null || x === undefined ? "—" : `${fmt(x * 100, d)}%`);

// 保存した結果の種類(session.evals の kind と同じ見分け方)
function kindOf(d) {
  if (d.recipe && d.suite) return "eval";
  return "target_from" in d ? "compare-methods" : "alloc" in d ? "nheads" : "neurons" in d ? "compare-ablation" : "other";
}

function series(d) {
  const s = [
    { key: "novelty", label: t("性格の変化量(無介入に無い語の割合)"), color: C.amber, v: (r) => r.persona.novelty * 100 },
    { key: "drop", label: t("能力の低下(pt)"), color: C.minus, v: (r) => r.capability_drop_pt },
    { key: "ppl", label: t("ppl の悪化(%、100 で頭打ち)"), color: C.ppl, dash: [4, 4], v: (r) => Math.min(100, Math.max(0, r.ppl_worse_pct)) },
  ];
  if (d.judge) s.push({ key: "judge", label: t("judge(1〜5 を 0〜100 に)"), color: C.judge, v: (r) => (r.judge === null ? null : ((r.judge - 1) / 4) * 100) });
  if (d.direction_metrics && d.direction_metrics.behavior) {          // 方向を使うレシピ(CHANGE-06 F): 判定の一致率
    s.push({ key: "rate", label: t("判定の一致率(%)"), color: C.react, dash: [2, 3], v: (r) => (r.rate === undefined || r.rate === null ? null : r.rate * 100) });
  }
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
  constructor(root, { recipe, onRun, onLoadProj, onLoadRecipe, onRunJob }) {
    this.root = root;
    this.recipe = recipe;              // UI のレシピ(recipe.js の Recipe)
    this.onRun = onRun;                // (params) → ジョブ eval を投げる
    this.onLoadProj = onLoadProj || (() => {});       // (名前, ["L:Hh:sS@NAME"]) → ヘッド数の曲線の組をレシピに読み込む
    this.onLoadRecipe = onLoadRecipe || (() => {});   // (レシピの dict) → 方式の比較の行のレシピを読み込む
    this.list = [];
    this.data = null;
    this.dose = null;                  // 下に出力の例を出す倍率
    // 読み込みの順番: 後から始めた読み込みがあるか、待つ間に表示が替わった(ほかの eval を選んだなど)なら、先に始めた読み込みの
    // 結果は捨てる(遅れて届いた古い結果で、選び直した表示を上書きしないように)
    this.loadSeq = 0;                  // load の通し番号
    this.refreshSeq = 0;               // refresh の通し番号
    this.want = null;                  // 最後に読み込みを始めた eval の名前
    root.innerHTML = `<div class="evx-tools">
        <label class="muted">${t("保存した eval")} <select id="evx-pick"></select></label>
        <span class="evx-run"><span class="muted" id="evx-rc"></span>
          <label class="muted">${t("倍率")} <input id="evx-doses" value="0,0.25,0.5,0.75,1" size="16" spellcheck="false"></label>
          <label class="muted">when <select id="evx-when"><option>all</option><option>gen</option><option>assistant</option></select></label>
          <label class="muted" title="${t("各倍率で同じ大きさのランダムな向きのレシピも測る(時間はおよそ本数倍)")}">${t("対照")} <input class="w4" id="evx-rc-n" type="number" min="0" max="64" value="0"></label>
          <label class="muted" title="${t("反応性: reactivity の対(トリガーあり/なし)と validate.match を持つ対照集合の YAML。空なら測らない")}">${t("反応性")} <input id="evx-react" placeholder="configs/contrast/….yaml" size="22" spellcheck="false"></label>
          <label class="muted" title="${t("レシピのヘッドを同じ k のランダムなヘッドに替えた対照で反応性も測る本数(反応性があるときだけ)")}">${t("ヘッドの対照")} <input class="w4" id="evx-hc-n" type="number" min="0" max="64" value="0"></label>
          <label class="muted" title="${t("方向を使うレシピ(ヘッドの proj など)で、方向の指標(射影の値の下がり方・ずれ・判定の一致率)も測る。外すと能力・ppl・混入・性格の変化量だけで早く終わる")}"><input type="checkbox" id="evx-dm" checked> ${t("方向の指標")}</label>
          <label class="muted" title="${t("方向の指標で読む質問の数(A・B それぞれ。一致率の返事もこの数まで)。多い対照集合は等間隔に間引く。0 で全部")}">${t("問")} <input class="w45" id="evx-dm-n" type="number" min="0" max="4096" value="64"></label>
          <span id="evx-kx" class="evx-kx-in hidden">
            <label class="muted" title="${t("keyed のユニットがあるとき、鍵の対照集合の保留した例で eval の選択性(書き込みたい位置の残差を動かし、書き込みたくない位置・dont_touch を動かさないか)と、add・gain・proj との比較、比較用の鍵との比較を測る(prefill だけ)。key-check の選択性(鍵の反応の強さの差)とは別の量")}"><input type="checkbox" id="evx-km" checked> ${t("keyed の選択性")}</label>
            <label class="muted" title="${t("鍵の対照集合。自動 = 方向を作った対照集合が鍵と同じものならそれ(違えば測らずに理由を出す)")}">${t("鍵の対照集合")} <select id="evx-kc"><option value="">${t("自動")}</option></select></label>
            <label class="muted" title="${t("会話の組ごとに読む例の数")}">${t("例")} <input class="w45" id="evx-ke" type="number" min="1" max="4096" value="8"></label>
            <label class="muted" title="${t("比べる比較用の鍵の本数(0 で比べない)")}">${t("比較用の鍵")} <input class="w4" id="evx-kp" type="number" min="0" max="256" value="8"></label>
            <label class="muted" title="${t("読む質問。自動 = 鍵の作成と反対側(even ↔ odd)")}">subset <select id="evx-ks"><option value="">${t("自動")}</option><option>odd</option><option>even</option><option>all</option></select></label>
            <label class="muted" title="${t("add・gain・proj の強さを探して、書き込みたい位置の変化を keyed にそろえる(外すと同じ α の add とだけ比べて早く終わる)")}"><input type="checkbox" id="evx-kmatch" checked> ${t("大きさ合わせ")}</label>
          </span>
          <button class="main" id="evx-go">${t("今のレシピで eval")}</button></span></div>
      <div class="evx-help">${noteHTML(t("倍率はレシピ全体に掛ける係数(0 = 無介入)。能力 = exact と format の正答率、ppl は参照テキスト、混入は日本語の回答に出た簡体字・ハングルなどの割合。4B で数分。"))}</div>
      <div id="evx-body"></div>
      <details class="evx-ft" id="evx-ft-wrap"><summary>${t("実地テスト(本番と同じ動かし方で、ファイルどうしを比べる)")}</summary><div id="evx-ft"></div></details>`;
    this.$ = (s) => root.querySelector(s);
    this.ft = new FieldtestPanel(this.$("#evx-ft"), { onRun: onRunJob });
    this.$("#evx-ft-wrap").addEventListener("toggle", () => { if (this.$("#evx-ft-wrap").open) this.ft.refresh(); });
    this.$("#evx-pick").addEventListener("change", (ev) => this.load(ev.target.value));
    this.$("#evx-go").addEventListener("click", () => this.run());
    new ResizeObserver(() => this.draw()).observe(root);
    // 左に作業の一覧(能力と ppl・実地テスト)、右に 入力 → 実行 → 結果
    this.steps = new Steps(root, {
      store: "gne.eval.step", label: t("評価の作業"),
      steps: [{ key: "main", label: t("能力と ppl"), els: () => [this.$(".evx-tools"), this.$(".evx-help"), this.$("#evx-body")] },
              { key: "ft", label: t("実地テスト"), els: () => [this.$("#evx-ft-wrap")] }],
      onSelect: (k) => { if (k === "ft" && !this.$("#evx-ft-wrap").open) this.$("#evx-ft-wrap").open = true; },
    });
  }

  async refresh(select) {
    const r = this.recipe;
    this.$("#evx-rc").textContent = r && r.units.length ? t("レシピ {name}({n} ユニット)", { name: r.name || "—", n: r.units.length }) : t("レシピが空です");
    this.$("#evx-go").disabled = !(r && r.units.length);
    const hasKeyed = !!(r && r.keyed && r.keyed().length);            // CHANGE-07: keyed の選択性の欄はレシピに keyed があるときだけ
    this.$("#evx-kx").classList.toggle("hidden", !hasKeyed);
    if (hasKeyed && !this.contrastsLoaded) {
      this.contrastsLoaded = true;
      getJSON("/api/contrasts").then((cs) => {
        const sel = this.$("#evx-kc");
        for (const c of Array.isArray(cs) ? cs : []) {
          const o = document.createElement("option");
          o.value = c.path; o.textContent = c.label;
          sel.appendChild(o);
        }
      }).catch(() => { this.contrastsLoaded = false; });
    }
    const rseq = ++this.refreshSeq, lseq = this.loadSeq, before = this.data;
    let list;
    try { list = await getJSON("/api/evals"); } catch { list = []; }
    if (rseq !== this.refreshSeq) return;                             // 後から始めた refresh に任せる
    this.list = list;
    const pick = this.$("#evx-pick");
    const tag = (k) => ({ "compare-methods": ` · ${t("方式の比較")}`, nheads: ` · ${t("ヘッド数の曲線")}`, "compare-ablation": ` · ${t("消去の比較")}`,
                          other: ` · ${t("その他")}` }[k] || "");
    pick.innerHTML = this.list.length
      ? this.list.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}(${esc((x.created || "").replace("T", " ").slice(0, 16))}${tag(x.kind)})</option>`).join("")
      : `<option value="">${t("(なし)")}</option>`;
    if (this.loadSeq !== lseq || this.data !== before) {             // 一覧を待つ間に別の eval を選んだか、表示が替わった: それを残す
      const cur = this.loadSeq !== lseq ? this.want : this.data && this.data.name;
      if (cur && this.list.some((x) => x.name === cur)) pick.value = cur;
      return;
    }
    const name = select || (this.data && this.data.name) || (this.list[0] && this.list[0].name);
    if (name && this.list.some((x) => x.name === name)) { pick.value = name; await this.load(name); }
    else this.render();
  }

  async load(name) {
    const seq = ++this.loadSeq, before = this.data;
    this.want = name || null;
    if (!name) { this.data = null; this.render(); return; }
    const stale = () => seq !== this.loadSeq || this.data !== before;   // 古い読み込み(上の this.loadSeq の説明)
    let d;
    try { d = await getJSON(`/api/evals/${encodeURIComponent(name)}`); }
    catch (e) {
      if (stale()) return;
      this.data = null; this.$("#evx-body").innerHTML = `<p class="err">${esc(e.message)}</p>`; return;
    }
    if (stale()) return;
    this.data = d;
    if (kindOf(this.data) === "eval") {
      const ds = this.data.rows.map((r) => r.dose);
      if (!ds.includes(this.dose)) this.dose = ds[ds.length - 1];
    }
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
    if (!this.$("#evx-dm").checked) p.direction_metrics = false;
    else {
      const dmn = Math.max(0, Math.floor(+this.$("#evx-dm-n").value || 0));
      if (dmn !== 64) p.dir_max_examples = dmn || null;                 // 0 = 全部の質問
    }
    if (this.recipe.keyed && this.recipe.keyed().length) {          // CHANGE-07 E: keyed の選択性
      p.keyed_metrics = this.$("#evx-km").checked;
      if (p.keyed_metrics) {
        const kc = this.$("#evx-kc").value, ks = this.$("#evx-ks").value;
        if (kc) p.keyed_contrast = kc;
        if (ks) p.keyed_subset = ks;
        p.keyed_examples = Math.max(1, Math.floor(+this.$("#evx-ke").value || 8));
        p.keyed_pseudo = Math.max(0, Math.floor(+this.$("#evx-kp").value || 0));
        p.keyed_match = this.$("#evx-kmatch").checked;
      }
    }
    if (this.recipe.name) p.name = this.recipe.name;
    this.onRun(p);
  }

  render() {
    const body = this.$("#evx-body");
    const d = this.data;
    this.cv = null;
    this.nh = null;
    if (!d) { body.innerHTML = `${noteHTML(t("eval がまだありません。上の「今のレシピで eval」で作ります(gne eval / MCP の eval でも)。"), "muted note")}`; return; }
    const kind = kindOf(d);
    if (kind === "compare-methods") { this.renderMethods(body, d); return; }
    if (kind === "nheads") { this.renderNheads(body, d); return; }
    if (kind !== "eval") {
      body.innerHTML = `<p class="muted note">${t("この種類の結果の表示はありません。中身をそのまま出します。")}</p>
        <details open><summary class="muted note">JSON</summary><pre class="mono note evx-raw">${esc(JSON.stringify(d, null, 1).slice(0, 40000))}</pre></details>`;
      return;
    }
    const th = d.suite.thresholds;
    // ユニットの名前(MoE は E37 N511・S N5・E3、CHANGE-05 G)。route は保存の形では delta
    const who = (u) => (u.kind === "head" ? `H${u.head}` : u.kind === "random" ? `random${u.seed}` : u.kind === "direction" ? `dir:${u.direction}`
      : u.kind === "expert_neuron" ? `E${u.expert} N${u.neuron}` : u.kind === "shared" ? `S N${u.neuron}` : u.kind === "expert" ? `E${u.expert}` : `N${u.neuron}`);
    const pm = (x) => `${x >= 0 ? "+" : "−"}${fmt(Math.abs(x), 2)}`;
    const units = d.recipe.units.map((u) => (u.mode === "query" ? `L${u.layer} H${u.head} query ${u.direction}→${String(u.seek || "").split("/").pop().replace(/\.ya?ml$/, "")} ×${fmt(u.k, 2)}`
      : u.mode === "proj" ? `L${u.layer} H${u.head} s${+(+u.k).toFixed(3)} @${u.direction || d.recipe.default_direction || "?"}`
      : u.mode === "keyed" ? `L${u.layer} ${u.kind === "shared" ? "S" : u.kind === "expert" ? `E${u.expert}` : u.kind} keyed a${+(+u.k).toFixed(3)} @${
        u.direction || d.recipe.default_direction || "?"} #${u.key || d.recipe.default_key || "?"}`                    // CHANGE-07
      : u.mode === "gain" ? `L${u.layer} ${who(u)} ×${fmt(u.k, 2)}`
      : u.mode === "route" ? `L${u.layer} ${who(u)} route ${pm(u.delta ?? u.dose)}`
      : `L${u.layer} ${who(u)} ${pm(u.dose)}`));
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
    // 方向の指標(CHANGE-06 F): A 側の射影の値の下がり方、直交成分のずれ(A・B)、判定の一致率とその減少。対照があれば null
    const dm = d.direction_metrics, hasDir = d.rows.some((r) => r.dir_drop_a !== undefined), rc = !!d.random_controls;
    const nullTd = (x) => `<td class="${x === null || x === undefined ? "" : x >= 0.95 ? "ok" : "ng"}">${fmt(x, 2)}</td>`;
    const dirCells = (r) => (hasDir ? `<td>${fmt(r.dir_drop_a, 2)}</td><td>${fmt(r.drift_orth_a, 3)}</td><td>${fmt(r.drift_orth_b, 3)}</td>${
      dm && dm.behavior ? `<td>${pct(r.rate, 0)}</td><td>${pct(r.rate_drop, 0)}</td>` : ""}${
      rc ? (r.ctrl ? `${nullTd(r.dir_null_pct)}${nullTd(r.drift_orth_b_null_pct)}${dm && dm.behavior ? nullTd(r.rate_null_pct) : ""}` : `<td></td><td></td>${dm && dm.behavior ? "<td></td>" : ""}`) : ""}` : "");
    const dirHead = hasDir ? `<th title="${t("A 側の射影の値の下がり方(無介入 − 介入、射影の値を測る層)")}">${t("A 下がり")}</th><th title="${t("A 側の残差の差のうち方向に直交する成分(ずれの大きさ ÷ 残差のノルム)")}">A ⊥</th><th title="${t("B 側(関係のない入力)の残差の差のうち方向に直交する成分。小さいほど他を壊さない")}">B ⊥</th>${
      dm && dm.behavior ? `<th title="${t("対照集合の検証プロンプトで性格の判定(validate.match)に当たった割合")}">${t("一致率")}</th><th title="${t("判定の一致率の減少(無介入 − 介入)")}">${t("一致減")}</th>` : ""}${
      rc ? `<th title="${t("A 側の下がり方が対照(同数・同じ s のランダムなヘッド)の何割より大きいか")}">${t("下がり null")}</th><th title="${t("B 側の直交成分のずれが対照の何割より大きいか(大きいほど他を壊している)")}">B⊥ null</th>${
        dm && dm.behavior ? `<th title="${t("一致率の減少が対照の何割より大きいか")}">${t("一致 null")}</th>` : ""}` : ""}` : "";
    // eval の選択性(CHANGE-07 E): 目的の層での、書き込みたい位置 − 書き込みたくない位置の r̂ 方向の変化。比較用の鍵の対照があればその p95 と null
    const hasK = d.rows.some((r) => r.keyed), kctrl = hasK && d.rows.some((r) => r.keyed && r.keyed.null_pct !== undefined);
    const kCells = (r) => (hasK ? `<td>${r.keyed ? fmt(r.keyed.selectivity, 2) : "—"}</td><td>${r.keyed ? fmt(r.keyed.b_norm, 2) : "—"}</td>${
      kctrl ? (r.keyed && r.keyed.null_pct !== undefined ? `<td class="muted">${fmt(r.keyed.ctrl_p95, 2)}</td>${nullTd(r.keyed.null_pct)}` : "<td></td><td></td>") : ""}` : "");
    const kHead = hasK ? `<th class="wrap" title="${t("eval の選択性: 目的の層の残差の変化の r̂ 方向の成分の、書き込みたい位置 − 書き込みたくない位置(書く向きにそろえた値。大きいほど書き込みたい位置だけを動かす)。key-check の選択性とは別の量")}">${t("eval の選択性")}</th><th class="wrap" title="${t("書き込みたくない位置の残差の変化の大きさ ‖Δ‖")}">${t("書き込みたくない位置の ‖Δ‖")}</th>${
      kctrl ? `<th class="wrap" title="${t("鍵を比較用の鍵に替えた対照の eval の選択性の 95% 点")}">${t("比較用の鍵の p95")}</th><th title="${t("eval の選択性が、比較用の鍵の対照の何割より大きいか(0.95 未満は鍵が状態を選んでいるとは言えない)")}">null</th>` : ""}` : "";
    const notes = [...(d.notes || [])];
    const ok = d.rows.filter((r) => r.shippable && r.dose > 0).map((r) => r.dose);
    const rows = d.rows.map((r) => `<tr class="pick${r.dose === this.dose ? " on" : ""}" data-dose="${r.dose}">
        <td>${fmt(r.dose, 2)}</td><td>${pct(r.capability)}</td><td>${fmt(r.capability_drop_pt, 1)}</td>
        <td>${fmt(r.ppl_mean, 2)}</td><td>${r.ppl_worse_pct >= 0 ? "+" : ""}${fmt(r.ppl_worse_pct, 1)}%</td>
        <td>${pct(r.mix_zh + r.mix_other, 2)}</td><td>${fmt(r.persona.novelty, 2)}</td><td>${fmt(r.persona.distinct4, 2)}</td>
        <td>${pct(r.persona.capped, 0)}</td>${d.judge ? `<td>${fmt(r.judge, 2)}</td>` : ""}${d.random_controls ? (r.ctrl
          ? `<td class="muted">${fmt(r.ctrl_novelty_p95, 2)}</td><td class="${r.novelty_null_pct >= 0.95 ? "ok" : "ng"}">${fmt(r.novelty_null_pct, 2)}</td><td class="muted">${fmt(r.ctrl_capability_drop_p50, 1)}</td><td class="muted">${fmt(r.ctrl_ppl_worse_p50, 1)}%</td>`
          : "<td></td><td></td><td></td><td></td>") : ""}${rxCells(r)}${dirCells(r)}${kCells(r)}
        <td class="${r.shippable ? "ok" : "ng"}">${r.shippable ? t("配布可") : Object.entries(r.ok).filter(([, v]) => !v).map(([k]) => ({ capability: t("能力"), ppl: "ppl", mix: t("混入") }[k])).join("・") + " ×"}</td></tr>`).join("");
    body.innerHTML = `
      <p class="note mono">${esc(units.join(" · "))} <span class="muted">· when=${esc(d.when)} · ${t("{n} タスク", { n: d.suite.n_tasks })} · ${fmt(d.seconds, 0)} s${d.random_controls ? ` · ${t("対照 {n} 本", { n: d.random_controls })}` : ""}${
        rx ? ` · ${t("反応性 {name}({n} 対)", { name: esc(rx.name), n: rx.n_pairs })}${hc ? ` · ${t("ヘッドの対照 {n} 本", { n: rx.head_controls })}` : ""}` : ""}${
        dm ? ` · ${t("方向 {d}(射影の値を測る層 {L})", { d: esc(dm.direction), L: (dm.layers || []).map((L) => `L${L}`).join(", ") })}${dm.behavior ? ` · ${t("判定の一致率(無介入 {p})", { p: pct(dm.base_rate, 0) })}` : ""}` : ""}</span></p>
      ${notes.length ? `<ul class="evx-notes">${notes.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      <p class="note">${ok.length ? t("配布可の最大倍率: {d}", { d: fmt(Math.max(...ok), 2) }) : t("配布可の倍率はありません(0 を除く)")}
        <span class="muted">${t("基準: 能力 −{c}pt 以内・ppl +{p}% 以内・混入が増えない", { c: th.capability_drop_pt, p: th.ppl_worse_pct })}</span></p>
      <div class="evx-legend note">${series(d).map((s) => `<span><i style="background:${s.color}${s.thin ? ";opacity:.45;height:1px" : ""}"></i>${esc(s.label)}</span>`).join("")}
        <span><i class="dot" style="background:${C.ok}"></i>${t("配布可")}</span></div>
      <canvas id="evx-canvas"></canvas>
      <div class="evx-scroll"><table class="evx mono"><thead><tr><th>${t("倍率")}</th><th>${t("能力")}</th><th>${t("低下pt")}</th><th>ppl</th><th>${t("悪化")}</th><th>${t("混入")}</th>
        <th title="${t("無介入の出力に無い語の割合(プロンプトごとの平均)")}">${t("新語")}</th><th title="distinct-4">d4</th><th title="${t("max_tokens で打ち切られた割合")}">${t("打切")}</th>${d.judge ? "<th>judge</th>" : ""}${d.random_controls ? `<th title="${t("同じ大きさのランダムな向きの対照レシピの新語の 95% 点")}">${t("対照p95")}</th><th title="${t("新語が対照の何割より多いか(0.95 未満は対照と区別できない)")}">null</th><th title="${t("対照の能力の低下(中央値、pt)")}">${t("対照低下")}</th><th title="${t("対照の ppl の悪化(中央値)")}">${t("対照悪化")}</th>` : ""}${rx ? `<th title="${t("トリガーありのプロンプトで性格の判定(validate.match)に当たった割合")}">${t("あり")}</th><th title="${t("トリガーなしのプロンプトで判定に当たった割合")}">${t("なし")}</th><th title="${t("反応性 = あり − なし({n} 対)", { n: rx.n_pairs })}">${t("反応性")}</th>${
          hc ? `<th title="${t("レシピのヘッドを同じ k のランダムなヘッドに替えた対照の反応性の 95% 点")}">${t("対照p95")}</th><th title="${t("反応性が対照の何割より大きいか(0.95 未満は対照と区別できない)")}">null</th>` : ""}` : ""}${dirHead}${kHead}<th></th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      ${d.keyed ? this.keyedBlock(d.keyed) : ""}
      <div id="evx-sample"></div>`;
    for (const tr of body.querySelectorAll("tr[data-dose]")) tr.addEventListener("click", () => { this.dose = +tr.dataset.dose; this.render(); });
    for (const b of body.querySelectorAll("[data-krc]")) b.addEventListener("click", () => this.onLoadRecipe(d.keyed.methods[+b.dataset.krc].recipe));
    this.cv = body.querySelector("#evx-canvas");
    this.draw();
    this.sample();
  }

  // --- eval の選択性(CHANGE-07 E、結果の keyed): 方式の比較・比・比較用の鍵・注 -----------------------------------------
  keyedBlock(k) {
    const head = `<h4 class="note evx-kx-h">${t("eval の選択性(方式の比較と比較用の鍵)")}</h4>`;
    if (k.skipped) {
      return `<section class="evx-kx">${head}<p class="note pp-warn">${t("測っていません: {r}", { r: esc(k.skipped) })}</p>${
        k.hint ? `<p class="muted note">${esc(k.hint)}</p>` : ""}</section>`;
    }
    const label = (m) => ({ keyed: t("keyed(このレシピ)"), add_same_alpha: t("add(同じ α)"), add: t("add(書き込みたい位置をそろえた)"), gain: "gain", proj: "proj" }[m] || m);
    const sg = (x) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${x >= 0 ? "+" : "−"}${fmt(Math.abs(x), 3)}`);
    const c = k.counts || {}, ca = c.A || {}, cb = c.B || {};
    const dts = Object.keys(c).filter((g) => g.startsWith("dt:"));
    const ms = k.methods || [];
    const rows = ms.map((m, i) => (m.excluded
      ? `<tr class="rnd"><td class="l">${esc(label(m.method))}</td><td class="desc">${esc(m.desc || "")}</td><td class="excl" colspan="9">${t("除外")}: ${esc(m.excluded)}</td></tr>`
      : `<tr class="${m.method === "keyed" ? "best" : ""}"><td class="l">${esc(label(m.method))}</td><td class="desc">${esc(m.desc || "")}</td>
          <td>${fmt(m.beta, 3)}</td><td>${m.matched === true ? `<span class="ok">○</span>` : m.matched === false ? `<span class="ng" title="${t("書き込みたい位置の変化を ±10% でそろえられなかった(b_over_a で比べる)")}">×</span>` : "—"}</td>
          <td>${sg(m.a_par)}</td><td>${sg(m.b_par)}</td><td>${fmt(m.b_norm, 3)}</td><td>${fmt(m.dt_norm, 3)}</td><td>${sg(m.selectivity)}</td><td>${fmt(m.b_over_a, 3)}</td>
          <td>${m.recipe && (m.method === "keyed" || m.method === "proj") ? `<button class="small" data-krc="${i}" title="${t("この方式のレシピでレシピを置き換える")}">${t("レシピを置き換える")}</button>` : ""}</td></tr>`)).join("");
    const cmp = k.compare || {};
    const ratio = (o) => Object.entries(o || {}).map(([m, v]) => `${esc(label(m))} ${fmt(v, 2)}`).join(" · ");
    const p = k.pseudo;
    const verdict = (b) => (b === true ? `<span class="ok">○</span> ${t("超える")}` : b === false ? `<span class="ng">×</span> ${t("超えない")}` : t("判定なし"));
    return `<section class="evx-kx">${head}
      <p class="note mono">${t("鍵 {k} · 方向 {d} · L{L} の残差", { k: esc(k.key), d: esc(k.direction), L: k.objective_layer })} · ${t("対照集合 {c}", { c: esc(k.contrast || "—") })} (subset ${esc(k.subset || "—")}, ${
        k.held_out ? t("保留") : `<span class="ng">${t("作成と重なる")}</span>`}) · ${t("書き込みたい位置")} <span class="muted">${esc(k.positions || "")}</span> ${t("{e} 例 {p} 位置", { e: ca.examples ?? 0, p: ca.positions ?? 0 })} / ${
        t("書き込みたくない位置")} <span class="muted">${esc(k.positions_b || "")}</span> ${t("{e} 例 {p} 位置", { e: cb.examples ?? 0, p: cb.positions ?? 0 })}${
        dts.map((g) => ` / ${esc(g.slice(3))} ${t("{e} 例 {p} 位置", { e: c[g].examples, p: c[g].positions })}`).join("")}${
        k.cost ? `<span class="muted"> · prefill ${k.cost.prefills} · ${fmt(k.cost.seconds, 0)} s</span>` : ""}</p>
      ${noteHTML(t("同じ方向を方式ごとに当て、目的の層の残差の変化を位置の群ごとに比べる。書き込みたい位置の r̂ = そこでの r̂ 方向の変化、書き込みたくない位置の ‖Δ‖・dont_touch の ‖Δ‖ = そこでの変化の大きさ、eval の選択性 = 書く向きにそろえた、書き込みたい位置 − 書き込みたくない位置の r̂ 方向の変化、b_over_a = 書き込みたい位置を 1 押すあたりの書き込みたくない位置の乱れ(小さいほど選択的)。β = 書き込みたい位置の変化を keyed にそろえた強さ(○ = ±10% でそろった)。"), "muted note")}
      <div class="evx-scroll"><table class="evx mono evx-kx-tab"><thead><tr><th class="l">${t("方式")}</th><th class="l">${t("説明")}</th><th>β</th><th title="${t("書き込みたい位置の変化を keyed に ±10% でそろえられたか")}">${t("そろった")}</th>
        <th title="${t("書き込みたい位置での r̂ 方向の変化(par)")}">${t("書き込みたい位置の r̂")}</th><th title="${t("書き込みたくない位置での r̂ 方向の変化(par)")}">${t("書き込みたくない位置の r̂")}</th><th title="${t("書き込みたくない位置の残差の変化の大きさ")}">${t("書き込みたくない位置の ‖Δ‖")}</th>
        <th title="${t("dont_touch の群の残差の変化の大きさ(群の平均)")}">${t("dont_touch の ‖Δ‖")}</th><th title="${t("書く向きにそろえた、書き込みたい位置 − 書き込みたくない位置の r̂ 方向の変化。key-check の選択性(鍵の反応の強さの差)とは別の量")}">${t("eval の選択性")}</th><th title="${t("書き込みたくない位置の ‖Δ‖ ÷ |書き込みたい位置の r̂ 方向の変化|")}">b_over_a</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      ${ratio(cmp.b_norm_ratio) ? `<p class="note">${t("書き込みたくない位置の ‖Δ‖ の比(keyed ÷ 対照、同じ α の add と、書き込みたい位置をそろえた対照)")}: <span class="mono">${ratio(cmp.b_norm_ratio)}</span></p>` : ""}
      ${ratio(cmp.dt_norm_ratio) ? `<p class="note">${t("dont_touch の ‖Δ‖ の比(keyed ÷ 対照)")}: <span class="mono">${ratio(cmp.dt_norm_ratio)}</span></p>` : ""}
      ${ratio(cmp.b_over_a_ratio) ? `<p class="note">${t("書き込みたい位置を 1 押すあたりの、書き込みたくない位置の ‖Δ‖ の比(keyed ÷ 対照)")}: <span class="mono">${ratio(cmp.b_over_a_ratio)}</span></p>` : ""}
      ${cmp.keyed_smaller === true ? `<p class="note"><span class="ok">○</span> ${t("keyed の、書き込みたくない位置の乱れは、比べたどの対照より小さい")}</p>`
        : cmp.keyed_smaller === false ? `<p class="note"><span class="ng">×</span> ${t("keyed の、書き込みたくない位置の乱れが対照より小さくない方式があります(keyed の利点が見えません)")}</p>` : ""}
      ${p ? `<p class="note">${t("比較用の鍵 {n} 本の eval の選択性: p50 {a} · p95 {b} → 本物 {r}", { n: p.n, a: sg(p.p50), b: sg(p.p95), r: sg(p.real) })} ${verdict(p.beats)}${
        p.null_pct !== undefined && p.null_pct !== null ? ` <span class="muted">(null ${fmt(p.null_pct, 2)})</span>` : ""}</p>` : ""}
      ${(k.notes || []).length ? `<ul class="evx-notes">${k.notes.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      ${(k.warnings || []).length ? `<ul class="evx-notes">${k.warnings.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}</section>`;
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
      ${(r.notes || []).length ? `<ul class="evx-notes">${r.notes.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      ${r.direction && r.direction.layers ? `<details><summary class="muted note">${t("方向の指標(射影の値を測る層ごと)")}</summary><table class="evx mono"><thead><tr><th>${t("層")}</th><th>A</th><th>B</th>
        <th>${t("A 下がり")}</th><th>B Δ</th><th>A ∥</th><th>A ⊥</th><th>B ∥</th><th>B ⊥</th></tr></thead><tbody>${Object.entries(r.direction.layers).map(([L, x]) =>
          `<tr><td>L${esc(L)}</td><td>${fmt(x.score_a, 2)}</td><td>${fmt(x.score_b, 2)}</td><td>${fmt(x.drop_a, 2)}</td><td>${fmt(x.delta_b, 2)}</td><td>${fmt(x.drift_par_a, 3)}</td><td>${fmt(x.drift_orth_a, 3)}</td><td>${fmt(x.drift_par_b, 3)}</td><td>${fmt(x.drift_orth_b, 3)}</td></tr>`).join("")}</tbody></table></details>` : ""}
      ${r.reactivity && r.reactivity.samples && r.reactivity.samples.length ? `<details><summary class="muted note">${t("反応性の例(トリガーあり / なし)")}</summary>${
        r.reactivity.samples.map((x) => `<p class="note"><span class="muted mono">${t("あり")}</span> ${esc(x.with)}<br><span class="muted mono">${t("なし")}</span> ${esc(x.without)}</p>`).join("")}</details>` : ""}`;
  }

  // --- 方式の比較(CHANGE-06 F.3): 効果(A 側の射影の値)をそろえたときの副作用。外した方式は理由を出す -----------------------
  renderMethods(body, d) {
    const label = (m) => ({ ablate: t("全層の射影消去"), proj: t("ヘッドの方向成分(proj)"), head_gain: t("ヘッドの丸ごと倍率"), neurons: t("ニューロンの消去") }[m] || m);
    const live = d.rows.filter((r) => !r.excluded && Number.isFinite(r.drift_orth_b));
    const best = live.length ? live.reduce((a, b) => (b.drift_orth_b < a.drift_orth_b ? b : a)) : null;
    const hasRate = d.rows.some((r) => r.rate !== undefined && r.rate !== null);
    const nCol = 9 + (hasRate ? 2 : 0);
    const rows = d.rows.map((r, i) => (r.excluded
      ? `<tr><td class="l">${esc(label(r.method))}</td><td class="desc">${esc(r.desc || "")}</td><td>${fmt(r.alpha, 3)}</td><td>${fmt(r.score_a, 2)}</td>
          <td class="excl" colspan="${nCol - 4}">${t("除外")}: ${esc(r.excluded)}</td></tr>`
      : `<tr class="${r === best ? "best" : ""}"><td class="l">${esc(label(r.method))}</td><td class="desc">${esc(r.desc || "")}</td><td>${fmt(r.alpha, 3)}</td><td>${fmt(r.score_a, 2)}</td>
          <td>${fmt(r.drift_orth_b, 3)}</td><td>${fmt(r.capability_drop_pt, 1)}</td><td>${r.ppl_worse_pct >= 0 ? "+" : ""}${fmt(r.ppl_worse_pct, 1)}%</td><td>${pct(r.mix_increase, 2)}</td>${
          hasRate ? `<td>${pct(r.rate, 0)}</td><td>${pct(r.rate_drop, 0)}</td>` : ""}<td class="${r.shippable ? "ok" : "ng"}">${r.shippable ? t("配布可") : "×"}${
          r.recipe ? ` <button class="small" data-rc="${i}" title="${t("この方式のレシピでレシピを置き換える")}">${t("レシピを置き換える")}</button>` : ""}</td></tr>`)).join("");
    body.innerHTML = `
      <p class="note mono">${t("方式の比較")} · ${t("方向 {d}", { d: esc(d.direction || "—") })} · ${t("射影の値を測る層 L{L}", { L: d.score_layer ?? "—" })} · ${
        t("目標の A 側の射影の値 {v}({src})", { v: fmt(d.target, 2), src: d.target_from === "match_effect" ? "--match-effect" : t("このレシピの水準") })} · ${
        t("無介入 A {a} / B {b}", { a: fmt(d.base_score_a, 2), b: fmt(d.base_score_b, 2) })}${d.base_rate !== undefined && d.base_rate !== null ? ` · ${t("判定の一致率(無介入 {p})", { p: pct(d.base_rate, 0) })}` : ""}<span class="muted"> · ${fmt(d.seconds, 0)} s</span></p>
      ${noteHTML(t("同じ方向を、方式ごとに強さ α を合わせて A 側の射影の値を目標までそろえ、副作用を比べる。B ⊥ = 関係のない入力(B 側)の残差の、方向に直交する成分のずれ(小さいほど他を壊さない)。緑の行 = B ⊥ が最も小さい方式。"), "muted note")}
      <div class="evx-scroll"><table class="evx mono evx-methods"><thead><tr><th class="l">${t("方式")}</th><th class="l">${t("説明")}</th><th title="${t("目標にそろえた強さ")}">α</th><th title="${t("A 側の射影の値(測る層)")}">A</th>
        <th title="${t("B 側の残差の直交成分のずれ")}">B ⊥</th><th>${t("低下pt")}</th><th>${t("悪化")}</th><th title="${t("混入の増加")}">${t("混入")}</th>${
        hasRate ? `<th>${t("一致率")}</th><th>${t("一致減")}</th>` : ""}<th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
    for (const b of body.querySelectorAll("[data-rc]")) b.addEventListener("click", () => this.onLoadRecipe(d.rows[+b.dataset.rc].recipe));
  }

  // --- ヘッド数の曲線(CHANGE-06 F.4): gne heads allocate の組をヘッド数ごとに eval。推奨の組に印 ------------------------
  renderNheads(body, d) {
    const rows = [...d.rows].sort((a, b) => a.n_heads - b.n_heads);
    const rec = rows.find((r) => r.recommended);
    const tr = rows.map((r, i) => `<tr class="${r.recommended ? "rec" : ""}"><td>${r.n_heads}${r.recommended ? ` <span class="badge">${t("推奨")}</span>` : ""}</td>
        <td>${fmt(r.s_max, 2)}</td><td>${r.remain ?? "—"}</td><td>${fmt(r.capability_drop_pt, 1)}</td><td>${r.ppl_worse_pct >= 0 ? "+" : ""}${fmt(r.ppl_worse_pct, 1)}%</td>
        <td>${pct(r.mix_increase, 2)}</td><td>${fmt(r.dir_drop_a, 2)}</td><td>${fmt(r.drift_orth_b, 3)}</td><td>${pct(r.rate, 0)}</td>
        <td class="${r.shippable ? "ok" : r.shippable === false ? "ng" : ""}">${r.shippable ? t("配布可") : r.shippable === false ? "×" : ""}</td>
        <td class="units">${esc((r.units || []).join(" "))}</td><td>${(r.units || []).length ? `<button class="small" data-nh="${i}" title="${t("この組(proj のユニット)で今のレシピを置き換える")}">${t("レシピを置き換える")}</button>` : ""}</td></tr>`).join("");
    const notes = rows.flatMap((r) => (r.notes || []).map((x) => `${t("{n} ヘッド", { n: r.n_heads })}: ${x}`));
    body.innerHTML = `
      <p class="note mono">${t("ヘッド数の曲線")} · ${t("方向 {d}", { d: esc(d.direction || "—") })} · ${t("配分 {a}", { a: esc(String(d.alloc || "").split("/").pop()) })}${
        d.score_layer !== undefined ? ` · ${t("L{L} の射影の値を {v} まで", { L: d.score_layer, v: fmt(d.target, 2) })}` : ""}${rec ? ` · ${t("推奨 {n} ヘッド", { n: rec.n_heads })}` : ""}${
        d.random_controls ? ` · ${t("対照 {n} 本", { n: d.random_controls })}` : ""}<span class="muted"> · ${fmt(d.seconds, 0)} s</span></p>
      ${notes.length ? `<ul class="evx-notes">${notes.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      ${noteHTML(t("x = ヘッドの数(gne heads allocate の組)。少ないヘッドほど 1 ヘッドの s が大きくなる。残り = 目標に届かず残った効きにくい例の数。白の破線 = 推奨の組。"), "muted note")}
      <canvas id="evx-nh"></canvas>
      <div class="evx-scroll"><table class="evx mono"><thead><tr><th>${t("ヘッド数")}</th><th>s max</th><th title="${t("目標に届かず残った例の数")}">${t("残り")}</th><th>${t("低下pt")}</th><th>${t("悪化")}</th><th>${t("混入")}</th>
        <th title="${t("A 側の射影の値の下がり方")}">${t("A 下がり")}</th><th title="${t("B 側の残差の直交成分のずれ")}">B ⊥</th><th>${t("一致率")}</th><th></th><th class="l">${t("ユニット")}</th><th></th></tr></thead><tbody>${tr}</tbody></table></div>`;
    for (const b of body.querySelectorAll("[data-nh]")) {
      b.addEventListener("click", () => { const r = rows[+b.dataset.nh]; this.onLoadProj(`${d.name || "nheads"}-n${r.n_heads}`, r.units); });
    }
    this.nh = { cv: body.querySelector("#evx-nh"), rows, rec };
    this.drawNheads();
  }

  // 小さな折れ線 3 つ(能力の低下・B ⊥・残り)。x はヘッド数で共通、y はそれぞれの単位
  drawNheads() {
    const x = this.nh;
    if (!x || !x.cv || !x.cv.isConnected) return;
    const panels = [[t("能力の低下(pt)"), (r) => r.capability_drop_pt, C.minus, 1], [t("B ⊥(B 側の直交成分のずれ)"), (r) => r.drift_orth_b, C.judge, 3],
                    [t("残り(届かなかった例の数)"), (r) => r.remain, C.amber, 1]];
    const PH = 104, w = Math.max(320, this.root.clientWidth - 24), H2 = PH * panels.length + 26, dpr = window.devicePixelRatio || 1, cv = x.cv;
    cv.width = w * dpr; cv.height = H2 * dpr; cv.style.width = `${w}px`; cv.style.height = `${H2}px`;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, H2);
    const ns = x.rows.map((r) => r.n_heads), n0 = Math.min(...ns), n1 = Math.max(...ns);
    const LP = 64, X = (n) => LP + (n1 === n0 ? (w - LP - R0) / 2 : ((n - n0) / (n1 - n0)) * (w - LP - R0 - 10));
    g.font = "11px ui-monospace, monospace";
    panels.forEach(([label, f, col, dg], pi) => {
      const top = pi * PH + 24, bot = top + PH - 30;                      // 見出しは枠の上、目盛りは左
      const vs = x.rows.map(f).filter((v) => v !== null && v !== undefined && Number.isFinite(+v));
      const mx = Math.max(0, ...vs), lo = Math.min(0, ...vs);
      const hi = mx <= 0 ? 1 : vs.every((v) => Number.isInteger(+v)) ? Math.ceil(mx * 1.15) : mx * 1.15;   // 数(残り)は整数の目盛り
      const Y = (v) => top + (1 - (v - lo) / (hi - lo || 1)) * (bot - top);
      g.fillStyle = col; g.textAlign = "left"; g.fillText(label, LP, top - 8);
      g.strokeStyle = C.grid; g.lineWidth = 1;
      for (const v of [lo, (lo + hi) / 2, hi]) { g.beginPath(); g.moveTo(LP, Y(v)); g.lineTo(w - R0, Y(v)); g.stroke(); g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(fmt(v, dg), LP - 6, Y(v) + 4); }
      if (x.rec) { g.strokeStyle = "#f2f5f8"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(x.rec.n_heads), top); g.lineTo(X(x.rec.n_heads), bot); g.stroke(); g.setLineDash([]); }
      const pts = x.rows.map((r) => [r.n_heads, f(r)]).filter(([, v]) => v !== null && v !== undefined && Number.isFinite(+v));
      g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2;
      g.beginPath();
      pts.forEach(([n, v], i) => (i ? g.lineTo(X(n), Y(v)) : g.moveTo(X(n), Y(v))));
      g.stroke();
      for (const [n, v] of pts) { g.beginPath(); g.arc(X(n), Y(v), x.rec && n === x.rec.n_heads ? 4.5 : 3, 0, 2 * Math.PI); g.fill(); }
      g.lineWidth = 1;
    });
    g.textAlign = "center";
    for (const n of ns) { g.fillStyle = x.rec && n === x.rec.n_heads ? C.text : C.muted; g.fillText(`${n}`, X(n), H2 - 8); }
    g.textAlign = "left"; g.fillStyle = C.muted; g.fillText(t("ヘッド数"), 8, H2 - 8);
  }

  draw() {
    if (this.nh) { this.drawNheads(); return; }
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
