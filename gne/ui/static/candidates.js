// direction ビューの「強さの候補」(heads_candidates のジョブ、Session.heads_candidates)。
// 配分の組 × 全体の倍率で作った返事を、候補の名前を伏せて採点し(P = 目標の振る舞い、C = 自然さ、1〜5)、候補ごとに集計する。
// 要約: 候補 × 採点する側の表(一致率・繰り返し・反対側の変化・P・C・目的、向きごとにいちばん良い候補に印)、
// 返事を並べる: 質問ごとに候補の返事を並べる(名前を伏せると順番も混ぜる)、
// 名前を伏せて採点する: 1 本ずつ出し、P と C を付けるたびに保存する(途中でやめても続きから)。全部付けると集計を出す。
// いちばん良い候補から、その倍率のレシピを recipe タブに読み込む・1 列チャットで試す。
import { getJSON, postJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc, qHTML } from "./panels.js";
import { parseProj } from "./recipe.js";
import { composing } from "./ui.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const pct = (x) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${Math.round(x * 100)}%`);
const stripThink = (s) => String(s ?? "").replace(/<\/?think>/g, "").trim();
const goalLabel = (g) => (g === "remove" ? t("消す") : g === "add" ? t("強める") : "—");

// 決まった順に混ぜる(同じ種なら同じ順。名前を伏せて並べるとき、質問ごとに違う順にする)
function shuffled(arr, seed) {
  let a = (seed * 2654435761) >>> 0;
  const rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let x = Math.imul(a ^ (a >>> 15), 1 | a); x ^= x + Math.imul(x ^ (x >>> 7), 61 | x); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
  return out;
}

// 候補のその倍率のユニット("L:Hh:sS@NAME" の s に倍率を掛けたもの)
export function candidateSpecs(c, direction) {
  return (c.units || []).map((u) => parseProj(u, direction)).filter(Boolean)
    .map((u) => `${u.layer}:H${u.head}:s${+(u.s * c.scale).toFixed(4)}@${u.direction}`);
}

export class CandidatesPanel {
  constructor(root, { onLoadCandidate, onChatCandidate } = {}) {
    this.root = root;
    this.onLoadCandidate = onLoadCandidate || (async () => null);   // ({ name, specs, direction, withRecipe }) → recipe タブに読み込む(注意の文か null)
    this.onChatCandidate = onChatCandidate || (async () => null);   // 同じ引数 → 読み込んで 1 列チャットで開く
    this.list = [];
    this.data = null;                  // /api/candidates/<name>
    this.openSeq = 0;                  // open の通し番号(後から別の結果を開いたら、先に始めた読み込みの結果は捨てる)
    this.mode = "summary";             // summary / grid / score
    this.side = "A";                   // 返事を並べる側
    this.named = false;                // 返事を並べるとき、候補の名前を出すか
    this.blind = null;                 // /api/candidates/<name>/blind(採点の項目と、いま付いている点)
    this.pos = 0;                      // 採点している項目の位置
    this.cur = { P: null, C: null };   // いま付けている点(両方そろうと保存)
    this.saving = Promise.resolve();   // 保存は 1 つずつ(速く付けても順に書く)
    this.done = false;                 // 全部付けた(集計を出している)
    root.innerHTML = `
      <p class="muted note pp-help">${t("配分の組を全体の倍率で何段か当てた返事を、どの候補か伏せて採点し(P = 目標の振る舞いがどれだけ出ているか、C = 自然さ、1〜5)、候補ごとに集計します。候補はパイプラインの結果の「候補を作る」で作ります。")}</p>
      <h4>${t("保存した候補")}</h4><div id="cd-list"><p class="muted note">${t("読み込み中…")}</p></div>
      <div id="cd-result"></div>`;
    this.$ = (s) => root.querySelector(s);
    this._onKey = (ev) => this.key(ev);
    document.addEventListener("keydown", this._onKey);
  }

  async refreshList() {
    try { this.list = await getJSON("/api/candidates"); } catch { this.list = []; }
    const el = this.$("#cd-list");
    if (!this.list.length) { el.innerHTML = `<p class="muted note">${t("まだありません")}</p>`; return; }
    el.innerHTML = `<table class="evx mono pp-past"><thead><tr><th class="l">${t("名前")}</th><th class="l">${t("方向")}</th><th>${t("候補")}</th><th>${t("採点する返事")}</th><th>${t("集計")}</th><th class="l">${t("更新")}</th></tr></thead><tbody>${
      this.list.map((x) => `<tr data-cd="${esc(x.name)}" class="pick${this.data && this.data.name === x.name ? " on" : ""}"><td class="l">${esc(x.name)}</td><td class="l">${esc(x.direction || "—")}</td>
        <td>${x.n_candidates ?? "—"}</td><td>${x.n_judge ?? "—"}</td><td>${x.scored ? "✓" : ""}</td><td class="l">${esc((x.updated || "").replace("T", " ").slice(0, 16))}</td></tr>`).join("")}</tbody></table>`;
    for (const tr of el.querySelectorAll("tr[data-cd]")) tr.addEventListener("click", () => this.open(tr.dataset.cd));
  }

  // app.js の onJobFinished から(direction.js 経由)。mine = このパネルの「候補を作る」で投げたもの
  async jobEnded(j, mine) {
    if (j.status !== "done") return;
    await this.refreshList();
    if (mine && j.result && j.result.name) await this.open(j.result.name);
  }

  async open(name, mode = "summary") {
    const tok = ++this.openSeq;
    const out = this.$("#cd-result");
    out.innerHTML = `<p class="muted">${t("読み込み中…")}</p>`;
    let d;
    try { d = await getJSON(`/api/candidates/${encodeURIComponent(name)}`); }
    catch (e) { if (tok === this.openSeq) out.innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    if (tok !== this.openSeq) return;              // 待つ間に別の結果を開いた
    this.data = d;
    this.blind = null;
    this.mode = mode;
    const sides = new Set(this.data.candidates.flatMap((c) => [...(c.judge_sides || "")]));
    this.side = sides.has("A") ? "A" : "B";
    for (const tr of this.$("#cd-list").querySelectorAll("tr[data-cd]")) tr.classList.toggle("on", tr.dataset.cd === name);
    if (mode === "score") await this.startScoring(); else this.render();
  }

  render() {
    const d = this.data, out = this.$("#cd-result");
    if (!d) { out.innerHTML = ""; return; }
    const ho = d.held_out || {}, sp = d.sampling;
    const warn = !(ho.A && ho.B) ? `<p class="pp-warn note">${t("注意: 調整に使った質問で生成しています({sides} 側)。強さを決めるには、調整に使っていない質問で作り直してください。", { sides: ["A", "B"].filter((k) => !ho[k]).join("・") })}</p>` : "";
    const seg = `<span class="seg" id="cd-mode">${[["summary", t("要約")], ["grid", t("返事を並べる")], ["score", t("名前を伏せて採点する")]].map(([k, l]) =>
      `<button data-cm="${k}" class="${this.mode === k ? "on" : ""}">${l}</button>`).join("")}</span>`;
    out.innerHTML = `<div class="pp-res">
      <h3>${t("{name} の候補", { name: esc(d.name) })} <span class="muted note mono">${esc((d.updated || "").replace("T", " ").slice(0, 16))}${d.seconds ? ` · ${t("{n} 秒", { n: Math.round(d.seconds) })}` : ""}</span></h3>
      <p class="note">${t("方向 {d}", { d: esc(d.direction) })} · ${t("候補 {n} 個(無介入を含む)", { n: d.candidates.length })} · ${t("質問 A {a} / B {b}", { a: d.prompts.A.length, b: d.prompts.B.length })} · ${
        sp ? t("サンプリング 温度 {t}・種 {s}", { t: fmt(sp.temperature, 2), s: sp.seed }) : "greedy"} · ${t("最大 {n} トークン", { n: d.max_tokens })}${
        d.with_recipe ? ` · ${t("重ねたレシピ {r}", { r: esc(String(d.with_recipe).split("/").pop()) })}` : ""}</p>
      <p class="muted note">${t("目標の振る舞い: {t}", { t: esc(d.target || "—") })}</p>
      ${warn}
      <div class="pp-actions">${seg} <span class="muted note" id="cd-prog">${t("採点 {k} / {n}", { k: d.n_scored ?? 0, n: (d.judge || {}).n ?? "—" })}</span></div>
      <div id="cd-body"></div></div>`;
    for (const b of out.querySelectorAll("[data-cm]")) {
      b.addEventListener("click", () => { if (b.dataset.cm === "score") this.startScoring(); else { this.mode = b.dataset.cm; this.render(); } });
    }
    const body = this.$("#cd-body");
    if (this.mode === "grid") this.grid(body);
    else if (this.mode === "score") this.drawScore(body);
    else this.summary(body);
  }

  // --- 要約 ---------------------------------------------------------------------------------------
  summary(el) {
    const d = this.data, sc = d.scores || {}, best = d.best || {}, has = Object.keys(sc).length > 0;
    const rows = [];
    for (const c of d.candidates) {
      for (const side of c.judge_sides || "") {
        const lo = side.toLowerCase(), other = side === "A" ? "b" : "a";
        const v = (sc[c.name] || {})[side];
        const g = c.goal || (side === "A" ? "remove" : "add");
        const isBest = Object.values(best).some((b) => b.name === c.name && b.side === side);
        rows.push(`<tr class="${isBest ? "best" : c.name === "base" ? "rnd" : ""}"><td class="l">${esc(c.name)}${c.recommended && c.scale === 1 ? ` <span class="badge" title="${t("配分(allocate)が推した組の、そのままの強さ")}">${t("推奨")}</span>` : ""}</td>
          <td>${c.name === "base" ? "—" : fmt(c.scale, 2)}</td><td>${side}</td><td>${goalLabel(c.goal)}</td><td>${pct(c[`rate_${lo}`])}</td><td>${fmt(c[`d4_${lo}`], 2)}</td>
          <td>${c.name === "base" ? "—" : pct(1 - c[`same_as_base_${other}`])}</td>
          ${has ? `<td>${v ? fmt(v.P, 2) : "—"}</td><td>${v ? fmt(v.C, 2) : "—"}</td><td><b>${v ? fmt(v[g], 2) : "—"}</b></td><td>${v ? v.n : ""}</td>` : ""}
          <td class="l">${c.name === "base" ? "" : `<button class="small" data-load="${esc(c.name)}">${t("レシピを置き換える")}</button>`}</td></tr>`);
      }
    }
    const bests = Object.entries(best).map(([g, b]) => `<div class="cd-best"><b>${t("{goal}向きでいちばん良い: {name}", { goal: goalLabel(g), name: esc(b.name) })}</b>
      <span class="mono">${t("目的 {v}(無介入 {b})", { v: fmt(b.value, 2), b: fmt(b.base, 2) })}</span>
      <button class="small" data-load="${esc(b.name)}">${t("レシピを置き換える")}</button> <button class="small" data-chat="${esc(b.name)}">${t("1 列チャットで試す")}</button></div>`).join("");
    const missing = (d.scores_missing || []).length;
    el.innerHTML = `${bests}
      ${!has ? `<p class="note">${t("まだ集計していません。「名前を伏せて採点する」で全部の返事に点を付けると、候補ごとに集計します。")}</p>` : missing ? `<p class="pp-warn note">${t("点の無い返事が {n} 本あります(集計はある分だけ)。", { n: missing })}</p>` : ""}
      <div class="evx-scroll"><table class="evx mono cd-sum"><thead><tr><th class="l">${t("候補")}</th><th title="${t("全体の倍率(正 = A 側で消す向き、負 = B 側で強める向き)")}">${t("倍率")}</th>
        <th title="${t("採点した側(A = 振る舞いが出ている状態、B = ふつうの状態)")}">${t("側")}</th><th>${t("向き")}</th>
        <th title="${t("判定(validate.match)に合う返事の割合")}">${t("一致")}</th><th title="${t("4-gram の重複の少なさ(1 = 繰り返し無し)")}">d4</th>
        <th title="${t("採点しない側で、無介入と違う返事の割合(同じ種なので、変わらなければ一致する)")}">${t("反対側の変化")}</th>
        ${has ? `<th title="${t("目標の振る舞いの平均(1〜5)")}">P</th><th title="${t("自然さの平均(1〜5)")}">C</th><th title="${t("消す向きは min(6 − P, C)、強める向きは min(P, C) の平均")}">${t("目的")}</th><th>n</th>` : ""}<th></th></tr></thead><tbody>${rows.join("")}</tbody></table></div>
      <p class="muted note">${t("一致 = 判定(validate.match)に合う返事の割合、d4 = 繰り返しの少なさ、反対側の変化 = 採点しない側で無介入と違う返事の割合。目的 = 消す向きは min(6 − P, C)、強める向きは min(P, C) の平均。緑の行 = 向きごとにいちばん良い候補。")}</p>
      <p class="muted note" id="cd-note"></p>`;
    for (const b of el.querySelectorAll("[data-load]")) b.addEventListener("click", () => this.use(b.dataset.load, false));
    for (const b of el.querySelectorAll("[data-chat]")) b.addEventListener("click", () => this.use(b.dataset.chat, true));
  }

  // 候補のその倍率のレシピを recipe タブに(chat なら 1 列チャットも開く)
  async use(name, chat) {
    const d = this.data, c = d.candidates.find((x) => x.name === name);
    if (!c) return;
    const arg = { name: `${d.name}-${c.name}`, specs: candidateSpecs(c, d.direction), direction: d.direction, withRecipe: d.with_recipe };
    const note = await (chat ? this.onChatCandidate(arg) : this.onLoadCandidate(arg));
    const el = this.$("#cd-note");
    if (el) el.textContent = note === false ? t("やめました") : note || t("{name} でレシピを置き換えました({n} ユニット)", { name: arg.name, n: arg.specs.length });
  }

  // --- 返事を並べる -------------------------------------------------------------------------------
  grid(el) {
    const d = this.data, side = this.side, qs = d.prompts[side] || [];
    const cands = d.candidates.filter((c) => (d.replies || {})[`${c.name}|${side}`]);
    const blocks = qs.map((q, i) => {
      const order = this.named ? cands : shuffled(cands, i * 2 + (side === "B" ? 1 : 0) + 7);
      const cells = order.map((c, k) => `<div class="cd-cell"><div class="cd-lab mono">${this.named ? `${esc(c.name)} <span class="muted">${goalLabel(c.goal)}</span>` : `#${k + 1}`}</div>
        <div class="cd-txt">${esc(stripThink(d.replies[`${c.name}|${side}`][i]))}</div></div>`).join("");
      return `<div class="cd-q"><div class="cd-prompt">${qHTML(q, `${side}${i + 1}`)}</div><div class="cd-grid">${cells}</div></div>`;
    }).join("");
    el.innerHTML = `<div class="pp-actions"><span class="seg" id="cd-side">${["A", "B"].map((s) => `<button data-sd="${s}" class="${side === s ? "on" : ""}" title="${
      s === "A" ? t("A 側(振る舞いが出ている状態)の質問と返事") : t("B 側(ふつうの状態)の質問と返事")}">${s}</button>`).join("")}</span>
      <label class="muted"><input type="checkbox" id="cd-named"${this.named ? " checked" : ""}> ${t("候補の名前を出す")}</label>
      <span class="muted note">${this.named ? t("候補の順は表と同じ") : t("名前を伏せ、質問ごとに順番を混ぜています")}</span></div>${blocks}`;
    for (const b of el.querySelectorAll("[data-sd]")) b.addEventListener("click", () => { this.side = b.dataset.sd; this.render(); });
    el.querySelector("#cd-named").addEventListener("change", (ev) => { this.named = ev.target.checked; this.render(); });
  }

  // --- 名前を伏せて採点する ---------------------------------------------------------------------------
  async startScoring() {
    const d = this.data;
    let blind;
    try { blind = await getJSON(`/api/candidates/${encodeURIComponent(d.name)}/blind`); }
    catch (e) { if (this.data === d) this.$("#cd-result").insertAdjacentHTML("beforeend", `<p class="err">${esc(e.message)}</p>`); return; }
    if (this.data !== d) return;                   // 待つ間に別の結果を開いた
    this.blind = blind;
    this.mode = "score";
    const first = this.blind.items.findIndex((it) => !this.blind.scores[String(it.id)]);
    this.pos = first >= 0 ? first : 0;
    this.done = first < 0;                         // 全部付いていれば集計を出す(「見直す」で 1 本目から)
    this.cur = { P: null, C: null };
    this.render();
  }

  nScored() { return this.blind ? this.blind.items.filter((it) => this.blind.scores[String(it.id)]).length : 0; }

  drawScore(el) {
    const b = this.blind;
    if (!b) { el.innerHTML = `<p class="muted">${t("読み込み中…")}</p>`; return; }
    const n = b.items.length, k = this.nScored();
    this.$("#cd-prog").textContent = t("採点 {k} / {n}", { k, n });
    if (k === n && this.done) {
      el.innerHTML = `<p class="note ok">${t("全部採点しました。集計:")} <button class="small" id="cd-review">${t("見直す")}</button></p><div id="cd-done"></div>`;
      el.querySelector("#cd-review").addEventListener("click", () => { this.done = false; this.pos = 0; this.render(); });
      this.summary(el.querySelector("#cd-done"));
      return;
    }
    const it = b.items[this.pos], have = b.scores[String(it.id)] || null;
    const P = this.cur.P ?? (have ? have[0] : null), C = this.cur.C ?? (have ? have[1] : null);
    const btns = (which, val) => [1, 2, 3, 4, 5].map((v) => `<button class="cd-sc${val === v ? " on" : ""}" data-${which}="${v}">${v}</button>`).join("");
    el.innerHTML = `<div class="cd-score">
      <div class="pp-actions"><button class="small" id="cd-prev" title="${t("前の返事(←)")}">← ${t("前へ")}</button>
        <button class="small" id="cd-skip" title="${t("点を付けずに次へ(→)")}">${t("とばす")} →</button>
        <span class="mono">#${it.id}</span> <span class="muted note">${t("{i} 本目 / {n} 本", { i: this.pos + 1, n })}${have ? ` · ${t("付けた点 P {p} C {c}", { p: have[0], c: have[1] })}` : ""}</span>
        <span class="cd-bar"><span style="width:${(100 * k / Math.max(n, 1)).toFixed(1)}%"></span></span> <span class="muted note" id="cd-saved"></span></div>
      <div class="cd-prompt">${qHTML(it.prompt)}</div>
      <div class="cd-reply">${esc(stripThink(it.text)) || `<span class="muted">${t("(空の返事)")}</span>`}</div>
      <div class="cd-pc"><span class="cd-pc-l" title="${t("目標の振る舞いがどれだけ出ているか(1 = まったく出ていない、5 = 典型的に全体に出ている)")}">P</span>${btns("p", P)}
        <span class="cd-pc-l" title="${t("自然さ(1 = 崩れている、5 = 完全に自然)")}">C</span>${btns("c", C)}</div>
      <p class="muted note">${t("キー: 数字 1〜5 で P、続けてもう一度押すと C(そろうと保存して次へ)。← 前へ、→ とばす、Esc 付けかけを消す。どの候補の返事かは伏せてあります。")}</p>
      <details class="cd-rub"><summary class="muted note">${t("採点基準")}</summary><pre class="note">${esc(b.rubric || "")}</pre></details></div>`;
    const $ = (s) => el.querySelector(s);
    $("#cd-prev").addEventListener("click", () => this.move(-1));
    $("#cd-skip").addEventListener("click", () => this.move(1));
    for (const x of el.querySelectorAll("[data-p]")) x.addEventListener("click", () => this.setPC("P", +x.dataset.p));
    for (const x of el.querySelectorAll("[data-c]")) x.addEventListener("click", () => this.setPC("C", +x.dataset.c));
  }

  move(dir) {
    const n = this.blind.items.length;
    this.pos = (this.pos + dir + n) % n;
    this.cur = { P: null, C: null };
    this.render();
  }

  // P と C の両方をこの返事で付けたら保存する(付け直しも両方)
  setPC(which, v) {
    this.cur[which] = v;
    if (this.cur.P !== null && this.cur.C !== null) this.save(this.blind.items[this.pos].id, this.cur.P, this.cur.C);
    else this.render();
  }

  // 1 本の点を保存して、次の点の無い返事へ(全部付いたら集計を出す)
  save(id, P, C) {
    const had = !!this.blind.scores[String(id)];   // 付け直し(見直し)なら、点の有無にかかわらず次の返事へ
    this.blind.scores[String(id)] = [P, C];
    this.cur = { P: null, C: null };
    const name = this.data.name;
    this.saving = this.saving.then(async () => {
      try {
        const r = await postJSON(`/api/candidates/${encodeURIComponent(name)}/scores`, { scores: { [String(id)]: [P, C] } });
        if (this.data && this.data.name === name) { this.data = r; const s = this.$("#cd-saved"); if (s) s.textContent = t("#{id} を保存しました", { id }); }
      } catch (e) {
        delete this.blind.scores[String(id)];
        const s = this.$("#cd-saved");
        if (s) { s.textContent = `${t("保存できませんでした")}: ${e.message}`; s.classList.add("err"); }
      }
    });
    const n = this.blind.items.length;
    const next = [...Array(n).keys()].map((k) => (this.pos + 1 + k) % n).find((k) => !this.blind.scores[String(this.blind.items[k].id)]);
    if (next === undefined && had && this.pos < n - 1) { this.pos += 1; this.render(); return; }
    if (next === undefined) {                       // 全部付いた: 保存を待ってから集計を出す
      this.done = true;
      this.saving.then(() => { if (this.mode === "score") this.render(); });
      this.render();
      return;
    }
    this.pos = next;
    this.render();
  }

  key(ev) {
    if (this.mode !== "score" || !this.blind || this.done || !this.root.offsetParent) return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || ev.ctrlKey || ev.metaKey || ev.altKey || composing(ev)) return;
    if (/^[1-5]$/.test(ev.key)) { ev.preventDefault(); this.setPC(this.cur.P === null ? "P" : "C", +ev.key); }
    else if (ev.key === "ArrowLeft") { ev.preventDefault(); this.move(-1); }
    else if (ev.key === "ArrowRight") { ev.preventDefault(); this.move(1); }
    else if (ev.key === "Escape") { this.cur = { P: null, C: null }; this.render(); }
  }
}
