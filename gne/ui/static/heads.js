// HEAD パネル(CHANGE-04 G): attention ヘッドを 1 つ開く、NEURON パネルと同列の右パネル。タブは Pattern / Contrast / Details。
// Pattern: 文章を流して、そのヘッドのトークン × トークンの注目(行 = 見ている位置、列 = 見られている位置)を Canvas 2D で描く。
//   チャットの形でペルソナ(system)あり/なしを比べると、共通の末尾(ユーザーの発話と返事の始まり)で差を出す
//   (system の部分は 1 列にまとめる。gne heads contrast の pattern_shift と同じまとめ方)。
//   トリガー(部分文字列、"re:" で正規表現)の列に印を付け、トリガー区間への注目を表にする。
// Contrast: 保存済みのヘッドの対照分析(gne heads contrast)ごとの指標・順位と基準値(中立どうしの差の p95)、上位のヘッド、対照集合の会話を Pattern で見るリンク。
// Details: W_O の列ブロックの logit lens(主成分 2 本の上位・下位の語)、当てたことのある gain、大きさ。
// Listen: system + ユーザーの発話 + 返事を流し、返事のトークンが選んだヘッド(複数可、例 14:H0-3)で文脈のどのトークンに
//   注目しているか(注目)と、各トークンの方向への寄与(注目 × ゲート × OV、gne heads listen)を文章の上に色で出す。
// Causal(CHANGE-06 G): 保存済みの因果スクリーニング(gne heads causal)での、このヘッドの行(止めたときの下がり方・直接の寄与・
//   ratio・効率 η・B 側への影響、射影の値を測る層ごと)と、選んだスクリーニングの上位のヘッドの表(並べ替え・表示する測る層)、
//   強さの配分(gne heads allocate)の組。見出しの「射影 s1」でこのヘッドの方向成分の射影(L:Hh:s1@方向)をレシピに足す。
import { getJSON, postJSON } from "./api.js";
import { t, tl } from "./i18n.js";
import { esc, toksHTML } from "./panels.js";
import { plainEnter, noteHTML } from "./ui.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const sgn = (x, d = 3) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
const pct = (x) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${(x * 100).toFixed(1)}%`);
const AMBER = [255, 196, 84], PLUS = [242, 210, 75], MINUS = [201, 79, 167], MASKED = [10, 12, 16];
const kindLabel = (k) => (k === "attn" ? "attention" : t("線形注意"));
// gne heads contrast の指標(heads.py の METRICS)。signed は符号つき(基準値と比べるのは絶対値)
const METRICS = [
  ["trigger_mass", () => t("トリガーへの注目"), () => t("返事の各位置から、共通部分(ユーザーの発話 + 返事)への注目のうちトリガー区間が占める割合の平均。ペルソナあり − なし"), true],
  ["pattern_shift", () => t("注目のずれ"), () => t("返事の各行の注目の分布の JS ダイバージェンス(bit)の平均。system の部分は 1 つにまとめる"), false],
  ["out_shift", () => t("出力の差"), () => t("ヘッドの出力の平均の差のノルム ÷ そのヘッドの典型的な出力ノルム"), false],
  ["dir_cos", () => t("方向との cos"), () => t("出力の差を W_O の列ブロックで残差に写したものと、その層の方向(gne direction)の cos"), true],
];

// 画面に描くトークン(空白は ␣、改行は ↵)。長いものは切る
const tokText = (s, n = 10) => { const x = String(s ?? "").replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥"); return x.length > n ? `${x.slice(0, n - 1)}…` : x || "∅"; };

// トリガーの文字区間(gne/attn.py の find_spans と同じ: 部分文字列、"re:" で正規表現、重なりは 1 つに)
function findSpans(text, trigs) {
  const spans = [];
  for (const tg of trigs) {
    if (tg.startsWith("re:")) {
      let re;
      try { re = new RegExp(tg.slice(3), "g"); } catch { continue; }
      for (let m = re.exec(text); m; m = re.exec(text)) {
        if (!m[0].length) { re.lastIndex++; continue; }
        spans.push([m.index, m.index + m[0].length]);
      }
    } else if (tg) {
      for (let i = text.indexOf(tg); i >= 0; i = text.indexOf(tg, i + 1)) spans.push([i, i + tg.length]);
    }
  }
  spans.sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const [s, e] of spans) { if (out.length && s <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], e); else out.push([s, e]); }
  return out;
}

// トークン列(文字列)の中のトリガー → [{ text, toks: [添字] }]
function triggerTokens(toks, trigs) {
  const offs = [];
  let at = 0;
  for (const x of toks) { offs.push([at, at + x.length]); at += x.length; }
  const text = toks.join("");
  return findSpans(text, trigs).map(([s, e]) => ({ text: text.slice(s, e), toks: offs.map(([a, b], i) => (b > a && a < e && b > s ? i : -1)).filter((i) => i >= 0) }))
    .filter((x) => x.toks.length);
}

// 共通の末尾(トークンの文字列で後ろから比べる)の長さ
function suffixLen(a, b) {
  let n = 0;
  while (n < Math.min(a.length, b.length) && a[a.length - 1 - n] === b[b.length - 1 - n]) n++;
  return n;
}

// パターン(T × T)を共通の末尾 S の行・列に切り出す。列 0 は末尾より前(system の部分)への注目の合計
function shared(view, S) {
  const P = view.pattern, p0 = view.tokens.length - S;
  return Array.from({ length: S }, (_, i) => {
    const row = P[p0 + i], out = new Float32Array(S + 1);
    let pre = 0;
    for (let c = 0; c < p0; c++) pre += row[c];
    out[0] = pre;
    for (let j = 0; j <= i; j++) out[j + 1] = row[p0 + j];
    return out;
  });
}

export class HeadPanel {
  constructor(root, { onRecipe, inRecipe, onNamed, onOverlay, onGoto, directions, onProj, inProj, onLoadProj, projDirections, causalList }) {
    this.root = root;
    this.onRecipe = onRecipe;          // (L, h, k) → レシピに足す・外す
    this.inRecipe = inRecipe;          // (L, h) → レシピの中の倍率 k か null
    this.onNamed = onNamed;            // (names) → 名前の一覧が変わった
    this.onOverlay = onOverlay;        // (field) → 地図の色を heads:<name> に
    this.onGoto = onGoto;              // ({ layer, head }) → 別のヘッドを開く
    this.directions = directions || (() => []);   // 保存済みの方向の名前(Listen の「寄与」)
    this.onProj = onProj || (() => {});            // (L, h, 方向) → 射影 s1 をレシピに足す・外す(CHANGE-06)
    this.inProj = inProj || (() => null);          // (L, h) → レシピの射影のユニット { s, direction } か null
    this.onLoadProj = onLoadProj || (() => {});    // (名前, ["L:Hh:sS@NAME"], 方向) → 強さの配分の組でレシピを置き換える
    this.projDirections = projDirections || (() => []);   // proj に使える方向(gne direction の方向セット)
    this.causalList = causalList || (() => []);    // 保存済みの因果スクリーニングの一覧(/api/model の causal)
    this.causalSums = new Map();                   // 名前 → /api/causal/<name>(Causal タブ)
    this._czPending = new Set();
    this.cz = { name: null, sort: "causal_max", layer: null };   // Causal タブの選択(スクリーニング・並べ替え・射影の値を測る層)
    this.lis = { heads: "", system: "", user: "", reply: "", dir: "", mode: "contrib", maxTokens: 64 };   // Listen の入力
    this.lres = null;                  // Listen の結果
    this.tab = "pattern";
    this.d = null;
    this.sums = new Map();             // 対照分析の名前 → /api/heads/<name>(Contrast タブ)
    this.pat = { text: "", chat: true, system: "", compare: true, triggers: "", view: "a" };   // Pattern の入力(ヘッドを替えても残す)
    this.res = null;                   // { key, compare, a, b, S }
    this.busy = false;
    this.status = "";
    this._token = 0;
  }

  async open({ layer, head, tab = null }) {
    if (tab) this.tab = tab;
    const tok = ++this._token;
    this.root.innerHTML = `<p class="muted">${t("{id} を読み込み中…", { id: `L${layer} H${head}` })}</p>`;
    let d;
    try { d = await getJSON("/api/head", { layer, head, lens: false }); }
    catch (e) { if (tok === this._token) this.root.innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    if (tok !== this._token) return;
    this.d = d;
    this.render();
    if (!d.logit_lens) {                           // W_O ブロックの logit lens は初回だけ数秒かかるので後から
      getJSON("/api/head", { layer, head }).then((x) => {
        if (tok !== this._token) return;
        this.d.logit_lens = x.logit_lens;
        if (this.tab === "details") this.render();
      }).catch(() => {});
    }
  }

  key() { const p = this.pat, d = this.d; return JSON.stringify([d.layer, d.head, p.text, p.chat, p.chat ? p.system.trim() : "", p.chat && p.compare && !!p.system.trim()]); }

  render() {
    const d = this.d;
    if (!d) return;
    const best = d.contrasts[0];
    const stats = best ? `#${best.rank}/${best.n_ranked} · z ${fmt(best.z, 2)} · null ${fmt(best.null_pct, 2)} · ${esc(best.name)}` : t("ヘッドの対照分析なし");
    this.root.innerHTML = `
      <div class="n-head">
        <div class="n-id mono">L${d.layer} H${d.head} <span class="muted h-kind">${kindLabel(d.kind)}</span></div>
        <div class="n-stats mono muted">${stats}</div>
        <div class="n-row"><input class="n-name h-name" placeholder="${t("名前(入力して Enter か欄の外で保存)")}" value="${esc(d.name || "")}" spellcheck="false">
          ${this.recipeButtons(d)}</div>
      </div>
      <div class="tabs"><button data-tab="pattern">${tl("注目", "Pattern")}</button><button data-tab="contrast">${tl("対照分析", "Contrast")}${d.contrasts.length ? ` <span class="muted">${d.contrasts.length}</span>` : ""}</button><button data-tab="details">${tl("くわしく", "Details")}</button><button data-tab="listen">${tl("聞く", "Listen")}</button><button data-tab="causal">${tl("効き目", "Causal")}${(d.causal || []).length ? ` <span class="muted">${d.causal.length}</span>` : ""}</button></div>
      <div class="n-tab" id="h-tab"></div>`;
    for (const b of this.root.querySelectorAll(".tabs button")) {
      b.classList.toggle("on", b.dataset.tab === this.tab);
      b.addEventListener("click", () => { this.tab = b.dataset.tab; this.render(); });
    }
    const name = this.root.querySelector(".h-name");
    const save = async () => {
      if ((name.value.trim() || null) === (d.name || null)) return;
      const r = await postJSON("/api/head/name", { layer: d.layer, head: d.head, name: name.value });
      d.name = r.name;
      name.classList.add("saved");
      setTimeout(() => name.classList.remove("saved"), 800);
      this.onNamed(r.names);
    };
    name.addEventListener("keydown", (ev) => { if (plainEnter(ev)) save(); });
    name.addEventListener("blur", save);
    for (const b of this.root.querySelectorAll("[data-hk]")) b.addEventListener("click", () => { this.onRecipe(d.layer, d.head, +b.dataset.hk); this.render(); });
    const pb = this.root.querySelector("[data-hp]");
    if (pb) pb.addEventListener("click", () => { this.onProj(d.layer, d.head, pb.dataset.hp); this.render(); });
    const body = this.root.querySelector("#h-tab");
    if (this.tab === "contrast") this.contrast(body);
    else if (this.tab === "details") body.innerHTML = this.details();
    else if (this.tab === "listen") this.listen(body);
    else if (this.tab === "causal") this.causal(body);
    else this.pattern(body);
  }

  // ×0(消す)と ×2(倍)。同じ倍率でもう一度押すと外す。recipe タブで別の倍率にしたものは横に出す
  recipeButtons(d) {
    const cur = this.inRecipe(d.layer, d.head);
    const b = (k) => {
      const on = cur !== null && cur === k;
      return `<button data-hk="${k}" class="${on ? `on ${k < 1 ? "minus" : "plus"}` : ""}" title="${on ? t("もう一度押すとレシピから外す") : t("強さ ×{k} でレシピに足す(強さはレシピのパネルで変えられる)", { k })}">${
        on ? `${t("レシピ")} ×${k}` : t("レシピに足す(×{k})", { k })}</button>`;
    };
    const other = cur !== null && cur !== 0 && cur !== 2 ? `<span class="muted mono h-cur" title="${t("レシピの中の倍率")}">${t("レシピ")} ×${fmt(cur, 2)}</span>` : "";
    return `${b(0)}${b(2)}${other}${this.projButton(d)}`;
  }

  // 射影 s1(CHANGE-06): このヘッドの書き込みのうち方向 r̂ に沿った成分だけを消す。方向は、このヘッドの因果スクリーニング・
  // ヘッドの対照分析の方向、無ければ最初の方向セット。レシピにあれば s と方向を出し、押すと外す
  projButton(d) {
    const dirs = this.projDirections();
    const cur = this.inProj(d.layer, d.head);
    if (cur) {
      return `<button data-hp="${esc(cur.direction || "")}" class="on minus" title="${t("もう一度押すとレシピから外す(強さ s と方向はレシピのパネルで変えられる)")}">${t("レシピ")} s${fmt(cur.s, 2)} @${esc(cur.direction || "")}</button>`;
    }
    const want = [...(d.causal || []).map((c) => c.direction), ...d.contrasts.map((c) => c.direction)].find((x) => x && dirs.includes(x)) || dirs[0];
    if (!want) return "";
    return `<button data-hp="${esc(want)}" title="${t("このヘッドの書き込みのうち方向 {d} に沿った成分だけを消す射影(L:Hh:s1@{d})をレシピに足す(強さ s はレシピのパネルで変えられる)", { d: want })}">${t("レシピに足す(射影 s1)")}</button>`;
  }

  // --- Pattern -------------------------------------------------------------------------------------
  pattern(el) {
    const p = this.pat, d = this.d;
    if (d.kind !== "attn") {
      el.innerHTML = `${noteHTML(t("線形注意(Gated DeltaNet など)の層には softmax の注目のパターンがありません。Details の W_O ブロックの lens と、Contrast の出力の差(out_shift・dir_cos)で見てください。"), "muted note")}`;
      return;
    }
    const canCmp = p.chat && !!p.system.trim();
    el.innerHTML = `
      <div class="ch-input"><textarea id="hp-text" rows="3" placeholder="${t("文章を入れて「流す」(このヘッドがどのトークンを見ているか)")}">${esc(p.text)}</textarea>
        <button class="main" id="hp-run"${this.busy ? " disabled" : ""}>${t("流す")}</button></div>
      <div class="ch-tools hp-opts">
        <label class="muted" title="${t("文章をユーザーの発話としてチャットテンプレートに入れる")}"><input type="checkbox" id="hp-chat"${p.chat ? " checked" : ""}> ${t("チャットの形")}</label>
        <label class="muted" title="${t("同じユーザーの発話を system あり/なしで流し、共通の末尾で注目の差を見る")}"><input type="checkbox" id="hp-compare"${p.compare ? " checked" : ""}${canCmp ? "" : " disabled"}> ${t("ペルソナあり/なしを比べる")}</label></div>
      <textarea id="hp-system" class="hp-sys" rows="2" placeholder="${t("ペルソナ(system)。空なら system なし")}"${p.chat ? "" : " disabled"}>${esc(p.system)}</textarea>
      <div class="list-tools"><input id="hp-trig" placeholder="${t("トリガー(カンマ区切り、re: で正規表現)")}" value="${esc(p.triggers)}" spellcheck="false"></div>
      <div class="ch-status muted note" id="hp-status">${esc(this.status)}</div>
      <div id="hp-out"></div>`;
    const $ = (s) => el.querySelector(s);
    $("#hp-text").addEventListener("input", (ev) => { p.text = ev.target.value; });
    $("#hp-text").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) this.run(); });
    $("#hp-system").addEventListener("input", (ev) => {
      p.system = ev.target.value;
      $("#hp-compare").disabled = !(p.chat && p.system.trim());
    });
    $("#hp-chat").addEventListener("change", (ev) => { p.chat = ev.target.checked; this.pattern(el); });
    $("#hp-compare").addEventListener("change", (ev) => { p.compare = ev.target.checked; });
    $("#hp-trig").addEventListener("change", (ev) => { p.triggers = ev.target.value; this.drawOut(); });
    $("#hp-run").addEventListener("click", () => this.run());
    if (this.res && this.res.key === this.key()) this.drawOut();
    else if (p.text.trim() && !this.busy && this.res && this.res.head !== `${d.layer}:${d.head}`) this.run();   // 別のヘッドを開いたら同じ文章で流し直す
  }

  async run() {
    const p = this.pat, d = this.d;
    if (!p.text.trim() || this.busy) return;
    const key = this.key(), cmp = p.chat && p.compare && !!p.system.trim();
    this.busy = true;
    this.status = t("流しています(GPU を待つことがあります)…");
    if (this.tab === "pattern") this.render();
    const tok = this._token;
    try {
      const x = await postJSON("/api/attention", { text: p.text, layer: d.layer, head: d.head, chat: p.chat,
                                                   system: p.chat ? p.system : null, compare: cmp });
      if (tok !== this._token) return;
      if (cmp) {
        const S = suffixLen(x.with.tokens, x.without.tokens);
        this.res = { key, head: `${d.layer}:${d.head}`, compare: true, a: x.with, b: x.without, S };
        this.status = t("{a} / {b} トークン · 共通の末尾 {s}", { a: x.with.n_tokens, b: x.without.n_tokens, s: S });
      } else {
        this.res = { key, head: `${d.layer}:${d.head}`, compare: false, a: x, b: null, S: null };
        this.status = t("{n} トークン", { n: x.n_tokens });
        if (p.view !== "a") p.view = "a";
      }
      if (x.warnings && x.warnings.length) this.status += ` · ${x.warnings.join(" / ")}`;
    } catch (e) { this.status = `${t("エラー")}: ${e.message}`; }
    finally { this.busy = false; }
    if (tok === this._token && this.tab === "pattern") this.render();
  }

  // 結果の描画: 見る行列を選び(あり・なし・差)、ヒートマップとトリガーの表
  drawOut() {
    const out = this.root.querySelector("#hp-out"), r = this.res, p = this.pat;
    if (!out || !r || r.key !== this.key()) return;
    if (!r.a.pattern) { out.innerHTML = `<p class="muted note">${t("このヘッドのパターンを捕捉できませんでした")}</p>`; return; }
    const trigs = p.triggers.split(",").map((s) => s.trim()).filter(Boolean);
    let M, rowLab, colLab, diverging = false, note, trigCols = new Set(), sysCol = false;
    if (r.compare && p.view === "diff") {
      const A = shared(r.a, r.S), B = shared(r.b, r.S);
      M = A.map((row, i) => row.map((v, j) => v - B[i][j]));
      const suf = r.a.tokens.slice(r.a.tokens.length - r.S);
      rowLab = suf; colLab = [t("(system)"), ...suf];
      diverging = true; sysCol = true;
      for (const tr of triggerTokens(suf, trigs)) for (const i of tr.toks) trigCols.add(i + 1);
      let mx = 0;
      for (const row of M) for (const v of row) mx = Math.max(mx, Math.abs(v));
      note = t("差 = ペルソナあり − なし(共通の末尾の行と列。左端の列は末尾より前 = system への注目の合計)。黄 = ありで増えた、赤紫 = 減った。明るさ = √(|差| / 最大 {m})", { m: fmt(mx, 3) });
    } else {
      const v = p.view === "b" && r.b ? r.b : r.a;
      M = v.pattern; rowLab = colLab = v.tokens;
      const from = r.compare ? v.tokens.length - r.S : 0;         // 比べるときのトリガーは共通の末尾(ユーザーの発話)の中だけ
      for (const tr of triggerTokens(v.tokens.slice(from), trigs)) for (const i of tr.toks) trigCols.add(i + from);
      note = t("行 = 見ている位置、列 = 見られている位置(右上は因果マスクで 0)。明るさ = √(注目の重み)、0〜1");
    }
    const seg = r.compare ? `<span class="seg hp-view">${[["a", t("ペルソナあり")], ["b", t("なし")], ["diff", t("差(共通の末尾)")]].map(([k, l]) =>
      `<button data-v="${k}" class="${p.view === k ? "on" : ""}">${l}</button>`).join("")}</span>` : "";
    out.innerHTML = `${seg}<div class="hp-wrap"><canvas id="hp-canvas"></canvas></div>
      <div class="hp-info mono muted" id="hp-info">${t("マウスを当てると、その行と列のトークンと値")}</div>
      <p class="muted note">${note}${trigCols.size ? ` ${t("水色の印 = トリガーの列")}` : ""}</p>
      ${this.trigTable(trigs)}`;
    for (const b of out.querySelectorAll("[data-v]")) b.addEventListener("click", () => { p.view = b.dataset.v; this.drawOut(); });
    this.heat = { M, rowLab, colLab, diverging, trigCols, sysCol };
    this.drawHeat(out.querySelector("#hp-canvas"));
  }

  // トリガー区間への注目(共通部分の中の割合)。最後の行 = 返事の最初の語を決める位置、後ろの行 = トリガーより後ろの行の平均
  trigTable(trigs) {
    const r = this.res;
    if (!trigs.length) return "";
    const measure = (v, from) => {
      const toks = v.tokens.slice(from), P = v.pattern, T = v.tokens.length;
      return triggerTokens(toks, trigs).map((tr) => {
        const idx = tr.toks.map((i) => i + from), lastTok = Math.max(...idx);
        const share = (q) => { let s = 0, all = 0; for (let c = from; c <= q; c++) all += P[q][c]; for (const c of idx) if (c <= q) s += P[q][c]; return all > 0 ? s / all : 0; };
        const rows = [];
        for (let q = lastTok + 1; q < T; q++) rows.push(share(q));
        return { text: tr.text, last: share(T - 1), after: rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : null };
      });
    };
    const A = measure(r.a, r.compare ? r.a.tokens.length - r.S : 0);
    if (!A.length) return `<p class="muted note">${t("トリガーが文章の中に見つかりません")}</p>`;
    const B = r.compare ? measure(r.b, r.b.tokens.length - r.S) : null;
    const head = r.compare ? `<tr><th>${t("トリガー")}</th><th colspan="3">${t("最後の行(あり / なし / 差)")}</th><th colspan="3">${t("後ろの行の平均(あり / なし / 差)")}</th></tr>`
      : `<tr><th>${t("トリガー")}</th><th>${t("最後の行")}</th><th>${t("後ろの行の平均")}</th></tr>`;
    const rows = A.map((a, i) => {
      const b = B && B[i];
      const cell = (x, y) => (b ? `<td>${pct(x)}</td><td>${pct(y)}</td><td class="${x - y >= 0 ? "plus" : "minus"}">${x === null || y === null ? "—" : sgn(x - y, 3)}</td>` : `<td>${pct(x)}</td>`);
      return `<tr><td class="w">${esc(tokText(a.text, 24))}</td>${cell(a.last, b && b.last)}${cell(a.after, b && b.after)}</tr>`;
    }).join("");
    return `<h4>${t("トリガー区間への注目(共通部分の中の割合)")}</h4><table class="kv mono hp-trig">${head}${rows}</table>
      ${noteHTML(t("割合 = その行の注目のうち、トリガーのトークンへのもの(比べるときは system の部分を除いた共通部分の中で。gne heads contrast の trigger_mass と同じ分母)。最後の行 = 返事の最初のトークンを決める位置。"), "muted note")}`;
  }

  drawHeat(cv) {
    const H = this.heat, rows = H.M.length, cols = H.colLab.length;
    const width = Math.max(200, (cv.parentElement.clientWidth || this.root.clientWidth || 420) - 4);
    const cell0 = Math.floor((width - 8) / cols), labels = cell0 >= 9;
    const lab = labels ? 70 : 0, cell = Math.max(1, Math.min(16, Math.floor((width - lab - 8) / cols)));
    const W = lab + cell * cols + 8, top = 6, Hh = rows * cell, bottom = labels ? 64 : 10, dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = (top + Hh + bottom) * dpr; cv.style.width = `${W}px`; cv.style.height = `${top + Hh + bottom}px`;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#000"; g.fillRect(0, 0, W, top + Hh + bottom);
    let mx = 0;
    if (H.diverging) for (const row of H.M) for (const v of row) mx = Math.max(mx, Math.abs(v));
    const img = new ImageData(cols, rows);                         // 1 マス = 1 画素で作って、拡大して描く(T が大きくても速い)
    const offset = H.sysCol ? 1 : 0;
    for (let i = 0; i < rows; i++) {
      const row = H.M[i];
      for (let j = 0; j < cols; j++) {
        const v = row[j] ?? 0, masked = j - offset > i;
        let c;
        if (masked) c = MASKED;
        else if (H.diverging) { const b = Math.sqrt(Math.abs(v) / Math.max(mx, 1e-9)); c = (v >= 0 ? PLUS : MINUS).map((u) => u * b); }
        else { const b = Math.sqrt(Math.max(0, Math.min(1, v))); c = AMBER.map((u) => u * b); }
        const o = 4 * (i * cols + j);
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = cols; off.height = rows;
    off.getContext("2d").putImageData(img, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(off, lab, top, cols * cell, rows * cell);
    g.fillStyle = "#6fe3ff";                                        // トリガーの列の印(下端)
    for (const j of H.trigCols) g.fillRect(lab + j * cell, top + Hh + 1, Math.max(1, cell), 3);
    if (labels) {
      g.font = "10px ui-monospace, monospace";
      g.textAlign = "right"; g.textBaseline = "middle";
      for (let i = 0; i < rows; i++) { g.fillStyle = "#8f9bab"; g.fillText(tokText(H.rowLab[i]), lab - 4, top + (i + 0.5) * cell); }
      g.save();
      for (let j = 0; j < cols; j++) {
        g.save();
        g.translate(lab + (j + 0.5) * cell, top + Hh + 6);
        g.rotate(-Math.PI / 2);
        g.fillStyle = H.trigCols.has(j) ? "#6fe3ff" : "#8f9bab";
        g.fillText(tokText(H.colLab[j]), 0, 0);
        g.restore();
      }
      g.restore();
    }
    this.heatGeom = { lab, cell, top, rows, cols };
    const info = this.root.querySelector("#hp-info");
    cv.onmousemove = (ev) => {
      const r = cv.getBoundingClientRect(), x = ev.clientX - r.left - lab, y = ev.clientY - r.top - top;
      const i = Math.floor(y / cell), j = Math.floor(x / cell);
      if (i < 0 || j < 0 || i >= rows || j >= cols) return;
      const v = H.M[i][j];
      info.innerHTML = `${t("行")} ${i}${tokHTMLs(H.rowLab[i])} → ${t("列")} ${H.sysCol ? (j ? j - 1 : "—") : j}${tokHTMLs(H.colLab[j])}: <b>${
        j - offset > i ? t("(未来は見えない)") : H.diverging ? sgn(v, 4) : fmt(v, 4)}</b>`;
    };
  }

  // --- Contrast ------------------------------------------------------------------------------------
  contrast(el) {
    const d = this.d;
    if (!d.contrasts.length) {
      el.innerHTML = `<p class="muted note">${t("このヘッドを含むヘッドの対照分析はまだありません({cmd})。", { cmd: "<code>gne heads contrast --contrast configs/contrast/….yaml</code>" })}</p>`;
      return;
    }
    const need = d.contrasts.filter((c) => !this.sums.has(c.name)).map((c) => c.name);
    if (need.length) {
      const tok = this._token;
      Promise.all(need.map((n) => getJSON(`/api/heads/${encodeURIComponent(n)}`).then((x) => this.sums.set(n, x)).catch(() => this.sums.set(n, null))))
        .then(() => { if (tok === this._token && this.tab === "contrast") this.render(); });
    }
    el.innerHTML = d.contrasts.map((c) => {
      const sm = this.sums.get(c.name), fl = c.floors || {};
      const mrows = METRICS.map(([k, label, tip, signed]) => {
        const v = c[k], f = fl[k], hot = v !== null && v !== undefined && f !== undefined && (signed ? Math.abs(v) : v) > f;
        return `<tr title="${esc(tip())}"><td>${label()}</td><td class="mono ${hot ? "hot" : ""}">${v === null || v === undefined ? "—" : signed ? sgn(v, 4) : fmt(v, 4)}</td><td class="mono muted">${f === undefined ? "—" : fmt(f, 4)}</td><td class="muted">${k}</td></tr>`;
      }).join("");
      const top = sm ? sm.rows.slice(0, 12).map((x, i) => `<tr data-go="${x.layer}:${x.head}" class="pick${x.layer === d.layer && x.head === d.head ? " sel" : ""}"><td>${i + 1}</td><td>L${x.layer} H${x.head}</td><td>${x.kind === "attn" ? "attn" : "linear"}</td><td>${fmt(x.z, 2)}</td><td>${fmt(x.null_pct, 2)}</td></tr>`).join("") : "";
      const samples = sm && sm.samples ? sm.samples.filter((s) => s.triggers && s.triggers.length) : [];
      return `<div class="h-con">
        <h4><b class="mono">${esc(c.name)}</b> <span class="mono">#${c.rank}/${c.n_ranked} · z ${fmt(c.z, 2)} · null ${fmt(c.null_pct, 2)}</span>
          <button class="small" data-ov="${esc(c.name)}" title="${t("地図の板の上辺のヘッドをこの対照分析の z で色付けする")}">${t("地図で見る")}</button></h4>
        <table class="kv h-met"><tr><th>${t("指標")}</th><th>${t("値")}</th><th title="${t("中立どうし(system を別の中立の文に替えただけ)で測った同じ指標の、全ヘッドの 95% 点")}">${t("基準値(中立どうしの差の p95)")}</th><th></th></tr>${mrows}</table>
        <p class="muted note">${t("琥珀 = 基準値を超えた指標。z = 測れた指標ごとの対照に対する z の平均、null = 対照のヘッドの何割より大きいか。")}${c.direction ? ` ${t("方向 {d}", { d: esc(c.direction) })}` : ""}</p>
        ${samples.length ? `<h4>${t("トリガーのある会話(Pattern で、このヘッドの注目をペルソナあり/なしで見る)")}</h4>${samples.slice(0, 8).map((s, i) =>
          `<div class="h-sample"><button class="small" data-pat="${esc(c.name)}" data-i="${i}">${t("Pattern で見る")}</button> <span class="note">${esc(s.prompt)}</span> <span class="muted mono">${s.triggers.map((x) => esc(tokText(x, 16))).join(" · ")}</span></div>`).join("")}` : ""}
        ${top ? `<h4>${t("この対照分析の上位")}</h4><table class="rank mono h-top"><tr><th>#</th><th>${t("ヘッド")}</th><th>kind</th><th>z</th><th>null</th></tr>${top}</table>` : sm === undefined ? `<p class="muted">${t("読み込み中…")}</p>` : ""}
      </div>`;
    }).join("");
    for (const b of el.querySelectorAll("[data-ov]")) b.addEventListener("click", () => this.onOverlay(`heads:${b.dataset.ov}`));
    for (const tr of el.querySelectorAll("tr[data-go]")) {
      tr.addEventListener("click", () => { const [L, h] = tr.dataset.go.split(":").map(Number); this.onGoto({ layer: L, head: h }); });
    }
    for (const b of el.querySelectorAll("[data-pat]")) {
      b.addEventListener("click", () => {
        const sm = this.sums.get(b.dataset.pat), s = sm.samples.filter((x) => x.triggers && x.triggers.length)[+b.dataset.i];
        const trig = (sm.triggers && sm.triggers[s.prompt]) || [...new Set(s.triggers.map((x) => x.trim()).filter(Boolean))];
        Object.assign(this.pat, { text: s.prompt, chat: true, system: sm.system_a || this.pat.system, compare: !!(sm.system_a || this.pat.system),
                                  triggers: trig.join(", "), view: "diff" });
        this.tab = "pattern";
        this.res = null;
        this.render();
        this.run();
      });
    }
  }

  // --- Causal(CHANGE-06 G) -----------------------------------------------------------------------------
  causal(el) {
    const d = this.d, list = this.causalList(), mine = d.causal || [];
    if (!list.length && !mine.length) {
      el.innerHTML = `<p class="muted note">${t("因果スクリーニングはまだありません({cmd})。", { cmd: "<code>gne heads causal --direction NAME</code>" })}</p>`;
      return;
    }
    const names = [...new Set([...mine.map((c) => c.name), ...list.map((c) => c.name)])];
    const cz = this.cz;
    if (!names.includes(cz.name)) { cz.name = names[0]; cz.layer = null; }
    const sm = this.causalSums.get(cz.name);
    if (sm === undefined && !this._czPending.has(cz.name)) {
      const tok = this._token, name = cz.name;
      this._czPending.add(name);
      getJSON(`/api/causal/${encodeURIComponent(name)}`).then((x) => this.causalSums.set(name, x)).catch(() => this.causalSums.set(name, null))
        .then(() => { this._czPending.delete(name); if (tok === this._token && this.tab === "causal") this.render(); });
    }
    const info = list.find((c) => c.name === cz.name) || mine.find((c) => c.name === cz.name) || {};
    const SL = (sm && sm.score_layers) || info.score_layers || [], P = (sm && sm.primary) ?? info.primary;
    if (cz.layer === null || !SL.includes(cz.layer)) cz.layer = P ?? SL[0] ?? null;
    const at = cz.layer;
    const sel = `<select id="cz-name">${names.map((n) => `<option${n === cz.name ? " selected" : ""}>${esc(n)}</option>`).join("")}</select>`;
    const modes = info.modes || (sm ? sm.modes : []) || [];
    const ov = `${modes.includes("ablate") ? `<button class="small" data-czov="causal" title="${t("地図の板の上辺のヘッドを、止めたときの下がり方(causal_max)で色付けする")}">${t("地図: 効き")}</button>` : ""}${
      modes.includes("marginal") ? `<button class="small" data-czov="eta" title="${t("地図の板の上辺のヘッドを、効率 η で色付けする")}">${t("地図: 効率 η")}</button>` : ""}`;
    const head = sm ? `<p class="muted note mono">${t("方向 {d}", { d: esc(sm.direction || "—") })} · ${t("射影の値を測る層")} ${SL.map((L) => (L === P ? `<b>L${L}</b>` : `L${L}`)).join(" ")} · A ${sm.n_a ?? "—"} / B ${sm.n_b ?? "—"}${
      sm.spearman_ablate_marginal !== null && sm.spearman_ablate_marginal !== undefined ? ` · ${t("順位相関(ablate・marginal)")} ${fmt(sm.spearman_ablate_marginal, 2)}` : ""} · ${esc((sm.updated || "").replace("T", " "))}</p>` : "";
    const layerSel = SL.length > 1 ? `<label class="muted" title="${t("表に出す下がり方・寄与・η を測る層(太字 = 主の層)")}">${t("射影の値を測る層")} <select id="cz-layer">${
      SL.map((L) => `<option value="${L}"${L === at ? " selected" : ""}>L${L}${L === P ? " *" : ""}</option>`).join("")}</select></label>` : "";
    el.innerHTML = `<div class="list-tools cz-tools">${sel}${ov}${layerSel}</div>${head}
      ${this.causalMine(mine.find((c) => c.name === cz.name), at, P)}
      ${sm ? this.causalTop(sm, at, P) : sm === undefined ? `<p class="muted">${t("読み込み中…")}</p>` : `<p class="err">${t("読み込めませんでした")}</p>`}`;
    const $ = (q) => el.querySelector(q);
    $("#cz-name").addEventListener("change", (ev) => { cz.name = ev.target.value; cz.layer = null; this.render(); });
    const ls = $("#cz-layer");
    if (ls) ls.addEventListener("change", (ev) => { cz.layer = +ev.target.value; this.render(); });
    for (const b of el.querySelectorAll("[data-czov]")) b.addEventListener("click", () => this.onOverlay(`${b.dataset.czov}:${cz.name}`));
    for (const th of el.querySelectorAll("th[data-czsort]")) th.addEventListener("click", () => { cz.sort = th.dataset.czsort; this.render(); });
    for (const tr of el.querySelectorAll("tr[data-go]")) {
      tr.addEventListener("click", () => { const [L, h] = tr.dataset.go.split(":").map(Number); this.onGoto({ layer: L, head: h, tab: "causal" }); });
    }
    for (const b of el.querySelectorAll("[data-alloc]")) {
      b.addEventListener("click", () => {
        const row = sm.alloc.table.find((x) => String(x.n_heads) === b.dataset.alloc);
        if (row) this.onLoadProj(`${cz.name}-n${row.n_heads}`, row.units, sm.alloc.direction || sm.direction);
      });
    }
  }

  // このヘッドの行(ablate と marginal)。at = 表示する測る層、P = 主の層
  causalMine(c, at, P) {
    const d = this.d;
    if (!c) return `<p class="muted note">${t("L{L} H{h} はこのスクリーニングで測っていません(層単位のスクリーニングで選ばれた層のヘッドだけを測ります)。", { L: d.layer, h: d.head })}</p>`;
    const k = String(at), drops = (r) => Object.entries(r.score_drop || {}).map(([L, v]) => `${+L === P ? "<b>" : ""}L${L} ${sgn(v, 2)}${+L === P ? "</b>" : ""}`).join(" · ");
    const out = [`<h4>${t("このヘッド(L{L} H{h})", { L: d.layer, h: d.head })}</h4>`];
    for (const r of c.rows) {
      const m = r.mode || "ablate", rk = (c.rank || {})[m];
      const contrib = (r.contrib_by_layer || {})[k], drop = (r.score_drop || {})[k];
      const ratio = Number.isFinite(drop) && Number.isFinite(contrib) && Math.abs(contrib) > 1e-9 ? drop / contrib : null;
      const b = `<tr><td>${t("B 側の射影の値の変化")}</td><td class="mono">${sgn(r.b_side_delta, 3)}</td><td>${t("B 側の直交成分のずれ")}</td><td class="mono">${fmt(r.drift_orth_b, 3)}</td></tr>`;
      if (m === "marginal") {
        out.push(`<div class="h-con"><h4><b>${t("効率(marginal)")}</b> <span class="mono">${rk ? `#${rk.rank}/${rk.n}` : ""} · η ${fmt((r.eta_by_layer || {})[k] ?? r.eta, 3)}</span>${
          r.in_base ? ` <span class="badge">${t("いまの組")}</span>` : ""}${r.excluded ? ` <span class="badge minus" title="${t("B 側の変化が上限を超えたので配分から外した")}">${t("除外")}</span>` : ""}</h4>
          <table class="kv h-met"><tr><td>η(L${esc(k)})</td><td class="mono">${fmt((r.eta_by_layer || {})[k] ?? r.eta, 3)}</td><td>Δs</td><td class="mono">${fmt(r.delta_s, 1)}</td></tr>
          <tr><td>${t("下がり方")}</td><td class="mono" colspan="3">${drops(r)}</td></tr>${b}</table></div>`);
      } else {
        out.push(`<div class="h-con"><h4><b>${t("止める(ablate)")}</b> <span class="mono">${rk ? `#${rk.rank}/${rk.n}` : ""} · causal_max ${fmt(r.causal_max, 2)}</span></h4>
          <table class="kv h-met"><tr><td title="${t("出力を 0 にしたときの A 側の射影の値の下がり方")}">causal(L${esc(k)})</td><td class="mono">${sgn(drop, 3)}</td>
            <td title="${t("無介入でこのヘッドが、射影の値を測る層の方向に直接書く量(A 側の平均)")}">${t("直接の寄与")}(L${esc(k)})</td><td class="mono">${sgn(contrib, 3)}</td></tr>
          <tr><td title="${t("causal ÷ 直接の寄与。1 より大きい = 後段のユニットを通る間接の経路、1 より小さい = 補い合い")}">ratio</td><td class="mono">${fmt(ratio, 2)}</td><td>causal_max</td><td class="mono">${fmt(r.causal_max, 3)}</td></tr>
          <tr><td>${t("下がり方")}</td><td class="mono" colspan="3">${drops(r)}</td></tr>${b}</table></div>`);
      }
    }
    out.push(`${noteHTML(t("下がり方 = 介入の前 − 後の A 側の射影の値(正 = このヘッドが A 側の向きに押していた)。ratio > 1 は間接の経路、< 1 は補い合い。η = いまの組から s を Δs 上げたときの、効きにくい例での下がり方 ÷ Δs。"), "muted note")}`);
    return out.join("");
  }

  // 選んだスクリーニングの上位のヘッド(ablate と marginal を 1 行にまとめる)。並べ替え: causal_max・causal・contrib・ratio・η
  causalTop(sm, at, P) {
    const d = this.d, k = String(at), cz = this.cz;
    const by = new Map();
    for (const r of sm.rows) {
      const key = `${r.layer}:${r.head}`;
      if (!by.has(key)) by.set(key, { layer: r.layer, head: r.head, kind: r.kind, a: null, m: null });
      by.get(key)[(r.mode || "ablate") === "marginal" ? "m" : "a"] = r;
    }
    const val = (x, what) => {
      const a = x.a, m = x.m, base = a || m;
      const drop = a ? (a.score_drop || {})[k] : null, contrib = base ? (base.contrib_by_layer || {})[k] : null;
      return { causal_max: a ? a.causal_max : null, causal: drop, contrib,
               ratio: Number.isFinite(drop) && Number.isFinite(contrib) && Math.abs(contrib) > 1e-9 ? drop / contrib : null,
               eta: m ? (m.eta_by_layer || {})[k] ?? m.eta : null }[what];
    };
    const rows = [...by.values()].sort((x, y) => (val(y, cz.sort) ?? -Infinity) - (val(x, cz.sort) ?? -Infinity));
    const th = (key, label, tip) => `<th data-czsort="${key}" class="${cz.sort === key ? "on" : ""}" title="${esc(tip)}">${label}${cz.sort === key ? " ▾" : ""}</th>`;
    const body = rows.slice(0, 40).map((x, i) => {
      const b = x.a || x.m;
      return `<tr data-go="${x.layer}:${x.head}" class="pick${x.layer === d.layer && x.head === d.head ? " sel" : ""}${x.m && x.m.excluded ? " rnd" : ""}"><td>${i + 1}</td><td>L${x.layer} H${x.head}</td>
        <td>${fmt(val(x, "causal_max"), 2)}</td><td>${fmt(val(x, "causal"), 2)}</td><td>${fmt(val(x, "contrib"), 2)}</td><td>${fmt(val(x, "ratio"), 2)}</td>
        <td>${fmt(val(x, "eta"), 3)}${x.m && x.m.in_base ? "*" : ""}</td><td>${fmt(b.b_side_delta, 2)}</td><td>${fmt(b.drift_orth_b, 3)}</td></tr>`;
    }).join("");
    const lr = (sm.layer_rows || []).slice().sort((a, b) => (b.causal_max ?? 0) - (a.causal_max ?? 0));
    const lrows = lr.map((r) => `<tr><td>L${r.layer}</td><td>${r.kind === "attn" ? "attn" : "linear"}</td><td>${r.n_heads ?? "—"}</td><td>${fmt(r.causal_max, 2)}</td>
      <td>${fmt((r.score_drop || {})[k], 2)}</td><td>${fmt(r.b_side_delta, 2)}</td><td>${fmt(r.drift_orth_b, 3)}</td></tr>`).join("");
    return `<h4>${t("このスクリーニングの上位({n} ヘッド)", { n: by.size })}</h4>
      <div class="cz-scroll"><table class="rank mono h-top cz-top"><tr><th>#</th><th>${t("ヘッド")}</th>${th("causal_max", "causal_max", t("止めたときの下がり方の、射影の値を測る層での最大"))}${
        th("causal", `causal L${esc(k)}`, t("止めたときの下がり方(表示中の測る層)"))}${th("contrib", `contrib L${esc(k)}`, t("直接の寄与(表示中の測る層)"))}${
        th("ratio", "ratio", t("causal ÷ 直接の寄与(表示中の測る層)"))}${th("eta", `η L${esc(k)}`, t("効率 η(marginal、表示中の測る層)。* = いまの組のヘッド"))}<th title="${t("B 側の射影の値の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th></tr>${body}</table></div>
      <p class="muted note">${t("見出しをクリックで並べ替え、行をクリックでそのヘッドを開く。灰色 = 配分から外したヘッド。")}</p>
      ${lrows ? `<details><summary class="muted note">${t("層単位のスクリーニング({n} 層、層のヘッドを全部止めたとき)", { n: lr.length })}</summary>
        <table class="kv mono cz-layers"><tr><th>${t("層")}</th><th>kind</th><th>${t("ヘッド")}</th><th>causal_max</th><th>L${esc(k)}</th><th>B Δ</th><th>B ⊥</th></tr>${lrows}</table></details>` : ""}
      ${this.allocHTML(sm)}`;
  }

  // 強さの配分(gne heads allocate)の組。ヘッド数ごとの s の最大・残った例の数・B 側のずれ。行のボタンでレシピに読み込む
  allocHTML(sm) {
    const al = sm.alloc;
    if (!al || !(al.table || []).length) return "";
    const rows = al.table.filter((r) => !r.skipped).map((r) => `<tr class="${r.recommended ? "on" : ""}"><td>${r.n_heads}${r.recommended ? ` <span class="badge">${t("推奨")}</span>` : ""}</td>
      <td>${fmt(r.s_max, 2)}</td><td>${r.remain ?? "—"}</td><td>${fmt(r.a_max, 2)}</td><td>${fmt(r.b_drift_orth, 3)}</td><td>${fmt(r.b_side_delta, 2)}</td>
      <td class="w">${esc((r.units || []).join(" "))}</td><td><button class="small" data-alloc="${r.n_heads}" title="${t("この組(proj のユニット)で今のレシピを置き換える")}">${t("レシピを置き換える")}</button></td></tr>`).join("");
    return `<h4>${t("強さの配分(gne heads allocate、L{L} の射影の値を {v} まで)", { L: al.score_layer ?? "—", v: fmt(al.target, 2) })}</h4>
      <div class="cz-scroll"><table class="kv mono cz-alloc"><tr><th>${t("ヘッド数")}</th><th>s max</th><th title="${t("目標に届かず残った例の数")}">${t("残り")}</th><th title="${t("A 側の射影の値の最大")}">A max</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th><th title="${t("B 側の射影の値の変化")}">B Δ</th><th>${t("ユニット")}</th><th></th></tr>${rows}</table></div>`;
  }

  // --- Listen --------------------------------------------------------------------------------------
  // 見るヘッドの既定: 開いているヘッド。「K/V グループ」でそのヘッドと K/V を共有するヘッドの組にする
  groupSpec() {
    const d = this.d;
    if (!d.n_head_kv || d.kv_group === null || d.kv_group === undefined) return `${d.layer}:H${d.head}`;
    const g = d.n_head / d.n_head_kv, a = d.kv_group * g;
    return g > 1 ? `${d.layer}:H${a}-${a + g - 1}` : `${d.layer}:H${a}`;
  }

  listen(el) {
    const L = this.lis, d = this.d;
    if (d.kind !== "attn") {
      el.innerHTML = `<p class="muted note">${t("線形注意の層には softmax の注目がないので、Listen は使えません。")}</p>`;
      return;
    }
    if (!L.heads) L.heads = `${d.layer}:H${d.head}`;
    if (!L.system && this.pat.system) L.system = this.pat.system;
    const dirs = this.directions();
    el.innerHTML = `
      ${noteHTML(t("返事のトークンが、選んだヘッドで文脈のどのトークンに注目しているか。方向を選ぶと、文脈の各トークンがヘッドの出力を通してその方向にどれだけ寄与したか(注目 × ゲート × OV)も出す。"), "muted note")}
      <textarea id="ls-system" class="hp-sys" rows="3" placeholder="${t("system(空なら system なし)")}">${esc(L.system)}</textarea>
      <div class="ch-input"><textarea id="ls-user" rows="2" placeholder="${t("ユーザーの発話")}">${esc(L.user)}</textarea>
        <button class="main" id="ls-run"${this.busy ? " disabled" : ""}>${t("流す")}</button></div>
      <textarea id="ls-reply" class="hp-sys" rows="2" placeholder="${t("返事(空なら greedy に作る)")}">${esc(L.reply)}</textarea>
      <div class="list-tools ls-opts">
        <label class="muted">${t("ヘッド")} <input id="ls-heads" class="mono" value="${esc(L.heads)}" spellcheck="false" title="${t("例 14:H0-3,16:H2(L:H* で層の全ヘッド)")}"></label>
        <button class="small" id="ls-one" title="${t("開いているヘッドだけにする")}">L${d.layer} H${d.head}</button>
        <button class="small" id="ls-grp" title="${t("開いているヘッドと K/V を共有するヘッドの組にする")}">${t("K/V グループ")}</button>
        <label class="muted">${t("方向")} <select id="ls-dir"><option value="">${t("(なし)")}</option>${dirs.map((x) => `<option${x === L.dir ? " selected" : ""}>${esc(x)}</option>`).join("")}</select></label>
        <label class="muted">${t("長さ")} <input id="ls-max" type="number" min="1" max="256" value="${L.maxTokens}" class="mono ls-num"></label></div>
      <div class="ch-status muted note" id="ls-status">${esc(this.lstatus || "")}</div>
      <div id="ls-out"></div>`;
    const $ = (q) => el.querySelector(q);
    for (const [id, k] of [["#ls-system", "system"], ["#ls-user", "user"], ["#ls-reply", "reply"], ["#ls-heads", "heads"]]) {
      $(id).addEventListener("input", (ev) => { L[k] = ev.target.value; });
    }
    $("#ls-user").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) this.runListen(); });
    $("#ls-dir").addEventListener("change", (ev) => { L.dir = ev.target.value; });
    $("#ls-max").addEventListener("change", (ev) => { L.maxTokens = Math.max(1, Math.min(256, +ev.target.value || 64)); });
    $("#ls-one").addEventListener("click", () => { L.heads = `${d.layer}:H${d.head}`; $("#ls-heads").value = L.heads; });
    $("#ls-grp").addEventListener("click", () => { L.heads = this.groupSpec(); $("#ls-heads").value = L.heads; });
    $("#ls-run").addEventListener("click", () => this.runListen());
    this.drawListen();
  }

  async runListen() {
    const L = this.lis;
    if (!L.user.trim() || !L.heads.trim() || this.busy) return;
    this.busy = true;
    this.lstatus = L.reply.trim() ? t("流しています(GPU を待つことがあります)…") : t("返事を作って流しています(GPU を待つことがあります)…");
    if (this.tab === "listen") this.render();
    const tok = this._token;
    try {
      const x = await postJSON("/api/listen", { user: L.user, heads: L.heads, system: L.system || null, reply: L.reply.trim() ? L.reply : null,
                                                max_tokens: L.maxTokens, direction: L.dir || null });
      this.lres = x;
      if (!x.contrib) L.mode = "attn";
      else if (this.lres && L.mode === "attn" && !this._modeChosen) L.mode = "contrib";
      const share = this.contribShare(x);
      this.lstatus = t("{n} トークン(返事 {r})", { n: x.n_tokens, r: x.n_tokens - x.reply_start })
        + (share ? ` · ${t("層の寄与のうち選んだヘッドの分 {p}", { p: share })}` : "")
        + (x.warnings && x.warnings.length ? ` · ${x.warnings.join(" / ")}` : "");
    } catch (e) { this.lstatus = `${t("エラー")}: ${e.message}`; }
    finally { this.busy = false; }
    if (tok === this._token && this.tab === "listen") this.render();
  }

  // 選んだヘッドの寄与 ÷ その層の全ヘッドの寄与(層ごと)
  contribShare(x) {
    if (!x.layers) return "";
    return Object.entries(x.layers).map(([Lk, info]) => {
      const sel = x.heads.filter((h) => String(h.layer) === Lk).reduce((a, h) => a + info.contrib_by_head[h.head], 0);
      return `L${Lk} ${pct(info.contrib_total ? sel / info.contrib_total : null)}`;
    }).join(" · ");
  }

  drawListen() {
    const out = this.root.querySelector("#ls-out"), x = this.lres, L = this.lis;
    if (!out || !x) return;
    const contrib = L.mode === "contrib" && x.contrib, vals = contrib ? x.contrib : x.attn;
    let mx = 0;
    for (let i = 0; i < x.reply_start; i++) mx = Math.max(mx, Math.abs(vals[i]));
    const keys = Object.keys(contrib ? x.contrib_by_head : x.attn_by_head);
    const color = (v) => {
      const b = Math.sqrt(Math.min(1, Math.abs(v) / Math.max(mx, 1e-9)));
      const c = contrib ? (v >= 0 ? PLUS : MINUS) : AMBER;
      return `rgba(${c[0]},${c[1]},${c[2]},${(0.08 + 0.8 * b).toFixed(3)})`;
    };
    const span = (i) => {
      const per = keys.map((k) => `${k} ${contrib ? sgn(x.contrib_by_head[k][i], 4) : fmt(x.attn_by_head[k][i], 4)}`).join("\n");
      const tip = `#${i} ${tokText(x.tokens[i], 30)} (${x.roles[i]})\n${contrib ? t("寄与") : t("注目")} ${contrib ? sgn(vals[i], 4) : fmt(vals[i], 4)}\n${per}`;
      const txt = esc(x.tokens[i]).replace(/\n/g, "↵<br>");
      return `<span class="lt r-${x.roles[i]}" style="background:${color(vals[i])}" title="${esc(tip)}">${txt || "∅"}</span>`;
    };
    const ctxHTML = Array.from({ length: x.reply_start }, (_, i) => span(i)).join("");
    const repHTML = x.tokens.slice(x.reply_start).map((tk) => `<span class="lt r-reply">${esc(tk).replace(/\n/g, "↵<br>")}</span>`).join("");
    const seg = `<span class="seg hp-view">${[["attn", t("注目")], ["contrib", t("寄与")]].map(([k, l]) =>
      `<button data-m="${k}" class="${(contrib ? "contrib" : "attn") === k ? "on" : ""}"${k === "contrib" && !x.contrib ? " disabled" : ""}>${l}</button>`).join("")}</span>`;
    const top = (contrib ? x.top_contrib : x.top_attn) || [];
    const rows = top.map((r) => `<tr><td>${r.pos}</td><td class="w">${esc(tokText(r.token, 20))}</td><td class="muted">${r.role}</td><td>${contrib ? sgn(r.value, 4) : fmt(r.value, 4)}</td>${
      keys.map((k) => `<td class="muted">${contrib ? sgn(x.contrib_by_head[k][r.pos], 3) : fmt(x.attn_by_head[k][r.pos], 3)}</td>`).join("")}</tr>`).join("");
    const role = Object.entries(x.by_role).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v.attn, 3)}</td><td>${v.contrib === null ? "—" : sgn(v.contrib, 4)}</td></tr>`).join("");
    const layers = x.layers ? Object.entries(x.layers).map(([Lk, info]) => {
      const sel = new Set(x.heads.filter((h) => String(h.layer) === Lk).map((h) => h.head));
      const m = Math.max(...info.contrib_by_head.map(Math.abs), 1e-9);
      return `<h4>${t("L{L} のヘッドごとの寄与(層全体 {tot}、分解の誤差 {err})", { L: Lk, tot: sgn(info.contrib_total, 3), err: info.check_rel_err === null ? "—" : pct(info.check_rel_err) })}</h4>
        <div class="ls-bars">${info.contrib_by_head.map((v, h) => `<div class="ls-bar${sel.has(h) ? " sel" : ""}" title="H${h} ${sgn(v, 4)}"><span style="height:${(100 * Math.abs(v) / m).toFixed(1)}%;background:${v >= 0 ? "rgb(242,210,75)" : "rgb(201,79,167)"}"></span><i>${h}</i></div>`).join("")}</div>`;
    }).join("") : "";
    out.innerHTML = `${seg}
      <div class="ls-toks">${ctxHTML}</div>
      <div class="ls-reply"><span class="muted">${t("返事")}:</span> ${repHTML}</div>
      <p class="muted note">${contrib ? t("色 = そのトークンの方向への寄与(返事のトークンの平均)。黄 = 方向の向き、赤紫 = 逆向き。明るさ = √(|値| / 文脈の最大)。")
        : t("色 = 返事の各トークンからの注目の平均(選んだヘッドの合計)。明るさ = √(値 / 文脈の最大)。")} ${t("トークンにマウスを当てるとヘッドごとの値。")}</p>
      ${layers}
      <h4>${contrib ? t("寄与の上位(文脈のトークン)") : t("注目の上位(文脈のトークン)")}</h4>
      <table class="kv mono hp-trig"><tr><th>pos</th><th>${t("トークン")}</th><th>${t("役")}</th><th>${t("合計")}</th>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr>${rows}</table>
      <h4>${t("役ごとの合計")}</h4><table class="kv mono"><tr><th>${t("役")}</th><th>${t("注目")}</th><th>${t("寄与")}</th></tr>${role}</table>`;
    for (const b of out.querySelectorAll("[data-m]")) b.addEventListener("click", () => { L.mode = b.dataset.m; this._modeChosen = true; this.drawListen(); });
  }

  // --- Details -------------------------------------------------------------------------------------
  details() {
    const d = this.d, ll = d.logit_lens, out = [];
    out.push(`<h4>${t("大きさ")}</h4><table class="kv mono">
      <tr><td>kind</td><td>${kindLabel(d.kind)}</td><td>n_head</td><td>${d.n_head}${d.n_head_kv ? ` / kv ${d.n_head_kv}` : ""}</td></tr>
      <tr><td>head_dim</td><td>${d.head_dim}</td><td>${d.kind === "attn" ? "kv group" : ""}</td><td>${d.kv_group ?? ""}</td></tr>
      <tr><td>${t("書き込み")}</td><td colspan="3">${esc(d.writer)} [${d.block[0]}, ${d.block[1]})</td></tr></table>`);
    if (ll) {
      out.push(`<h4>${t("W_O の列ブロックの logit lens(主成分 1 本目の割合 {p})", { p: pct(ll.sv_share) })}</h4>`);
      ll.components.forEach((c, i) => {
        out.push(`<div class="muted note mono">PC${i + 1} · σ ${fmt(c.sv, 3)}</div>
          <div class="lens"><span class="sg plus">+</span>${toksHTML(c.up_tokens)}</div><div class="lens"><span class="sg minus">−</span>${toksHTML(c.down_tokens)}</div>`);
      });
      out.push(`${noteHTML(t("このヘッドが残差に書ける向き(W_O の列ブロック)を特異値分解し、主成分を出力層に通したときに上がる語(+)と下がる語(−)。主成分の符号は任意なので、+ と − は入れ替わりうる。"), "muted note")}`);
    } else out.push(`<h4>W_O logit lens</h4><p class="muted">${t("計算中…")}</p>`);
    out.push(d.gain_k && d.gain_k.length ? `<p class="muted note mono">${t("当てたことのある gain")}: ${d.gain_k.map((k) => `×${fmt(k, 2)}`).join(" ")}</p>`
      : `${noteHTML(t("このヘッドの gain はまだ当てていません(上の「×0 レシピ」「×2 レシピ」から 3 列チャットや eval で試せます)"), "muted note")}`);
    return out.join("");
  }
}

// ヒートマップの説明の行のトークン(空白・改行を見える形に)
function tokHTMLs(s) { return ` <span class="tok">${esc(tokText(s, 24))}</span>`; }
