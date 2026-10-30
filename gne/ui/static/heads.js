// HEAD パネル(CHANGE-04 G): attention ヘッドを 1 つ開く、NEURON パネルと同列の右パネル。タブは Pattern / Contrast / Details。
// Pattern: 文章を流して、そのヘッドのトークン × トークンの注目(行 = 見ている位置、列 = 見られている位置)を Canvas 2D で描く。
//   チャットの形でペルソナ(system)あり/なしを比べると、共通の末尾(ユーザーの発話と返事の始まり)で差を出す
//   (system の部分は 1 列にまとめる。gne heads contrast の pattern_shift と同じまとめ方)。
//   トリガー(部分文字列、"re:" で正規表現)の列に印を付け、トリガー区間への注目を表にする。
// Contrast: 保存済みのヘッドの対照分析(gne heads contrast)ごとの指標・順位と床、上位のヘッド、対照集合の会話を Pattern で見るリンク。
// Details: W_O の列ブロックの logit lens(主成分 2 本の上位・下位の語)、当てたことのある gain、大きさ。
// Listen: system + ユーザーの発話 + 返事を流し、返事のトークンが選んだヘッド(複数可、例 14:H0-3)で文脈のどのトークンに
//   注目しているか(注目)と、各トークンの方向への寄与(注目 × ゲート × OV、gne heads listen)を文章の上に色で出す。
import { getJSON, postJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc, toksHTML } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const sgn = (x, d = 3) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
const pct = (x) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${(x * 100).toFixed(1)}%`);
const AMBER = [255, 196, 84], PLUS = [242, 210, 75], MINUS = [201, 79, 167], MASKED = [10, 12, 16];
const kindLabel = (k) => (k === "attn" ? "attention" : t("線形注意"));
// gne heads contrast の指標(heads.py の METRICS)。signed は符号つき(床と比べるのは絶対値)
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
  constructor(root, { onRecipe, inRecipe, onNamed, onOverlay, onGoto, directions }) {
    this.root = root;
    this.onRecipe = onRecipe;          // (L, h, k) → レシピに足す・外す
    this.inRecipe = inRecipe;          // (L, h) → レシピの中の倍率 k か null
    this.onNamed = onNamed;            // (names) → 名前の一覧が変わった
    this.onOverlay = onOverlay;        // (field) → 地図の色を heads:<name> に
    this.onGoto = onGoto;              // ({ layer, head }) → 別のヘッドを開く
    this.directions = directions || (() => []);   // 保存済みの方向の名前(Listen の「寄与」)
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
      <div class="tabs"><button data-tab="pattern">Pattern</button><button data-tab="contrast">Contrast${d.contrasts.length ? ` <span class="muted">${d.contrasts.length}</span>` : ""}</button><button data-tab="details">Details</button><button data-tab="listen">Listen</button></div>
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
    name.addEventListener("keydown", (ev) => { if (ev.key === "Enter") save(); });
    name.addEventListener("blur", save);
    for (const b of this.root.querySelectorAll("[data-hk]")) b.addEventListener("click", () => { this.onRecipe(d.layer, d.head, +b.dataset.hk); this.render(); });
    const body = this.root.querySelector("#h-tab");
    if (this.tab === "contrast") this.contrast(body);
    else if (this.tab === "details") body.innerHTML = this.details();
    else if (this.tab === "listen") this.listen(body);
    else this.pattern(body);
  }

  // ×0(消す)と ×2(倍)。同じ倍率でもう一度押すと外す。recipe タブで別の倍率にしたものは横に出す
  recipeButtons(d) {
    const cur = this.inRecipe(d.layer, d.head);
    const b = (k) => {
      const on = cur !== null && cur === k;
      return `<button data-hk="${k}" class="${on ? `on ${k < 1 ? "minus" : "plus"}` : ""}" title="${on ? t("もう一度押すとレシピから外す") : t("レシピに ×{k} で足す(倍率は recipe タブで変えられる)", { k })}">${
        on ? `${t("レシピ")} ×${k}` : t("×{k} レシピ", { k })}</button>`;
    };
    const other = cur !== null && cur !== 0 && cur !== 2 ? `<span class="muted mono h-cur" title="${t("レシピの中の倍率")}">${t("レシピ")} ×${fmt(cur, 2)}</span>` : "";
    return `${b(0)}${b(2)}${other}`;
  }

  // --- Pattern -------------------------------------------------------------------------------------
  pattern(el) {
    const p = this.pat, d = this.d;
    if (d.kind !== "attn") {
      el.innerHTML = `<p class="muted note">${t("線形注意(Gated DeltaNet など)の層には softmax の注目のパターンがありません。Details の W_O ブロックの lens と、Contrast の出力の差(out_shift・dir_cos)で見てください。")}</p>`;
      return;
    }
    const canCmp = p.chat && !!p.system.trim();
    el.innerHTML = `
      <div class="ch-input"><textarea id="hp-text" rows="3" placeholder="${t("文章を入れて「流す」(このヘッドがどのトークンを見ているか)")}">${esc(p.text)}</textarea>
        <button id="hp-run"${this.busy ? " disabled" : ""}>${t("流す")}</button></div>
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
      <p class="muted note">${t("割合 = その行の注目のうち、トリガーのトークンへのもの(比べるときは system の部分を除いた共通部分の中で。gne heads contrast の trigger_mass と同じ分母)。最後の行 = 返事の最初のトークンを決める位置。")}</p>`;
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
      const top = sm ? sm.rows.slice(0, 12).map((x, i) => `<tr data-go="${x.layer}:${x.head}" class="${x.layer === d.layer && x.head === d.head ? "sel" : ""}"><td>${i + 1}</td><td>L${x.layer} H${x.head}</td><td>${x.kind === "attn" ? "attn" : "linear"}</td><td>${fmt(x.z, 2)}</td><td>${fmt(x.null_pct, 2)}</td></tr>`).join("") : "";
      const samples = sm && sm.samples ? sm.samples.filter((s) => s.triggers && s.triggers.length) : [];
      return `<div class="h-con">
        <h4><b class="mono">${esc(c.name)}</b> <span class="mono">#${c.rank}/${c.n_ranked} · z ${fmt(c.z, 2)} · null ${fmt(c.null_pct, 2)}</span>
          <button class="small" data-ov="${esc(c.name)}" title="${t("地図の板の上辺のヘッドをこの対照分析の z で色付けする")}">${t("地図で見る")}</button></h4>
        <table class="kv h-met"><tr><th>${t("指標")}</th><th>${t("値")}</th><th title="${t("中立どうし(system を別の中立の文に替えただけ)で測った同じ指標の、全ヘッドの 95% 点")}">${t("床(対照 p95)")}</th><th></th></tr>${mrows}</table>
        <p class="muted note">${t("琥珀 = 床を超えた指標。z = 測れた指標ごとの対照に対する z の平均、null = 対照のヘッドの何割より大きいか。")}${c.direction ? ` ${t("方向 {d}", { d: esc(c.direction) })}` : ""}</p>
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
      <p class="muted note">${t("返事のトークンが、選んだヘッドで文脈のどのトークンに注目しているか。方向を選ぶと、文脈の各トークンがヘッドの出力を通してその方向にどれだけ寄与したか(注目 × ゲート × OV)も出す。")}</p>
      <textarea id="ls-system" class="hp-sys" rows="3" placeholder="${t("system(空なら system なし)")}">${esc(L.system)}</textarea>
      <div class="ch-input"><textarea id="ls-user" rows="2" placeholder="${t("ユーザーの発話")}">${esc(L.user)}</textarea>
        <button id="ls-run"${this.busy ? " disabled" : ""}>${t("流す")}</button></div>
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
      out.push(`<p class="muted note">${t("このヘッドが残差に書ける向き(W_O の列ブロック)を特異値分解し、主成分を出力層に通したときに上がる語(+)と下がる語(−)。主成分の符号は任意なので、+ と − は入れ替わりうる。")}</p>`);
    } else out.push(`<h4>W_O logit lens</h4><p class="muted">${t("計算中…")}</p>`);
    out.push(d.gain_k && d.gain_k.length ? `<p class="muted note mono">${t("当てたことのある gain")}: ${d.gain_k.map((k) => `×${fmt(k, 2)}`).join(" ")}</p>`
      : `<p class="muted note">${t("このヘッドの gain はまだ当てていません(上の「×0 レシピ」「×2 レシピ」から 3 列チャットや eval で試せます)")}</p>`);
    return out.join("");
  }
}

// ヒートマップの説明の行のトークン(空白・改行を見える形に)
function tokHTMLs(s) { return ` <span class="tok">${esc(tokText(s, 24))}</span>`; }
