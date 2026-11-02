// direction ビューの「JSON から方向とヘッドを作る」(head_pipeline のジョブ、Session.heads_pipeline)。
// 対照集合と同じ中身の JSON(name・user_prompts・system_a / system_b か recipe_a / recipe_b)を読み込むか貼り付け、
// ブラウザの中で形を確かめて要約を出し、設定(目標・ヘッド数・max_examples・positions・top_layers・hard)を添えてジョブを投げる。
// 終わったら(app.js の onJobFinished から)、または保存済みの結果(/api/pipelines)を開くと、段ごとに結果を出す:
// 1. 方向と基準層(層ごとの cohen_d)、2. 因果スクリーニング(層 → ヘッド、外したものと理由)、3. 最初の比率、
// 4. 効率での配り直し(η の表と、ヘッドの数ごとの組の比較。推す組に印。行からレシピに読み込む・eval する・ヘッドを開く)。
// 組の比較の下の「候補を作る」で heads_candidates のジョブを投げる(組 × 全体の倍率の返事を作り、名前を伏せて採点する。
// 結果は同じビューの「強さの候補」に出る)。
import { getJSON } from "./api.js";
import { LANG, t } from "./i18n.js";
import { esc } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const sgn = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
const short = (s, n = 80) => { const x = String(s ?? ""); return x.length > n ? `${x.slice(0, n - 1)}…` : x; };
const NAME_RE = /^[\w.-]+$/;
const PIPE_KEYS = ["positions", "score_positions", "target", "n_heads", "hard", "top_layers", "max_examples", "s_max", "direction_layer", "resume", "name"];
const DRAFT = "gne.pipeline.draft";               // 入力欄の下書き(このブラウザだけ)
// 設定の欄 → ジョブの引数(spec の pipeline に値があれば、読み込んだときに欄に入れる)
const FIELDS = [["target", "#pp-target"], ["n_heads", "#pp-nheads"], ["max_examples", "#pp-mx"], ["positions", "#pp-pos"],
                ["top_layers", "#pp-tl"], ["hard", "#pp-hard"]];
const DEFAULTS = { target: "", n_heads: "6,5,4", max_examples: 64, positions: "last", top_layers: 4, hard: 16 };
const ETA_SHOWN = 12;                             // η の表で最初に出す行の数(残りはボタンで)

// 「ひな形」で入れる小さな例(system で A / B を作る。レシピで作るときは recipe_a: ["14:8949:+1.0"] など)
export const TEMPLATE = {
  name: "pirate",
  system_a: "あなたは海賊です。いつも荒っぽい海賊の口調で話してください。",
  system_b: "あなたは丁寧で役に立つアシスタントです。",
  user_prompts: ["あなたは誰ですか?", "今日の予定を教えてください。", "好きな食べ物は何ですか?", "おすすめの本を教えてください。",
                 "Who are you?", "What do you think about the sea?", "Give me advice for my day.", "What is your favorite food?"],
  validate: { match: "だぜ|野郎|海賊|arr|matey", target: "a pirate" },
};

// 片側(a / b)の作り方の短い説明
function sideText(d, s) {
  const sys = d[`system_${s}`], rec = d[`recipe_${s}`], parts = [];
  if (rec !== undefined && rec !== null) {
    parts.push(Array.isArray(rec) ? t("レシピ(ユニット {n} 個: {u})", { n: rec.length, u: rec.slice(0, 3).join(", ") + (rec.length > 3 ? " …" : "") })
      : typeof rec === "object" ? t("レシピ(表、ユニット {n} 個)", { n: (rec.units || []).length }) : t("レシピ {r}", { r: rec }));
  }
  parts.push(sys ? `system「${short(sys)}」` : t("system なし"));
  const own = d[`prompts_${s}`];
  if (Array.isArray(own)) parts.push(t("この側だけの質問 {n} 個", { n: own.length }));
  return parts.join(" + ");
}

// 入力欄の文字列を確かめる。{ spec, errors, warns, summary }。サーバ(gne/reverse.py の ContrastSet.load と heads_pipeline)と同じ決まり
export function checkSpec(text) {
  const out = { spec: null, errors: [], warns: [], summary: null };
  if (!String(text || "").trim()) return out;
  let d;
  try { d = JSON.parse(text); } catch (e) { out.errors.push(t("JSON として読めません: {m}", { m: e.message })); return out; }
  if (!d || typeof d !== "object" || Array.isArray(d)) { out.errors.push(t("中身は表(オブジェクト)にしてください")); return out; }
  out.spec = d;
  if (!NAME_RE.test(String(d.name ?? ""))) out.errors.push(t("name は英数字・_・-・. だけの名前にしてください(方向と結果のファイル名になります)"));
  // 質問: 両側で同じなら user_prompts、側ごとに違うなら prompts_a / prompts_b(片側だけ書けばもう片側は user_prompts)
  // 1 問は文字列か、会話の履歴(user と assistant が交互で、user で始まり user で終わるメッセージのリスト)
  const isTurns = (q) => Array.isArray(q) && q.length > 0 && q.length % 2 === 1
    && q.every((m, i) => m && typeof m === "object" && m.role === (i % 2 ? "assistant" : "user") && typeof m.content === "string");
  const isList = (v) => Array.isArray(v) && v.every((u) => typeof u === "string" || isTurns(u));
  let listOk = true;
  for (const k of ["user_prompts", "prompts_a", "prompts_b"]) {
    if (d[k] !== undefined && d[k] !== null && !isList(d[k])) {
      out.errors.push(t("{k} は質問のリストです。1 問は文字列か、会話の履歴([{\"role\": \"user\", \"content\": …}, {\"role\": \"assistant\", …}, …] を user で始めて user で終える)", { k }));
      listOk = false;
    }
  }
  const pa = isList(d.prompts_a) ? d.prompts_a : isList(d.user_prompts) ? d.user_prompts : null;
  const pb = isList(d.prompts_b) ? d.prompts_b : isList(d.user_prompts) ? d.user_prompts : null;
  if (listOk && (!pa || !pb)) out.errors.push(t("質問がありません: 両側で同じなら user_prompts、側ごとに違うなら prompts_a と prompts_b を書いてください"));
  else if (pa && pb) {
    const n = Math.min(pa.length, pb.length);
    if (n < 4) out.errors.push(t("質問は 4 個以上にしてください(いま {n} 個)", { n }));
    else if (n < 16) out.warns.push(t("質問が {n} 個と少ないので、測った値のぶれが大きくなります(数十〜数百個を勧めます)", { n }));
  }
  const samePrompts = !!(pa && pb) && pa.length === pb.length && pa.every((u, i) => JSON.stringify(u) === JSON.stringify(pb[i]));
  if (pa && pb && !samePrompts) out.warns.push(t("A と B で質問が違います。方向には狙った違いのほかに、話題・長さ・言語の差も混ざるので、それ以外はできるだけそろえてください"));
  const hasRecipe = (d.recipe_a !== undefined && d.recipe_a !== null) || (d.recipe_b !== undefined && d.recipe_b !== null);
  const sysA = d.system_a !== undefined && d.system_a !== null, sysB = d.system_b !== undefined && d.system_b !== null;
  if (!hasRecipe && samePrompts && !sysA && !sysB) out.errors.push(t("A と B の作り方がありません: system_a か system_b(片方だけでも可)、recipe_a / recipe_b、または側ごとの質問 prompts_a / prompts_b を書いてください"));
  else if (!hasRecipe && samePrompts && d.system_a === d.system_b) out.errors.push(t("A と B が同じです(system も recipe も違いがありません)"));
  if (d.pipeline !== undefined && d.pipeline !== null) {
    if (typeof d.pipeline !== "object" || Array.isArray(d.pipeline)) out.errors.push(t("pipeline は表にしてください"));
    else {
      const bad = Object.keys(d.pipeline).filter((k) => !PIPE_KEYS.includes(k));
      if (bad.length) out.errors.push(t("pipeline に {k} は使えません", { k: bad.join(", ") }));
    }
  }
  const nMulti = [...(pa || []), ...(samePrompts ? [] : pb || [])].filter((q) => Array.isArray(q)).length;
  if (nMulti) out.warns.push(t("会話の履歴を書いた質問が {n} 個あります(最後の user の発話への返事の直前で測ります)", { n: nMulti }));
  if (pa && pb) out.summary = { n: samePrompts ? pa.length : `${pa.length} / ${pb.length}`, A: sideText(d, "a"), B: sideText(d, "b"), validate: d.validate && d.validate.match };
  return out;
}

// "16:H8:s15.65@evil" → { layer, head, s, direction }
const parseUnit = (u) => { const m = String(u).match(/^(\d+):H(\d+):s([^@]+)(?:@(.+))?$/i); return m ? { layer: +m[1], head: +m[2], s: +m[3], direction: m[4] || null } : null; };
const headChip = (L, h, extra = "") => `<a href="#" class="pp-head mono" data-head="${L}:${h}" title="${t("HEAD パネルで開く")}">L${L} H${h}${extra}</a>`;

// 層ごとの cohen_d の小さな棒グラフ(SVG)。基準層は白、cohen_d 最大の層は緑。棒にマウスを当てると値
function dChart(layers, base, best) {
  if (!layers || !layers.length) return "";
  const W = 720, H = 120, P = 26, n = layers.length, bw = (W - 2 * P) / n;
  const mx = Math.max(1e-9, ...layers.map((r) => r.cohen_d || 0));
  const every = Math.max(1, Math.ceil(n / 24));
  const bars = layers.map((r, i) => {
    const v = Math.max(0, r.cohen_d || 0), h = (v / mx) * (H - P - 14), x = P + i * bw;
    const col = r.layer === base ? "#f2f5f8" : r.layer === best ? "#6fcf8f" : "#e8b04a";
    const lab = i % every === 0 || r.layer === base ? `<text x="${x + bw / 2}" y="${H - 8}" text-anchor="middle">${r.layer}</text>` : "";
    return `<rect x="${x + bw * 0.12}" y="${H - P - h}" width="${bw * 0.76}" height="${h}" fill="${col}"><title>L${r.layer} cohen_d ${fmt(r.cohen_d, 2)} · cos ${fmt(r.cos_base, 3)} · A ${fmt(r.A, 1)} / B ${fmt(r.B, 1)}</title></rect>${lab}`;
  }).join("");
  return `<svg class="pp-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="cohen_d"><text x="${P}" y="10" class="ax">cohen_d ${fmt(mx, 1)}</text>${bars}</svg>`;
}

export class PipelinePanel {
  constructor(root, { onRun, onShowDirection, onOpenHead, onLoadProj, onEval, onOverlay, onOpenCandidates } = {}) {
    this.root = root;
    this.onRun = onRun;                          // (kind, params) → Promise<ジョブ | null>(jobs.js の start。見積もりの確認つき)
    this.onShowDirection = onShowDirection || (() => {});   // (名前) → 上の「方向を見る」でその方向を開く
    this.onOpenHead = onOpenHead || (() => {});   // ({ layer, head, tab }) → HEAD パネル
    this.onLoadProj = onLoadProj || (() => {});   // (名前, ["L:Hh:sS@NAME"], 方向) → レシピを置き換える
    this.onEval = onEval || (() => {});           // (params) → ジョブ eval
    this.onOverlay = onOverlay || (() => {});     // (field) → 地図の色(causal:<name>)
    this.onOpenCandidates = onOpenCandidates || (() => {});   // (名前) → 「強さの候補」で開く
    this.list = [];
    this.data = null;                            // 開いている結果(/api/pipelines/<name>)
    this.causal = null;                          // その因果スクリーニングの全部の行(/api/causal/<name>)
    this.jobId = null;
    this.candJobId = null;                       // 「候補を作る」で投げた heads_candidates のジョブ
    this.cf = { scales: "0.5,1,1.5,2,3", n_heads: "", max_prompts: 24, max_tokens: 120, sampling: "llama_server", seed: 0,
                judge_b: false, qa: "", qb: "", with_recipe: "" };   // 「候補を作る」の欄(結果を開き直しても残す)
    this.lastPipe = null;                        // 最後に欄に入れた spec の pipeline(変わったときだけ設定の欄を書き換える)
    root.innerHTML = `
      <p class="muted note pp-help">${t("1 つの JSON(対照集合と同じ中身)から、方向 → 基準層 → 因果スクリーニング → 強さの配分 を 1 回のジョブで回します。A と B は system_a / system_b(system の文)か recipe_a / recipe_b(レシピの名前・パス・ユニットのリスト)で作ります。片側の system を省くと system なしで読ませます。")}</p>
      <div class="pp-in">
        <label class="muted">${t("JSON ファイル")} <input type="file" id="pp-file" accept=".json,application/json"></label>
        <label class="muted" title="${t("用途別のテンプレート(configs/pipeline/)。選ぶと入力欄に入る")}">${t("ひな形")}
          <select id="pp-tpl-sel"><option value="small">${t("小さな例(海賊、system を両側に)")}</option></select></label>
        <button class="small" id="pp-tpl" title="${t("選んだひな形を入力欄に入れる")}">${t("入れる")}</button>
        <span class="muted note">${t("または下に貼り付け")}</span></div>
      <div id="pp-tpl-about" class="muted note"></div>
      <textarea id="pp-text" class="mono" rows="9" spellcheck="false" placeholder='{"name": "…", "system_a": "…", "system_b": "…", "user_prompts": ["…", "…", "…", "…"]}'></textarea>
      <div id="pp-check" class="note"></div>
      <div class="pp-set">
        <label class="muted" title="${t("配分で A 側の点数をここまで下げる(物差しの最後の層)。数か、B(B 側の平均まで = 消し切る)。空なら A と B の平均の真ん中(半分ほど消す)")}">${t("目標")} <input id="pp-target" size="7" placeholder="${t("真ん中")}" spellcheck="false"></label>
        <label class="muted" title="${t("比べる組のヘッドの数(カンマ区切り)。少ない組ほど 1 ヘッドの強さが大きくなる")}">${t("ヘッド数")} <input id="pp-nheads" size="8" value="${DEFAULTS.n_heads}" spellcheck="false"></label>
        <label class="muted" title="${t("止めて測る段(物差し・因果・配分)で使う質問の数(等間隔に間引く)。方向づくりには全部使う")}">max_examples <input id="pp-mx" type="number" min="4" max="4096" value="${DEFAULTS.max_examples}"></label>
        <label class="muted" title="${t("方向を作る位置。last = 質問の最後のトークン、response = 返事を書かせてその位置(遅い)")}">positions <select id="pp-pos"><option>last</option><option>response</option></select></label>
        <label class="muted" title="${t("層ごとのスクリーニングのあと、ヘッドを 1 個ずつ止めて測る層の数")}">top_layers <input id="pp-tl" type="number" min="1" max="64" value="${DEFAULTS.top_layers}"></label>
        <label class="muted" title="${t("効率を測るときの、点数が残りやすい質問の数")}">hard <input id="pp-hard" type="number" min="1" max="1024" value="${DEFAULTS.hard}"></label>
        <label class="muted" title="${t("同じ JSON・同じ設定で作り済みの方向と因果の結果があっても使い回さず、最初から作り直す(止めて流し直すときは外したままにすると配分からやり直せる)")}"><input type="checkbox" id="pp-fresh"> ${t("最初から作り直す")}</label></div>
      <div class="pp-go"><button id="pp-go" disabled>${t("方向とヘッドを作る")}</button> <span class="muted note" id="pp-status"></span></div>
      <h4>${t("これまでの実行")}</h4><div id="pp-past"><p class="muted note">${t("読み込み中…")}</p></div>
      <div id="pp-result"></div>`;
    this.$ = (s) => root.querySelector(s);
    const ta = this.$("#pp-text");
    try { ta.value = localStorage.getItem(DRAFT) || ""; } catch { /* 下書きが無くても使える */ }
    ta.addEventListener("input", () => this.check());
    this.$("#pp-file").addEventListener("change", async (ev) => {
      const f = ev.target.files && ev.target.files[0];
      if (!f) return;
      ta.value = await f.text();
      this.$("#pp-tpl-about").textContent = "";
      this.check();
    });
    this.templates = [];                         // 用途別のテンプレート(/api/pipeline-templates)
    this.tplText = null;                         // 最後に入れたひな形の文字列(書き換えていなければ確かめずに差し替える)
    const sel = this.$("#pp-tpl-sel");
    this.$("#pp-tpl").addEventListener("click", () => this.insertTemplate(sel.value, false));
    sel.addEventListener("change", () => this.insertTemplate(sel.value, true));
    getJSON("/api/pipeline-templates").then((list) => {
      this.templates = Array.isArray(list) ? list : [];
      for (const x of this.templates) {
        const o = document.createElement("option");
        o.value = x.file;
        o.textContent = (x[LANG] && x[LANG].title) || (x.ja && x.ja.title) || x.file;
        sel.appendChild(o);
      }
    }).catch(() => { /* テンプレートが無くても小さな例は使える */ });
    this.$("#pp-go").addEventListener("click", () => this.run());
    this.check();
  }

  status(s, err = false) { const el = this.$("#pp-status"); el.textContent = s; el.classList.toggle("err", err); }

  // ひな形を入力欄に入れる(key = "small" か configs/pipeline/ のファイル名)。選び直したときは、書きかけの JSON を消す前に確かめる
  insertTemplate(key, ask) {
    const ta = this.$("#pp-text"), about = this.$("#pp-tpl-about");
    const x = this.templates.find((y) => y.file === key);
    const spec = x ? x.spec : TEMPLATE;
    const cur = ta.value.trim();
    if (ask && cur && cur !== this.tplText && !window.confirm(t("入力欄の JSON をひな形で置き換えます。よいですか?"))) return;
    ta.value = JSON.stringify(spec, null, 2);
    this.tplText = ta.value.trim();
    const a = x ? (x[LANG] && x[LANG].text) || (x.ja && x.ja.text) || "" : "";
    about.textContent = a;
    about.title = x ? `configs/pipeline/${x.file}` : "";
    this.check();
  }

  // 入力欄を確かめて要約を出す。spec の pipeline が変わったら、その値を設定の欄に入れる
  check() {
    const text = this.$("#pp-text").value;
    try { localStorage.setItem(DRAFT, text); } catch { /* 置けなくても使える */ }
    const c = checkSpec(text);
    this.checked = c;
    const el = this.$("#pp-check");
    if (!c.spec && !c.errors.length) { el.innerHTML = `<span class="muted">${t("JSON を読み込むか貼り付けてください(「ひな形」で用途別の例を選べます)")}</span>`; }
    else {
      const s = c.summary;
      el.innerHTML = `${s ? `<div class="pp-sum"><b>${esc(String(c.spec.name ?? ""))}</b> · ${t("質問 {n} 個", { n: s.n })}<br>A: ${esc(s.A)}<br>B: ${esc(s.B)}${
        s.validate ? `<br><span class="muted">${t("判定(validate.match)")}: <span class="mono">${esc(short(s.validate, 100))}</span></span>` : ""}</div>` : ""}${
        c.errors.map((x) => `<div class="err">${esc(x)}</div>`).join("")}${c.warns.map((x) => `<div class="pp-warn">${esc(x)}</div>`).join("")}`;
    }
    const pipe = c.spec && c.spec.pipeline && typeof c.spec.pipeline === "object" ? c.spec.pipeline : null;
    const key = JSON.stringify(pipe || {});
    if (c.spec && key !== this.lastPipe) {
      this.lastPipe = key;
      for (const [k, sel] of FIELDS) {
        const v = pipe && pipe[k] !== undefined && pipe[k] !== null ? pipe[k] : DEFAULTS[k];
        this.$(sel).value = Array.isArray(v) ? v.join(",") : String(v);
      }
    }
    this.$("#pp-go").disabled = !c.spec || c.errors.length > 0;
  }

  // ジョブの引数。欄の値が読めなければ null(理由を出す)
  params() {
    const c = this.checked;
    if (!c || !c.spec || c.errors.length) return null;
    const nh = this.$("#pp-nheads").value.split(/[\s,]+/).filter(Boolean).map(Number);
    if (!nh.length || !nh.every((x) => Number.isInteger(x) && x >= 1)) { this.status(t("ヘッド数は 1 以上の整数をカンマ区切りで(例 6,5,4)"), true); return null; }
    const num = (sel, lo) => { const v = Number(this.$(sel).value); return Number.isInteger(v) && v >= lo ? v : null; };
    const mx = num("#pp-mx", 4), tl = num("#pp-tl", 1), hard = num("#pp-hard", 1);
    if (mx === null || tl === null || hard === null) { this.status(t("max_examples・top_layers・hard は整数で(max_examples は 4 以上)"), true); return null; }
    const p = { spec: c.spec, n_heads: nh, max_examples: mx, positions: this.$("#pp-pos").value, top_layers: tl, hard };
    if (this.$("#pp-fresh").checked) p.resume = false;
    const tg = this.$("#pp-target").value.trim();
    if (tg) {
      if (/^b$/i.test(tg)) p.target = "B";
      else if (/^mid$/i.test(tg)) p.target = "mid";
      else if (!Number.isFinite(Number(tg))) { this.status(t("目標は数か B(B 側の平均)で(空なら A と B の平均の真ん中)"), true); return null; }
      else p.target = Number(tg);
    }
    return p;
  }

  async run() {
    const p = this.params();
    if (!p) return;
    this.$("#pp-go").disabled = true;
    this.status(t("ジョブを投げています…"));
    try {
      const j = await this.onRun("head_pipeline", p);
      if (!j) this.status(t("やめました"));
      else {
        this.jobId = j.id;
        this.status(t("ジョブを投げました。進み具合は下のジョブのバーに出ます(終わるとここに結果を出します)"));
      }
    } catch (e) { this.status(`${t("エラー")}: ${e.message}`, true); }
    finally { this.$("#pp-go").disabled = !this.checked || !this.checked.spec || this.checked.errors.length > 0; }
  }

  // 「残りから追加」: その組で残った質問だけで、追加のヘッドを探す(heads_refine のジョブ)
  async runRefine(name, nHeads, btn) {
    try {
      const j = await this.onRun("heads_refine", { name, n_heads: nHeads });
      if (!j) return;
      this.refineJobId = j.id;
      if (btn) { btn.textContent = t("探しています…"); btn.disabled = true; }
      this.status(t("{n} ヘッドの組で残った質問から、追加のヘッドを探すジョブを投げました(終わると新しい結果を開きます)", { n: nHeads }));
    } catch (e) { this.status(`${t("エラー")}: ${e.message}`, true); }
  }

  // 「残りから追加」で作った結果の説明と、元の組と合わせた組(レシピに読み込む・eval する)
  refineNote(d) {
    const r = d.refine;
    if (!r) return "";
    const rm = r.remaining || {}, c = r.combined;
    return `<div class="pp-refine note"><p>${t("{base} の {n} ヘッドの組で目標まで下がらなかった {k} 問(A 側 {nA} 問のうち)から作った、追加のヘッドの結果です。A と B の両側に元の組を当てたまま、方向を作り直しています。",
      { base: esc(r.base), n: r.n_heads, k: rm.n ?? "—", nA: rm.n_a ?? "—" })}${rm.padded ? ` ${t("(残りが少ないので、点数の高い順に {u} 問で作りました)", { u: rm.used })}` : ""}</p>
      ${c ? `<p><b>${t("元の組と合わせた組({m} ヘッド)", { m: c.n_heads })}</b>${c.chosen_by === "fewest_remaining" ? ` <span class="muted">${t("(目標に届いた組が無いので、合わせた残りが最も少ない組)")}</span>` : ""}: ${t("残り {a} → {b} 問", { a: c.remain_before, b: c.remain_after })} ·
        ${t("A 平均 {a} → {b}", { a: fmt(c.a_mean_before, 2), b: fmt(c.a_mean_after, 2) })} · ${t("B ⊥ {a} → {b}", { a: fmt(c.b_drift_orth_base, 3), b: fmt(c.b_drift_orth, 3) })}
        <button class="small" id="pp-rf-load" title="${t("元の組と追加の組を合わせたユニットで今のレシピを置き換える")}">${t("合わせた組をレシピに読み込む")}</button>
        <button class="small" id="pp-rf-eval" title="${t("合わせた組のレシピで eval を投げる(倍率 0 と 1、ランダムな対照 8 本)")}">${t("合わせた組で eval する")}</button></p>`
        : `<p class="muted">${t("追加の組を作れなかったので、合わせた組はありません。")}</p>`}
      <p class="muted">${t("下の比較表の「元の組と合わせて読み込む」「合わせて eval する」は、どの行も元の組とその行の組を合わせて使います(追加の組は元の組を当てた状態で探したので、単独では使いません)。")}</p></div>`;
  }

  // app.js の onJobFinished から(このパネルで投げたものでなくても一覧は取り直す)
  async jobEnded(j) {
    const mine = j.id === this.jobId || (j.kind === "heads_refine" && j.id === this.refineJobId);
    if (j.status === "done") {
      const name = j.result && j.result.name;
      if (mine) this.status(t("できました({s} 秒)", { s: fmt(j.seconds, 0) }));
      await this.refreshList();
      if (name && (mine || !this.data)) await this.open(name);
    } else if (mine) {
      this.status(j.status === "error" ? `${t("失敗")}: ${(j.error && j.error.message) || ""}` : t("止めました"), j.status === "error");
    }
  }

  async refreshList() {
    try { this.list = await getJSON("/api/pipelines"); } catch { this.list = []; }
    const el = this.$("#pp-past");
    if (!this.list.length) { el.innerHTML = `<p class="muted note">${t("まだありません")}</p>`; return; }
    el.innerHTML = `<table class="evx mono pp-past"><thead><tr><th class="l">${t("名前")}</th><th>${t("質問")}</th><th>${t("推す組")}</th><th>${t("目標")}</th><th>${t("時間")}</th><th class="l">${t("更新")}</th></tr></thead><tbody>${
      this.list.map((x) => `<tr data-pp="${esc(x.name)}" class="${this.data && this.data.name === x.name ? "on" : ""}"><td class="l">${esc(x.name)}</td><td>${x.n_prompts ?? "—"}</td>
        <td>${x.recommended ? t("{n} ヘッド", { n: x.recommended.n_heads }) : "—"}</td><td>${fmt(x.target, 2)}</td><td>${x.seconds ? t("{n} 秒", { n: Math.round(x.seconds) }) : "—"}</td>
        <td class="l">${esc((x.updated || "").replace("T", " ").slice(0, 16))}</td></tr>`).join("")}</tbody></table>`;
    for (const tr of el.querySelectorAll("tr[data-pp]")) tr.addEventListener("click", () => this.open(tr.dataset.pp));
  }

  async open(name) {
    const out = this.$("#pp-result");
    out.innerHTML = `<p class="muted">${t("読み込み中…")}</p>`;
    let d, c = null;
    try {
      let cl;
      [d, c, cl] = await Promise.all([getJSON(`/api/pipelines/${encodeURIComponent(name)}`),
                                      getJSON(`/api/causal/${encodeURIComponent(name)}`).catch(() => null),
                                      getJSON("/api/candidates").catch(() => [])]);
      this.candSaved = cl.some((x) => x.name === name);
    } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    this.data = d;
    this.causal = c;
    for (const tr of this.$("#pp-past").querySelectorAll("tr[data-pp]")) tr.classList.toggle("on", tr.dataset.pp === name);
    this.render();
  }

  render() {
    const d = this.data, out = this.$("#pp-result");
    if (!d) { out.innerHTML = ""; return; }
    const sides = d.sides || {};
    const side = (x) => (x ? [x.recipe ? (Array.isArray(x.recipe) ? t("レシピ(ユニット {n} 個: {u})", { n: x.recipe.length, u: x.recipe.slice(0, 3).join(", ") })
      : typeof x.recipe === "object" ? t("レシピ(表)") : t("レシピ {r}", { r: x.recipe })) : null, x.system ? `system「${short(x.system)}」` : t("system なし")].filter(Boolean).join(" + ") : "—");
    out.innerHTML = `
      <div class="pp-res">
      <h3>${t("{name} の結果", { name: esc(d.name) })} <span class="muted note mono">${esc((d.updated || "").replace("T", " ").slice(0, 16))} · ${t("{n} 秒", { n: Math.round(d.seconds || 0) })}</span></h3>
      <p class="note">${t("質問 {n} 個", { n: d.n_prompts ?? "—" })} · A: ${esc(side(sides.A))} · B: ${esc(side(sides.B))}</p>
      ${this.refineNote(d)}
      ${(d.reused || []).length ? `<p class="note muted">${t("同じ JSON・同じ設定で作り済みの{w}を使い回しました(作り直すときは「最初から作り直す」に印を付けて流し直す)",
        { w: d.reused.map((x) => (x === "direction" ? t("方向") : x === "causal" ? t("因果の結果") : x)).join(t("と")) })}</p>` : ""}
      <p class="note">${t("目標: L{L} の A 側の点数を {v} まで({src})", { L: (d.causal && (d.causal.score_layers || []).slice(-1)[0]) ?? "—", v: fmt(d.target, 2),
        src: esc(d.target_from === "settings" ? t("設定") : d.target_from ? t(d.target_from) : "—") })}${d.recommended ? ` · <b>${t("推す組: {n} ヘッド", { n: d.recommended.n_heads })}</b>` : ` · ${t("推す組なし(どの組も目標に届かない)")}`}</p>
      <div class="pp-actions"><button class="small" id="pp-showdir" title="${t("上の「方向を見る」で、この方向の層ごとの点数と分布を見る")}">${t("方向を見る")}</button>
        <button class="small" id="pp-map" title="${t("地図の板の上辺のヘッドを、止めたときの下がり方で色付けする")}">${t("地図: 効き")}</button>
        ${d.spec ? `<button class="small" id="pp-respec" title="${t("この実行に使った JSON を入力欄に戻す(設定を変えて流し直すため)")}">${t("入力欄に戻す")}</button>` : ""}</div>
      ${this.stage1(d)}${this.stage2(d)}${this.stage3(d)}${this.stage4(d)}</div>`;
    const $ = (s) => out.querySelector(s);
    $("#pp-showdir").addEventListener("click", () => this.onShowDirection(d.name));
    $("#pp-map").addEventListener("click", () => this.onOverlay(`causal:${d.name}`));
    const rs = $("#pp-respec");
    if (rs) {
      rs.addEventListener("click", () => {                              // JSON と、その実行の設定を欄に戻す
        this.$("#pp-text").value = JSON.stringify({ ...d.spec, pipeline: undefined }, null, 2);
        this.check();
        const st = d.settings || {};
        for (const [k, sel] of FIELDS) if (st[k] !== undefined) this.$(sel).value = st[k] === null ? "" : Array.isArray(st[k]) ? st[k].join(",") : String(st[k]);
        this.$("#pp-text").scrollIntoView({ block: "center" });
      });
    }
    for (const a of out.querySelectorAll("[data-head]")) {
      a.addEventListener("click", (ev) => { ev.preventDefault(); const [L, h] = a.dataset.head.split(":").map(Number); this.onOpenHead({ layer: L, head: h, tab: "causal" }); });
    }
    this.wireCand(out, d);
    const more = $("#pp-eta-more");
    if (more) more.addEventListener("click", () => { for (const tr of out.querySelectorAll("table.pp-eta tr.pp-more")) tr.classList.remove("hidden"); more.remove(); });
    const table = (d.allocate && d.allocate.table) || [];
    const crows = (d.refine && d.refine.combined_rows) || [];
    const crow = (n) => crows.find((c) => String(c.n_new) === String(n));
    for (const b of out.querySelectorAll("[data-load]")) {
      b.addEventListener("click", () => {
        const c = crow(b.dataset.load);
        if (c) { this.onLoadProj(`${d.name}-n${c.n_new}-combined`, c.units, d.name); return; }   // 残りから追加: 元の組と合わせて
        const r = table.find((x) => String(x.n_heads) === b.dataset.load);
        if (r) this.onLoadProj(`${d.name}-n${r.n_heads}`, r.units, d.name);
      });
    }
    for (const b of out.querySelectorAll("[data-refine]")) {
      b.addEventListener("click", () => this.runRefine(d.name, +b.dataset.refine, b));
    }
    const rf = d.refine && d.refine.combined;
    const rl = $("#pp-rf-load"), re = $("#pp-rf-eval");
    if (rf && rl) rl.addEventListener("click", () => this.onLoadProj(`${d.name}-combined`, rf.units, d.name));
    if (rf && re) {
      re.addEventListener("click", () => {
        this.onEval({ recipe: rf.recipe, doses: [0, 1], random_controls: 8 });
        re.textContent = t("eval を投げました");
        re.disabled = true;
      });
    }
    for (const b of out.querySelectorAll("[data-eval]")) {
      b.addEventListener("click", () => {
        const r = table.find((x) => String(x.n_heads) === b.dataset.eval);
        if (!r || !r.recipe) return;
        const c = crow(b.dataset.eval);
        this.onEval({ recipe: c ? c.recipe : r.recipe, doses: [0, 1], random_controls: 8 });
        b.textContent = t("eval を投げました");
        b.disabled = true;
      });
    }
  }

  // 「候補を作る」: 組 × 全体の倍率で返事を作り、名前を伏せて採点する(heads_candidates のジョブ)
  candForm(d) {
    const f = this.cf, ok = (d.allocate && (d.allocate.table || []).some((r) => !r.skipped));
    if (!ok) return "";
    const sp = [["llama_server", t("llama-server の既定")], ["common", t("よく使う設定")], ["greedy", "greedy"]];
    return `<div class="pp-cand"><h4 class="pp-sub">${t("強さの候補を作る(名前を伏せて採点する)")}</h4>
      <p class="muted note">${t("比較表の組それぞれに全体の倍率を何段か掛けて、A 側(振る舞いが出ている状態)と B 側(ふつうの状態)で返事を作ります。正の倍率は A 側で振る舞いを消す向き(A 側の返事を採点)、負の倍率は符号を返して B 側で強める向き(B 側の返事を採点)。無介入(base)は必ず入ります。どの候補の返事か伏せて採点し、候補ごとに集計します。調整に使っていない質問で作ると、強さを決めるのに使えます。")}</p>
      <div class="pp-set">
        <label class="muted" title="${t("カンマ区切り。正 = A 側で消す向き、負 = B 側で強める向き。0 は使えない(無介入は base)")}">${t("倍率")} <input id="pc-scales" size="16" value="${esc(f.scales)}" spellcheck="false"></label>
        <label class="muted" title="${t("比較表のどの組を使うか(ヘッド数、カンマ区切り)。空なら全部")}">${t("組(ヘッド数)")} <input id="pc-nheads" size="6" value="${esc(f.n_heads)}" placeholder="${t("全部")}" spellcheck="false"></label>
        <label class="muted" title="${t("各側の質問の数の上限(多ければ等間隔に間引く)")}">max_prompts <input id="pc-mp" type="number" min="1" max="1000" value="${f.max_prompts}"></label>
        <label class="muted" title="${t("返事の長さの上限(トークン)")}">max_tokens <input id="pc-mt" type="number" min="1" max="2048" value="${f.max_tokens}"></label>
        <label class="muted" title="${t("全部の候補(無介入も)で同じ種を使うので、違いは介入だけになる")}">${t("サンプリング")} <select id="pc-sp">${sp.map(([k, l]) => `<option value="${k}"${f.sampling === k ? " selected" : ""}>${esc(l)}</option>`).join("")}</select></label>
        <label class="muted">seed <input id="pc-seed" type="number" min="0" value="${f.seed}"></label>
        <label class="muted" title="${t("全部の候補の両側の返事を採点する(既定は、消す向きは A 側、強める向きは B 側だけ)")}"><input type="checkbox" id="pc-jb"${f.judge_b ? " checked" : ""}> ${t("両側を採点する")}</label></div>
      <div class="pp-cand-q"><textarea id="pc-qa" rows="4" spellcheck="false" placeholder="${t("A 側の質問(調整に使っていないもの。1 行に 1 つ。空なら対照集合の質問)")}">${esc(f.qa)}</textarea>
        <textarea id="pc-qb" rows="4" spellcheck="false" placeholder="${t("B 側の質問(空なら A と同じ質問)")}">${esc(f.qb)}</textarea></div>
      <div class="pp-set"><label class="muted" title="${t("全部の候補(無介入も)に重ねるレシピ(ニューロンの種火など)。保存したレシピの名前か .yaml のパス")}">${t("重ねるレシピ")} <input id="pc-with" size="28" value="${esc(f.with_recipe)}" placeholder="${t("なし")}" spellcheck="false"></label></div>
      <div class="pp-go"><button id="pc-go">${t("候補を作る")}</button>${this.candSaved ? ` <button class="small" id="pc-open">${t("保存した候補を開く")}</button>` : ""} <span class="muted note" id="pc-status">${esc(this.candStatus || "")}</span></div></div>`;
  }

  wireCand(out, d) {
    const go = out.querySelector("#pc-go");
    if (!go) return;
    const f = this.cf, q = (s) => out.querySelector(s);
    const bind = (sel, key, fn = (x) => x.value) => q(sel).addEventListener("input", (ev) => { f[key] = fn(ev.target); });
    bind("#pc-scales", "scales"); bind("#pc-nheads", "n_heads"); bind("#pc-mp", "max_prompts"); bind("#pc-mt", "max_tokens");
    bind("#pc-seed", "seed"); bind("#pc-qa", "qa"); bind("#pc-qb", "qb"); bind("#pc-with", "with_recipe");
    q("#pc-sp").addEventListener("change", (ev) => { f.sampling = ev.target.value; });
    q("#pc-jb").addEventListener("change", (ev) => { f.judge_b = ev.target.checked; });
    go.addEventListener("click", () => this.runCand(d));
    const op = q("#pc-open");
    if (op) op.addEventListener("click", () => this.onOpenCandidates(d.name));
  }

  candStatusSet(s, err = false) {
    this.candStatus = s;
    const el = this.$("#pc-status");
    if (el) { el.textContent = s; el.classList.toggle("err", err); }
  }

  // ジョブの引数。読めない欄があれば null(理由を出す)
  candParams(d) {
    const f = this.cf, nums = (s) => String(s).split(/[\s,]+/).filter(Boolean).map(Number);
    const scales = nums(f.scales);
    if (!scales.length || !scales.every((x) => Number.isFinite(x) && x !== 0)) { this.candStatusSet(t("倍率は 0 以外の数をカンマ区切りで(例 0.5,1,-1)"), true); return null; }
    const int = (v, lo, hi) => { const x = Number(v); return Number.isInteger(x) && x >= lo && x <= hi ? x : null; };
    const mp = int(f.max_prompts, 1, 100000), mt = int(f.max_tokens, 1, 2048), seed = int(f.seed, 0, 2 ** 31 - 1);
    if (mp === null || mt === null || seed === null) { this.candStatusSet(t("max_prompts・max_tokens・seed は整数で"), true); return null; }
    const p = { alloc: d.name, scales, max_prompts: mp, max_tokens: mt, sampling: f.sampling, seed, judge_b: !!f.judge_b };
    if (String(f.n_heads).trim()) {
      const nh = nums(f.n_heads);
      if (!nh.every((x) => Number.isInteger(x) && x >= 1)) { this.candStatusSet(t("組(ヘッド数)は整数をカンマ区切りで"), true); return null; }
      p.n_heads = nh;
    }
    const lines = (s) => String(s).split("\n").map((x) => x.trim()).filter(Boolean);
    const qa = lines(f.qa), qb = lines(f.qb);
    if (qb.length && !qa.length) { this.candStatusSet(t("B 側の質問だけでは作れません(A 側の質問も書いてください)"), true); return null; }
    if (qa.length) p.prompts = qb.length ? { A: qa, B: qb } : qa;
    if (String(f.with_recipe).trim()) p.with_recipe = String(f.with_recipe).trim();
    return p;
  }

  async runCand(d) {
    const p = this.candParams(d);
    if (!p) return;
    this.candStatusSet(t("ジョブを投げています…"));
    try {
      const j = await this.onRun("heads_candidates", p);
      if (!j) this.candStatusSet(t("やめました"));
      else { this.candJobId = j.id; this.candStatusSet(t("ジョブを投げました。進み具合は下のジョブのバーに出ます(終わると「強さの候補」で開きます)")); }
    } catch (e) { this.candStatusSet(`${t("エラー")}: ${e.message}`, true); }
  }

  // direction.js から(heads_candidates のジョブが終わった)
  candJobEnded(j) {
    if (j.id !== this.candJobId) return;
    if (j.status === "done") { this.candSaved = true; this.candStatusSet(t("できました({s} 秒)", { s: fmt(j.seconds, 0) })); }
    else this.candStatusSet(j.status === "error" ? `${t("失敗")}: ${(j.error && j.error.message) || ""}` : t("止めました"), j.status === "error");
  }

  // 1. 方向と基準層
  stage1(d) {
    const sc = d.score || {}, dir = d.direction || {}, nr = dir.n_rows || {};
    return `<section class="pp-stage"><h4>${t("1. 方向と基準層")}</h4>
      <p class="muted note">${t("A と B で同じ質問を読ませ、層ごとに残差の平均の差を方向にします(質問は全部使う)。cohen_d は、A と B の点数(残差 · 方向)がどれだけ離れているか。いちばん離れる層を基準層として記録します。")}</p>
      <p class="note mono">${t("基準層")} <b>L${sc.base_layer ?? "—"}</b> · ${t("cohen_d 最大")} L${sc.best_layer ?? "—"} · positions ${esc(dir.positions || "—")} · ${t("方向を作った例")} A ${nr.A ?? "—"} / B ${nr.B ?? "—"} · ${
        t("点数を測った例")} A ${(sc.n || {}).A ?? "—"} / B ${(sc.n || {}).B ?? "—"}</p>
      ${dChart(sc.layers, sc.base_layer, sc.best_layer)}
      <p class="muted note">${t("棒 = 層ごとの cohen_d(白 = 基準層、緑 = cohen_d 最大)。棒にマウスを当てると値。")}</p></section>`;
  }

  // 2. 因果スクリーニング: 層ごと(外した層と理由)→ ヘッドを 1 個ずつ(外したヘッドと理由)
  stage2(d) {
    const c = d.causal;
    if (!c) return "";
    const SL = c.score_layers || [], P = c.primary, k = (d.settings || {}).top_layers;
    const all = this.causal && this.causal.rows ? this.causal.rows.filter((r) => (r.mode || "ablate") === "ablate") : c.top || [];
    const measured = new Set(all.map((r) => r.layer));
    const lr = (c.layers || []).slice().sort((a, b) => (b.causal_max ?? -Infinity) - (a.causal_max ?? -Infinity));
    const broke = lr.filter((r) => r.excluded);
    const dropCells = (r) => SL.map((L) => `<td>${sgn((r.score_drop || {})[String(L)], 2)}</td>`).join("");
    const why = (r) => (r.excluded ? `<span class="pp-ex">${t("外した: モデルを壊す")}</span>` : "");
    const layerRows = lr.map((r) => `<tr class="${r.excluded ? "rnd" : measured.has(r.layer) ? "on" : ""}"><td>L${r.layer}</td>${dropCells(r)}<td>${fmt(r.causal_max, 2)}</td>
      <td>${fmt(r.drift_orth_b, 3)}</td><td class="l">${r.excluded ? why(r) : measured.has(r.layer) ? t("ヘッドを測った") : ""}</td></tr>`).join("");
    const heads = (c.top || []).map((r, i) => `<tr class="${r.excluded ? "rnd" : ""}"><td>${i + 1}</td><td class="l">${headChip(r.layer, r.head)}</td><td>${fmt(r.causal_max, 2)}</td>${dropCells(r)}
      <td>${fmt(r.contrib, 2)}</td><td>${fmt(r.ratio, 2)}</td><td>${sgn(r.b_side_delta, 2)}</td><td>${fmt(r.drift_orth_b, 3)}</td><td class="l">${why(r)}</td></tr>`).join("");
    const slh = SL.map((L) => `<th title="${t("止めたときの、L{L} での A 側の点数の下がり方", { L })}">L${L}${L === P ? "*" : ""}</th>`).join("");
    return `<section class="pp-stage"><h4>${t("2. 因果スクリーニング")}</h4>
      <p class="muted note">${t("層ごとにその層のヘッドを全部止め、後ろの層(物差しの層)で A 側の点数がどれだけ下がるかを測って、よく下がる層を選びます(top_layers = {k})。止めると中立の入力(B 側)の残差まで大きく動く層とヘッドは、効いているのではなくモデルを壊しているだけなので外します。選んだ層のヘッドを 1 個ずつ止めて、同じように下がり方(因果の効き目)を測ります。", { k: k ?? "—" })}</p>
      <p class="note mono">${t("物差しの層")} ${SL.map((L) => (L === P ? `<b>L${L}</b>` : `L${L}`)).join(" ")} · ${t("測った例")} A ${c.n_a ?? "—"} / B ${c.n_b ?? "—"} · ${t("ヘッドの行 {n}", { n: c.n_rows ?? "—" })} · ${
        t("ヘッドを測った層")} ${[...measured].sort((a, b) => a - b).map((L) => `L${L}`).join(", ") || "—"}${broke.length ? ` · ${t("外した層")} ${broke.map((r) => `L${r.layer}`).join(", ")}` : ""}</p>
      <h4 class="pp-sub">${t("止めたときの下がり方が大きいヘッド(上位 {n})", { n: (c.top || []).length })}</h4>
      <div class="evx-scroll"><table class="evx mono pp-top"><thead><tr><th>#</th><th class="l">${t("ヘッド")}</th><th title="${t("止めたときの下がり方の、物差しの層での最大")}">causal_max</th>${slh}
        <th title="${t("無介入でこのヘッドが主の物差しの層の方向に直接書く量")}">contrib</th><th title="${t("causal ÷ 直接の寄与。1 より大きい = 後段のユニットを通る間接の経路")}">ratio</th>
        <th title="${t("B 側の点数の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th><th></th></tr></thead><tbody>${heads}</tbody></table></div>
      ${lr.length ? `<details><summary class="muted note">${t("層ごとのスクリーニング({n} 層、層のヘッドを全部止めたとき)", { n: lr.length })}</summary>
        <div class="evx-scroll"><table class="evx mono pp-layers"><thead><tr><th>${t("層")}</th>${slh}<th>causal_max</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th><th></th></tr></thead><tbody>${layerRows}</tbody></table></div></details>` : ""}
      <p class="muted note">${t("* = 主の物差しの層。下がり方 = 止める前 − 後の A 側の点数(正 = A 側の向きに押していた)。")}</p></section>`;
  }

  // 3. 最初の比率
  stage3(d) {
    const a = d.allocate || {}, init = a.initial;
    const intro = `<p class="muted note">${t("前の段の因果の効き目(ヘッドを 1 個ずつ止めたときに、後ろの層の点数がどれだけ下がるか)が大きいヘッドを選び、強さはこの効き目に比例させます。全体の大きさ α だけを、返事を書かせない速い点数が目標に届くよう二分探索で決めます。")}</p>`;
    if (!init) return `<section class="pp-stage"><h4>${t("3. 最初の比率")}</h4>${intro}<p class="muted note">${t("(この実行には最初の組がありません)")}</p></section>`;
    const us = (init.heads || []).map(parseUnit).filter(Boolean);
    const mx = Math.max(1e-9, ...us.map((u) => u.s));
    const rows = us.map((u) => `<tr><td class="l">${headChip(u.layer, u.head)}</td><td>${fmt(u.s, 2)}</td><td>${fmt(u.s / mx, 2)}</td></tr>`).join("");
    return `<section class="pp-stage"><h4>${t("3. 最初の比率")}</h4>${intro}
      <p class="note mono">α ${fmt(init.alpha, 2)} · ${t("A 側の点数の平均 {v}(目標 {g})", { v: fmt(init.mean_a, 2), g: fmt(d.target, 2) })} · ${init.reached ? `<span class="ok">${t("目標に届いた")}</span>` : `<span class="ng">${t("届かない(強さの上限)")}</span>`}</p>
      <table class="evx mono pp-init"><thead><tr><th class="l">${t("ヘッド")}</th><th title="${t("射影の強さ s(= α × 比率)")}">s</th><th title="${t("効き目の比率(最大のヘッドを 1)")}">${t("比率")}</th></tr></thead><tbody>${rows}</tbody></table></section>`;
  }

  // 4. 効率での配り直し: η の表と、ヘッドの数ごとの組の比較
  stage4(d) {
    const a = d.allocate || {}, hard = a.hard || {}, S = a.score_layer;
    const ab = new Map();                                                // 丸ごと止めたときの効き目(配分の物差しの層)
    for (const r of (this.causal && this.causal.rows ? this.causal.rows.filter((x) => (x.mode || "ablate") === "ablate") : (d.causal || {}).top || [])) {
      ab.set(`${r.layer}:${r.head}`, (r.score_drop || {})[String(S)] ?? r.causal);
    }
    const etaRows = a.eta || [];
    const byAb = etaRows.map((r) => ab.get(`${r.layer}:${r.head}`)).filter((x) => Number.isFinite(x)).sort((x, y) => y - x);
    const abRank = (r) => { const v = ab.get(`${r.layer}:${r.head}`); return Number.isFinite(v) ? byAb.indexOf(v) + 1 : null; };
    const eta = etaRows.map((r, i) => {
      const rk = abRank(r), moved = rk !== null && Math.abs(rk - (i + 1)) >= 2;
      return `<tr class="${r.excluded ? "rnd" : ""}${i >= ETA_SHOWN ? " pp-more hidden" : ""}"><td>${i + 1}</td><td class="l">${headChip(r.layer, r.head)}</td><td>${fmt(r.eta, 3)}</td>
        <td>${fmt(ab.get(`${r.layer}:${r.head}`), 2)}</td><td class="${moved ? "pp-moved" : ""}">${rk ?? "—"}</td><td>${sgn(r.b_side_delta, 2)}</td><td>${fmt(r.drift_orth_b, 3)}</td>
        <td>${r.in_base ? "✓" : ""}</td><td class="l">${r.excluded ? `<span class="pp-ex">${t("外した: 中立の残差が大きく動く")}</span>` : ""}</td></tr>`;
    }).join("");
    const nA = a.n_a;
    const crows = (d.refine && d.refine.combined_rows) || [];             // 残りから追加の結果: 行は「元の組 + この組」で使う
    const crow = (n) => crows.find((c) => String(c.n_new) === String(n));
    const cmp = (a.table || []).slice().sort((x, y) => y.n_heads - x.n_heads).map((r) => {
      if (r.skipped) return `<tr><td>${r.n_heads}</td><td class="excl" colspan="9">${esc(r.skipped)}</td></tr>`;
      const chips = (r.heads || []).map((h) => headChip(h.layer, h.head, ` <span class="muted">s${fmt(h.s, 1)}</span>`)).join(" ");
      return `<tr class="${r.recommended ? "rec" : ""}" data-n="${r.n_heads}"><td>${r.n_heads}${r.recommended ? ` <span class="badge">${t("推奨")}</span>` : ""}</td><td class="units">${chips}</td>
        <td>${fmt(r.s_max, 2)}</td><td>${r.remain ?? "—"}${nA ? ` / ${nA}` : ""}</td><td>${fmt(r.a_max, 2)}</td><td>${fmt(r.a_mean, 2)}</td><td>${sgn(r.b_side_delta, 2)}</td><td>${fmt(r.b_drift_orth, 3)}</td>
        <td class="${r.reached ? "ok" : "ng"}">${r.reached ? "✓" : "×"}</td>
        <td class="l">${crow(r.n_heads) ? `<div class="muted note">${t("元の組と合わせると: 残り {a} → {b} 問 · B ⊥ {c}", { a: crow(r.n_heads).remain_before, b: crow(r.n_heads).remain_after, c: fmt(crow(r.n_heads).b_drift_orth, 3) })}</div>` : ""}
          <button class="small" data-load="${r.n_heads}" title="${crow(r.n_heads) ? t("元の組とこの組を合わせたユニットで今のレシピを置き換える(この組は元の組を当てた状態で探したので、単独では使わない)") : t("この組(proj のユニット)で今のレシピを置き換える")}">${crow(r.n_heads) ? t("元の組と合わせて読み込む") : t("レシピに読み込む")}</button>
          <button class="small" data-eval="${r.n_heads}" title="${crow(r.n_heads) ? t("元の組とこの組を合わせたレシピで eval を投げる(倍率 0 と 1、ランダムな対照 8 本)") : t("この組のレシピで eval を投げる(倍率 0 と 1、ランダムな対照 8 本)")}"${r.recipe ? "" : " disabled"}>${crow(r.n_heads) ? t("合わせて eval する") : t("eval する")}</button>
          <button class="small" data-refine="${r.n_heads}" title="${t("この組を当てても目標まで下がらない質問(残り)だけで、組を当てたまま方向を作り直し、因果 → 配分で足すヘッドを探す。結果は新しい一括の結果(名前 _r1 …)として出る")}"${r.remain > 0 ? "" : " disabled"}>${t("残りから追加")}</button></td></tr>`;
    }).join("");
    const hp = hard.prompts || [], hs = hard.scores || [];
    return `<section class="pp-stage"><h4>${t("4. 効率での配り直し")}</h4>
      <p class="muted note">${t("最初の組を当てた状態で、点数が残りやすい質問を選び、ヘッドごとに強さを少しだけ上げて、その質問での下がり方(効率 η)を測ります。強さを効率に比例させて 0 から配り直し、もう一度二分探索で目標に合わせます。丸ごと止めたときの効き目と効率では順位が入れ替わるので、この段が要ります。ヘッドの数を減らした組も作って比べます。")}</p>
      ${hp.length ? `<details><summary class="muted note">${t("点数が残りやすい質問 {n} 個(L{L} の点数)", { n: hard.n ?? hp.length, L: hard.layer ?? S ?? "—" })}</summary>
        <table class="evx mono dv-ex"><tbody>${hp.map((q, i) => `<tr><td class="w">${esc(q)}</td><td>${fmt(hs[i], 2)}</td></tr>`).join("")}</tbody></table></details>` : ""}
      <h4 class="pp-sub">${t("効率 η(大きい順)")}</h4>
      <div class="evx-scroll"><table class="evx mono pp-eta"><thead><tr><th>#</th><th class="l">${t("ヘッド")}</th><th title="${t("s を Δs 上げたときの、点数が残りやすい質問での下がり方 ÷ Δs")}">η</th>
        <th title="${t("丸ごと止めたときの下がり方(L{L})", { L: S ?? "—" })}">${t("止めた効き目")}</th><th title="${t("丸ごと止めたときの効き目での順位。黄 = 効率の順位と 2 つ以上ずれている")}">${t("効き目の順位")}</th>
        <th title="${t("B 側の点数の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th><th title="${t("最初の組に入っていたヘッド")}">${t("最初の組")}</th><th></th></tr></thead><tbody>${eta}</tbody></table></div>
      ${etaRows.length > ETA_SHOWN ? `<button class="small" id="pp-eta-more">${t("残り {n} 行も見る", { n: etaRows.length - ETA_SHOWN })}</button>` : ""}
      <h4 class="pp-sub">${t("ヘッドの数ごとの組の比較")}</h4>
      <div class="evx-scroll"><table class="evx mono pp-cmp"><thead><tr><th>${t("ヘッド数")}</th><th class="l">${t("ヘッドと強さ")}</th><th>s max</th>
        <th title="${t("目標を超えて残った A 側の例の数")}">${t("残り")}</th><th title="${t("A 側の点数の最大")}">A max</th><th title="${t("A 側の点数の平均")}">${t("A 平均")}</th>
        <th title="${t("B 側の点数の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ(小さいほど他を壊さない)")}">B ⊥</th><th title="${t("点数が残りやすい質問の平均が目標に届いたか")}">${t("届いた")}</th><th></th></tr></thead><tbody>${cmp}</tbody></table></div>
      <p class="muted note">${t("推奨 = 目標に届いた組のうち、残る例の数が最も少ない組と同程度(A 側の 5% まで)で、B 側のずれが最小の 1.25 倍までの組の中で、ヘッドが最も少ない組。ヘッドの名前をクリックで HEAD パネル。")}</p>
      ${this.candForm(d)}</section>`;
  }
}
