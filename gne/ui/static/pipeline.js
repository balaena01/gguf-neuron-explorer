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
import { Steps } from "./steps.js";
import { noteHTML, askConfirm, aboutHTML } from "./ui.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const sgn = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
const short = (s, n = 80) => { const x = String(s ?? ""); return x.length > n ? `${x.slice(0, n - 1)}…` : x; };
const NAME_RE = /^[\w.-]+$/;
const PIPE_KEYS = ["positions", "score_positions", "target", "n_heads", "hard", "top_layers", "max_examples", "s_max", "direction_layer", "resume", "name",
                   "rank", "unit_kinds", "verify_top", "verify_sample", "verify_mode", "n_units", "check_replies", "check_tokens",
                   "score_layers", "select_layer", "candidate_layers", "always_layers", "include_score_layers", "same_reply", "loop_direction", "holdout"];
const DRAFT = "gne.pipeline.draft";               // 入力欄の下書き(このブラウザだけ)
// 設定の欄 → ジョブの引数(spec の pipeline に値があれば、読み込んだときに欄に入れる)
const FIELDS = [["target", "#pp-target"], ["n_heads", "#pp-nheads"], ["max_examples", "#pp-mx"], ["positions", "#pp-pos"],
                ["top_layers", "#pp-tl"], ["hard", "#pp-hard"], ["rank", "#pp-rank"], ["unit_kinds", "#pp-kinds"],
                ["n_units", "#pp-nunits"], ["verify_top", "#pp-vtop"], ["check_replies", "#pp-ckn"], ["check_tokens", "#pp-ckt"],
                ["holdout", "#pp-holdout"], ["loop_direction", "#pp-loopdir"]];
const DEFAULTS = { target: "", n_heads: "6,5,4", max_examples: 64, positions: "last", top_layers: 4, hard: 16, rank: "writes", unit_kinds: "heads",
                   n_units: "24,12,6", verify_top: 64, check_replies: 16, check_tokens: 512, holdout: false, loop_direction: "first" };
// 欄に値を入れる(チェックボックスは checked、並びはカンマ区切り。null・undefined は既定)
function setField(el, v, def) {
  const x = v === undefined || v === null || v === "" ? def : v;
  if (el.type === "checkbox") el.checked = !!x;
  else el.value = Array.isArray(x) ? x.join(",") : String(x ?? "");
}
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
  if (nMulti) out.warns.push(t("会話の履歴を書いた質問が {n} 個あります(最後の user の発話への応答の直前で測ります)", { n: nMulti }));
  if (pa && pb) out.summary = { n: samePrompts ? pa.length : `${pa.length} / ${pb.length}`, A: sideText(d, "a"), B: sideText(d, "b"), validate: d.validate && d.validate.match };
  return out;
}

// "16:H8:s15.65@evil" → { layer, head, s, direction }
const parseUnit = (u) => { const m = String(u).match(/^(\d+):H(\d+):s([^@]+)(?:@(.+))?$/i); return m ? { layer: +m[1], head: +m[2], s: +m[3], direction: m[4] || null } : null; };
// 部品の印: ヘッドは HEAD パネルへのリンク、ニューロン(L:Nn)は文字だけ
const unitChip = (u, extra = "") => (u.kind === "neuron" ? `<span class="mono">L${u.layer} N${u.head ?? u.index}${extra}</span>`
  : u.kind === "layer_attn" ? `<span class="mono" title="${t("層の注意の出力全体(attn_output、線形注意の層は ssm_out)")}">L${u.layer} attn${extra}</span>`
  : u.kind === "layer_mlp" ? `<span class="mono" title="${t("層の MLP の出力全体(ffn_down)")}">L${u.layer} ffn${extra}</span>`
  : headChip(u.layer, u.head ?? u.index, extra));
const KIND_LABEL = { attn: "注意ヘッド", linear: "線形注意のヘッド", neuron: "ニューロン", layer_attn: "層の注意全体", layer_mlp: "層の MLP 全体" };
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
  constructor(root, { onRun, onShowDirection, onOpenHead, onLoadProj, onEval, onOverlay, onOpenCandidates, isLoop } = {}) {
    this.isLoop = isLoop || (() => false);       // 同じ層を 2 回実行するモデル(ループの中の部品の方向の欄を出す)
    this.root = root;
    this.onRun = onRun;                          // (kind, params) → Promise<ジョブ | null>(jobs.js の start。見積もりの確認つき)
    this.onShowDirection = onShowDirection || (() => {});   // (名前) → 上の「方向を見る」でその方向を開く
    this.onOpenHead = onOpenHead || (() => {});   // ({ layer, head, tab }) → HEAD パネル
    this.onLoadProj = onLoadProj || (() => {});   // (名前, ["L:Hh:sS@NAME"], 方向) → レシピを置き換える
    this.onEval = onEval || (() => {});           // (params, 名前) → 評価の画面へ送る(そこで設定を見て、変えて、始める)
    this.onOverlay = onOverlay || (() => {});     // (field) → 地図の色(causal:<name>)
    this.onOpenCandidates = onOpenCandidates || (() => {});   // (名前) → 「強さの候補」で開く
    this.list = [];
    this.data = null;                            // 開いている結果(/api/pipelines/<name>)
    this.openSeq = 0;                            // open の通し番号(後から別の結果を開いたら、先に始めた読み込みの結果は捨てる)
    this.causal = null;                          // その因果スクリーニングの全部の行(/api/causal/<name>)
    this.jobId = null;
    this.candJobId = null;                       // 「候補を作る」で投げた heads_candidates のジョブ
    this.cf = { scales: "0.5,1,1.5,2,3", n_heads: "", max_prompts: 24, max_tokens: 120, sampling: "llama_server", seed: 0,
                judge_b: false, qa: "", qb: "", with_recipe: "" };   // 「候補を作る」の欄(結果を開き直しても残す)
    this.lastPipe = null;                        // 最後に欄に入れた spec の pipeline(変わったときだけ設定の欄を書き換える)
    root.innerHTML = `
      <div id="pp-input">
      ${noteHTML(t("1 つの JSON(対照集合と同じ中身)から、方向 → 基準層 → 候補づくり → 強さの配分 を 1 回のジョブで回します。候補づくりは、既定では全部の部品を方向への書き込みの量で並べます(「止めて選ぶ」で v0.7.0 までの選び方)。A と B は system_a / system_b(system の文)か recipe_a / recipe_b(レシピの名前・パス・ユニットのリスト)で作ります。片側の system を省くと system なしで読ませます。"))}
      <div class="pp-in">
        <label class="muted">${t("JSON ファイル")} <input type="file" id="pp-file" accept=".json,application/json"></label>
        <label class="muted" title="${t("用途別のテンプレート(configs/pipeline/)。選ぶと入力欄に入る")}">${t("ひな形")}
          <select id="pp-tpl-sel"><option value="">${t("(ひな形を選ぶ)")}</option><option value="small">${t("小さな例(海賊、system を両側に)")}</option></select></label>
        <span class="muted note">${t("または下に貼り付け")}</span></div>
      <div id="pp-tpl-about" class="muted note"></div>
      <textarea id="pp-text" class="mono" rows="9" spellcheck="false" placeholder='{"name": "…", "system_a": "…", "system_b": "…", "user_prompts": ["…", "…", "…", "…"]}'></textarea>
      <div id="pp-check" class="note"></div>
      </div>
      <div id="pp-shared"><p class="stage-shared-h">${t("共通の設定(方向とヘッドを作る が使う)")}</p>
      <div class="pp-set">
        <label class="muted" title="${t("配分で A 側の射影の値をここまで下げる(射影の値を測る最後の層)。数か、B(B 側の平均まで = 消し切る)。空なら A と B の平均の真ん中(半分ほど消す)")}">${t("射影の目標")} <input id="pp-target" size="7" placeholder="${t("真ん中")}" spellcheck="false"></label>
        <label class="muted" title="${t("射影の値を測る段(基準層・因果・配分)で使う質問の数(等間隔に間引く)。方向づくりには全部使う")}">${t("測る質問の数")} <input id="pp-mx" type="number" min="4" max="4096" value="${DEFAULTS.max_examples}"></label>
        <label class="muted" title="${t("方向を作る位置。last = 質問の最後のトークン、response = 応答を書かせてその位置(遅い)")}">${t("方向を作る位置")} <select id="pp-pos"><option value="last">${t("質問の最後")}</option><option value="response">${t("応答(遅い)")}</option></select></label>
        <label class="muted" title="${t("候補の探し方。書き込みの量で全部から(既定) = 全部の層の注意ヘッド・線形注意のヘッド(部品で選べばニューロンも)を方向への書き込みの量で並べ、上位を s=1 で確かめてから、s=1 で組に足す(足りないときだけ全体の強さを上げる)。止めて選ぶ = v0.7.0 までの選び方(層ごとに止めた効き目の上位の層のヘッドを 1 個ずつ止め、効率で配り直す)。応答が変わらないときは、もう一方も試す")}">${t("候補の探し方")} <select id="pp-rank"><option value="writes">${t("書き込みの量で全部から(既定)")}</option><option value="causal">${t("止めて選ぶ(v0.7.0 まで)")}</option></select></label>
        <label class="muted" title="${t("同じ JSON・同じ設定で作り済みの方向と因果の結果があっても使い回さず、最初から作り直す(止めて流し直すときは外したままにすると配分からやり直せる)")}"><input type="checkbox" id="pp-fresh"> ${t("最初から作り直す")}</label></div>
      <fieldset class="pp-set fx-part" id="pp-fs-writes" aria-describedby="pp-fs-writes-off"><legend class="muted note fx-part-h">${t("書き込みの量で全部から選ぶときの設定")} <span class="pp-fs-off" id="pp-fs-writes-off" hidden>${t("(「止めて選ぶ」では使いません)")}</span></legend>
        <label class="muted" title="${t("書き込みの量で全部から選ぶときの部品の種類")}">${t("部品")} <select id="pp-kinds"><option value="heads">${t("ヘッドだけ(既定)")}</option><option value="all">${t("ヘッド・線形注意・ニューロン")}</option><option value="heads,layers">${t("ヘッドと層全体")}</option><option value="all,layers">${t("ヘッド・ニューロン・層全体")}</option></select></label>
        <label class="muted" title="${t("比べる組の部品の数の上限(カンマ区切り、既定 24,12,6)。上限ごとに組を作り、表で比べる")}">${t("組の大きさの上限")} <input id="pp-nunits" size="8" value="${DEFAULTS.n_units}" spellcheck="false"></label>
        <details class="more" id="pp-more"><summary class="muted">${t("細かい設定")}</summary><div class="pp-set">
          <label class="muted" title="${t("点数の上位から、1 個ずつ s=1 で射影して流し直し、効き目を確かめる部品の数(既定 64)。多いほど時間がかかる")}">${t("確かめる部品の数")} <input id="pp-vtop" type="number" min="1" max="1024" value="${DEFAULTS.verify_top}"></label>
          <label class="muted" title="${t("選んだ組と無介入で応答を書かせ、validate.match に一致した数を数える質問の数(側ごと、既定 16)。0 で行わない")}">${t("応答での確かめの質問の数")} <input id="pp-ckn" type="number" min="0" max="256" value="${DEFAULTS.check_replies}"></label>
          <label class="muted" title="${t("応答での確かめで書かせる長さの上限(トークン、既定 512)")}">${t("上限のトークン数")} <input id="pp-ckt" type="number" min="16" max="4096" value="${DEFAULTS.check_tokens}"></label>
          <label class="muted" title="${t("質問を半分に分け、偶数番目で選び、奇数番目(選ぶのに使っていない質問)で表の数と応答の確かめを測る。各側 16 問以上が要る")}"><input type="checkbox" id="pp-holdout"> ${t("選ぶ質問と測る質問を分ける(各側 16 問以上)")}</label>
          <label class="muted" title="${t("同じ層を 2 回実行するモデル(sensenova_loop)で書き換えを両方の回に当てるとき、ループの中の部品を、1 回目の段(既定)と 2 回目の段のどちらの方向で測って射影するか")}">${t("ループの中の部品の方向")} <select id="pp-loopdir" aria-describedby="pp-loopdir-off"><option value="first">${t("1 回目の段(既定)")}</option><option value="second">${t("2 回目の段")}</option></select>
            <span class="muted note" id="pp-loopdir-off" hidden>${t("同じ層を 2 回実行するモデルだけ")}</span></label></div></details>
      </fieldset>
      <fieldset class="pp-set fx-part" id="pp-fs-causal" aria-describedby="pp-fs-causal-off"><legend class="muted note fx-part-h">${t("止めて選ぶ(v0.7.0 まで)ときの設定")} <span class="pp-fs-off" id="pp-fs-causal-off" hidden>${t("(「書き込みの量で全部から」では使いません)")}</span></legend>
        <label class="muted" title="${t("比べる組のヘッドの数(カンマ区切り)。少ない組ほど 1 ヘッドの強さが大きくなる")}">${t("ヘッド数")} <input id="pp-nheads" size="8" value="${DEFAULTS.n_heads}" spellcheck="false"></label>
        <label class="muted" title="${t("層ごとのスクリーニングのあと、ヘッドを 1 個ずつ止めて測る層の数")}">${t("ヘッドを測る層の数")} <input id="pp-tl" type="number" min="1" max="64" value="${DEFAULTS.top_layers}"></label>
        <label class="muted" title="${t("効率を測るときの、射影の値が残りやすい質問の数")}">${t("残りやすい質問の数")} <input id="pp-hard" type="number" min="1" max="1024" value="${DEFAULTS.hard}"></label>
      </fieldset>
      <p class="muted note" id="pp-prio">${t("欄の値が使われます。JSON の pipeline に同じ項目があれば、読み込んだときにその値が欄に入ります(欄を書き換えれば欄の値が優先)。")}</p>
      </div>
      <div class="pp-go" id="pp-run"><button class="main" id="pp-go" disabled>${t("方向とヘッドを作る")}</button> <span class="muted note" id="pp-status"></span></div>
      <div id="pp-past-wrap"><div id="pp-past"><p class="muted note">${t("読み込み中…")}</p></div></div>
      <p class="muted note" id="pp-nores">${t("まだ結果がありません。「1. 対照の文」で「方向とヘッドを作る」を押すか、「これまでの実行」から開いてください。")}</p>
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
    this.$("#pp-rank").addEventListener("change", () => this.rankFields());
    this.rankFields();
    // 左に手順、右に選んだ手順。結果は 1 回のジョブ(方向とヘッドを作る)の段ごとに分けて見せる
    const res = (sel) => () => (this.data ? [this.$("#pp-result .pp-res-head"), ...this.$$(`#pp-result ${sel}`)] : [this.$("#pp-nores")]);
    this.steps = new Steps(root, {
      shared: this.$("#pp-shared"), store: "gne.chips.step", label: t("CHiPS の手順"),
      steps: [{ key: "input", label: t("1. 対照の文"), els: () => [this.$("#pp-input"), this.$("#pp-run")], shared: true },
              { key: "dir", label: t("2. 方向と基準層"), els: res("section.pp-stage:nth-of-type(1)") },
              { key: "causal", label: t("3. 効くヘッドを探す"), els: res("section.pp-stage:nth-of-type(2)") },
              { key: "alloc", label: t("4. 配り方を決める"), els: res("section.pp-stage:nth-of-type(3), section.pp-stage:nth-of-type(4)") },
              { key: "cand", label: t("5. 強さの候補を作る"), els: res(".pp-cand") },
              { key: "past", label: t("これまでの実行"), els: () => [this.$("#pp-past-wrap")], sep: true }],
    });
    const check = this.check.bind(this);                            // 入力を確かめるたびに 1. の印も付け直す
    this.check = (...a) => { const r = check(...a); this.marks(); return r; };
    this.check();
  }

  $$(s) { return [...this.root.querySelectorAll(s)]; }

  // 手順の印(済・実行中・まだ)
  marks() {
    if (!this.steps) return;
    const run = (id) => !!(id && !(this.endedJobs || new Set()).has(id));
    const d = this.data, pipe = run(this.jobId) || run(this.refineJobId);
    const m = (k, running, done) => this.steps.mark(k, running ? "run" : done ? "done" : "todo");
    m("input", false, !this.$("#pp-go").disabled);
    m("dir", pipe, !!(d && d.direction));
    m("causal", pipe, !!(d && d.causal));
    m("alloc", pipe, !!(d && d.allocate));
    m("cand", run(this.candJobId), !!this.candSaved);
  }

  status(s, err = false) { const el = this.$("#pp-status"); el.textContent = s; el.classList.toggle("err", err); }

  // ひな形を入力欄に入れる(key = "small" か configs/pipeline/ のファイル名)。選び直したときは、書きかけの JSON を消す前に確かめる
  async insertTemplate(key, ask) {
    if (!key) return;
    const ta = this.$("#pp-text"), about = this.$("#pp-tpl-about");
    const x = this.templates.find((y) => y.file === key);
    const spec = x ? x.spec : TEMPLATE;
    const cur = ta.value.trim();
    if (ask && cur && cur !== this.tplText && !(await askConfirm(t("入力欄の JSON をひな形で置き換えます。よいですか?")))) return;
    ta.value = JSON.stringify(spec, null, 2);
    this.tplText = ta.value.trim();
    about.innerHTML = x ? aboutHTML((x[LANG] && x[LANG].text) ? x[LANG] : x.ja) : "";   // 要約と、節ごとにたたんだ説明
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
      for (const [k, sel] of FIELDS) setField(this.$(sel), pipe ? pipe[k] : undefined, DEFAULTS[k]);
      this.rankFields();
    }
    this.$("#pp-go").disabled = !c.spec || c.errors.length > 0;
  }

  // 候補の探し方に合わせて、使わない設定のまとまりを無効にする(値は残すが、ジョブには渡さない)。ループの中の部品の方向は、
  // 同じ層を 2 回実行するモデルのときだけ選べる
  rankFields() {
    const writes = this.$("#pp-rank").value === "writes";
    this.$("#pp-fs-writes").disabled = !writes;
    this.$("#pp-fs-causal").disabled = writes;
    this.$("#pp-fs-writes-off").hidden = writes;
    this.$("#pp-fs-causal-off").hidden = !writes;
    const loop = !!this.isLoop();
    this.$("#pp-loopdir").disabled = !loop;
    this.$("#pp-loopdir-off").hidden = loop;
  }

  // ジョブの引数。欄の値が読めなければ null(理由を出す)。候補の探し方で使う欄の値だけを渡す
  params() {
    const c = this.checked;
    if (!c || !c.spec || c.errors.length) return null;
    const ints = (sel) => this.$(sel).value.split(/[\s,]+/).filter(Boolean).map(Number);
    const num = (sel, lo) => { const v = Number(this.$(sel).value); return Number.isInteger(v) && v >= lo ? v : null; };
    const mx = num("#pp-mx", 4);
    if (mx === null) { this.status(t("測る質問の数は 4 以上の整数で"), true); return null; }
    const p = { spec: c.spec, max_examples: mx, positions: this.$("#pp-pos").value };
    if (this.$("#pp-fresh").checked) p.resume = false;
    p.rank = this.$("#pp-rank").value;                                   // writes(既定)/ causal(v0.7.0 まで)
    if (p.rank === "writes") {
      const nu = ints("#pp-nunits");
      if (!nu.length || !nu.every((x) => Number.isInteger(x) && x >= 1)) { this.status(t("組の大きさの上限は 1 以上の整数をカンマ区切りで(例 24,12,6)"), true); return null; }
      const vt = num("#pp-vtop", 1), ck = num("#pp-ckn", 0), ct = num("#pp-ckt", 16);
      if (vt === null || ck === null || ct === null) { this.status(t("確かめる部品の数は 1 以上、応答での確かめの質問の数は 0 以上、上限のトークン数は 16 以上の整数で"), true); return null; }
      Object.assign(p, { unit_kinds: this.$("#pp-kinds").value, n_units: nu, verify_top: vt, check_replies: ck, check_tokens: ct,
                         holdout: this.$("#pp-holdout").checked });
      if (this.isLoop()) p.loop_direction = this.$("#pp-loopdir").value;
    } else {
      const nh = ints("#pp-nheads");
      if (!nh.length || !nh.every((x) => Number.isInteger(x) && x >= 1)) { this.status(t("ヘッド数は 1 以上の整数をカンマ区切りで(例 6,5,4)"), true); return null; }
      const tl = num("#pp-tl", 1), hard = num("#pp-hard", 1);
      if (tl === null || hard === null) { this.status(t("ヘッドを測る層の数と残りやすい質問の数は 1 以上の整数で"), true); return null; }
      Object.assign(p, { n_heads: nh, top_layers: tl, hard });
    }
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
        this.marks();
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
      this.marks();
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
      { base: esc(r.base), n: r.n_heads, k: rm.n ?? "—", nA: rm.n_a ?? "—" })}${rm.padded ? ` ${t("(残りが少ないので、射影の値の高い順に {u} 問で作りました)", { u: rm.used })}` : ""}</p>
      ${c ? `<p><b>${t("元の組と合わせた組({m} ヘッド)", { m: c.n_heads })}</b>${c.chosen_by === "fewest_remaining" ? ` <span class="muted">${t("(目標に届いた組が無いので、合わせた残りが最も少ない組)")}</span>` : ""}: ${t("残り {a} → {b} 問", { a: c.remain_before, b: c.remain_after })} ·
        ${t("A 平均 {a} → {b}", { a: fmt(c.a_mean_before, 2), b: fmt(c.a_mean_after, 2) })} · ${t("B ⊥ {a} → {b}", { a: fmt(c.b_drift_orth_base, 3), b: fmt(c.b_drift_orth, 3) })}
        <button class="small" id="pp-rf-load" title="${t("元の組と追加の組を合わせたユニットで今のレシピを置き換える")}">${t("合わせた組でレシピを置き換える")}</button>
        <button class="small" id="pp-rf-eval" title="${t("合わせた組のレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)")}">${t("合わせた組を評価へ送る")}</button></p>`
        : `<p class="muted">${t("追加の組を作れなかったので、合わせた組はありません。")}</p>`}
      <p class="muted">${t("下の比較表の「元の組と合わせて読み込む」「合わせて eval する」は、どの行も元の組とその行の組を合わせて使います(追加の組は元の組を当てた状態で探したので、単独では使いません)。")}</p></div>`;
  }

  // app.js の onJobFinished から(このパネルで投げたものでなくても一覧は取り直す)
  async jobEnded(j) {
    (this.endedJobs || (this.endedJobs = new Set())).add(j.id);
    this.marks();
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
      this.list.map((x) => `<tr data-pp="${esc(x.name)}" class="pick${this.data && this.data.name === x.name ? " on" : ""}"><td class="l">${esc(x.name)}</td><td>${x.n_prompts ?? "—"}</td>
        <td>${x.recommended ? t("{n} ヘッド", { n: x.recommended.n_heads }) : "—"}</td><td>${fmt(x.target, 2)}</td><td>${x.seconds ? t("{n} 秒", { n: Math.round(x.seconds) }) : "—"}</td>
        <td class="l">${esc((x.updated || "").replace("T", " ").slice(0, 16))}</td></tr>`).join("")}</tbody></table>`;
    for (const tr of el.querySelectorAll("tr[data-pp]")) tr.addEventListener("click", () => this.open(tr.dataset.pp));
  }

  async open(name) {
    const tok = ++this.openSeq;
    const out = this.$("#pp-result");
    out.innerHTML = `<p class="muted">${t("読み込み中…")}</p>`;
    let d, c = null, cl;
    try {
      [d, c, cl] = await Promise.all([getJSON(`/api/pipelines/${encodeURIComponent(name)}`),
                                      getJSON(`/api/causal/${encodeURIComponent(name)}`).catch(() => null),
                                      getJSON("/api/candidates").catch(() => [])]);
    } catch (e) { if (tok === this.openSeq) out.innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    if (tok !== this.openSeq) return;              // 待つ間に別の結果を開いた(ジョブの終わりに開く結果と、行のクリックなど)
    this.candSaved = cl.some((x) => x.name === name);
    this.data = d;
    this.causal = c;
    if (this.steps && ["input", "past"].includes(this.steps.current)) this.steps.select("dir");
    for (const tr of this.$("#pp-past").querySelectorAll("tr[data-pp]")) tr.classList.toggle("on", tr.dataset.pp === name);
    this.render();
  }

  render() {
    this._render();
    if (this.steps) { this.steps.apply(); this.marks(); }
  }

  _render() {
    const d = this.data, out = this.$("#pp-result");
    if (!d) { out.innerHTML = ""; return; }
    const sides = d.sides || {};
    const side = (x) => (x ? [x.recipe ? (Array.isArray(x.recipe) ? t("レシピ(ユニット {n} 個: {u})", { n: x.recipe.length, u: x.recipe.slice(0, 3).join(", ") })
      : typeof x.recipe === "object" ? t("レシピ(表)") : t("レシピ {r}", { r: x.recipe })) : null, x.system ? `system「${short(x.system)}」` : t("system なし")].filter(Boolean).join(" + ") : "—");
    out.innerHTML = `
      <div class="pp-res"><div class="pp-res-head">
      <h3>${t("{name} の結果", { name: esc(d.name) })} <span class="muted note mono">${esc((d.updated || "").replace("T", " ").slice(0, 16))} · ${t("{n} 秒", { n: Math.round(d.seconds || 0) })}</span></h3>
      <p class="note">${t("質問 {n} 個", { n: d.n_prompts ?? "—" })} · A: ${esc(side(sides.A))} · B: ${esc(side(sides.B))}</p>
      <p class="note">${d.writes ? t("選び方: 書き込みの量で全部の部品から(--rank writes、既定)") : t("選び方: 止めて選ぶ(v0.7.0 までの選び方、--rank causal)")}${d.rank_note ? ` <span class="muted">${esc(d.rank_note)}</span>` : ""}</p>
      ${this.refineNote(d)}
      ${(d.reused || []).length ? `<p class="note muted">${t("同じ JSON・同じ設定で作り済みの{w}を使い回しました(作り直すときは「最初から作り直す」に印を付けて流し直す)",
        { w: d.reused.map((x) => (x === "direction" ? t("方向") : x === "causal" ? t("因果の結果") : x)).join(t("と")) })}</p>` : ""}
      <p class="note">${t("目標: L{L} の A 側の射影の値を {v} まで({src})", { L: (d.causal && (d.causal.score_layers || []).slice(-1)[0]) ?? "—", v: fmt(d.target, 2),
        src: esc(d.target_from === "settings" ? t("設定") : d.target_from ? t(d.target_from) : "—") })}${this.recText(d)}</p>
      <div class="pp-actions"><button class="small" id="pp-showdir" title="${t("上の「方向を見る」で、この方向の層ごとの射影の値と分布を見る")}">${t("方向を見る")}</button>
        ${d.writes ? "" : `<button class="small" id="pp-map" title="${t("地図の板の上辺のヘッドを、止めたときの下がり方で色付けする")}">${t("地図: 効き")}</button>`}
        ${d.spec ? `<button class="small" id="pp-respec" title="${t("この実行に使った JSON を入力欄に戻す(設定を変えて流し直すため)")}">${t("入力欄に戻す")}</button>` : ""}</div>
      </div>${this.stage1(d)}${d.writes ? this.writesStage(d) : `${this.stage2(d)}${this.stage3(d)}${this.stage4(d)}`}
      ${this.candForm(d) || `<div class="pp-cand"><p class="muted note">${t("この実行には、候補を作れる組がありません(4. の組がどれも作れていない)。")}</p></div>`}</div>`;
    const $ = (s) => out.querySelector(s);
    $("#pp-showdir").addEventListener("click", () => this.onShowDirection(d.name));
    if ($("#pp-map")) $("#pp-map").addEventListener("click", () => this.onOverlay(`causal:${d.name}`));
    const rs = $("#pp-respec");
    if (rs) {
      rs.addEventListener("click", () => {                              // JSON と、その実行の設定を欄に戻す
        this.$("#pp-text").value = JSON.stringify({ ...d.spec, pipeline: undefined }, null, 2);
        this.check();
        const st = d.settings || {};
        // その実行の設定を欄に入れる(rank causal の結果には writes の設定が無いので、それらの欄は既定に戻す)
        for (const [k, sel] of FIELDS) setField(this.$(sel), st[k] === undefined && k === "rank" ? "causal" : st[k], DEFAULTS[k]);
        this.rankFields();
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
        if (b.dataset.load === "s1") { const s1 = (d.allocate || {}).s1; if (s1) this.onLoadProj(`${d.name}-s1`, s1.units, d.name); return; }
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
    if (rf && re) re.addEventListener("click", () => this.onEval({ recipe: rf.recipe, doses: [0, 1], random_controls: 8 }, `${d.name}-combined`));
    for (const b of out.querySelectorAll("[data-eval]")) {
      b.addEventListener("click", () => {
        if (b.dataset.eval === "s1") {                                   // 倍率を上げる前の組
          const s1 = (d.allocate || {}).s1;
          if (s1 && s1.recipe) this.onEval({ recipe: s1.recipe, doses: [0, 1], random_controls: 8 }, `${d.name}-s1`);
          return;
        }
        const r = table.find((x) => String(x.n_heads) === b.dataset.eval);
        if (!r || !r.recipe) return;
        const c = crow(b.dataset.eval);
        this.onEval({ recipe: c ? c.recipe : r.recipe, doses: [0, 1], random_controls: 8 }, c ? `${d.name}-n${c.n_new}-combined` : `${d.name}-n${r.n_heads}`);
      });
    }
  }

  // 「候補を作る」: 組 × 全体の倍率で返事を作り、名前を伏せて採点する(heads_candidates のジョブ)
  candForm(d) {
    const f = this.cf, ok = (d.allocate && (d.allocate.table || []).some((r) => !r.skipped));
    if (!ok) return "";
    const sp = [["llama_server", t("llama-server の既定")], ["common", t("よく使う設定")], ["greedy", "greedy"]];
    return `<div class="pp-cand"><h4 class="pp-sub">${t("強さの候補を作る(名前を伏せて採点する)")}</h4>
      ${noteHTML(t("比較表の組それぞれに全体の倍率を何段か掛けて、A 側(振る舞いが出ている状態)と B 側(ふつうの状態)で応答を作ります。正の倍率は A 側で振る舞いを消す向き(A 側の応答を採点)、負の倍率は符号を返して B 側で強める向き(B 側の応答を採点)。無介入(base)は必ず入ります。どの候補の応答か伏せて採点し、候補ごとに集計します。調整に使っていない質問で作ると、強さを決めるのに使えます。"), "muted note")}
      <div class="pp-set">
        <label class="muted" title="${t("カンマ区切り。正 = A 側で消す向き、負 = B 側で強める向き。0 は使えない(無介入は base)")}">${t("強さ")} <input id="pc-scales" size="16" value="${esc(f.scales)}" spellcheck="false"></label>
        <label class="muted" title="${t("比較表のどの組を使うか(ヘッド数、カンマ区切り)。空なら全部")}">${t("組(ヘッド数)")} <input id="pc-nheads" size="6" value="${esc(f.n_heads)}" placeholder="${t("全部")}" spellcheck="false"></label>
        <label class="muted" title="${t("各側の質問の数の上限(多ければ等間隔に間引く)")}">${t("質問の数")} <input id="pc-mp" type="number" min="1" max="1000" value="${f.max_prompts}"></label>
        <label class="muted" title="${t("応答の長さの上限(トークン)")}">${t("応答の長さ")} <input id="pc-mt" type="number" min="1" max="2048" value="${f.max_tokens}"></label>
        <label class="muted" title="${t("全部の候補(無介入も)で同じ種を使うので、違いは介入だけになる")}">${t("サンプリング")} <select id="pc-sp">${sp.map(([k, l]) => `<option value="${k}"${f.sampling === k ? " selected" : ""}>${esc(l)}</option>`).join("")}</select></label>
        <label class="muted">${t("種")} <input id="pc-seed" type="number" min="0" value="${f.seed}"></label>
        <label class="muted" title="${t("全部の候補の両側の応答を採点する(既定は、消す向きは A 側、強める向きは B 側だけ)")}"><input type="checkbox" id="pc-jb"${f.judge_b ? " checked" : ""}> ${t("両側を採点する")}</label></div>
      <div class="pp-cand-q"><textarea id="pc-qa" rows="4" spellcheck="false" placeholder="${t("A 側の質問(調整に使っていないもの。1 行に 1 つ。空なら対照集合の質問)")}">${esc(f.qa)}</textarea>
        <textarea id="pc-qb" rows="4" spellcheck="false" placeholder="${t("B 側の質問(空なら A と同じ質問)")}">${esc(f.qb)}</textarea></div>
      <div class="pp-set"><label class="muted" title="${t("全部の候補(無介入も)に重ねるレシピ(ニューロンの種火など)。保存したレシピの名前か .yaml のパス")}">${t("重ねるレシピ")} <input id="pc-with" size="28" value="${esc(f.with_recipe)}" placeholder="${t("なし")}" spellcheck="false"></label></div>
      <div class="pp-go"><button class="main" id="pc-go">${t("候補を作る")}</button>${this.candSaved ? ` <button class="small" id="pc-open">${t("保存した候補を開く")}</button>` : ""} <span class="muted note" id="pc-status">${esc(this.candStatus || "")}</span></div></div>`;
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
      else { this.candJobId = j.id; this.marks(); this.candStatusSet(t("ジョブを投げました。進み具合は下のジョブのバーに出ます(終わると「強さの候補」で開きます)")); }
    } catch (e) { this.candStatusSet(`${t("エラー")}: ${e.message}`, true); }
  }

  // direction.js から(heads_candidates のジョブが終わった)
  candJobEnded(j) {
    if (j.id !== this.candJobId) return;
    (this.endedJobs || (this.endedJobs = new Set())).add(j.id);
    setTimeout(() => this.marks());
    if (j.status === "done") { this.candSaved = true; this.candStatusSet(t("できました({s} 秒)", { s: fmt(j.seconds, 0) })); }
    else this.candStatusSet(j.status === "error" ? `${t("失敗")}: ${(j.error && j.error.message) || ""}` : t("止めました"), j.status === "error");
  }

  // 見出しの「推す組」: rank writes で、どれも届かないときは組の効きがいちばん大きい組、推す組が無いときはその理由と次の手
  recText(d) {
    const r = d.recommended;
    if (r && r.reached === false) return ` · <b>${t("推す組: {n} 個(どれも目標に届かないので、組の効きがいちばん大きい組)", { n: r.n_heads })}</b>`;
    if (r) return ` · <b>${t("推す組: {n} ヘッド", { n: r.n_heads })}</b>`;
    if (d.no_recommend) return ` · ${t("推す組なし({r})", { r: t(d.no_recommend.reason) })}<br><span class="muted">${
      t("次の手: 「止めて選ぶ」(v0.7.0 までの選び方、--rank causal)、部品の種類を変える、組の行の「残りから追加」")}</span>`;
    return ` · ${t("推す組なし(どの組も目標に届かない)")}`;
  }

  // 1. 方向と基準層
  stage1(d) {
    const sc = d.score || {}, dir = d.direction || {}, nr = dir.n_rows || {};
    return `<section class="pp-stage"><h4>${t("1. 方向と基準層")}</h4>
      ${noteHTML(t("A と B で同じ質問を読ませ、層ごとに残差の平均の差を方向にします(質問は全部使う)。cohen_d は、A と B の射影の値(残差 · 方向)がどれだけ離れているか。いちばん離れる層を基準層として記録します。"), "muted note")}
      <p class="note mono">${t("基準層")} <b>L${sc.base_layer ?? "—"}</b> · ${t("cohen_d 最大")} L${sc.best_layer ?? "—"} · positions ${esc(dir.positions || "—")} · ${t("方向を作った例")} A ${nr.A ?? "—"} / B ${nr.B ?? "—"} · ${
        t("射影の値を測った例")} A ${(sc.n || {}).A ?? "—"} / B ${(sc.n || {}).B ?? "—"}</p>
      ${dChart(sc.layers, sc.base_layer, sc.best_layer)}
      <p class="muted note">${t("棒 = 層ごとの cohen_d(白 = 基準層、緑 = cohen_d 最大)。棒にマウスを当てると値。")}</p></section>`;
  }

  // rank writes の 3〜5 段: 書き込みの順位づけ → 上位を 1 個ずつ確かめる → s=1 から組を作る(→ 応答での確かめ)
  writesStage(d) {
    const w = d.writes || {}, v = d.verify || {}, a = d.allocate || {}, nA = a.n_a;
    const kinds = (o) => Object.entries(o || {}).filter(([, n]) => n).map(([k, n]) => `${t(KIND_LABEL[k] || k)} ${n}`).join(" · ");
    const share = Object.entries(w.totals_share || {}).filter(([k, x]) => x !== null && (w.n_units || {})[k])
      .map(([k, x]) => `${t(KIND_LABEL[k] || k)} ${sgn(100 * x, 0)}%`).join(" · ");
    const corr = Object.entries(v.spearman || {}).map(([k, x]) => `${k === "all" ? t("部品全体") : t(KIND_LABEL[k] || k)} ${sgn(x.score_vs_gap_drop, 2)}(${t("{n} 個", { n: x.n })})`).join(" · ");
    const vrows = (v.top || []).map((r, i) => `<tr class="${r.excluded ? "rnd" : ""}"><td>${i + 1}</td><td class="l">${unitChip(r)}</td><td>${t(KIND_LABEL[r.kind] || r.kind)}</td>
      <td>${sgn(r.score, 3)}</td><td>${sgn(r.gap_drop, 3)}</td><td>${sgn(r.causal, 3)}</td><td>${sgn(r.b_side_delta, 3)}</td><td>${fmt(r.drift_orth_b, 3)}</td></tr>`).join("");
    const stop = { target: t("全部 1"), raised: t("全体を上げた"), s_max: t("上限で止めた"), drift: t("B 側の直交成分のずれで止めた"),
      b_move: t("B 側が A 側へ動くので止めた"), no_gain: t("上げても良くならないので止めた") };
    // 倍率ごとの A 平均・B Δ・B ⊥(格子の点と、上限の手前を詰めた点。超 = 上限を超えた点)
    const X = a.loop_exit;                                               // ループのモデル: 1 回目の出口の段
    const steps = (r) => {
      // 上限の手前を詰めた点は、上限の中で最後の点と超えた最初の点だけ
      const edge = (r.raise || []).filter((p) => p.phase === "edge");
      const lastOk = edge.filter((p) => p.feasible).sort((x, y) => y.alpha - x.alpha)[0], firstNg = edge.filter((p) => !p.feasible).sort((x, y) => x.alpha - y.alpha)[0];
      const pts = (r.raise || []).filter((p) => ["s1", "grid"].includes(p.phase)).concat([lastOk, firstNg].filter(Boolean)).sort((x, y) => x.alpha - y.alpha);
      if (pts.length < 2) return "";
      const ex = (p) => { const q = X !== null && X !== undefined ? (p.layers || {})[String(X)] : null; return X === null || X === undefined ? "" : q ? `<td>${sgn(q.b_delta, 2)}</td><td>${fmt(q.b_orth, 3)}</td>` : "<td>—</td><td>—</td>"; };
      const tr = pts.map((p) => `<tr class="${p.feasible ? "" : "ng"}"><td>×${fmt(p.alpha, 2)}</td><td>${fmt(p.a_mean, 2)}</td><td>${sgn(p.b_delta, 2)}</td><td>${fmt(p.b_orth, 3)}</td>${ex(p)}<td>${fmt(p.effect, 2)}</td><td>${p.feasible ? "" : esc(t(p.over === "b_move" ? "B 側が A 側へ動く" : "直交成分のずれ"))}</td></tr>`).join("");
      const exh = X === null || X === undefined ? "" : `<th>${t("段 {L} の B Δ", { L: X })}</th><th>${t("段 {L} の B ⊥", { L: X })}</th>`;
      return `<details class="note"><summary>${t("倍率ごと({n} 点)", { n: pts.length })}</summary><table class="evx mono"><thead><tr><th>${t("倍率")}</th><th>${t("A 平均")}</th><th>B Δ</th><th>B ⊥</th>${exh}<th>${t("効き")}</th><th>${t("上限")}</th></tr></thead><tbody>${tr}</tbody></table></details>`;
    };
    const hoNote = (r) => {                                              // --holdout: 選ぶのに使っていない質問での数
      const h = r.holdout, H = a.holdout;
      if (!h || !H) return "";
      return `<div class="muted note">${t("選ぶのに使っていない質問: A 平均 {a}、残り {r} / {n}、B Δ {b}、B ⊥ {o}、効き {e}{x}", { a: fmt(h.a_mean, 2), r: h.remain, n: H.n_a, b: sgn(h.b_side_delta, 2), o: fmt(h.b_drift_orth, 3), e: fmt(h.effect, 2), x: h.reached ? "" : t("(目標に届かない)") })}</div>`;
    };
    const exitNote = (r) => {                                            // 1 回目の出口での変化(ループの中の部品は 1 回目にも効く)
      const bl = (r.by_layer || {})[String(X)], b0 = (a.base_by_layer || {})[String(X)];
      if (X === null || X === undefined || !bl || !b0) return "";
      return `<div class="muted note">${t("段 {L}(1 回目の出口): A {a0} → {a1}、B Δ {bd}、B ⊥ {bo}", { L: X, a0: fmt(b0.A, 2), a1: fmt(bl.A, 2), bd: sgn(bl.B - b0.B, 2), bo: bl.b_orth === undefined ? "—" : fmt(bl.b_orth, 3) })}</div>`;
    };
    const btns = (key, r, refine) => `<td class="l"><button class="small" data-load="${key}" title="${t("この組(proj のユニット)で今のレシピを置き換える")}">${t("レシピを置き換える")}</button>
          <button class="small" data-eval="${key}" title="${t("この組のレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)")}"${r.recipe ? "" : " disabled"}>${t("評価へ送る")}</button>
          ${refine ? `<button class="small" data-refine="${key}" title="${t("この組を当てても目標まで下がらない質問(残り)だけで、組を当てたまま方向を作り直し、足す部品を探す。結果は新しい一括の結果(名前 _r1 …)として出る")}"${r.remain > 0 ? "" : " disabled"}>${t("残りから追加")}</button>` : ""}</td>`;
    const cells = (r) => `<td>${r.remain ?? "—"}${nA ? ` / ${nA}` : ""}</td><td>${fmt(r.a_mean, 2)}</td>
        <td>${sgn(r.b_side_delta, 2)}</td><td>${fmt(r.b_drift_orth, 3)}</td><td>${r.effect === undefined ? "—" : fmt(r.effect, 2)}</td><td class="${r.reached ? "ok" : "ng"}">${r.reached ? "✓" : "×"}</td>`;
    const chipsOf = (r) => (r.heads || []).map((h) => unitChip(h, ` <span class="muted">s${fmt(h.s, 2)}</span>`)).join(" ");
    const s1 = a.s1 ? `<tr class="pp-s1"><td>s=1</td><td class="units">${chipsOf(a.s1)}<div class="muted note">${kinds(a.s1.kinds)} · ${t("倍率を上げる前(取り除くだけ)")}</div>${hoNote(a.s1)}${exitNote(a.s1)}</td>
        <td>1 / 1</td>${cells(a.s1)}${btns("s1", a.s1, false)}</tr>` : "";
    const cmp = s1 + (a.table || []).slice().sort((x, y) => y.n_heads - x.n_heads).map((r) => {
      if (r.skipped) return `<tr><td>${r.n_heads}</td><td class="excl" colspan="8">${esc(r.skipped)}</td></tr>`;
      return `<tr class="${r.recommended ? "rec" : ""}" data-n="${r.n_heads}"><td>${r.n_heads}${r.recommended ? ` <span class="badge">${t("推奨")}</span>` : ""}</td><td class="units">${chipsOf(r)}<div class="muted note">${kinds(r.kinds)}</div>${hoNote(r)}${exitNote(r)}${steps(r)}</td>
        <td>${fmt(r.s_max, 2)} / ${fmt(r.s_median, 2)}<div class="muted note">${esc(stop[r.stop] || r.stop || "")}</div></td>${cells(r)}${btns(r.n_heads, r, true)}</tr>`;
    }).join("");
    const ck = d.check;
    const cell = (x) => (x.match_finished !== null && x.match_finished !== undefined ? `${x.match_finished}/${x.finished}` : `${x.finished}/${x.n}`);
    const ckRows = ck ? ["none", "recipe"].map((k) => { const x = ck.rows[k]; return `<tr><td class="l">${k === "none" ? t("無介入") : t("組")}</td><td>${cell(x.A)}</td><td>${cell(x.B)}</td>
      <td>${x.A.empty} / ${x.B.empty}</td><td>${x.A.loop} / ${x.B.loop}</td><td>${x.A.at_limit} / ${x.B.at_limit}</td></tr>`; }).join("") : "";
    return `<section class="pp-stage"><h4>${t("3. 書き込みの順位づけ")}</h4>
      ${noteHTML(t("質問 1 問につき 1 回流すだけで、全部の部品(注意ヘッド・線形注意のヘッド・MLP のニューロン)が、その層の方向にどれだけ書いたかを読みます。点数 = 選ぶ層で「s=1 の射影で直接に縮まる A と B の差」。後ろの層での変換は見ないので、後ろの書き手を動かしているだけの部品は点数に出ません。"), "muted note")}
      <p class="note mono">${t("選ぶ層")} L${w.select_layer ?? "—"} · A ${(w.n || {}).A ?? "—"} / B ${(w.n || {}).B ?? "—"} · ${t("A−B の差")} ${fmt(w.gap, 2)} · ${kinds(w.n_units)}</p>
      ${share ? `<p class="note">${t("直接の書き込みの和 ÷ 差")}: ${share}</p>` : ""}
      ${(w.layer_scores || []).length ? `<p class="note" title="${t("その層の注意の出力全体・MLP の出力全体を s=1 で射影したときに、選ぶ層で直接に縮む A−B の差の割合(層全体の部品の選び方の目安)")}">${t("層ごとの直接の書き込み ÷ 差(上位 6 層)")}: ${w.layer_scores.slice(0, 6).map((x) => `L${x.layer} ${t("注意全体")} ${sgn(100 * x.attn_share, 0)}% · MLP ${sgn(100 * x.mlp_share, 0)}%`).join(" | ")}</p>` : ""}</section>
      <section class="pp-stage"><h4>${t("4. 上位を 1 個ずつ確かめる")}</h4>
      ${noteHTML(t("点数の上位を 1 個ずつ s=1 で射影して(または止めて)流し直し、因果スクリーニングと同じ物差しで測ります。測った「A−B の差の縮み」と点数の順位相関を出します。"), "muted note")}
      <p class="note mono">${kinds(v.kinds)}${corr ? ` · ${t("順位相関")} ${corr}` : ""}</p>
      <div class="evx-scroll"><table class="evx mono pp-top"><thead><tr><th>#</th><th class="l">${t("部品")}</th><th>${t("種類")}</th><th title="${t("選ぶ層で s=1 の射影で直接に縮まる A−B の差")}">${t("点数")}</th>
        <th title="${t("流し直して測った A−B の差の縮み")}">${t("測った縮み")}</th><th title="${t("A 側の射影の値の下がり方")}">${t("A の下がり")}</th><th title="${t("B 側の射影の値の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th></tr></thead><tbody>${vrows}</tbody></table></div></section>
      <section class="pp-stage"><h4>${t("5. s=1 から組を作る")}</h4>
      ${noteHTML(t("確かめた部品を、A 側の下がり − B 側の動きの大きい順に、s=1 で 1 個ずつ組に足します(A 側の平均が下がり、組の効きが増え、上限の中のときだけ)。目標に届けば強さは全部 1。届かなければ組の全部の強さを同じ倍率で上げます。s が 1 を超えると成分を反転させるので、B 側(ふつうの入力)が A 側へ動くことがあります。倍率ごとに測り、目標に届けば届く最小の倍率、届かなければ上限の中で組の効きがいちばん大きい倍率を使います。効き = A 側の下がり − B 側の動きの大きさ。上限 = B 側の直交成分のずれ {d}、B 側が A 側へ動く量は A と B の差の {m}%。", { d: fmt(a.drift_max ?? 0.5, 2), m: fmt(100 * (a.b_move_max ?? 0.1), 0) }), "muted note")}
      ${a.holdout ? `<p class="note">${t("質問を半分に分けました: 偶数番目で選び、奇数番目(選ぶのに使っていない質問、A {a} 問・B {b} 問)で各行の数と応答の確かめを測り直しています(目標 {t})", { a: a.holdout.n_a, b: a.holdout.n_b, t: fmt(a.holdout.target, 2) })}</p>` : ""}
      <div class="evx-scroll"><table class="evx mono pp-cmp"><thead><tr><th>${t("上限")}</th><th class="l">${t("部品と強さ")}</th><th>${t("s 最大 / 中央値")}</th>
        <th title="${t("目標を超えて残った A 側の例の数")}">${t("残り")}</th><th title="${t("A 側の射影の値の平均")}">${t("A 平均")}</th><th title="${t("B 側の射影の値の変化")}">B Δ</th>
        <th title="${t("B 側の直交成分のずれ(小さいほど他を壊さない)")}">B ⊥</th><th title="${t("A 側の下がり − B 側の動きの大きさ")}">${t("効き")}</th><th>${t("届いた")}</th><th></th></tr></thead><tbody>${cmp}</tbody></table></div>
      ${(a.notes || []).map((x) => `<p class="note ng">${x.code === "s1_low"
        ? t("倍率を上げる前の組(取り除くだけ)は、目標までの差の {p}% しか縮めません。直接書き込む部品の成分を取り除くだけでは届かない題材で、強さを上げて(成分を反転させて)届かせています", { p: fmt(100 * x.closed, 0) })
        : x.code === "loop_inner" ? t("組にループの中の部品(段 {a}〜{b} の重み)が入っています。重みは 2 回とも使われるので、1 回目(段 {c}〜{d})にも効きます", { a: x.lo, b: x.hi, c: x.first[0], d: x.first[1] })
        : esc(x.text || "")}</p>`).join("")}</section>
      ${ck ? `<section class="pp-stage"><h4>${t("6. 応答での確かめ")}</h4>
        ${noteHTML(t("選んだ組と無介入で応答を書かせ、書き終えた応答のうち validate.match に一致した数を数えます(A 側 / B 側)。射影の値が目標に届いても応答が変わらないことがあるので、その確かめです。"), "muted note")}
        <table class="evx mono"><thead><tr><th></th><th>A</th><th>B</th><th>${t("空")} A / B</th><th>${t("繰り返し")} A / B</th><th>${t("上限で切れた")} A / B</th></tr></thead><tbody>${ckRows}</tbody></table>
        ${ck.match_warning ? `<p class="note ng">${t("無介入の A 側で validate.match に一致したのは {m}/{f} だけです。正規表現が応答に合っていないか、モデルが A 側の振る舞いをしていない疑いがあり、この確かめの数は当てになりません", { m: ck.rows.none.A.match_finished ?? 0, f: ck.rows.none.A.finished })}</p>` : ""}
        ${Object.entries(ck.break_warning || {}).map(([side, x]) => `<p class="note ng">${t("組を当てると {s} 側の応答が壊れています(繰り返し {l}・空 {e} / {n} 問、無介入では {l0}・{e0})。強さを下げるか、別の組・別の部品の種類を試してください", { s: side, l: x.loop, e: x.empty, n: x.n, l0: x.loop_none, e0: x.empty_none })}</p>`).join("")}
        ${ck.note ? `<p class="note ng">${t(ck.note)}</p>` : ""}
        ${ck.suggest ? `<p class="note">${t("応答が変わらないときは「止めて選ぶ」(v0.7.0 までの選び方、--rank causal)も試してください。Qwen3.5-4B の注意書き(disclaimer)では、この選び方は射影の値が届いても応答を変えず、v0.7.0 までの選び方は注意書きを減らした(ただし軽い質問にも注意書きを足した)")}</p>` : ""}</section>` : ""}`;
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
    const slh = SL.map((L) => `<th title="${t("止めたときの、L{L} での A 側の射影の値の下がり方", { L })}">L${L}${L === P ? "*" : ""}</th>`).join("");
    return `<section class="pp-stage"><h4>${t("2. 因果スクリーニング")}</h4>
      ${noteHTML(t("層ごとにその層のヘッドを全部止め、後ろの層(射影の値を測る層)で A 側の射影の値がどれだけ下がるかを測って、よく下がる層を選びます(top_layers = {k})。止めると中立の入力(B 側)の残差まで大きく動く層とヘッドは、効いているのではなくモデルを壊しているだけなので外します。選んだ層のヘッドを 1 個ずつ止めて、同じように下がり方(因果の効き目)を測ります。", { k: k ?? "—" }), "muted note")}
      <p class="note mono">${t("射影の値を測る層")} ${SL.map((L) => (L === P ? `<b>L${L}</b>` : `L${L}`)).join(" ")} · ${t("測った例")} A ${c.n_a ?? "—"} / B ${c.n_b ?? "—"} · ${t("ヘッドの行 {n}", { n: c.n_rows ?? "—" })} · ${
        t("ヘッドを測った層")} ${[...measured].sort((a, b) => a - b).map((L) => `L${L}`).join(", ") || "—"}${broke.length ? ` · ${t("外した層")} ${broke.map((r) => `L${r.layer}`).join(", ")}` : ""}</p>
      <h4 class="pp-sub">${t("止めたときの下がり方が大きいヘッド(上位 {n})", { n: (c.top || []).length })}</h4>
      <div class="evx-scroll"><table class="evx mono pp-top"><thead><tr><th>#</th><th class="l">${t("ヘッド")}</th><th title="${t("止めたときの下がり方の、射影の値を測る層での最大")}">causal_max</th>${slh}
        <th title="${t("無介入でこのヘッドが、射影の値を測る主の層の方向に直接書く量")}">contrib</th><th title="${t("causal ÷ 直接の寄与。1 より大きい = 後段のユニットを通る間接の経路")}">ratio</th>
        <th title="${t("B 側の射影の値の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th><th></th></tr></thead><tbody>${heads}</tbody></table></div>
      ${lr.length ? `<details><summary class="muted note">${t("層ごとのスクリーニング({n} 層、層のヘッドを全部止めたとき)", { n: lr.length })}</summary>
        <div class="evx-scroll"><table class="evx mono pp-layers"><thead><tr><th>${t("層")}</th>${slh}<th>causal_max</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th><th></th></tr></thead><tbody>${layerRows}</tbody></table></div></details>` : ""}
      <p class="muted note">${t("* = 射影の値を測る主の層。下がり方 = 止める前 − 後の A 側の射影の値(正 = A 側の向きに押していた)。")}</p></section>`;
  }

  // 3. 最初の比率
  stage3(d) {
    const a = d.allocate || {}, init = a.initial;
    const intro = `${noteHTML(t("前の段の因果の効き目(ヘッドを 1 個ずつ止めたときに、後ろの層の射影の値がどれだけ下がるか)が大きいヘッドを選び、強さはこの効き目に比例させます。全体の大きさ α だけを、応答を書かせずに測る射影の値が目標に届くよう二分探索で決めます。"), "muted note")}`;
    if (!init) return `<section class="pp-stage"><h4>${t("3. 最初の比率")}</h4>${intro}<p class="muted note">${t("(この実行には最初の組がありません)")}</p></section>`;
    const us = (init.heads || []).map(parseUnit).filter(Boolean);
    const mx = Math.max(1e-9, ...us.map((u) => u.s));
    const rows = us.map((u) => `<tr><td class="l">${headChip(u.layer, u.head)}</td><td>${fmt(u.s, 2)}</td><td>${fmt(u.s / mx, 2)}</td></tr>`).join("");
    return `<section class="pp-stage"><h4>${t("3. 最初の比率")}</h4>${intro}
      <p class="note mono">α ${fmt(init.alpha, 2)} · ${t("A 側の射影の値の平均 {v}(目標 {g})", { v: fmt(init.mean_a, 2), g: fmt(d.target, 2) })} · ${init.reached ? `<span class="ok">${t("目標に届いた")}</span>` : `<span class="ng">${t("届かない(強さの上限)")}</span>`}</p>
      <table class="evx mono pp-init"><thead><tr><th class="l">${t("ヘッド")}</th><th title="${t("射影の強さ s(= α × 比率)")}">s</th><th title="${t("効き目の比率(最大のヘッドを 1)")}">${t("比率")}</th></tr></thead><tbody>${rows}</tbody></table></section>`;
  }

  // 4. 効率での配り直し: η の表と、ヘッドの数ごとの組の比較
  stage4(d) {
    const a = d.allocate || {}, hard = a.hard || {}, S = a.score_layer;
    const ab = new Map();                                                // 丸ごと止めたときの効き目(配分で射影の値を測る層)
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
          <button class="small" data-load="${r.n_heads}" title="${crow(r.n_heads) ? t("元の組とこの組を合わせたユニットで今のレシピを置き換える(この組は元の組を当てた状態で探したので、単独では使わない)") : t("この組(proj のユニット)で今のレシピを置き換える")}">${crow(r.n_heads) ? t("元の組と合わせてレシピを置き換える") : t("レシピを置き換える")}</button>
          <button class="small" data-eval="${r.n_heads}" title="${crow(r.n_heads) ? t("元の組とこの組を合わせたレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)") : t("この組のレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)")}"${r.recipe ? "" : " disabled"}>${crow(r.n_heads) ? t("合わせて評価へ送る") : t("評価へ送る")}</button>
          <button class="small" data-refine="${r.n_heads}" title="${t("この組を当てても目標まで下がらない質問(残り)だけで、組を当てたまま方向を作り直し、因果 → 配分で足すヘッドを探す。結果は新しい一括の結果(名前 _r1 …)として出る")}"${r.remain > 0 ? "" : " disabled"}>${t("残りから追加")}</button></td></tr>`;
    }).join("");
    const hp = hard.prompts || [], hs = hard.scores || [];
    return `<section class="pp-stage"><h4>${t("4. 効率での配り直し")}</h4>
      ${noteHTML(t("最初の組を当てた状態で、射影の値が残りやすい質問を選び、ヘッドごとに強さを少しだけ上げて、その質問での下がり方(効率 η)を測ります。強さを効率に比例させて 0 から配り直し、もう一度二分探索で目標に合わせます。丸ごと止めたときの効き目と効率では順位が入れ替わるので、この段が要ります。ヘッドの数を減らした組も作って比べます。"), "muted note")}
      ${hp.length ? `<details><summary class="muted note">${t("射影の値が残りやすい質問 {n} 個(L{L} の射影の値)", { n: hard.n ?? hp.length, L: hard.layer ?? S ?? "—" })}</summary>
        <table class="evx mono dv-ex"><tbody>${hp.map((q, i) => `<tr><td class="w">${esc(q)}</td><td>${fmt(hs[i], 2)}</td></tr>`).join("")}</tbody></table></details>` : ""}
      <h4 class="pp-sub">${t("効率 η(大きい順)")}</h4>
      <div class="evx-scroll"><table class="evx mono pp-eta"><thead><tr><th>#</th><th class="l">${t("ヘッド")}</th><th title="${t("s を Δs 上げたときの、射影の値が残りやすい質問での下がり方 ÷ Δs")}">η</th>
        <th title="${t("丸ごと止めたときの下がり方(L{L})", { L: S ?? "—" })}">${t("止めた効き目")}</th><th title="${t("丸ごと止めたときの効き目での順位。黄 = 効率の順位と 2 つ以上ずれている")}">${t("効き目の順位")}</th>
        <th title="${t("B 側の射影の値の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th><th title="${t("最初の組に入っていたヘッド")}">${t("最初の組")}</th><th></th></tr></thead><tbody>${eta}</tbody></table></div>
      ${etaRows.length > ETA_SHOWN ? `<button class="small" id="pp-eta-more">${t("残り {n} 行も見る", { n: etaRows.length - ETA_SHOWN })}</button>` : ""}
      <h4 class="pp-sub">${t("ヘッドの数ごとの組の比較")}</h4>
      <div class="evx-scroll"><table class="evx mono pp-cmp"><thead><tr><th>${t("ヘッド数")}</th><th class="l">${t("ヘッドと強さ")}</th><th>s max</th>
        <th title="${t("目標を超えて残った A 側の例の数")}">${t("残り")}</th><th title="${t("A 側の射影の値の最大")}">A max</th><th title="${t("A 側の射影の値の平均")}">${t("A 平均")}</th>
        <th title="${t("B 側の射影の値の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ(小さいほど他を壊さない)")}">B ⊥</th><th title="${t("射影の値が残りやすい質問の平均が目標に届いたか")}">${t("届いた")}</th><th></th></tr></thead><tbody>${cmp}</tbody></table></div>
      ${noteHTML(t("推奨 = 目標に届いた組のうち、残る例の数が最も少ない組と同程度(A 側の 5% まで)で、B 側のずれが最小の 1.25 倍までの組の中で、ヘッドが最も少ない組。ヘッドの名前をクリックでくわしく見る。"), "muted note")}
      </section>`;
  }
}
