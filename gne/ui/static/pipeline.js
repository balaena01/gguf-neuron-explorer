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
                   "score_layers", "select_layer", "candidate_layers", "always_layers", "include_score_layers", "same_reply", "loop_direction", "holdout",
                   "strength", "from_direction"];
const DRAFT = "gne.pipeline.draft";               // 入力欄の下書き(このブラウザだけ)
// 設定の欄 → ジョブの引数(spec の pipeline に値があれば、読み込んだときに欄に入れる)
const FIELDS = [["target", "#pp-target"], ["n_heads", "#pp-nheads"], ["max_examples", "#pp-mx"], ["positions", "#pp-pos"],
                ["top_layers", "#pp-tl"], ["hard", "#pp-hard"], ["rank", "#pp-rank"], ["unit_kinds", "#pp-kinds"],
                ["n_units", "#pp-nunits"], ["verify_top", "#pp-vtop"], ["check_replies", "#pp-ckn"], ["check_tokens", "#pp-ckt"],
                ["holdout", "#pp-holdout"], ["loop_direction", "#pp-loopdir"], ["strength", "#pp-strength"], ["from_direction", "#pp-fromdir"]];
const DEFAULTS = { target: "", n_heads: "6,5,4", max_examples: 64, positions: "last", top_layers: 4, hard: 16, rank: "writes", unit_kinds: "heads",
                   n_units: "24,12,6", verify_top: 64, check_replies: 16, check_tokens: 512, holdout: false, loop_direction: "first",
                   strength: "per_unit", from_direction: "" };
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
// Inkling 2 の段の表(/api/model の inkling。PipelinePanel が作られたときに入れる)。部品の名前に回を出し、パネルへのリンクは付けない
let INK = null;
const inkMark = (L) => (INK && L >= INK.core_start && L < INK.core_end ? ` ×${INK.n_passes}` : "");
const inkTip = (L) => (INK && L >= INK.core_start && L < INK.core_end
  ? t("層 {a}〜{b} の重みは {n} 回の繰り返しで共有なので、この部品の書き換えは {n} 回すべてに効きます", { a: INK.core_start, b: INK.core_end - 1, n: INK.n_passes })
  : t("繰り返しの外の層(1 回だけ実行)"));
const inkWho = (u) => (u.kind === "neuron" ? (u.expert !== undefined && u.expert !== null ? `E${u.expert} N${u.head ?? u.index}` : `N${u.head ?? u.index}`)
  : u.kind === "layer_attn" ? "attn" : u.kind === "layer_mlp" ? "ffn" : `H${u.head ?? u.index}`);
// 測る点の名前(段の番号から。サーバーの label は日本語なので、英語の画面ではここで作る)
const inkPoint = (x) => (x.layer === 48 ? t("最後に混ぜた出力") : t("段 {s}", { s: x.layer }));
// 部品の印: ヘッドは HEAD パネルへのリンク、ニューロン(L:Nn)は文字だけ
const unitChip = (u, extra = "") => (INK ? `<span class="mono" title="${esc(inkTip(u.layer))}">L${u.layer} ${inkWho(u)}${inkMark(u.layer)}${extra}</span>`
  : u.kind === "neuron" ? `<a href="#" class="pp-head mono" data-neuron="${u.layer}:${u.head ?? u.index}" title="${t("NEURON パネルで開く")}">L${u.layer} N${u.head ?? u.index}${extra}</a>`
  : u.kind === "layer_attn" ? `<span class="mono" title="${t("層の注意の出力全体(attn_output、線形注意の層は ssm_out)")}">L${u.layer} attn${extra}</span>`
  : u.kind === "layer_mlp" ? `<span class="mono" title="${t("層の MLP の出力全体(ffn_down)")}">L${u.layer} ffn${extra}</span>`
  : headChip(u.layer, u.head ?? u.index, extra));
const KIND_LABEL = { attn: "注意ヘッド", linear: "線形注意のヘッド", neuron: "ニューロン", layer_attn: "層の注意全体", layer_mlp: "層の MLP 全体" };
const headChip = (L, h, extra = "") => `<a href="#" class="pp-head mono" data-head="${L}:${h}" title="${t("HEAD パネルで開く")}">L${L} H${h}${extra}</a>`;
// 残りから追加の結果: 表の行(上限の数か "s1")の、土台と合わせたレシピ。v0.8.3 までの結果は key が無く n_new で引く
const combinedRow = (d, key) => ((d.refine && d.refine.combined_rows) || []).find((c) => String(c.key ?? c.n_new) === String(key)) || null;
// レシピの欄に読み込むときの名前: 土台(元の組 + それまでの回)と足した組を合わせたもの
// v0.8.4 からの残りから追加の結果か(合わせたレシピの行に判定がある)。v0.8.3 までは推す組を足した組だけの値で決めていた
const refineNew = (d) => ((d.refine && d.refine.combined_rows) || []).some((c) => c.verdict);
const combinedName = (d, c) => `${d.name}-${String(c.key ?? c.n_new) === "s1" ? "s1" : `n${c.key ?? c.n_new}`}-base-plus-added`;
const roundsUpTo = (k) => (k <= 1 ? t("1 回目") : t("1〜{k} 回目", { k }));
// 残りから追加の結果の行の判定(合わせたレシピの数で: better / worse / no_gain / replies_break)の注
const WHY = { remain_up: "残りが増える", effect_down: "組の効きが下がる", b_move: "B 側が A 側へ動く", drift: "B 側の直交成分のずれ", replies_break: "応答が壊れる" };
const verdictNote = (c) => {
  if (!c || !c.verdict) return "";
  const added = c.recommended_added ? ` <span class="muted">${t("(足した組だけの値では推していた)")}</span>` : "";
  if (c.verdict === "better") return c.recommended ? "" : `<div class="muted note">${t("合わせると良くなる(推す行のほうが良い)")}${added}</div>`;
  if (c.verdict === "no_gain") return `<div class="muted note">${t("合わせても良くならない(推さない)")}${added}</div>`;
  return `<div class="note ng">${t("合わせると悪くなる: {w}(推さない)", { w: (c.why || []).map((k) => t(WHY[k] || k)).join("・") })}${added}</div>`;
};
// 組の比較表の印(v0.9.0): 推奨(道具の規則で選んだ目安)と、A 側が最も下がった組(元の方向・元の目標で測った A 平均が最小)。
// 同じ行なら 2 つ並べる。応答での確かめで応答が壊れたと分かった行は、最も下がった印の横に警告を並べる
const recBadge = (on) => (on ? ` <span class="badge pp-rec" title="${t("推奨は、道具の規則(残りの数・B 側のずれ)で選んだ目安です。合否の判定ではありません。応答は見ていません")}">${t("推奨")}</span>` : "");
const mrBadge = (on, broke) => (on ? ` <span class="badge pp-mr" title="${t("元の方向・元の目標で測った A 側の射影の値の平均がいちばん小さい組。強さはチャットで調整してください")}">${t("A 側が最も下がった")}</span>${
  broke ? ` <span class="badge pp-mr-ng ng" title="${t("応答での確かめで、この組を当てると繰り返し・空の応答が増えました。強さを下げてから使ってください")}">${t("応答が壊れた")}</span>` : ""}` : "");
// 作り済みの方向から流した一括(--from-direction)の出どころと知らせ(結果の見出しに出す。どの手順でも見える)
const fromDirHTML = (dir) => (!dir || !dir.from ? "" : `<p class="note pp-from">${t("方向: 作り済みの方向 {n} を写して使いました(方向づくりは飛ばしました)", { n: esc(dir.from) })}</p>${
  dir.from_note ? `<p class="note pp-warn pp-from-note">${t("方向 {n} を作った対照集合と、この一括の対照集合の質問・system・レシピが同じとは確かめられません。違う対照集合で作った方向は、射影の値・目標の意味が変わります", { n: esc(dir.from) })}</p>` : ""}`);
const MR_NOTE = () => t("推す規則は射影の値(残りの数・B 側のずれ)で決めていて、応答は見ていません(応答での確かめは、壊れていないかを見るだけです)。「A 側が最も下がった」は、元の方向・元の目標で測った A 側の平均がいちばん小さい組です(推奨と同じ行なら印を 2 つ並べます)。どの組を使うときも、強さはチャットで調整してください。");
const rowBroke = (r, c) => (c ? c.verdict === "replies_break" || !!((c.check || {}).break_warning && Object.keys(c.check.break_warning).length) : !!(r && r.replies_break));
// 推す組・推す行が無いときの次の手(A 側が最も下がった組と、チャットで強さを調整する案内)
const mrNext = (mr, refine) => {
  if (!mr) return t("チャットで強さを調整して、応答を確かめてください(「この強さをレシピに反映」で保存できます)。");
  const which = refine ? (mr.key === "s1" ? t("s=1 の行の合わせたレシピ") : t("上限 {n} の行の合わせたレシピ", { n: mr.key ?? mr.n_added }))
    : (mr.s1 ? t("倍率を上げる前の組(s=1)") : t("{n} 個の組", { n: mr.n_heads }));
  const a = refine ? mr.a_mean_after : mr.a_mean;
  return t("A 側が最も下がったのは、{w}です(A 平均 {a})。チャットで強さを調整して、応答を確かめてください(「この強さをレシピに反映」で保存できます)。", { w: which, a: fmt(a, 2) })
    + (mr.replies_break ? ` <span class="ng">${t("ただし、この組は応答での確かめで応答が壊れました(繰り返し・空が増えた)。強さを下げてから使ってください。")}</span>` : "");
};
const partName = (q) => (q.kind === "neuron" ? `L${q.layer} ${q.expert !== undefined && q.expert !== null ? `E${q.expert} ` : ""}N${q.head}${inkMark(q.layer)}` : q.kind === "layer_attn" ? `L${q.layer} attn` : q.kind === "layer_mlp" ? `L${q.layer} ffn` : `L${q.layer} H${q.head}`);

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
  constructor(root, { onRun, onShowDirection, onOpenHead, onOpenNeuron, onLoadProj, onEval, onOverlay, onOpenCandidates, isLoop, writesBlocked,
                      inkling, onChatProj } = {}) {
    this.onChatProj = onChatProj || ((name, specs, dir) => this.onLoadProj(name, specs, dir));   // (名前, 部品, 方向) → レシピを置き換えて 3 列チャット
    this.isLoop = isLoop || (() => false);       // 同じ層を 2 回実行するモデル(ループの中の部品の方向の欄を出す)
    this.inkling = inkling || (() => null);      // Inkling 2 の段の表(答えの前に思考を書くモデル。位置の選択と、使えない欄を変える)
    // このモデルで書き込みの量での順位づけ(rank writes)を実行しない理由の文(実行できれば null)。サーバーの既定の切り替えと同じ条件:
    // 残差が複数本の流れ(hyper-connections)、注意の出力のあとに正規化がある層(v0.9.0)
    this.writesBlocked = writesBlocked || (() => null);
    this.rankForced = false;                     // 欄の既定(書き込みの量)を、そのモデルでは実行しないので止めて選ぶに替えたか
    this.root = root;
    this.onRun = onRun;                          // (kind, params) → Promise<ジョブ | null>(jobs.js の start。見積もりの確認つき)
    this.onShowDirection = onShowDirection || (() => {});   // (名前) → 上の「方向を見る」でその方向を開く
    this.onOpenHead = onOpenHead || (() => {});   // ({ layer, head, tab }) → HEAD パネル
    this.onOpenNeuron = onOpenNeuron || null;     // ({ layer, n }) → NEURON パネル(無ければ、ニューロンの印は文字だけ)
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
        <label class="muted" title="${t("候補の探し方。書き込みの量で全部から(既定) = 全部の層の注意ヘッド・線形注意のヘッド(部品で選べばニューロンも)を方向への書き込みの量で並べ、上位を s=1 で確かめてから、s=1 で組に足す(足りないときだけ全体の強さを上げる)。止めて選ぶ = v0.7.0 までの選び方(層ごとに止めた効き目の上位の層のヘッドを 1 個ずつ止め、効率で配り直す)。応答が変わらないときは、もう一方も試す")}">${t("候補の探し方")} <select id="pp-rank"><option value="writes">${t("書き込みの量で全部から(既定)")}</option><option value="causal">${t("止めて選ぶ(v0.7.0 まで)")}</option></select></label><span id="pp-rank-blocked" class="note pp-warn" hidden></span>
        <label class="muted" title="${t("効率を測るときの、射影の値が残りやすい質問(目標を超えたまま残っている A 側の質問)の数。どちらの探し方でも、強さを部品ごとに配るときに使う")}">${t("残りやすい質問の数")} <input id="pp-hard" type="number" min="1" max="1024" value="${DEFAULTS.hard}"></label>
        <label class="muted" title="${t("同じ JSON・同じ設定で作り済みの方向と因果の結果があっても使い回さず、最初から作り直す(止めて流し直すときは外したままにすると配分からやり直せる)")}"><input type="checkbox" id="pp-fresh"> ${t("最初から作り直す")}</label>
        <details class="more" id="pp-more-dir"><summary class="muted">${t("細かい設定(どちらの探し方でも)")}</summary><div class="pp-set">
          <label class="muted" title="${t("方向づくりを飛ばし、保存済みの方向の向きで基準層・候補づくり・配分をする(--from-direction)")}">${t("作り済みの方向で一括を流す")} <select id="pp-fromdir"><option value="">${t("使わない(対照の文から方向を作る、既定)")}</option></select></label>
          <span id="pp-fromdir-note" class="note pp-warn" hidden></span>
          <p class="muted note">${t("方向づくりを飛ばし、選んだ方向の向きで基準層・候補づくり・配分をします(層の幾何で保存した向き、別の一括の方向など)。方向は一括の名前の方向として写して使います。方向を作った対照集合と入力欄の対照集合が同じと確かめられないときは、結果に知らせが出ます(違う対照集合で作った方向では、射影の値と目標の意味が変わります)。一括の名前と同じ方向は選べません。方向を作った位置が「方向を作る位置」と違うと実行しません。")}</p>
        </div></details></div>
      <fieldset class="pp-set fx-part" id="pp-fs-writes" aria-describedby="pp-fs-writes-off"><legend class="muted note fx-part-h">${t("書き込みの量で全部から選ぶときの設定")} <span class="pp-fs-off" id="pp-fs-writes-off" hidden>${t("(「止めて選ぶ」では使いません)")}</span></legend>
        <label class="muted" title="${t("書き込みの量で全部から選ぶときの部品の種類")}">${t("部品")} <select id="pp-kinds"><option value="heads">${t("ヘッドだけ(既定)")}</option><option value="all">${t("ヘッド・線形注意・ニューロン")}</option><option value="heads,layers">${t("ヘッドと層全体")}</option><option value="all,layers">${t("ヘッド・ニューロン・層全体")}</option></select></label>
        <label class="muted" title="${t("比べる組の部品の数の上限(カンマ区切り、既定 24,12,6)。上限ごとに組を作り、表で比べる")}">${t("組の大きさの上限")} <input id="pp-nunits" size="8" value="${DEFAULTS.n_units}" spellcheck="false"></label>
        <label class="muted" title="${t("s=1 の組が目標に届かないときの強さの決め方。部品ごと(既定) = 部品ごとに s を少し上げてみて、射影の値が残りやすい質問での組の効き(A 側の下がり − B 側の動きの大きさ)の増え方に比例して上げ幅を配る(効く部品は高く、B 側を押す部品は低く)。全部を同じ倍率 = v0.8.2 までの決め方")}">${t("強さの決め方")} <select id="pp-strength"><option value="per_unit">${t("部品ごと(既定)")}</option><option value="uniform">${t("全部を同じ倍率(v0.8.2 まで)")}</option></select></label>
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
      </fieldset>
      <p class="muted note" id="pp-prio">${t("欄の値が使われます。JSON の pipeline に同じ項目があれば、読み込んだときにその値が欄に入ります(欄を書き換えれば欄の値が優先)。")}</p>
      </div>
      <div class="pp-go" id="pp-run"><button class="main" id="pp-go" disabled>${t("方向とヘッドを作る")}</button> <span class="muted note" id="pp-status"></span></div>
      <div id="pp-past-wrap"><div id="pp-past"><p class="muted note">${t("読み込み中…")}</p></div></div>
      <p class="muted note" id="pp-nores">${t("まだ結果がありません。「1. 対照の文」で「方向とヘッドを作る」を押すか、「これまでの実行」から開いてください。")}</p>
      <div id="pp-result"></div>`;
    this.$ = (s) => root.querySelector(s);
    this.inklingForm();
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
    this.dirs = [];                              // 保存済みの方向(/api/directions)。「作り済みの方向で一括を流す」の選択肢
    this.loadDirs();
    for (const sel of ["#pp-fromdir", "#pp-pos", "#pp-holdout"]) this.$(sel).addEventListener("change", () => this.check());
    this.$("#pp-rank").addEventListener("change", () => { this.rankForced = false; this.rankFields(); });
    this.rankFields();
    // 左に手順、右に選んだ手順。結果は 1 回のジョブ(方向とヘッドを作る)の段ごとに分けて見せる
    const res = (sel) => () => (this.data ? [this.$("#pp-result .pp-res-head"), ...this.$$(`#pp-result ${sel}`)] : [this.$("#pp-nores")]);
    this.steps = new Steps(root, {
      shared: this.$("#pp-shared"), store: "gne.chips.step", label: t("CHiPS の手順"), extra: () => [this.$("#pp-nores")],
      steps: [{ key: "input", label: t("1. 対照の文"), els: () => [this.$("#pp-input"), this.$("#pp-run")], shared: true },
              { key: "dir", label: this.inkling() ? t("2. 方向と測る点") : t("2. 方向と基準層"), els: res("section.pp-stage:nth-of-type(1)") },
              { key: "causal", label: t("3. 効くヘッドを探す"), els: res("section.pp-stage:nth-of-type(2)") },
              { key: "alloc", label: t("4. 配り方を決める"), els: res("section.pp-stage:nth-of-type(3), section.pp-stage:nth-of-type(4), section.pp-stage:nth-of-type(5)") },
              { key: "cand", label: t("5. 強さの候補を作る"), els: res(".pp-cand") },
              { key: "past", label: t("これまでの実行"), els: () => [this.$("#pp-past-wrap")], sep: true }],
    });
    const check = this.check.bind(this);                            // 入力を確かめるたびに 1. の印も付け直す
    this.check = (...a) => { const r = check(...a); this.marks(); return r; };
    this.check();
  }

  $$(s) { return [...this.root.querySelectorAll(s)]; }

  // Inkling 2: 位置の選択(答えの始まりなど)、使えない欄(止めて選ぶ・質問を分ける・ループの方向・作り済みの方向)を止め、このモデル用の既定を入れる
  inklingForm() {
    const ink = this.inkling();
    INK = ink || null;
    if (!ink) return;
    const pos = this.$("#pp-pos");
    pos.innerHTML = [["last", t("質問の最後(既定)")], ["answer", t("答えの始まり(思考の後)")], ["response", t("答えのトークン全部")],
                     ["reasoning", t("思考のトークン全部")], ["after", t("答えの中の一致の直後(正規表現)")], ["all", t("質問のトークン全部")]]
      .map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
    pos.closest("label").title = t("方向を作り、射影の値を測る位置。このモデルは答えの前に必ず思考を書くので、質問の最後で作った方向は、思考と答えを書く位置では差が小さくなることがあります。答えの口調や書き方を狙うときは「答えの始まり」か「答えのトークン全部」も試してください(どちらも、書き換える前のモデルに返事を書かせてから読むので遅い)");
    pos.closest("label").insertAdjacentHTML("afterend", `<label class="muted" id="pp-pos-re-l" hidden>${t("正規表現")} <input id="pp-pos-re" size="10" spellcheck="false" placeholder="[.!?]"></label>`
      + `<p class="muted note" id="pp-ink-note">${esc(t("Inkling 2: 方向は MLP が読む入力(層 6〜17 は 3 回目)で作り、部品はヘッド・注意全体・MLP 全体・専門家のニューロンから選びます。層 6〜17 の重みは 3 回の繰り返しで共有なので、その層の部品は 3 回すべてに効きます。名前の patch を当てた llama.cpp が要ります(docs/inkling2.md)"))}</p>`);
    pos.addEventListener("change", () => { this.$("#pp-pos-re-l").hidden = pos.value !== "after"; });
    const rank = this.$("#pp-rank");
    rank.value = "writes";
    rank.querySelector("option[value=causal]").disabled = true;
    const rb = this.$("#pp-rank-blocked");
    rb.hidden = false;
    rb.textContent = t("Inkling 2 では「止めて選ぶ」はまだ使えません");
    const kinds = this.$("#pp-kinds");                // このモデルの部品(線形注意は無い)と既定(ヘッド・ニューロン・層全体)
    kinds.innerHTML = [["heads", t("ヘッドだけ")], ["all", t("ヘッドと専門家のニューロン")], ["heads,layers", t("ヘッドと層全体")],
                       ["all,layers", t("ヘッド・専門家のニューロン・層全体(既定)")]].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
    kinds.value = "all,layers";
    this.$("#pp-mx").value = 32;
    this.$("#pp-mx").closest("label").title = t("このモデルでは、方向づくりと射影の値を測る段(順位づけ・確かめ・配分)の両方で、側ごとにこの数の質問を使う(等間隔に間引く。既定 32)。答えの始まりなどの位置では、質問 1 つごとに返事を書かせるので、全部の質問を使うと時間がかかるため");
    this.$("#pp-ckt").value = 1024;
    this.$("#pp-ckt").closest("label").title = t("応答での確かめで書かせる長さの上限(トークン、このモデルの既定 1024。答えの前に思考を書くので長め)");
    for (const id of ["#pp-holdout", "#pp-loopdir", "#pp-fromdir"]) { const l = this.$(id).closest("label"); if (l) l.hidden = true; }
    this.$("#pp-kinds").closest("label").insertAdjacentHTML("afterend",
      `<label class="muted" title="${t("層 0〜5(繰り返しの前)の部品も候補に入れる。既定は入れない")}"><input type="checkbox" id="pp-before"> ${t("層 0〜5 も候補に入れる")}</label>`);
  }

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
      for (const [k, sel] of FIELDS) {
        if (k === "from_direction") this.setFromDir(pipe && pipe[k] ? String(pipe[k]) : "");
        else setField(this.$(sel), pipe ? pipe[k] : undefined, DEFAULTS[k]);
      }
      this.rankFields();
    }
    this.$("#pp-go").disabled = !c.spec || c.errors.length > 0;
    if (this.fromDirFields() && c.spec) this.$("#pp-go").disabled = true;   // 作り済みの方向が使えない組み合わせ(理由は欄の下)
  }

  // 保存済みの方向の一覧を「作り済みの方向で一括を流す」の選択肢に入れる(選んでいた値は残す)
  async loadDirs() {
    try { this.dirs = await getJSON("/api/directions"); } catch { this.dirs = []; }
    const sel = this.$("#pp-fromdir");
    if (!sel) return;
    const cur = sel.value;
    sel.innerHTML = `<option value="">${t("使わない(対照の文から方向を作る、既定)")}</option>` + (this.dirs || []).map((x) =>
      `<option value="${esc(x.name)}">${esc(x.name)}${x.positions ? ` (${esc(x.positions)})` : ""}</option>`).join("");
    if (cur) this.setFromDir(cur);
    this.check();
  }

  // 値を選ぶ(一覧に無い名前は「見つからない」として足す。JSON の pipeline.from_direction から入れたとき)
  setFromDir(v) {
    const sel = this.$("#pp-fromdir");
    if (v && ![...sel.options].some((o) => o.value === v)) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = t("{n}(保存済みの方向に見つかりません)", { n: v });
      sel.appendChild(o);
    }
    sel.value = v || "";
  }

  // 「作り済みの方向で一括を流す」の確かめ。使えない組み合わせなら理由を出して true(押せなくする)。使えるが知らせがあれば出す
  fromDirFields() {
    const sel = this.$("#pp-fromdir"), note = this.$("#pp-fromdir-note");
    if (!sel || !note) return false;
    const v = sel.value, c = this.checked, x = (this.dirs || []).find((d) => d.name === v);
    let why = null, info = null;
    if (v) {
      const name = c && c.spec ? String(c.spec.name ?? "") : "";
      const pos = this.$("#pp-pos").value;
      if (name && v === name) why = t("一括の名前 {n} と同じ方向は使えません(一括はその名前の方向を書き直します)。JSON の name を別の名前にしてください", { n: v });
      else if (!x) why = t("方向 {n} は保存済みの方向に見つかりません", { n: v });
      else if (x.positions && x.positions !== pos) why = t("方向 {n} は「{p}」の位置で作った方向で、「方向を作る位置」({q})と違うので実行しません", { n: v, p: x.positions, q: pos });
      else {
        const src = x.source ? String(x.source).split(/[\\/]/).pop() : null;
        info = (src ? t("方向を作った対照集合: {s}。入力欄の対照集合と同じと確かめられないときは、結果に知らせが出ます", { s: src })
          : t("この方向には、作った対照集合の記録がありません。結果に知らせが出ます"))
          + (this.$("#pp-holdout").checked && this.$("#pp-rank").value === "writes" ? " " + t("選ぶ質問と測る質問を分けると、選ぶ質問は半分になるので、同じ対照集合とは確かめられず知らせが出ます") : "");
      }
    }
    note.hidden = !(why || info);
    note.textContent = why || info || "";
    note.classList.toggle("err", !!why);
    return !!why;
  }

  // 候補の探し方に合わせて、使わない設定のまとまりを無効にする(値は残すが、ジョブには渡さない)。ループの中の部品の方向は、
  // 同じ層を 2 回実行するモデルのときだけ選べる
  rankFields() {
    if (this.inkling()) {                          // Inkling 2: 書き込みの量で選ぶ流れだけ
      this.$("#pp-rank").value = "writes";
      this.$("#pp-fs-writes").disabled = false;
      this.$("#pp-fs-causal").disabled = true;
      this.$("#pp-fs-writes-off").hidden = true;
      this.$("#pp-fs-causal-off").hidden = false;
      return;
    }
    const why = this.writesBlocked(), opt = this.$("#pp-rank option[value=writes]"), note = this.$("#pp-rank-blocked");
    opt.disabled = !!why;
    note.hidden = !why;
    if (why) {                                     // このモデルでは書き込みの量での順位づけは実行しない(既定は止めて選ぶ)
      if (this.$("#pp-rank").value === "writes") { this.$("#pp-rank").value = "causal"; this.rankForced = true; }
      note.textContent = why;
    }
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
    const hard = num("#pp-hard", 1);
    if (hard === null) { this.status(t("残りやすい質問の数は 1 以上の整数で"), true); return null; }
    const p = { spec: c.spec, max_examples: mx, positions: this.$("#pp-pos").value, hard };
    if (this.inkling() && p.positions === "after") {                  // Inkling 2: 答えの中の一致の直後
      const re = this.$("#pp-pos-re").value;
      if (!re) { this.status(t("正規表現を入れてください(例 [.!?])"), true); return null; }
      p.positions = `after:${re}`;
    }
    if (this.$("#pp-fresh").checked) p.resume = false;
    const fd = this.$("#pp-fromdir").value;
    if (fd) p.from_direction = fd;                                    // 作り済みの方向で流す(方向づくりを飛ばす)
    p.rank = this.$("#pp-rank").value;                                   // writes(既定)/ causal(v0.7.0 まで)
    if (this.rankForced) {                       // 既定のまま: サーバーが止めて選ぶ流れに切り替え、結果に rank_note を入れる
      delete p.rank;
      const sp = c.spec.pipeline;
      if (sp && typeof sp === "object" && "rank" in sp) { const { rank: _r, ...rest } = sp; p.spec = { ...c.spec, pipeline: rest }; }
    }
    if (p.rank === "writes") {
      const nu = ints("#pp-nunits");
      if (!nu.length || !nu.every((x) => Number.isInteger(x) && x >= 1)) { this.status(t("組の大きさの上限は 1 以上の整数をカンマ区切りで(例 24,12,6)"), true); return null; }
      const vt = num("#pp-vtop", 1), ck = num("#pp-ckn", 0), ct = num("#pp-ckt", 16);
      if (vt === null || ck === null || ct === null) { this.status(t("確かめる部品の数は 1 以上、応答での確かめの質問の数は 0 以上、上限のトークン数は 16 以上の整数で"), true); return null; }
      Object.assign(p, { unit_kinds: this.$("#pp-kinds").value, n_units: nu, verify_top: vt, check_replies: ck, check_tokens: ct,
                         holdout: this.$("#pp-holdout").checked, strength: this.$("#pp-strength").value });
      if (this.isLoop()) p.loop_direction = this.$("#pp-loopdir").value;
      if (this.inkling()) {                                             // Inkling 2: 質問を分ける確かめは無い。層 0〜5 は指定したときだけ
        delete p.holdout;
        if (this.$("#pp-before").checked) p.include_before = true;
      }
    } else {
      const nh = ints("#pp-nheads");
      if (!nh.length || !nh.every((x) => Number.isInteger(x) && x >= 1)) { this.status(t("ヘッド数は 1 以上の整数をカンマ区切りで(例 6,5,4)"), true); return null; }
      const tl = num("#pp-tl", 1);
      if (tl === null) { this.status(t("ヘッドを測る層の数は 1 以上の整数で"), true); return null; }
      Object.assign(p, { n_heads: nh, top_layers: tl });
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
    const rm = r.remaining || {}, c = r.combined, rnd = r.round || 1;
    const base = rnd > 1 ? t("{base} の {n} 個の行の合わせたレシピ(元の組 + {r}の追加)", { base: esc(r.base), n: r.n_heads, r: roundsUpTo(rnd - 1) })
      : t("{base} の {n} 個の組", { base: esc(r.base), n: r.n_heads });
    const byRound = Object.entries(r.units_by_round || {}).map(([k, n]) => (k === "0" ? t("元の組 {n} 個", { n }) : t("{k} 回目 {n} 個", { k, n }))).join(" + ");
    return `<div class="pp-refine note"><p>${t("残りから追加の結果({k} 回目、元の一括 {root})です。土台 = {base}。土台を当てても元の方向・元の目標まで下がらなかった {m} 問(A 側 {nA} 問のうち)だけで、A と B の両側に土台を当てたまま方向を作り直し、土台に足す部品を探しました。",
      { k: rnd, root: esc(r.root || r.base), base, m: rm.n ?? "—", nA: rm.n_a ?? "—" })}${rm.padded ? ` ${t("(残りが少ないので、射影の値の高い順に {u} 問で作りました)", { u: rm.used })}` : ""}</p>
      ${r.rank_note ? `<p class="pp-warn">${t("選び方を切り替えました")}: ${esc(r.rank_note)}</p>` : ""}${r.postnorm
        ? `<p class="pp-warn">${t("組のヘッド・層全体の射影のうち、層 {l} は", { l: esc([...(r.postnorm.layers || []), ...(r.postnorm.unsure_layers || [])].slice(0, 8).join(", ")) })}${esc(r.postnorm.note || "")}</p>` : ""}
      ${c ? `<p><b>${t("代表の合わせたレシピ({m} 個 = {parts})", { m: c.n_heads, parts: byRound || "—" })}</b>${c.chosen_by === "fewest_remaining" ? ` <span class="muted">${t("(足した組に推す組が無いので、合わせた残りが最も少ない組)")}</span>` : ""}: ${t("残り {a} → {b} 問", { a: c.remain_before, b: c.remain_after })} ·
        ${t("A 平均 {a} → {b}", { a: fmt(c.a_mean_before, 2), b: fmt(c.a_mean_after, 2) })} · B Δ ${sgn(c.b_delta, 2)} · ${t("B ⊥ {a} → {b}", { a: fmt(c.b_drift_orth_base, 3), b: fmt(c.b_drift_orth, 3) })}
        <button class="small" id="pp-rf-load" title="${t("元の組と足した組を合わせたユニット(全部の回)で今のレシピを置き換える")}">${t("合わせた組でレシピを置き換える")}</button>
        ${INK ? `<button class="small" id="pp-rf-chat" title="${t("土台と足した組を合わせたレシピに置き換えて、3 列チャット(レシピ ×−1・無介入・×+1)を開く。このモデルは応答を読んで決めるので、まずここで試す")}">${t("合わせた組で 3 列チャット")}</button>` : ""}
        <button class="small" id="pp-rf-eval" title="${t("合わせた組のレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)")}">${t("合わせた組を評価へ送る")}</button></p>`
        : (r.combined_rows || []).length ? `<p class="ng"><b>${r.no_recommend === "replies_break" ? t("推せる行の合わせたレシピは、どれも応答が壊れました。") : t("どの行も、合わせると元の方向・元の目標で良くならない(残りが減らず、組の効きも上がらない)か、悪くなります。")}</b></p>
          <p class="note">${t("道具の基準では、足す行を 1 つに決められませんでした。土台のままが無難ですが、候補の行は下の表に残っていて選べます。")} ${mrNext(r.most_reduced, true)}</p>`
        : `<p class="muted">${t("足す組を作れなかったので、合わせた組はありません。")}</p>`}
      ${r.check && !d.writes ? this.refineCheck(r.check) : ""}
      <p class="muted">${t("下の表のどの行も、元の組に足した組です。「レシピを置き換える」「評価へ送る」は、どの行も土台と足した組を合わせたレシピを使います(足した組は土台を当てた状態で探したので、単独では使いません)。強さを上げるのは足した組だけで、土台の部品の強さはそのままです。")}</p></div>`;
  }

  // 止めて選ぶ(rank causal)の残りから追加の結果: 代表の合わせたレシピでの応答での確かめ(rank writes は 6. の段に出す)
  refineCheck(ck) {
    const x0 = ck.rows.none, x1 = ck.rows.recipe;
    const cell = (x) => (x.match_finished !== null && x.match_finished !== undefined ? `${x.match_finished}/${x.finished}` : `${x.finished}/${x.n}`);
    const warn = [ck.match_warning, ck.note && t(ck.note), ...Object.entries(ck.break_warning || {}).map(([side, x]) =>
      t("組を当てると {s} 側の応答が壊れています(繰り返し {l}・空 {e} / {n} 問、無介入では {l0}・{e0})。強さを下げるか、別の組・別の部品の種類を試してください", { s: side, l: x.loop, e: x.empty, n: x.n, l0: x.loop_none, e0: x.empty_none }))].filter(Boolean);
    return `<p class="note">${t("応答での確かめ(代表の合わせたレシピ、validate.match の一致 A / B): 無介入 {a0} / {b0} → 合わせたレシピ {a1} / {b1} · 空 {e1}(無介入 {e0}) · 繰り返し {l1}(無介入 {l0})", {
      a0: cell(x0.A), b0: cell(x0.B), a1: cell(x1.A), b1: cell(x1.B), e1: `${x1.A.empty}/${x1.B.empty}`, e0: `${x0.A.empty}/${x0.B.empty}`, l1: `${x1.A.loop}/${x1.B.loop}`, l0: `${x0.A.loop}/${x0.B.loop}` })}</p>${
      warn.map((w) => `<p class="note ng">${esc(w)}</p>`).join("")}`;
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
      this.loadDirs();                                                 // 一括が作った方向も「作り済みの方向」の選択肢に
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
      <p class="note">${t("質問 {n} 個", { n: d.n_prompts ?? "—" })} · A: ${esc(side(sides.A))} · B: ${esc(side(sides.B))}</p>${fromDirHTML(d.direction || {})}
      <p class="note">${d.writes ? t("選び方: 書き込みの量で全部の部品から(--rank writes、既定)") : t("選び方: 止めて選ぶ(v0.7.0 までの選び方、--rank causal)")}</p>${d.rank_note ? `<p class="note pp-warn">${t("選び方を切り替えました")}: ${esc(d.rank_note)}</p>` : ""}${d.postnorm
        ? `<p class="note pp-warn">${t("組のヘッド・層全体の射影のうち、層 {l} は", { l: esc([...(d.postnorm.layers || []), ...(d.postnorm.unsure_layers || [])].slice(0, 8).join(", ")) })}${esc(d.postnorm.note || "")}</p>` : ""}
      ${this.refineNote(d)}
      ${(d.reused || []).length ? `<p class="note muted">${t("同じ JSON・同じ設定で作り済みの{w}を使い回しました(作り直すときは「最初から作り直す」に印を付けて流し直す)",
        { w: d.reused.map((x) => (x === "direction" ? t("方向") : x === "causal" ? t("因果の結果") : x)).join(t("と")) })}</p>` : ""}
      <p class="note">${d.inkling ? t("目標: 最後に混ぜた出力の A 側の射影の値を {v} まで({src})", { v: fmt(d.target, 2),
        src: esc(d.target_from === "settings" ? t("設定") : d.target_from ? t(d.target_from) : "—") })
        : t("目標: L{L} の A 側の射影の値を {v} まで({src})", { L: (d.causal && (d.causal.score_layers || []).slice(-1)[0]) ?? "—", v: fmt(d.target, 2),
        src: esc(d.target_from === "settings" ? t("設定") : d.target_from ? t(d.target_from) : "—") })}${this.recText(d)}${this.oldRefineNote(d)}${d.refine
        ? `<br><span class="muted">${t("(この目標は、足す組を探したときの新しい方向でのものです。表の合わせたレシピの数と推す行は、元の方向 {dir} の目標 {thr} で決めています)", { dir: esc(d.refine.direction || d.refine.root || d.refine.base), thr: fmt(d.refine.threshold, 2) })}</span>` : ""}</p>
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
    // ニューロンの印: NEURON パネルで開く。同じ層を 2 回実行するモデルの段の番号は地図の層と一致しないので、文字だけにする
    for (const a of out.querySelectorAll("[data-neuron]")) {
      if (!this.onOpenNeuron || this.isLoop()) { const sp = document.createElement("span"); sp.className = "mono"; sp.innerHTML = a.innerHTML; a.replaceWith(sp); continue; }
      a.addEventListener("click", (ev) => { ev.preventDefault(); const [L, n] = a.dataset.neuron.split(":").map(Number); this.onOpenNeuron({ layer: L, n }); });
    }
    this.wireCand(out, d);
    const more = $("#pp-eta-more");
    if (more) more.addEventListener("click", () => { for (const tr of out.querySelectorAll("table.pp-eta tr.pp-more")) tr.classList.remove("hidden"); more.remove(); });
    const table = (d.allocate && d.allocate.table) || [];
    const crow = (key) => combinedRow(d, key);
    const ink = d.inkling || (d.refine && d.refine.inkling);
    if (ink) {                                     // Inkling 2: 評価は思考の後の回答で測る(ランダムな向きの比べる用はまだ無い)
      for (const b of out.querySelectorAll("[data-eval], #pp-rf-eval")) {
        b.title = t("このレシピを評価の画面へ送る(強さ 0 と 1。Inkling 2 は思考の後の回答で測り、問題を少なめにする。ランダムな向きの比べる用はまだ無い。始める前に変えられる)");
      }
    }
    const evp = (recipe) => ({ recipe, doses: [0, 1], random_controls: ink ? 0 : 8 });   // 評価の画面へ送る設定
    for (const b of out.querySelectorAll("[data-load]")) {
      b.addEventListener("click", () => {
        if (d.refine) {                                                  // 残りから追加の結果: どの行も土台と合わせたレシピだけ
          const c = crow(b.dataset.load);
          if (c) this.onLoadProj(combinedName(d, c), c.units, d.name);
          return;
        }
        if (b.dataset.load === "s1") { const s1 = (d.allocate || {}).s1; if (s1) this.onLoadProj(`${d.name}-s1`, s1.units, d.name); return; }
        const r = table.find((x) => String(x.n_heads) === b.dataset.load);
        if (r) this.onLoadProj(`${d.name}-n${r.n_heads}`, r.units, d.name);
      });
    }
    for (const b of out.querySelectorAll("[data-refine]")) {
      b.addEventListener("click", () => this.runRefine(d.name, +b.dataset.refine, b));
    }
    const rf = d.refine && d.refine.combined;
    const rl = $("#pp-rf-load"), re = $("#pp-rf-eval"), rc = $("#pp-rf-chat");
    if (rf && rl) rl.addEventListener("click", () => this.onLoadProj(combinedName(d, rf), rf.units, d.name));
    if (rf && rc) rc.addEventListener("click", () => this.onChatProj(combinedName(d, rf), rf.units, d.name));
    for (const b of out.querySelectorAll("[data-chat]")) {             // 表の行: 土台 + 足した組(残りから追加)か、その組で 3 列チャット
      b.addEventListener("click", () => {
        if (d.refine) { const c = crow(b.dataset.chat); if (c) this.onChatProj(combinedName(d, c), c.units, d.name); return; }
        if (b.dataset.chat === "s1") { const s1 = (d.allocate || {}).s1; if (s1) this.onChatProj(`${d.name}-s1`, s1.units, d.name); return; }
        const r = table.find((x) => String(x.n_heads) === b.dataset.chat);
        if (r) this.onChatProj(`${d.name}-n${r.n_heads}`, r.units, d.name);
      });
    }
    if (rf && re) re.addEventListener("click", () => this.onEval(evp(rf.recipe), combinedName(d, rf)));
    for (const b of out.querySelectorAll("[data-eval]")) {
      b.addEventListener("click", () => {
        if (d.refine) {                                                  // 残りから追加の結果: どの行も土台と合わせたレシピだけ
          const c = crow(b.dataset.eval);
          if (c && c.recipe) this.onEval(evp(c.recipe), combinedName(d, c));
          return;
        }
        if (b.dataset.eval === "s1") {                                   // 倍率を上げる前の組
          const s1 = (d.allocate || {}).s1;
          if (s1 && s1.recipe) this.onEval(evp(s1.recipe), `${d.name}-s1`);
          return;
        }
        const r = table.find((x) => String(x.n_heads) === b.dataset.eval);
        if (r && r.recipe) this.onEval(evp(r.recipe), `${d.name}-n${r.n_heads}`);
      });
    }
  }

  // 「候補を作る」: 組 × 全体の倍率で返事を作り、名前を伏せて採点する(heads_candidates のジョブ)
  candForm(d) {
    const f = this.cf, ok = (d.allocate && (d.allocate.table || []).some((r) => !r.skipped));
    if (!ok) return "";
    const sp = [["llama_server", t("llama-server の既定")], ["common", t("よく使う設定")], ["greedy", "greedy"]];
    return `<div class="pp-cand"><h4 class="pp-sub">${t("強さの候補を作る(名前を伏せて採点する)")}</h4>
      ${d.refine ? `<p class="note">${t("残りから追加の結果なので、候補は各行の合わせたレシピ(土台 + 足した組)に倍率を掛けたもので、元の一括の対照集合の質問で作ります(足した組だけでは作りません)。")}</p>` : ""}
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
    if (d.refine && refineNew(d)) {                                      // 残りから追加の結果: 合わせたレシピの数で推した行
      const c = d.refine.combined;
      const k = c ? c.key ?? c.n_new : null;
      if (c && (k === null || k === undefined)) return ` · <b>${t("推す行あり(合わせたレシピ)")}</b>`;
      if (c) return ` · <b>${t("推す行(合わせたレシピ): {k}", { k: k === "s1" ? "s=1" : t("上限 {n}", { n: k }) })}</b>`;
      return ` · <b>${t("推す行なし: 道具の基準では、足す行を 1 つに決められませんでした")}</b><br><span class="muted">${
        t("土台のままが無難ですが、候補の行は表に残っていて選べます。")} ${mrNext(d.refine.most_reduced, true)}</span>`;
    }
    const r = d.recommended;
    const brk = r && (r.after_replies_break || []).length ? ` <span class="ng">${t("(上限 {k} の組は応答が壊れたので、次の組)", { k: r.after_replies_break.join("・") })}</span>` : "";
    if (r && r.reached === false) return ` · <b>${t("推す組: {n} 個(どれも目標に届かないので、組の効きがいちばん大きい組)", { n: r.n_heads })}</b>${brk}`;
    if (r) return ` · <b>${t("推す組: {n} ヘッド", { n: r.n_heads })}</b>${brk}`;
    const why = d.no_recommend ? t(d.no_recommend.reason) : t("どの組も目標に届かない");
    return ` · <b>${t("推す組なし: 道具の基準では 1 つに決められませんでした({r})", { r: why })}</b><br><span class="muted">${mrNext(d.most_reduced, false)} ${
      d.writes ? t("ほかの手: 「止めて選ぶ」(v0.7.0 までの選び方、--rank causal)、部品の種類を変える、組の行の「残りから追加」")
        : t("ほかの手: 「書き込みの量で選ぶ」(--rank writes、既定)、組の行の「残りから追加」")}</span>`;
  }

  // v0.8.3 までの残りから追加の結果には、合わせたレシピの数での判定が無い(推す組は足した組だけの値)
  oldRefineNote(d) {
    return d.refine && !refineNew(d) ? `<p class="note muted">${t("v0.8.3 までの残りから追加の結果なので、推す組は足した組だけの値(新しい方向)で決めたものです。合わせたレシピの数で選び直すには、元の結果からもう一度残りから追加してください。")}</p>` : "";
  }

  // 1. 方向と基準層
  stage1(d) {
    const sc = d.score || {}, dir = d.direction || {}, nr = dir.n_rows || {};
    if (d.inkling) {                               // Inkling 2: 層ごとの基準は決まっているので、測る点の A と B の平均を並べる
      const ink = d.inkling, sk = dir.skipped || {};
      const rows = (sc.layers || []).map((x) => `<tr><td class="l">${esc(inkPoint(x))}</td><td>${fmt(x.A, 2)}</td><td>${fmt(x.B, 2)}</td><td>${fmt(x.A - x.B, 2)}</td></tr>`).join("");
      const nu = ink.n_questions || {};
      return `<section class="pp-stage"><h4>${t("1. 方向と測る点")}</h4>
        ${nu.direction ? `<p class="note">${t("使った質問: 方向 A {da} / B {db}、測る段 A {ma} / B {mb}(対照集合 A {na} / B {nb} から、側ごとに最大 {mx} 問を等間隔に)", {
          da: nu.direction.A, db: nu.direction.B, ma: nu.measure.A, mb: nu.measure.B, na: nu.total.A, nb: nu.total.B, mx: nu.max_examples ?? "—" })}</p>` : ""}
        ${noteHTML(t("A と B で同じ質問を読ませ、段ごとに MLP が読む入力の平均の差を方向にします(層 6〜17 は 3 回目の段、繰り返しのゲートの後)。部品の射影の基準はその層の最後の段の方向です。組は「最後に混ぜた出力」で A 側の値を下げるように作ります。"), "muted note")}
        <p class="note mono">positions ${esc(dir.positions || "—")} · ${t("方向を作った例")} A ${nr.A ?? "—"} / B ${nr.B ?? "—"}${
          (sk.A || sk.B) ? ` · ${t("答えの始まりが無く読めなかった会話")} A ${sk.A || 0} / B ${sk.B || 0}` : ""}</p>
        ${ink.position_kind === "last" ? `<p class="note">${t("このモデルは答えの前に必ず思考を書きます。質問の最後のトークンで作った方向は、思考と答えを書く位置では差が小さくなることがあります。答えの口調や書き方を狙うときは「答えの始まり」か「答えのトークン全部」も試してください")}</p>` : ""}
        <table class="tbl pp-tbl"><tr><th class="l">${t("測る点")}</th><th>A</th><th>B</th><th>A − B</th></tr>${rows}</table></section>`;
    }
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
    const vwhy = (r) => (r.excluded ? `<span class="pp-ex">${t("外した: B 側の直交成分のずれが {d} を超える(ふつうの入力の残差まで大きく動かす)", { d: fmt(a.drift_max ?? 0.5, 2) })}</span>`
      : r.effect !== undefined && r.effect !== null && r.effect <= 0 ? `<span class="muted">${t("組に使わない: A 側の下がり − B 側の動きの大きさが 0 以下")}</span>` : "");
    const vrows = (v.top || []).map((r, i) => `<tr class="${r.excluded ? "rnd" : ""}"><td>${i + 1}</td><td class="l">${unitChip(r)}</td><td>${t(KIND_LABEL[r.kind] || r.kind)}</td>
      <td>${sgn(r.score, 3)}</td><td>${sgn(r.gap_drop, 3)}</td><td>${sgn(r.causal, 3)}</td><td>${sgn(r.b_side_delta, 3)}</td><td>${fmt(r.drift_orth_b, 3)}</td><td class="l">${vwhy(r)}</td></tr>`).join("");
    const stop = { target: t("全部 1"), raised: t("上げて届いた"), s_max: t("上限で止めた"), drift: t("B 側の直交成分のずれで止めた"),
      b_move: t("B 側が A 側へ動くので止めた"), no_gain: t("上げても良くならないので止めた"), iters: t("回数の上限で止めた") };
    const STRENGTH = { per_unit: t("部品ごと"), uniform: t("全部を同じ倍率") };
    // 倍率ごとの A 平均・B Δ・B ⊥(格子の点と、上限の手前を詰めた点。超 = 上限を超えた点)
    const X = a.loop_exit;                                               // ループのモデル: 1 回目の出口の段
    const steps = (r) => {
      const st = (r.raise || []).filter((p) => p.phase === "step");          // 部品ごとの強さ: 回ごとに採った点
      if (st.length) {
        const tr = st.map((p) => `<tr><td>${p.step}</td><td>${fmt(p.s_max, 2)} / ${fmt(p.s_median, 2)} / ${fmt(p.s_min, 2)}</td><td>${fmt(p.a_mean, 2)}</td><td>${sgn(p.b_delta, 2)}</td><td>${fmt(p.b_orth, 3)}</td><td>${fmt(p.effect, 2)}</td></tr>`).join("");
        return `<details class="note"><summary>${t("回ごと({n} 回)", { n: st.length })}</summary><table class="evx mono"><thead><tr><th>${t("回")}</th><th>${t("s 最大 / 中央値 / 最小")}</th><th>${t("A 平均")}</th><th>B Δ</th><th>B ⊥</th><th>${t("効き")}</th></tr></thead><tbody>${tr}</tbody></table></details>`;
      }
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
    const BLOCK = { s_max: "s の上限", b_move: "B 側が A 側へ動く", drift: "直交成分のずれ", limit: "上限" };
    const probesHTML = (r) => {                                          // 部品ごとの強さ: 回ごとの、残りやすい質問と部品ごとの効率
      const P = r.probes || [], hs = r.heads || [];
      if (!P.length) return "";
      const cell = (p, i) => (p.eff[i] === null || p.eff[i] === undefined ? `<span class="muted">${esc(t(BLOCK[p.blocked[i]] || p.blocked[i] || "—"))}</span>`
        : `${sgn(p.eff[i], 3)} <span class="muted">s${fmt(p.s[i], 2)}</span>`);
      const rows = hs.map((h, i) => `<tr><td class="l">${unitChip(h)}</td>${P.map((p) => `<td>${cell(p, i)}</td>`).join("")}<td>${fmt(h.s, 2)}</td></tr>`).join("");
      const qs = P.map((p) => `<details><summary class="muted">${t("{k} 回目: 射影の値が残りやすい質問 {n} 問(その質問の A 平均 {a})", { k: p.step, n: (p.hard || []).length, a: fmt(p.a_hard, 2) })}</summary>
        <table class="evx mono dv-ex"><tbody>${(p.questions || []).map((q, j) => `<tr><td class="w">${esc(q)}</td><td>${fmt((p.a_vals || [])[j], 2)}</td></tr>`).join("")}</tbody></table></details>`).join("");
      return `<details class="note pp-probes"><summary>${t("強さの決め方(部品ごとの効率、{n} 回)", { n: P.length })}</summary>
        ${noteHTML(t("回ごとに、射影の値が残りやすい A 側の質問を選び直し、部品ごとに s を 1.5 倍にしてみて、その質問での組の効き(A 側の下がり − B 側の動きの大きさ)の増え方を s の増えで割った値(効率)を測ります。上げ幅は効率に比例して配り、効率が 0 以下の部品と上げられない部品は据え置きます。表の値は効率と、そのときの s。"), "muted note")}
        <table class="evx mono"><thead><tr><th class="l">${t("部品")}</th>${P.map((p) => `<th>${t("{k} 回目", { k: p.step })}</th>`).join("")}<th>${t("決まった s")}</th></tr></thead><tbody>${rows}</tbody></table>${qs}</details>`;
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
    const R = d.refine, rnd = (R && R.round) || 1;                      // 残りから追加の結果: 行は「土台 + 足した組」
    const nAroot = R ? (R.remaining || {}).n_a : null;
    const btns = (key, r, refine) => {
      const c = R ? combinedRow(d, key) : null, off = R && !c ? ` disabled` : "";
      const ttl = (x, y) => (R ? (c ? x : t("この行には土台と合わせたレシピがありません")) : y);
      return `<td class="l"><button class="small" data-load="${key}" title="${ttl(t("土台(元の組とそれまでの回の追加)と、この行の足した組を合わせたレシピで、今のレシピを置き換える(足した組は単独では使わない)"), t("この組(proj のユニット)で今のレシピを置き換える"))}"${off}>${R ? t("合わせてレシピを置き換える") : t("レシピを置き換える")}</button>
          ${INK ? `<button class="small" data-chat="${key}" title="${R ? t("土台とこの行の足した組を合わせたレシピに置き換えて、3 列チャットを開く") : t("この組のレシピに置き換えて、3 列チャットを開く")}"${off}>${t("3 列チャット")}</button>` : ""}
          <button class="small" data-eval="${key}" title="${ttl(t("土台と、この行の足した組を合わせたレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)"), t("この組のレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)"))}"${r.recipe && !off ? "" : " disabled"}>${R ? t("合わせて評価へ送る") : t("評価へ送る")}</button>
          ${refine ? `<button class="small" data-refine="${key}" title="${R ? t("この行の合わせたレシピ(土台 + この組)を当てても元の目標まで下がらない質問だけで、もう一度足す部品を探す(重ねる)。結果は新しい一括の結果(名前 _r1 …)として出る") : t("この組を当てても目標まで下がらない質問(残り)だけで、組を当てたまま方向を作り直し、足す部品を探す。結果は新しい一括の結果(名前 _r1 …)として出る")}"${(R ? c && c.remain_after > 0 : r.remain > 0) ? "" : " disabled"}>${t("残りから追加")}</button>` : ""}</td>`;
    };
    const cells = (r) => `<td>${r.remain ?? "—"}${nA ? ` / ${nA}` : ""}</td><td title="${t("A 側の射影の値の最大 {v}", { v: fmt(r.a_max, 2) })}">${fmt(r.a_mean, 2)}</td>
        <td>${sgn(r.b_side_delta, 2)}</td><td>${fmt(r.b_drift_orth, 3)}</td><td>${r.effect === undefined ? "—" : fmt(r.effect, 2)}</td><td class="${r.reached ? "ok" : "ng"}">${r.reached ? "✓" : "×"}</td>`;
    // 合わせたレシピの数(元の方向・元の目標で。前 = 土台だけ、後 = 合わせたもの)
    const ccells = (c) => {
      if (!c) return `<td colspan="6" class="ng">${t("合わせたレシピがありません")}</td>`;
      const rc = c.reached_combined;
      return `<td title="${t("前 = 土台だけ、後 = 合わせたレシピ。元の方向・元の目標で、A 側の全部の質問")}">${c.remain_before} → <b>${c.remain_after}</b>${nAroot ? ` / ${nAroot}` : ""}</td>
        <td title="${t("A 側の射影の値の最大 {a} → {b}", { a: fmt(c.a_max_before, 2), b: fmt(c.a_max_after, 2) })}">${fmt(c.a_mean_before, 2)} → <b>${fmt(c.a_mean_after, 2)}</b></td>
        <td title="${t("土台だけ {v}", { v: sgn(c.b_delta_base, 2) })}">${sgn(c.b_delta, 2)}</td><td title="${t("土台だけ {v}", { v: fmt(c.b_drift_orth_base, 3) })}">${fmt(c.b_drift_orth, 3)}</td>
        <td title="${t("土台だけ {v}", { v: fmt(c.effect_base, 2) })}">${fmt(c.effect, 2)}</td><td class="${rc === undefined ? "" : rc ? "ok" : "ng"}">${rc === undefined ? "—" : rc ? "✓" : "×"}</td>`;
    };
    const roundName = (k) => (+k === 0 ? t("元の組") : t("{k} 回目", { k }));
    const baseChips = (c, r) => {                                        // 土台の部品(薄く、回ごと。数が多いので畳む)
      if (!c) return "";
      if (!c.parts) {
        const us = c.units.slice(0, c.units.length - (r.units || []).length);
        return `<details class="pp-base"><summary>${t("土台 {n} 個(元の組)", { n: us.length })}</summary><div class="pp-base-r"><span class="pp-round">${t("元の組")}(${us.length})</span> <span class="mono">${esc(us.join(" "))}</span></div></details>`;
      }
      const by = new Map();
      for (const q of c.parts.filter((x) => x.round < rnd)) { if (!by.has(q.round)) by.set(q.round, []); by.get(q.round).push(q); }
      const sum = [...by.entries()].map(([k, qs]) => `${roundName(k)} ${qs.length}`).join(" + ");
      return `<details class="pp-base"><summary>${t("土台 {n} 個({parts})", { n: [...by.values()].reduce((x, qs) => x + qs.length, 0), parts: sum })}</summary>${
        [...by.entries()].map(([k, qs]) => `<div class="pp-base-r"><span class="pp-round">${roundName(k)}(${qs.length})</span> ${qs.map((q) => unitChip(q, ` s${fmt(q.s, 2)}`)).join(" ")}</div>`).join("")}</details>`;
    };
    const dupNote = (c) => (c && (c.dups || []).length ? `<div class="note pp-dup" title="${t("射影を両方当てます(どちらも元の重みから作って足し合わせる。探したときと同じ当て方)。方向: {d}", {
      d: esc(c.dups.map((x) => `${partName(x)}: ${x.directions.join(" / ")}`).join(" · ")) })}">${t("同じ部品が 2 つの方向で入っています: {u}", {
      u: c.dups.map((x) => `${partName(x)}(${x.rounds.map((k, i) => `${roundName(k)} s${fmt(x.s[i], 2)}`).join("、")})`).join(" · ") })}</div>` : "");
    const crossNote = (c) => (c && (c.cross_visit || []).length ? `<div class="note pp-dup" title="${t("探したときは土台と足した組を別々に当てて足し合わせたが、合わせたレシピでは同じ重みへの 2 つの射影を続けて掛ける。強さが 1 を超えると効き方が大きく違うことがある。表の数は合わせたレシピで測った値")}">${
      t("ループの同じ重みを、土台と足した組が別の回の段で使っています: {u}(探したときと当て方が違う)", { u: c.cross_visit.map((x) => `${esc(x.base)} / ${esc(x.added)}`).join(" · ") })}</div>` : "");
    const cExit = (c) => {                                               // ループのモデル: 合わせたレシピの、1 回目の出口の段
      const e0 = c && c.exit_before, e1 = c && c.exit_after;
      if (!e0 || !e1) return "";
      return `<div class="muted note">${t("段 {L}(1 回目の出口、合わせたレシピ): A {a0} → {a1}、B Δ {bd}、B ⊥ {bo}", { L: e1.layer, a0: fmt(e0.a_mean, 2), a1: fmt(e1.a_mean, 2), bd: sgn(e1.b_delta, 2), bo: fmt(e1.b_drift_orth, 3) })}</div>`;
    };
    const cHo = (c) => {                                                 // 元の一括が質問を分けていたら、選ぶのに使っていない質問での合わせたレシピの数
      const h = c && c.holdout;
      if (!h) return "";
      return `<div class="muted note">${t("選ぶのに使っていない質問(合わせたレシピ): 残り {a} → {b} / {n}、A 平均 {ma} → {mb}、B Δ {bd}、B ⊥ {bo}、効き {e}{x}", {
        a: h.remain_before, b: h.remain_after, n: h.n_a, ma: fmt(h.a_mean_before, 2), mb: fmt(h.a_mean_after, 2), bd: sgn(h.b_delta, 2), bo: fmt(h.b_drift_orth, 3),
        e: fmt(h.effect, 2), x: h.reached ? "" : t("(目標に届かない)") })}</div>`;
    };
    const addedNote = (r) => `<div class="muted note" title="${t("新しい方向 {dir} で、残りの {k} 問を、土台を当てた状態で測った値。足した組は単独では使いません", { dir: esc(d.name), k: nA ?? "—" })}">${
      t("足した組だけ(参考、新しい方向で): 残り {rm} / {k}、A 平均 {am}、B Δ {bd}、B ⊥ {bo}", { k: nA ?? "—", rm: r.remain ?? "—", am: fmt(r.a_mean, 2), bd: sgn(r.b_side_delta, 2), bo: fmt(r.b_drift_orth, 3) })}</div>`;
    const chipsOf = (r) => (r.heads || []).map((h) => unitChip(h, ` <span class="muted">s${fmt(h.s, 2)}</span>`)).join(" ");
    const unitsCell = (key, r, extra) => {
      if (!R) return `<td class="units">${chipsOf(r)}${extra}</td>`;
      const c = combinedRow(d, key);
      return `<td class="units">${verdictNote(c)}${baseChips(c, r)}<div class="pp-added"><span class="pp-round">${t("足した組({k} 回目)", { k: rnd })}</span> ${chipsOf(r)}</div>${dupNote(c)}${crossNote(c)}${cExit(c)}${cHo(c)}${extra}${addedNote(r)}</td>`;
    };
    const s1rec = R && refineNew(d) && !!(combinedRow(d, "s1") || {}).recommended;
    const s1c = R ? combinedRow(d, "s1") : null, s1mr = R ? !!(s1c || {}).most_reduced : !!(a.s1 || {}).most_reduced;
    const s1 = a.s1 ? `<tr class="pp-s1${s1rec ? " rec" : ""}${s1mr ? " mr" : ""}"><td>s=1${recBadge(s1rec)}${mrBadge(s1mr, rowBroke(a.s1, s1c))}</td>${unitsCell("s1", a.s1, `<div class="muted note">${kinds(a.s1.kinds)} · ${t("倍率を上げる前(取り除くだけ)")}</div>${R ? "" : hoNote(a.s1) + exitNote(a.s1)}`)}
        <td>1 / 1 / 1</td>${R ? ccells(combinedRow(d, "s1")) : cells(a.s1)}${btns("s1", a.s1, false)}</tr>` : "";
    const cmp = s1 + (a.table || []).slice().sort((x, y) => y.n_heads - x.n_heads).map((r) => {
      if (r.skipped) return `<tr><td>${r.n_heads}</td><td class="excl" colspan="8">${esc(r.skipped)}</td></tr>`;
      const rec = R && refineNew(d) ? !!(combinedRow(d, r.n_heads) || {}).recommended : r.recommended;
      const rc = R ? combinedRow(d, r.n_heads) : null, mr = R ? !!(rc || {}).most_reduced : !!r.most_reduced;
      return `<tr class="${[rec && "rec", mr && "mr"].filter(Boolean).join(" ")}" data-n="${r.n_heads}"><td>${r.n_heads}${recBadge(rec)}${mrBadge(mr, rowBroke(r, rc))}</td>${unitsCell(r.n_heads, r, `${r.replies_break ? `<div class="note ng">${t("応答での確かめで応答が壊れた(推さない)")}</div>` : ""}<div class="muted note">${kinds(r.kinds)}</div>${R ? "" : hoNote(r) + exitNote(r)}${steps(r)}${probesHTML(r)}`)}
        <td>${fmt(r.s_max, 2)} / ${fmt(r.s_median, 2)} / ${fmt(r.s_min ?? Math.min(...(r.heads || []).map((h) => h.s)), 2)}<div class="muted note">${esc(stop[r.stop] || r.stop || "")}</div>${R ? `<div class="muted note">${t("足した組の強さ(土台はそのまま)")}</div>` : ""}</td>${R ? ccells(combinedRow(d, r.n_heads)) : cells(r)}${btns(r.n_heads, r, true)}</tr>`;
    }).join("");
    const ck = d.check || (R && R.check);
    const cell = (x) => (x.match_finished !== null && x.match_finished !== undefined ? `${x.match_finished}/${x.finished}` : `${x.finished}/${x.n}`);
    const ckRows = ck ? ["none", "recipe"].map((k) => { const x = ck.rows[k]; return `<tr><td class="l">${k === "none" ? t("無介入") : t("組")}</td><td>${cell(x.A)}</td><td>${cell(x.B)}</td>
      <td>${x.A.empty} / ${x.B.empty}</td><td>${x.A.loop} / ${x.B.loop}</td><td>${x.A.at_limit} / ${x.B.at_limit}</td></tr>`; }).join("") : "";
    return `<section class="pp-stage"><h4>${t("3. 書き込みの順位づけ")}</h4>
      ${noteHTML(t("質問 1 問につき 1 回流すだけで、全部の部品(注意ヘッド・線形注意のヘッド・MLP のニューロン)が、その層の方向にどれだけ書いたかを読みます。点数 = 選ぶ層で「s=1 の射影で直接に縮まる A と B の差」。後ろの層での変換は見ないので、後ろの書き手を動かしているだけの部品は点数に出ません。"), "muted note")}
      <p class="note mono">${t("選ぶ層")} L${w.select_layer ?? "—"} · A ${(w.n || {}).A ?? "—"} / B ${(w.n || {}).B ?? "—"} · ${t("A−B の差")} ${fmt(w.gap, 2)} · ${kinds(w.n_units)}</p>
      ${(w.zero_layers || []).length ? `<p class="note pp-warn pp-zero">${t("方向の長さが 0 の層 {l} を外しました。A 側と B 側の残差がまったく同じ層です(短い畳み込みの層は数トークン前までしか見ないので、両側で違う system プロンプトがまだ届いていません)。この層では方向に沿った書き込みが決まりません。", { l: esc(w.zero_layers.slice(0, 8).join(", ")) })}</p>` : ""}
      ${share ? `<p class="note">${t("直接の書き込みの和 ÷ 差")}: ${share}</p>` : ""}
      ${(w.layer_scores || []).length ? `<p class="note" title="${t("その層の注意の出力全体・MLP の出力全体を s=1 で射影したときに、選ぶ層で直接に縮む A−B の差の割合(層全体の部品の選び方の目安)")}">${t("層ごとの直接の書き込み ÷ 差(上位 6 層)")}: ${w.layer_scores.slice(0, 6).map((x) => `L${x.layer} ${t("注意全体")} ${sgn(100 * x.attn_share, 0)}% · MLP ${sgn(100 * x.mlp_share, 0)}%`).join(" | ")}</p>` : ""}</section>
      <section class="pp-stage"><h4>${t("4. 上位を 1 個ずつ確かめる")}</h4>
      ${noteHTML(t("点数の上位を 1 個ずつ s=1 で射影して(または止めて)流し直し、因果スクリーニングと同じ物差しで測ります。測った「A−B の差の縮み」と点数の順位相関を出します。"), "muted note")}
      <p class="note mono">${kinds(v.kinds)}${corr ? ` · ${t("順位相関")} ${corr}` : ""}</p>
      <div class="evx-scroll"><table class="evx mono pp-top"><thead><tr><th>#</th><th class="l">${t("部品")}</th><th>${t("種類")}</th><th title="${t("選ぶ層で s=1 の射影で直接に縮まる A−B の差")}">${t("点数")}</th>
        <th title="${t("流し直して測った A−B の差の縮み")}">${t("測った縮み")}</th><th title="${t("A 側の射影の値の下がり方")}">${t("A の下がり")}</th><th title="${t("B 側の射影の値の変化")}">B Δ</th><th title="${t("B 側の直交成分のずれ")}">B ⊥</th><th></th></tr></thead><tbody>${vrows}</tbody></table></div></section>
      <section class="pp-stage"><h4>${t("5. s=1 から組を作る")}</h4>
      ${noteHTML(t("確かめた部品を、A 側の下がり − B 側の動きの大きい順に、s=1 で 1 個ずつ組に足します(A 側の平均が下がり、組の効きが増え、上限の中のときだけ)。目標に届けば強さは全部 1。届かなければ強さを上げます。部品ごと(既定)では、部品ごとに s を 1.5 倍にしてみて、射影の値が残りやすい質問での組の効きの増え方(効率)を測り、効率に比例して上げ幅を配ることを繰り返します(効く部品は高く、B 側を押す部品は低く。どの部品も 1 より下げず、組から外さない)。s が 1 を超えると成分を反転させるので、B 側(ふつうの入力)が A 側へ動くことがあります。目標に届けば届く最小の上げ幅、届かなければ上限の中で組の効きがいちばん大きい点を使います。効き = A 側の下がり − B 側の動きの大きさ。上限 = B 側の直交成分のずれ {d}、B 側が A 側へ動く量は A と B の差の {m}%。", { d: fmt(a.drift_max ?? 0.5, 2), m: fmt(100 * (a.b_move_max ?? 0.1), 0) }), "muted note")}
      <p class="note">${t("届かないときの強さの決め方")}: <b>${esc(STRENGTH[a.strength || "uniform"] || a.strength || "")}</b>${a.strength ? "" : ` <span class="muted">${t("(v0.8.2 までの結果は全部を同じ倍率)")}</span>`}</p>
      ${a.holdout ? `<p class="note">${t("質問を半分に分けました: 偶数番目で選び、奇数番目(選ぶのに使っていない質問、A {a} 問・B {b} 問)で各行の数と応答の確かめを測り直しています(目標 {t})", { a: a.holdout.n_a, b: a.holdout.n_b, t: fmt(a.holdout.target, 2) })}</p>` : ""}
      ${R ? `<p class="note pp-refine-h">${t("残りから追加の結果なので、表の数は、土台(元の組とそれまでの回の追加)とその行の足した組を合わせたレシピを、元の方向 {dir}・元の目標 {thr} で測った値です(前 = 土台だけ、後 = 合わせたもの)。足した組だけの値は、各行の補足に参考として出します。",
        { dir: esc(R.direction || R.root || R.base), thr: fmt(R.threshold, 2) })}</p>` : ""}
      <div class="evx-scroll"><table class="evx mono pp-cmp"><thead><tr><th title="${t("組の部品の数の上限")}">${t("上限")}</th><th class="l" title="${R ? t("薄い字 = 土台の部品(何回目に入ったか)、ふつうの字 = この行で足した部品") : t("組の部品と強さ s")}">${t("部品と強さ")}</th><th title="${t("部品ごとの強さ s の最大・中央値・最小")}">${t("s 最大 / 中央値 / 最小")}</th>
        <th title="${t("目標を超えて残った A 側の例の数")}">${R ? t("残り(前 → 後)") : t("残り")}</th><th title="${t("A 側の射影の値の平均")}">${R ? t("A 平均(前 → 後)") : t("A 平均")}</th><th title="${t("B 側の射影の値の変化")}">B Δ</th>
        <th title="${t("B 側の直交成分のずれ(小さいほど他を壊さない)")}">B ⊥</th><th title="${t("A 側の下がり − B 側の動きの大きさ")}">${t("効き")}</th><th title="${t("A 側の射影の値の平均が目標に届いたか")}">${t("届いた")}</th><th></th></tr></thead><tbody>${cmp}</tbody></table></div>
      ${noteHTML(t("推奨は目安です(合否の判定ではありません)。") + " " + t("推奨 = 目標に届いた組のうち、残る例の数が最も少ない組と同程度(A 側の 5% まで)で、B 側のずれが最小の 1.25 倍までの組の中で、部品が最も少ない組。どれも届かなければ、上限の中で組の効きがいちばん大きい組と同程度(差の 5% まで)で部品が最も少ない組。") + (R ? " " + t("残りから追加の結果では、推奨は合わせたレシピの数(元の方向・元の目標)で決めます。足すと残りが増える・組の効きが下がる・B 側の上限を超えて土台より悪くなる行は推さず、どの行も良くならなければ推しません(土台のままが無難ですが、候補の行は表に残っていて選べます)。推す行の応答が壊れていれば次の行にします。") : "") + " " + MR_NOTE() + (R ? " " + t("残りから追加の結果では、どちらの印も合わせたレシピの値で決めます。") : ""), "muted note")}
      ${(a.notes || []).map((x) => `<p class="note ng">${x.code === "s1_low"
        ? t("倍率を上げる前の組(取り除くだけ)は、目標までの差の {p}% しか縮めません。直接書き込む部品の成分を取り除くだけでは届かない題材で、強さを上げて(成分を反転させて)届かせています", { p: fmt(100 * x.closed, 0) })
        : x.code === "loop_inner" ? t("組にループの中の部品(段 {a}〜{b} の重み)が入っています。重みは 2 回とも使われるので、1 回目(段 {c}〜{d})にも効きます", { a: x.lo, b: x.hi, c: x.first[0], d: x.first[1] })
        : esc(x.text || "")}</p>`).join("")}</section>
      ${ck ? `<section class="pp-stage"><h4>${R ? t("6. 応答での確かめ(代表の合わせたレシピ)") : t("6. 応答での確かめ")}</h4>
        ${noteHTML(t("選んだ組と無介入で応答を書かせ、書き終えた応答のうち validate.match に一致した数を数えます(A 側 / B 側)。射影の値が目標に届いても応答が変わらないことがあるので、その確かめです。"), "muted note")}
        ${R ? `<p class="note">${t("残りから追加の結果では、代表の行({k})の合わせたレシピ(土台 + 足した組、{n} 個)で、元の一括と同じ質問・同じ長さで確かめています{h}。", { k: ck.key === "s1" ? "s=1" : ck.key ?? "—", n: ck.n_heads ?? "—", h: ck.held_out_contrast ? t("(元の一括が分けた、選ぶのに使っていない質問)") : "" })}</p>` : ""}
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
    const R = d.refine;                                                  // 残りから追加の結果: 行は「土台 + この組」で使う
    const crow = (n) => combinedRow(d, n);
    const cmp = (a.table || []).slice().sort((x, y) => y.n_heads - x.n_heads).map((r) => {
      if (r.skipped) return `<tr><td>${r.n_heads}</td><td class="excl" colspan="9">${esc(r.skipped)}</td></tr>`;
      const chips = (r.heads || []).map((h) => headChip(h.layer, h.head, ` <span class="muted">s${fmt(h.s, 1)}</span>`)).join(" ");
      const c = crow(r.n_heads), off = R && !c ? " disabled" : "";
      const nb = c ? (c.n_base ?? (c.units.length - (r.units || []).length)) : 0;
      const rec = R && refineNew(d) ? !!(c || {}).recommended : r.recommended;
      const mr = R ? !!(c || {}).most_reduced : !!r.most_reduced;
      return `<tr class="${[rec && "rec", mr && "mr"].filter(Boolean).join(" ")}" data-n="${r.n_heads}"><td>${r.n_heads}${recBadge(rec)}${mrBadge(mr, rowBroke(r, R ? c : null))}</td><td class="units">${R ? verdictNote(c) : ""}${R && c ? `<div class="pp-base">${t("土台 {n} 個(元の組{k})", { n: nb, k: (R.round || 1) > 1 ? t(" + {r}の追加", { r: roundsUpTo(R.round - 1) }) : "" })}</div><div class="pp-added">` : ""}${chips}${R && c ? "</div>" : ""}</td>
        <td>${fmt(r.s_max, 2)}</td><td>${r.remain ?? "—"}${nA ? ` / ${nA}` : ""}</td><td>${fmt(r.a_max, 2)}</td><td>${fmt(r.a_mean, 2)}</td><td>${sgn(r.b_side_delta, 2)}</td><td>${fmt(r.b_drift_orth, 3)}</td>
        <td class="${r.reached ? "ok" : "ng"}">${r.reached ? "✓" : "×"}</td>
        <td class="l">${c ? `<div class="note">${t("元の組と合わせると(元の方向・元の目標で): 残り {a} → {b} 問 · A 平均 {ma} → {mb} · B Δ {bd} · B ⊥ {c}", { a: c.remain_before, b: c.remain_after, ma: fmt(c.a_mean_before, 2), mb: fmt(c.a_mean_after, 2), bd: sgn(c.b_delta, 2), c: fmt(c.b_drift_orth, 3) })}</div>` : R ? `<div class="note ng">${t("この行には土台と合わせたレシピがありません")}</div>` : ""}
          <button class="small" data-load="${r.n_heads}" title="${R ? t("土台(元の組とそれまでの回の追加)とこの組を合わせたユニットで今のレシピを置き換える(この組は土台を当てた状態で探したので、単独では使わない)") : t("この組(proj のユニット)で今のレシピを置き換える")}"${off}>${R ? t("元の組と合わせてレシピを置き換える") : t("レシピを置き換える")}</button>
          <button class="small" data-eval="${r.n_heads}" title="${R ? t("土台とこの組を合わせたレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)") : t("この組のレシピを評価の画面へ送る(強さ 0 と 1、ランダムな向きの比べる用 8 本が入る。始める前に変えられる)")}"${r.recipe && !off ? "" : " disabled"}>${R ? t("合わせて評価へ送る") : t("評価へ送る")}</button>
          <button class="small" data-refine="${r.n_heads}" title="${R ? t("この行の合わせたレシピ(土台 + この組)を当てても元の目標まで下がらない質問だけで、もう一度足すヘッドを探す(重ねる)。結果は新しい一括の結果(名前 _r1 …)として出る") : t("この組を当てても目標まで下がらない質問(残り)だけで、組を当てたまま方向を作り直し、因果 → 配分で足すヘッドを探す。結果は新しい一括の結果(名前 _r1 …)として出る")}"${(R ? c && c.remain_after > 0 : r.remain > 0) ? "" : " disabled"}>${t("残りから追加")}</button></td></tr>`;
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
      ${noteHTML(t("推奨は目安です(合否の判定ではありません)。") + " " + t("推奨 = 目標に届いた組のうち、残る例の数が最も少ない組と同程度(A 側の 5% まで)で、B 側のずれが最小の 1.25 倍までの組の中で、ヘッドが最も少ない組。ヘッドの名前をクリックでくわしく見る。") + " " + MR_NOTE() + (R ? " " + t("残りから追加の結果では、どちらの印も合わせたレシピの値で決めます。") : ""), "muted note")}
      </section>`;
  }
}
