// direction ビューの「FiSH」: FiSH(Formula-instantiated Scene-gated Hidden units、docs/fish.md。すでにある MLP のニューロンを、
// 学習なしで式から計算した重みで書き換えて、決まった場面でだけ方向を書くようにする)を、この画面だけで最初から最後まで作る。
// 1. 対照集合: 「ファイルを選ぶ」(configs/contrast/ と <root>/contrasts/)か「JSON を書く」(ひな形 configs/fish/ から)。
//    先に当てておくレシピ(iv)の欄(CHiPS のレシピを当てた状態で作るとき)。
// 2. 方向を作る(fish_direction のジョブ): 作り方(今まで / 同じ返事を両方の条件で読む)、返事の位置の区切り、層。
//    結果は層 × 区切りごとの長さと、今までの作り方との向きの近さ。
// 3. ニューロンの表(nw_neurons のジョブ): 層ごとに、ふだんの働きの小さい順。既定の候補(表の上から必要な個数)に印が付き、
//    チェックで選び直せる。
// 4. 鍵を作る(fish_keys のジョブ): 区切りごとの位置の鍵と、形に要る鍵をまとめて作る。比較用の鍵との比べの表。
// 5. レシピを組む(/api/fish/units): 層 × 区切りごとのユニット。「レシピに足す」「このレシピを当てた状態で鍵を作り直す」。
// 6. 確かめる: 式どおりか(fish_verify)と、場面・位置ごとに実際に足した量(fish_check)。
// 7. 返事を読む(fish_behave のジョブ): ユニットを当てて、狙う場面と書き込みたくない場面の質問に答えさせた返事と、数えた結果。
// 「まとめて作る」(fish_pipeline のジョブ)は 2〜6 を順に回す。層と強さは使う人が入れる(自動では決めない)。
// 強さは区切りごとの欄(区切り 0,1-3,4- なら 3 つ)。上限つきの形のしきい値は 1 つの欄(0.4-0.8 か、区切りごとに 0.4-0.8,0.3-0.7,0.3-0.7)。
// 「すでにある方向を使う」に名前を入れると、「まとめて作る」は方向を作り直さない。
// 6. の「足した量」には、位置ごとの平均の表のほかに、質問ごとの値の要約(区間ごとに、ほとんど書けていない質問の数・狙った量を
// 超えた質問の数と、書けていない質問の一覧)を出す。平均だけでは、一部の質問でまったく書けていないことが隠れるため。
// 「重みに入れる前に試す」(fish_try のジョブ): 2. で作った方向を、ニューロンにも鍵にも入れずに、狙う場面の質問の返事の位置どおりに
// 層の出力へ直接足して返事を作る。層・区切りごとの強さを変えて、返事と数(繰り返し・空・正規表現の一致)を並べて比べる。
// 「鍵の作り方」(表の fish.key_fit・key_c): 位置の鍵を、いまの作り方(既定)か、最小二乗の鍵(ますごとの重み。docs/fish.md §3)で作る。
// 「まとめて作る」「鍵をまとめて作る」「当てた状態で鍵を作り直す」に効く。どちらで作ったかは、鍵の表・まとめて作った結果・足した量の上に出る。
// 6. の「足した量」には、変えたくない場面の群ごとの漏れの表(群 × 区間ごとに、狙った量の 0.2 以上を書いた質問の数と書いた量の平均、
// 位置 0 だけの数)も出す。
// 遅れて届いた結果で、後から選んだもの・後から投げたジョブの結果を上書きしない(欄ごとに最後に投げたジョブだけを受け取る。
// 一覧の読み込みとユニットの組み立ては通し番号で見分ける。手で選び直したニューロンは、表を取り直しても残す)。
// 「CHiPS のヘッドの分を場面で取り除く」(fish_heads のジョブ、gne nw heads、docs/fish.md「ヘッドの分を場面で取り除く」): CHiPS のレシピ
// (proj のユニット)が弱めるヘッドの書き込みを、ヘッドの重みは変えずに、同じ層の MLP の 2 個 1 組が、場面の鍵が開いた位置でだけ
// 取り除くユニットを作る。対照集合は 1. の欄のもの。選ぶのは CHiPS のレシピ・場面の鍵・ほぼ 1 を返す鍵(どちらも MLP の入力の空間で、
// CHiPS のヘッドの層を全部含むもの)・名前・使う質問・返事の長さ・ニューロン(省略可)・保存先のレシピ名。共通の設定は読まない。
// 結果は層ごとの当たり方(5 分割の R²・行の数)、CHiPS との一致(どこでも開く鍵・本物の鍵の狙う場面・dont_touch の cos と大きさの比)、
// できたユニットとレシピの保存先。「レシピに足す」でレシピのタブに入る。
// 「セルに会話の形も入れる」(表の fish.key_conv_cells。最小二乗のキーのときだけ)は、キーの作り方の欄の横。共通の設定の細かい設定に
// 「前のやり取りの応答の上限」(fish.prior_max_tokens。対照集合に content: null と書いたアシスタントの応答を、読む前に無介入のモデルが
// 埋める長さ。空なら 256)。対照集合を読むジョブ(方向・キー・確かめ・応答を読む・試す・まとめて作る)に渡す。
// 「ユニットの書き方」(fish.kscale・remove): !kauto を付けるかと、ユニットの形(足す / 成分を取り除く rm / 基準値を超えた分だけ取り除く
// rmbase。取り除く形は上限つきの形だけ)。/api/fish/units と「まとめて作る」に渡す。
// 「方向のもと」(fish.token・token_delta): 対照集合の差(既定)か、出力層の行列のトークンの行(トークンと、目標のロジットの上がり Δ)。
// トークンの行なら 2. に「層を走査する」(direction_token_scan のジョブ。入力欄の表は /api/fish/contrast でファイルにしてから渡す):
// 層ごとに、狙う場面の入力の位置 0 でそのトークンが 1 位になる最小の Δ と、dont_touch の入力に 0.2 倍・0.5 倍を足したときの影響。
// 行の「使う」で、層と Δ を欄に入れる。
// 6. の「足した量」には、会話の形が 2 つ以上ある群ごとに、形 × 区間の表(狙う場面は ρ ≥ b の割合、dont_touch は書いてしまった数)と、
// 最後まで書き終えた応答の数も出す。4. のキーの表の上には、会話の形ごとの質問の数(形が 2 つ以上か、セルに会話の形を入れたとき)。
import { getJSON, postJSON } from "./api.js";
import { LANG, t } from "./i18n.js";
import { checkKeySpec, parseLayers } from "./keys.js";
import { esc } from "./panels.js";
import { parseNw } from "./recipe.js";
import { Steps } from "./steps.js";
import { noteHTML, askConfirm, layersInput, aboutHTML } from "./ui.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const NAME_RE = /^[\w.-]+$/;
const DRAFT = "gne.fish.draft";                                   // 入力欄の下書き(このブラウザだけ)
const SRC = "gne.fish.src";
const PER = { one: 1, two: 1, cap: 2 };                           // 区切り 1 つに要るニューロンの数(gne/fish.py の PER_SEGMENT)
const FISH_KEYS = ["layers", "segments", "form", "alpha", "cap", "same_reply", "stack", "stack_fast", "pseudo", "max_tokens", "response_tokens", "iv", "rekey",
                   "key_fit", "key_c", "key_conv_cells", "prior_max_tokens", "kscale", "remove", "token", "token_delta"];
// 会話の形(gne/prior.py の FORMS)の名前
const CONV_FORMS = { single: "1 ターン", multi_turn: "複数ターン", system: "system あり", system_multi_turn: "system あり・複数ターン" };
const formName = (f) => (CONV_FORMS[f] ? t(CONV_FORMS[f]) : f);
const CAP_RE = /^\d+(\.\d+)?-\d+(\.\d+)?$/;                       // 上限つきの形のしきい値 a-b
export const AMOUNT_HIGH = 0.2;                                   // 書き込みたくない場面・区切りの外で、足した量の平均がこれ以上なら赤
export const AMOUNT_LOW = 0.5;                                    // 狙う場面の区切りの中で、足した量の平均がこれ未満なら黄
const mark = (b) => (b === true ? `<span class="kx-ok">○</span>` : b === false ? `<span class="kx-ng">×</span>` : "—");

// 区切りの書き方("0,1-3,4-")→ [{ text, lo, hi }](hi は含まない。null は最後まで)。読めなければ null(gne/direction.py の parse_segments)
export function parseSegments(s) {
  const out = [];
  const b = String(s || "").trim().match(/^before:([\s\S]+)$/);       // 文字の直前に書く(keyed write の positions と同じ書き方)。1 つだけ
  if (b) return [{ text: `before:${b[1]}`, lo: null, hi: null, before: true }];
  for (const part of String(s || "").split(",").map((x) => x.trim()).filter(Boolean)) {
    const m = part.match(/^(\d+)(?:(-)(\d+)?)?$/);
    if (!m) return null;
    const lo = +m[1], hi = m[2] === undefined ? lo + 1 : m[3] === undefined ? null : +m[3] + 1;
    if (hi !== null && hi <= lo) return null;
    out.push({ text: hi === null ? `${lo}-` : hi === lo + 1 ? `${lo}` : `${lo}-${hi - 1}`, lo, hi });
  }
  return out.length ? out : null;
}

// しきい値の欄("0.4-0.8" か、区切りごとに "0.4-0.8,0.3-0.7,0.3-0.7")→ 並び。空なら [](既定)。読めない・数が合わなければ null
export function parseCaps(text, nSeg) {
  const xs = String(text || "").split(/[\s,/]+/).filter(Boolean);
  if (!xs.length) return [];
  if (!xs.every((x) => { if (!CAP_RE.test(x)) return false; const [a, b] = x.split("-").map(Number); return a < b; })) return null;
  return xs.length === 1 || xs.length === nSeg ? xs : null;
}

// 足した量の表のセルの色: group は scene か dont_touch:<名前>、seg は { lo, hi }、pos は "0".."7" か "8+"
export function amountClass(group, seg, pos, mean) {
  if (mean === null || mean === undefined || !Number.isFinite(+mean)) return "";
  if (seg && seg.before) {                                              // 文字の直前に書く: 列は match(一致の直前)と other
    if (group === "scene" && pos === "match") return +mean < AMOUNT_LOW ? "fx-low" : "fx-in";
    return Math.abs(+mean) >= AMOUNT_HIGH ? "fx-bad" : "";
  }
  const k = String(pos).endsWith("+") ? parseInt(pos, 10) : +pos;
  const tail = String(pos).endsWith("+");
  const inside = seg ? (tail ? seg.hi === null || seg.hi > k : k >= seg.lo && (seg.hi === null || k < seg.hi)) : false;
  if (group === "scene" && inside) return +mean < AMOUNT_LOW ? "fx-low" : "fx-in";
  return Math.abs(+mean) >= AMOUNT_HIGH ? "fx-bad" : "";
}

// 「CHiPS のヘッドの分を場面で取り除く」のニューロンの欄("8:101,102 12:201,202")→ { neurons: {層: [番号, 番号]} | null } か { error }。
// 空なら null(働きの小さい順の上 2 個)。書くなら CHiPS のヘッドの層(layers)を全部、層ごとに違う番号を 2 個ずつ
export function parseNeuronPairs(text, layers = []) {
  const s = String(text || "").trim().replace(/\s*([:,])\s*/g, "$1");
  if (!s) return { neurons: null };
  const out = {};
  for (const part of s.split(/[\s;]+/).filter(Boolean)) {
    const m = part.match(/^(\d+):(\d+),(\d+)$/);
    if (!m) return { error: t("ニューロンは 層:番号,番号 のように、層ごとに 2 個ずつ書きます(例 8:101,102 12:201,202)") };
    if (+m[2] === +m[3]) return { error: t("層 {l} のニューロンは、違う番号を 2 個にしてください(上限つきの 2 個 1 組)", { l: m[1] }) };
    out[+m[1]] = [+m[2], +m[3]];
  }
  const have = Object.keys(out).map(Number);
  const miss = layers.filter((L) => !have.includes(L)), extra = have.filter((L) => !layers.includes(L));
  if (miss.length) return { error: t("ニューロンを書くなら、CHiPS のヘッドの層 {l} を全部書いてください(足りない層 {m})", { l: layers.join(","), m: miss.join(",") }) };
  if (extra.length) return { error: t("層 {m} は CHiPS のヘッドの層({l})ではありません", { l: layers.join(","), m: extra.join(",") }) };
  return { neurons: out };
}

export class FishPanel {
  constructor(root, { onRun, onAddNw, nLayer } = {}) {
    this.root = root;
    this.onRun = onRun || (async () => null);       // (kind, params) → Promise<ジョブ | null>(jobs.js の start。見積もりの確認つき)
    this.onAddNw = onAddNw || (() => {});           // (レシピの行のリスト) → レシピに足す
    this.nLayer = nLayer || 0;
    this.contrasts = [];
    this.templates = [];
    this.tplText = null;
    this.lastSet = null;                            // 最後に欄に入れた表の fish
    this.jspec = { spec: null, errors: [], warns: [], summary: null };
    this.jobs = { all: null, dir: null, neu: null, keys: null, verify: null, check: null, behave: null, try: null, heads: null, scan: null };   // 欄ごとに、最後に投げたジョブの id
    // 「CHiPS のヘッドの分を場面で取り除く」で選べるもの: CHiPS のレシピ({ name, n: proj のユニットの数, layers })、鍵(mlpin の要約)、
    // 保存したレシピの名前(上書きの確かめ用)。一覧の読み込みは通し番号で見分ける
    this.hg = { recipes: [], keys: [], recipeNames: new Set() };
    this.hgSeq = 0;
    this.handled = new Set();
    this.seq = 0;                                   // 一覧の読み込みの通し番号
    this.unitSeq = 0;                               // ユニットの組み立ての通し番号
    this.loaded = false;
    // 作ったもの: name(方向・鍵の名前の元)、path(対照集合のパス)、dir・usage・keys・verify・check(各段の結果)、
    // picked({層: [番号…]})、touched(手で選び直した層)、keyBase(鍵の名前の頭)、units(書き方)、rows(レシピの行)
    this.st = { name: null, path: null, dir: null, usage: null, keys: null, verify: null, check: null, picked: {}, touched: new Set(),
                keyBase: null, units: [], rows: [], pipe: null, behave: null, tries: [], tryShown: -1, heads: null, scan: null };
    root.innerHTML = `
      <section class="kx-sec" data-step="src">
        <div class="pp-in"><span class="muted">${t("対照集合")}</span>
          <span class="seg" id="fx-src"><button data-src="file" title="${t("configs/contrast/ か <root>/contrasts/ の YAML を選ぶ")}">${t("ファイルを選ぶ")}</button><button data-src="json" title="${t("ひな形(configs/fish/)から始めて、入力欄の JSON を手で直す")}">${t("JSON を書く(ひな形)")}</button></span>
          <label class="muted" title="${t("FiSH 用のひな形(configs/fish/)。選ぶと入力欄に入る")}">${t("ひな形")}
            <select id="fx-tpl-sel"><option value="">${t("(ひな形を選ぶ)")}</option></select></label></div>
        <div id="fx-src-file">
          <div class="pp-in">
            <select id="fx-contrast" aria-label="${t("対照集合")}"></select>
            <input id="fx-cpath" size="34" placeholder="${t("または YAML のパス(入れるとこちらを使う)")}" spellcheck="false">
          </div>
          <div id="fx-cinfo" class="muted note"></div>
        </div>
        <div id="fx-src-json" class="hidden">
          <div id="fx-tpl-about" class="muted note"></div>
          <textarea id="fx-text" class="mono" rows="9" spellcheck="false" placeholder='{"name": "…", "system_a": "…", "user_prompts": ["…", "…", "…", "…"], "dont_touch": [{"name": "ordinary", "prompts": ["…"]}], "fish": {"segments": "0,1-3,4-", "form": "cap", "alpha": 0.9}}'></textarea>
          <div id="fx-jcheck" class="note"></div>
        </div>
      </section>
      <div id="fx-shared">
        <p class="stage-shared-h">${t("共通の設定(どの手順でも同じ値を使う)")}</p>
        ${noteHTML(t("FiSH(Formula-instantiated Scene-gated Hidden units): すでにある MLP のニューロンを、学習なしで、式から計算した重みで書き換えて、決まった場面の応答でだけ方向を書くようにする。ゲートの行にキー、入力の行にキー、出力の列に方向を入れる。ふだんの働きが小さいニューロンを選んで書き換える(元の働きは無くなる)。実行時は LoRA で当て、レシピのタブでチャット・保存・LoRA の書き出しができる。層と強さは使う人が決める。"))}
        <div class="pp-set">
          <label class="muted" title="${t("方向とキーの名前の元(方向 <名前>、キー <名前>_p0 など)。空なら対照集合の名前")}">${t("名前")} <input id="fx-name" size="16" spellcheck="false"></label>
          <label class="muted" title="${t("書き換える層(カンマ区切りか、続いた層は 11-23 のように範囲で)。層の番号はモデルで違うので、使う人が決める。Qwen3.5-4B の 1 例では、離れた 3 層より、続いた 13 層(11-23)のほうが良かった。層を増やすと、要るニューロンと作成時間が増える")}">${t("層")} ${layersInput("fx-layers", { placeholder: "11-23" })}</label>
          <label class="muted" title="${t("書く位置。応答の位置の区切り(位置 0 = 応答を書き始める直前、1-3 = 応答の最初の 3 トークン、4- = それ以降。区切りごとに方向・キー・ニューロンを分ける)か、before:<正規表現>(一致する文字の直前の位置だけ。例 before:(です|ます)。)")}">${t("位置の区切り")} <input id="fx-seg" size="14" value="0,1-3,4-" spellcheck="false"></label>
          <span class="muted" title="${t("強さ α(区切りごと)。1 で、その区切りの方向をそのままの長さで書く。使う人が決める(Qwen3.5-4B の 1 例では、位置 0 と 1-3 に 1、4 以降に 0.8 がよかった)")}">${t("強さ(区切りごと)")} <span id="fx-alphas"></span></span>
        </div>
        <details class="kx-adv"><summary class="muted note">${t("細かい設定(先に当てておくレシピ・キーと確かめの応答の長さ・前のやり取りの応答の上限)")}</summary><div class="pp-set">
          <label class="muted" title="${t("先に当てておく介入: 保存したレシピの名前か、ユニットを空白区切りで。CHiPS のレシピを当てた状態で作るときに入れる。空なら無介入")}">${t("先に当てておくレシピ(iv)")} <input id="fx-iv" size="18" spellcheck="false"></label>
          <label class="muted" title="${t("キーづくりと確かめで応答を生成する長さ")}">${t("応答の長さ")} <input class="w5" id="fx-mt" type="number" min="1" max="2048" value="96"></label>
          <label class="muted" title="${t("対照集合の会話で、前のやり取りのアシスタントの応答を content: null と書いたところは、読む前に無介入のモデルが応答を書いて埋める(その問の system と、その応答より前のメッセージだけを読む。greedy)。その応答の長さの上限(トークン)。空なら 256。埋めた応答は覚えておき、同じ会話・同じ上限なら作り直さない")}">${t("前のやり取りの応答の上限")} <input class="w5" id="fx-prior-mt" type="number" min="1" max="4096" step="1" placeholder="256" aria-describedby="fx-prior-note"></label>
        </div><p class="muted note" id="fx-prior-note">${t("content: null と書いたアシスタントの応答は、読む前に無介入のモデルが埋めます(この上限まで。空なら 256)")}</p></details>
      </div>
      <div class="pp-set fx-part" id="fx-formset"><span class="muted note fx-part-h">${t("形(ニューロンを選ぶ・キーを作る・レシピを組む・まとめて作るが使う)")}</span>
          <label class="muted" title="${t("キー 1 つ = キーが反応する位置で書く。キー 2 つ = 場面のキー かつ 位置のキー。上限つき = ニューロン 2 個を組にして、足す量が 1 倍あたりで頭打ちになる(実験でいちばん良かった形)")}">${t("形")}
            <select id="fx-form"><option value="cap">${t("上限つきの 2 個組")}</option><option value="two">${t("キー 2 つ(場面 かつ 位置)")}</option><option value="one">${t("キー 1 つ")}</option></select></label>
          <label class="muted" title="${t("上限つきの形のしきい値 a-b。キーの出力が a より小さいとほぼ書かず、b より大きいと狙った量で頭打ちになる。空なら既定の 0.4-0.8。1 つ書けば全部の区切りに同じ値、区切りごとに変えるなら区切りの順にカンマ区切りで(例 0.4-0.8,0.3-0.7,0.3-0.7)")}">${t("しきい値(上限つきの形)")} <input id="fx-cap" size="16" placeholder="0.4-0.8" spellcheck="false"></label>
      </div>
      <div class="pp-set fx-part" id="fx-keyset"><span class="muted note fx-part-h">${t("キーの作り方(キーを作る・作り直す・まとめて作るが使う)")}</span>
          <label class="muted" title="${t("位置のキーの作り方。いまの作り方(既定)= 書き込みたい位置の状態の平均から、書き込みたくない位置にも現れる成分を取り除く。最小二乗 = 書き込みたい位置で 1、書き込みたくない位置で 0 を返すキーを、正則化つきの式で解く。書き込みたくない位置の重みは「場面の群(対象外の場面の群と、対象の場面のほかの区切り)× 位置の区切り」のセルごとに同じにするので、数の少ない群(質問でない発言の群など)も見分ける。Qwen3.5-4B のツンデレの 1 例で確かめた段階。当てた状態でキーを作り直す段にも同じ作り方を使う。キーの計算は CPU で、4B の 13 層で数分かかる。書く位置が before:<正規表現> のときは使えない")}">${t("キーの作り方")}
            <select id="fx-keyfit"><option value="mean">${t("いまの作り方(既定)")}</option><option value="ridge">${t("最小二乗(セルごとの重み)")}</option></select></label>
          <label class="muted" title="${t("最小二乗のキーの正則化の強さ(行列の対角の平均の何倍か。例 1)。空なら、キーを作る質問だけを使った質問単位の 5 分割で、0.01・0.1・1・10・100 から決める(選ばれた値はキーの表の上に出る)")}">${t("正則化の強さ")} <input id="fx-keyc" size="7" placeholder="${t("5 分割で決める")}" spellcheck="false" disabled></label>
          <label class="muted" title="${t("最小二乗のキーで重みをそろえるセルを「群 × 会話の形(1 ターン・複数ターン・system あり・system ありの複数ターン)× 位置の区切り」にする(既定は群 × 位置の区切り)。対照集合に複数ターンや system つきの会話を混ぜたとき、数の少ない形も見分けるようにする。最小二乗のときだけ使える")}"><input type="checkbox" id="fx-convcells" disabled> ${t("セルに会話の形も入れる")}</label>
      </div>
      <div class="pp-set fx-part" id="fx-unitset"><span class="muted note fx-part-h">${t("ユニットの書き方(レシピを組む・まとめて作るが使う)")}</span>
          <label class="muted" title="${t("ユニットに !kauto を付ける。入力の行を 1/k、出力の列を k 倍にして(足す量は同じ)、量子化の丸めで書き込みが崩れにくくする。k は層の活性の大きさから自動で決める")}"><input type="checkbox" id="fx-kauto"> ${t("!kauto を付ける")}</label>
          <label class="muted" title="${t("足す = キーが開いた位置で α·方向 を足す(既定)。成分を取り除く(!rm)= キーが開いた位置で、その位置の方向の成分を α の割合だけ取り除く。基準値を超えた分だけ取り除く(!rmbase)= 中立の文での成分の平均(基準値)を超えた分だけを取り除く。取り除く形は上限つきの 2 個組だけ")}">${t("ユニットの形")}
            <select id="fx-uform"><option value="add">${t("足す(既定)")}</option><option value="rm">${t("成分を取り除く(rm)")}</option><option value="rmbase">${t("基準値を超えた分だけ取り除く(rmbase)")}</option></select></label>
          <span class="muted note" id="fx-uform-note"></span>
      </div>
      <fieldset class="pp-set fx-part" id="fx-dirset"><legend class="muted note fx-part-h">${t("方向のもと(方向を作る・まとめて作るが使う)")}</legend>
          <label class="muted" title="${t("対照集合の、指示あり(system_a)と指示なしの応答の差から方向を作る(今までどおり)")}"><input type="radio" name="fx-dirsrc" value="contrast" checked> ${t("対照集合の差(既定)")}</label>
          <label class="muted" title="${t("出力層の行列の、トークンの行から方向を作る。キーが開いた位置で足すと、応答の最初にそのトークンを出させる(終了トークンで応答しない・マーカーで相づち、のような入力の振り分け)。どの区切りにも同じ向きを使う。層ごとに同じ長さを書くので、層を増やすと足す量が重なる(ふつうは 1 層)")}"><input type="radio" name="fx-dirsrc" value="token"> ${t("トークンの行")}</label>
          <label class="muted" title="${t("足すトークン。1 トークンになる文字(例 <|im_end|>・◇)か #ID。空白で区切ると、複数のトークンの行の平均")}">${t("足すトークン")} <input id="fx-token" class="mono" size="12" placeholder="<|im_end|>" spellcheck="false" autocomplete="off" autocapitalize="none" autocorrect="off" disabled></label>
          <label class="muted" title="${t("目標のロジットの上がり Δ。位置 0 で、そのトークンのロジットがこれだけ上がる長さで方向を書く(後ろの層が足した分を変えないとしたときの見積もり)。2. の「層を走査する」で、層ごとに要る Δ が分かる")}">${t("Δ(ロジットの上がり)")} <input class="w5" id="fx-tdelta" type="number" min="0" step="any" placeholder="32" disabled></label>
      </fieldset>
      <div class="pp-set fx-part" id="fx-fastset"><span class="muted note fx-part-h">${t("読み方(方向を作る・まとめて作るが使う)")}</span>
          <label class="muted" title="${t("速い読み方(区切りごとにまとめて読む)。大きいモデルで方向づくりがずっと速くなるが、方向がわずかに変わる(今までの方向との cos 0.999 以上)。層や強さを比べる探索向け。配るものを作るときは切のまま。この読み方の方向での口調・正答は測っていない。効くのは方向を作る段だけで、「すでにある方向を使う」に名前を入れたときは効かない")}"><input type="checkbox" class="fx-fast" id="fx-fast"> ${t("速い読み方(区切りごとにまとめて読む。探索向け。方向がわずかに変わる)")}</label>
      </div>
      <section class="kx-sec" data-step="all">
        <div class="pp-go">
          <button class="main" id="fx-all" title="${t("方向 → ニューロンの表 → キー → レシピ → 当てた状態でキーを作り直す → 確かめ、を 1 回のジョブで回す。ニューロンは既定の候補(表の上から)を使い、あとで選び直せる")}">${t("まとめて作る")}</button>
          <label class="muted" title="${t("レシピを当てた状態でキーを作り直す段を入れる(例ではこのほうがよかった)")}"><input type="checkbox" id="fx-rekey-on" checked> ${t("当てた状態でキーを作り直す")}</label>
          <label class="muted" title="${t("前に作った方向の名前を入れると、「まとめて作る」は方向を作り直さずにそれを使う(層が多いと、方向づくりがいちばん時間がかかる)。区切り・層・重ねる層が合わなければエラーになる。空なら作る")}">${t("すでにある方向を使う")} <input id="fx-usedir" size="14" spellcheck="false"></label>
          <span class="muted note" id="fx-status-all"></span></div>
        <div id="fx-prog-all" class="note mono"></div>
        <div id="fx-pipe"></div>
      </section>
      <section class="kx-sec" data-step="dir">
        <div class="pp-set">
          <label class="muted" title="${t("今まで = それぞれの条件で作った応答を読む。同じ応答 = 指示を付けて作った応答を、指示あり・なしの両方で読む(読んでいる言葉の違いが方向に入らない)")}">${t("作り方")}
            <select id="fx-method"><option value="same">${t("同じ応答を両方の条件で読む")}</option><option value="own">${t("今までの作り方")}</option></select></label>
          <label class="muted" title="${t("下の層に書いた後に残る差を、上の層の方向にする(上の欄の層に重ねて書く前提)")}"><input type="checkbox" id="fx-stack" checked> ${t("下の層に書いた後の残りを使う")}</label>
          <label class="muted" title="${t("今までの作り方の方向も作って、向きの近さ(cos)を表に出す")}"><input type="checkbox" id="fx-compare" checked> ${t("今までの作り方と比べる")}</label>
          <label class="muted" title="${t("方向を作るとき、応答を生成する長さ")}">${t("方向の応答の長さ")} <input class="w5" id="fx-rt" type="number" min="1" max="2048" value="96"></label>
        </div>
        <div id="fx-scan-box" hidden>
          <h4 class="note">${t("層を走査する(トークンの行)")}</h4>
          ${noteHTML(t("層ごとに、対象の場面の入力(対照集合の user_prompts)の位置 0(応答の最初のトークンを決める位置)にだけトークンの行の方向を足して、そのトークンが 1 位になる入力が 95% 以上になる最小の Δ を探します。その Δ の 0.2 倍・0.5 倍を dont_touch の入力の位置 0 に足したときの、そのトークンが 1 位になった数(誤検出)・KL・1 位が変わった数も出します(キーが少し開いてしまったときの影響の目安)。応答は生成しません。"))}
          <div class="pp-set">
            <label class="muted" title="${t("走査する層(カンマ区切りか 11-23)。空なら最終層を除く全部の層")}">${t("走査する層")} ${layersInput("fx-scan-layers", { placeholder: t("空なら全部の層") })}</label>
          </div>
          <div class="pp-go"><button type="button" id="fx-scan-go" title="${t("方向のもとの欄のトークンと、1. の対照集合で、層ごとに要る Δ と dont_touch への影響を測る")}">${t("層を走査する")}</button>
            <span class="muted note" id="fx-status-scan" role="status"></span></div>
          <div id="fx-prog-scan" class="note mono"></div>
          <div id="fx-scan"></div>
        </div>
        <div class="pp-go"><button id="fx-dir-go">${t("方向を作る")}</button> <span class="muted note" id="fx-status-dir"></span></div>
        <div id="fx-prog-dir" class="note mono"></div>
        <div id="fx-dir"></div>
      </section>
      <section class="kx-sec" id="fx-try-sec" data-step="try">
        ${noteHTML(t("2. で作った方向を、ニューロンにもキーにも入れずに、層の出力へ直接足して応答を作る。足すのは対象の場面の質問の応答だけ(場面は質問の群で決める。重みに入れたときのキーの代わり)。層と区切りごとの強さを変えて何回か回し、応答と数を比べてから 3. 以降に進む"), "muted note")}
        <div class="pp-set">
          <label class="muted" title="${t("足す層(カンマ区切りか 11-23)。空なら上の欄の層。足せるのは層 1 から、最終層の 1 つ手前まで")}">${t("足す層")} ${layersInput("fx-try-layers", { placeholder: t("空なら共通の設定の層") })}</label>
          <label class="muted" title="${t("対象の場面・書き込みたくない場面それぞれで、応答を作る質問の数(対照集合の先頭から)")}">${t("質問の数")} <input class="w45" id="fx-try-n" type="number" min="1" max="200" value="12"></label>
          <label class="muted" title="${t("応答を生成する長さ")}">${t("応答の長さ")} <input class="w5" id="fx-try-mt" type="number" min="1" max="2048" value="120"></label>
          <label class="muted" title="${t("0 なら毎回同じ応答(greedy)。0 より大きいとサンプリング")}">${t("温度")} <input class="w45" id="fx-try-temp" type="number" min="0" max="2" step="0.1" value="0"></label>
          <label class="muted" title="${t("サンプリングの乱数の種")}">${t("種")} <input class="w45" id="fx-try-seed" type="number" min="0" value="0"></label>
          <label class="muted" title="${t("一致した応答を数える正規表現。空なら対照集合の validate.match。繰り返しの中の語も数えるので、応答は読んで確かめる")}">${t("数える正規表現")} <input id="fx-try-match" size="18" spellcheck="false"></label>
        </div>
        <details class="kx-adv"><summary class="muted note">${t("対象の場面の質問を自分で入れる")}</summary>
          <textarea id="fx-try-q" class="mono" rows="4" spellcheck="false" placeholder="${t("1 行に 1 問。空なら対照集合の質問の先頭から(方向づくりに使った質問なので、甘い)")}"></textarea></details>
        <div class="pp-go"><button id="fx-try-go" title="${t("方向を層の出力に直接足して、対象の場面の質問に答えさせる。同じ質問に何も足さない応答と、書き込みたくない場面の質問への応答(何も足さない)も作る")}">${t("直接足して応答を作る")}</button>
          <button class="small" id="fx-try-use" title="${t("この欄の足す層を、共通の設定の層に入れる(強さは共通の設定のものを使っています)")}">${t("この層を共通の設定に入れる")}</button>
          <span class="muted note" id="fx-status-try"></span></div>
        <div id="fx-prog-try" class="note mono"></div>
        <div id="fx-try"></div>
      </section>
      <section class="kx-sec" data-step="neu">
        <div class="pp-set">
          <label class="muted" title="${t("層ごとに表に出す数")}">${t("表の行数")} <input class="w5" id="fx-top" type="number" min="1" max="200" value="20"></label>
          <span class="muted note" id="fx-need"></span></div>
        <div class="pp-go"><button id="fx-neu-go" title="${t("ふつうの質問への応答と文章を読ませて、各ニューロンのふだんの働きを測る")}">${t("ニューロンの表を出す")}</button> <span class="muted note" id="fx-status-neu"></span></div>
        <div id="fx-prog-neu" class="note mono"></div>
        <div id="fx-neu"></div>
      </section>
      <section class="kx-sec" data-step="keys">
        <div class="pp-set">
          <label class="muted" title="${t("比較用のキーの本数。書き込みたい位置・書き込みたくない位置のラベルをでたらめに入れ替えて、同じ手順で作るキー")}">${t("比較用のキー")} <input class="w4" id="fx-pseudo" type="number" min="0" max="64" value="4"></label>
        </div>
        <div class="pp-go"><button id="fx-keys-go" title="${t("区切りごとの位置のキーと、形に要るキー(キー 2 つの形は場面のキー、上限つきの形はどの位置でもほぼ 1 を返すキー(定数キー))をまとめて作る")}">${t("キーをまとめて作る")}</button> <span class="muted note" id="fx-status-keys"></span></div>
        <div id="fx-prog-keys" class="note mono"></div>
        <div id="fx-keys"></div>
      </section>
      <section class="kx-sec" data-step="units">
        <div id="fx-units"><p class="muted note">${t("方向・ニューロン・キーがそろうと、ここにユニットが出ます")}</p></div>
        <div class="pp-go">
          <button id="fx-add" disabled title="${t("下のユニットをレシピに足す(同じ層・同じニューロンの行は置き換える)。レシピのタブでチャット・保存・LoRA の書き出しができる")}">${t("レシピに足す")}</button>
          <button id="fx-rekey" disabled title="${t("このユニットを当てた状態で応答を作り直し、キーを作り直す(名前は <名前>_r_…)。ニューロンが書いた後の状態でキーが反応するようになる")}">${t("このレシピを当てた状態でキーを作り直す")}</button>
          <span class="muted note" id="fx-status-units"></span></div>
      </section>
      <section class="kx-sec" data-step="check">
        <div class="pp-go">
          <button id="fx-verify-go" disabled title="${t("LoRA を当てて短い文を読ませ、書き換えたニューロンが式どおりに動くか、ほかのニューロンが変わっていないかを確かめる(数秒)")}">${t("式どおりか確かめる")}</button>
          <span class="muted note" id="fx-status-verify"></span>
          <button id="fx-check-go" disabled title="${t("レシピを当てて応答を作り、対象の場面と書き込みたくない場面で、応答の位置ごとに実際に足した量を測る")}">${t("足した量を測る")}</button>
          <span class="muted note" id="fx-status-check"></span></div>
        <div id="fx-prog-verify" class="note mono"></div>
        <div id="fx-verify"></div>
        <div id="fx-prog-check" class="note mono"></div>
        <div id="fx-check"></div>
      </section>
      <section class="kx-sec" data-step="behave">
        <div class="pp-set">
          <label class="muted" title="${t("対象の場面・書き込みたくない場面それぞれで、応答を作る質問の数(対照集合の先頭から)")}">${t("質問の数")} <input class="w45" id="fx-bn" type="number" min="1" max="200" value="12"></label>
          <label class="muted" title="${t("応答を生成する長さ")}">${t("応答の長さ")} <input class="w5" id="fx-bmt" type="number" min="1" max="2048" value="120"></label>
          <label class="muted" title="${t("一致した応答を数える正規表現。空なら対照集合の validate.match。繰り返しの中の語も数えるので、応答は読んで確かめる")}">${t("数える正規表現")} <input id="fx-bmatch" size="22" spellcheck="false"></label>
        </div>
        <details class="kx-adv"><summary class="muted note">${t("対象の場面の質問を自分で入れる")}</summary>
          <textarea id="fx-bq" class="mono" rows="4" spellcheck="false" placeholder="${t("1 行に 1 問。空なら対照集合の質問の先頭から(キーづくりに使った質問なので、甘い)")}"></textarea></details>
        <div class="pp-go"><button id="fx-behave-go" disabled title="${t("ユニットを当てて(system の指示は付けずに)、対象の場面の質問と、書き込みたくない場面の質問に答えさせる。結果は behave の結果としても保存する")}">${t("レシピを当てて応答を作る")}</button>
          <span class="muted note" id="fx-status-behave"></span></div>
        <div id="fx-prog-behave" class="note mono"></div>
        <div id="fx-behave"></div>
      </section>
      <section class="kx-sec" data-step="heads">
        ${noteHTML(t("CHiPS のレシピが弱めるヘッドの書き込みを、場面を選んで取り除くユニットを作る。ヘッドの重みは変えず、同じ層の MLP のニューロン 2 個 1 組が、場面のキーが開いた位置でだけ、CHiPS がその位置で取り除くはずの量を MLP の入力から当てて(切片つきのリッジ回帰)取り除く。対照集合の質問が取り除きたい場面、dont_touch が残したい場面。CHiPS を常に当てたときとの比べは、2 つのレシピを並べて eval・behave で行う。"))}
        ${noteHTML(t("分かっている限界: 場面を選んでも、取り除く場面の中では、抑えたい振る舞いと近い、残したい振る舞いまで弱まることがある。本文の主張の言い切りも増えることがある。強さを上げても解決しない。答えの質は、無介入をサンプリングで作り直した応答との読み比べで確かめる(docs/fish.md「ヘッドの分を場面で取り除く」)。"), "pp-warn note")}
        <p class="muted note" id="fx-hg-src"></p>
        <div class="pp-set">
          <label class="muted" title="${t("CHiPS のレシピ: proj のユニット(L:Hh:sS@方向)を持つ保存済みのレシピ。1 つの層のヘッドは同じ方向を使うこと。ヘッドの層は、nw を入れられる dense の MLP の層(MoE の層・最終層は不可)")}">${t("CHiPS のレシピ")} <select id="fx-hg-recipe"></select></label>
          <span class="muted note" id="fx-hg-layers"></span>
        </div>
        <div class="pp-set">
          <label class="muted" title="${t("場面のキー(MLP の入力の空間、role scene): 対照集合の質問への応答の位置で開き、dont_touch の質問への応答では開かないキー。CHiPS のヘッドの層を全部含むもの")}">${t("場面のキー")} <select id="fx-hg-scene"></select></label>
          <label class="muted" title="${t("定数キー(MLP の入力の空間、role one): どの位置でもほぼ 1 を返すキー。上限つきの形のしきい値と、取り除く量の切片に使う。CHiPS のヘッドの層を全部含むもの")}">${t("定数キー")} <select id="fx-hg-one"></select></label>
          <button type="button" class="small" id="fx-hg-reload" title="${t("保存したレシピとキーの一覧を取り直す")}">${t("一覧を取り直す")}</button>
        </div>
        <p class="muted note" id="fx-hg-keyhint"></p>
        <div class="pp-set">
          <label class="muted" title="${t("作るものの名前。方向 <名前>_dir(CHiPS が使う向き r̂)と、取り除く量のキー <名前>_amount を書く。空なら欄の薄い字の名前")}">${t("名前")} <input id="fx-hg-name" size="14" spellcheck="false" autocomplete="off" autocapitalize="none"></label>
          <label class="muted" title="${t("取り除きたい場面(対照集合の質問)のうち、取り除く量を当てる材料に使うもの。dont_touch の質問は全部使う")}">${t("使う質問")}
            <select id="fx-hg-subset"><option value="even">${t("偶数番目")}</option><option value="odd">${t("奇数番目")}</option><option value="all">${t("全部")}</option></select></label>
          <label class="muted" title="${t("材料にする無介入の応答を生成する長さ")}">${t("応答の長さ")} <input class="w5" id="fx-hg-mt" type="number" min="1" max="2048" value="96"></label>
          <label class="muted" title="${t("できたユニットを書くレシピの名前。空なら欄の薄い字の名前(<名前>_r)")}">${t("保存先のレシピ名")} <input id="fx-hg-save" size="14" spellcheck="false" autocomplete="off" autocapitalize="none"></label>
        </div>
        <details class="kx-adv"><summary class="muted note">${t("細かい設定(書き換えるニューロン)")}</summary><div class="pp-set">
          <label class="muted" title="${t("書き換えるニューロン。層ごとに 2 個(上限つきの 2 個 1 組)を、層:番号,番号 で空白区切りに書く。書くなら CHiPS のヘッドの層を全部。空なら、ふだんの働きの小さい順の上 2 個(3. の表と同じ測り方)。書き換えると、そのニューロンの元の働きは無くなる")}">${t("ニューロン")} <input id="fx-hg-neurons" class="mono" size="44" spellcheck="false" autocomplete="off" placeholder="${t("例 8:101,102 12:201,202(空なら働きの小さい順)")}"></label>
        </div></details>
        <div class="pp-go"><button type="button" class="main" id="fx-hg-go" title="${t("無介入の応答を作り、取り除く量のキー・方向・ユニットを作って、当たり方と CHiPS との一致を測る。できたユニットはレシピに保存する")}">${t("取り除くユニットを作る")}</button>
          <span class="muted note" id="fx-status-heads" role="status"></span></div>
        <div id="fx-prog-heads" class="note mono"></div>
        <div id="fx-heads"></div>
      </section>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#fx-contrast").addEventListener("change", () => this.contrastChanged());
    this.$("#fx-cpath").addEventListener("input", () => this.contrastChanged());
    for (const b of root.querySelectorAll("#fx-src button")) b.addEventListener("click", () => this.setSource(b.dataset.src));
    const ta = this.$("#fx-text");
    try { ta.value = localStorage.getItem(DRAFT) || ""; } catch { /* 下書きが無くても使える */ }
    ta.addEventListener("input", () => this.checkJSON());
    const sel = this.$("#fx-tpl-sel");
    sel.addEventListener("change", () => this.insertTemplate(sel.value, true));
    for (const id of ["#fx-layers", "#fx-seg", "#fx-form", "#fx-cap"]) this.$(id).addEventListener("input", () => this.settingsChanged());
    this.$("#fx-form").addEventListener("change", () => this.settingsChanged());
    this.$("#fx-keyfit").addEventListener("change", () => this.settingsChanged());
    for (const id of ["#fx-kauto", "#fx-uform"]) this.$(id).addEventListener("change", () => this.compose());
    for (const r of root.querySelectorAll("input[name=fx-dirsrc]")) r.addEventListener("change", () => this.dirMode());
    this.$("#fx-scan-go").addEventListener("click", () => this.runScan());
    for (const id of ["#fx-token", "#fx-tdelta", "#fx-prior-mt", "#fx-scan-layers"]) this.$(id).addEventListener("input", (e) => e.target.removeAttribute("aria-invalid"));
    root.addEventListener("input", (ev) => { if (ev.target.closest("input[data-fxa]")) this.compose(); });
    this.$("#fx-try-go").addEventListener("click", () => this.runTry());
    this.$("#fx-try-use").addEventListener("click", () => this.useTry());
    this.$("#fx-all").addEventListener("click", () => this.runAll());
    this.$("#fx-dir-go").addEventListener("click", () => this.runDirection());
    this.$("#fx-neu-go").addEventListener("click", () => this.runNeurons());
    this.$("#fx-keys-go").addEventListener("click", () => this.runKeys(false));
    this.$("#fx-rekey").addEventListener("click", () => this.runKeys(true));
    this.$("#fx-add").addEventListener("click", () => this.addToRecipe());
    this.$("#fx-verify-go").addEventListener("click", () => this.runVerify());
    this.$("#fx-check-go").addEventListener("click", () => this.runCheck());
    this.$("#fx-behave-go").addEventListener("click", () => this.runBehave());
    this.$("#fx-hg-go").addEventListener("click", () => this.runHeads());
    this.$("#fx-hg-reload").addEventListener("click", () => this.hgLoad());
    this.$("#fx-hg-recipe").addEventListener("change", () => this.hgRecipeChanged());
    this.$("#fx-hg-name").addEventListener("input", () => this.hgNames());
    for (const ev of ["input", "change"]) {                         // 直した欄の「直すところ」の印を消す
      this.$('section[data-step="heads"]').addEventListener(ev, (e) => { if (e.target.getAttribute("aria-invalid")) e.target.removeAttribute("aria-invalid"); });
    }
    root.addEventListener("change", (ev) => {
      const cb = ev.target.closest("input[data-fxn]");
      if (cb) this.togglePick(+cb.dataset.fxl, +cb.dataset.fxn, cb.checked);
    });
    root.addEventListener("click", (ev) => {
      const b = ev.target.closest("button[data-fxstop]");
      if (b) fetch(`/api/jobs/${b.dataset.fxstop}`, { method: "DELETE" }).catch(() => {});
      const d = ev.target.closest("button[data-fxdef]");
      if (d) this.resetPick(+d.dataset.fxdef);
      const tr = ev.target.closest("button[data-fxtry]");
      if (tr) { this.st.tryShown = +tr.dataset.fxtry; this.renderTry(); }
      if (ev.target.closest("#fx-hg-add")) this.addHeads();
      const sc = ev.target.closest("#fx-scan-tab tr[data-fxscan]");     // 走査の表: 行(か行の「使う」)で層と Δ を欄に入れる
      if (sc) this.useScan(+sc.dataset.fxscan);
    });
    let src = "json";
    try { src = localStorage.getItem(SRC) === "file" ? "file" : "json"; } catch { /* 既定は JSON */ }
    this.setSource(src, false);
    this.renderAlphas("fx-alphas", [1, 1, 0.8]);
    this.showNeed();
    // 左に手順、右に選んだ手順。共通の設定(名前・層・区切り・強さ)は上に 1 回だけ。形・鍵の作り方・読み方は、それを読む手順のときだけ出す
    const part = { form: ["neu", "keys", "units", "all"], key: ["keys", "units", "all"], unit: ["units", "all"], dsrc: ["dir", "all"], fast: ["dir", "all"] };   // 一部の手順だけが使う欄
    const sec = (k) => () => [root.querySelector(`section[data-step="${k}"]`),
                              ...(part.form.includes(k) ? [this.$("#fx-formset")] : []), ...(part.key.includes(k) ? [this.$("#fx-keyset")] : []),
                              ...(part.unit.includes(k) ? [this.$("#fx-unitset")] : []), ...(part.dsrc.includes(k) ? [this.$("#fx-dirset")] : []),
                              ...(part.fast.includes(k) ? [this.$("#fx-fastset")] : [])];
    // 「CHiPS のヘッドの分を場面で取り除く」は別の流れ(区切りで分け、共通の設定は読まない。対照集合は 1. のもの)。開いたら一覧を取り直す
    this.steps = new Steps(root, {
      shared: this.$("#fx-shared"), store: "gne.fish.step", label: t("FiSH の手順"),
      steps: [["src", t("1. 対照集合")], ["dir", t("2. 方向を作る")], ["try", t("重みに入れる前に試す")], ["neu", t("3. 書き換えるニューロンを選ぶ")],
              ["keys", t("4. キーを作る")], ["units", t("5. レシピを組む")], ["check", t("6. 確かめる")], ["behave", t("7. 応答を読む")],
              ["all", t("まとめて作る(2〜6)")], ["heads", t("CHiPS のヘッドの分を場面で取り除く")]]
        .map(([key, label]) => ({ key, label, els: sec(key), shared: key !== "heads", sep: key === "all" || key === "heads" })),
      onSelect: (k) => { if (k === "heads") { this.hgSrcLine(); this.hgLoad(); } },
    });
    this.marks();
    for (const ev of ["input", "change"]) root.addEventListener(ev, () => this.marks());   // 対照集合を選んだ・書いた
  }

  // 手順の印(済・実行中・まだ)を、作ったものと走っているジョブから付け直す
  marks() {
    if (!this.steps) return;
    const st = this.st, run = (w) => !!(this.jobs[w] && !this.handled.has(this.jobs[w])), all = run("all");
    const m = (k, running, done) => this.steps.mark(k, running ? "run" : done ? "done" : "todo");
    const src = this.src === "json" ? !!(this.jspec.spec && !(this.jspec.errors || []).length)
      : !!(this.$("#fx-contrast").value || this.$("#fx-cpath").value.trim());
    m("src", false, !!st.path || src);
    m("dir", run("dir") || run("scan") || all, !!st.dir);
    m("try", run("try"), st.tries.length > 0);
    m("neu", run("neu") || all, !!st.usage);
    m("keys", run("keys") || all, !!st.keys);
    m("units", false, (st.units || []).length > 0);
    m("check", run("verify") || run("check") || all, !!(st.verify || st.check));
    m("behave", run("behave"), !!st.behave);
    m("all", all, !!st.pipe);
    m("heads", run("heads"), !!st.heads);
    if (this.steps.current === "heads") this.hgSrcLine();
  }

  // 強さの欄を、区切りの数だけ並べる(box = fx-alphas か fx-try-alphas)。values を渡せばその値(数 1 つ・"1,1,0.8"・並び。足りない分は
  // 最後の値)、無ければいまの欄の値を残す。区切りの数も書き方も同じなら作り直さない(入力中の欄を消さない)
  renderAlphas(box, values) {
    const el = this.$(`#${box}`), main = box === "fx-alphas";
    const segs = parseSegments(this.$("#fx-seg").value);
    if (!segs) return;                                         // 区切りを書いている途中: 欄はそのまま
    this.alphaMem = this.alphaMem || {};
    const mem = this.alphaMem[box] || (this.alphaMem[box] = []);     // 区切りを減らして戻したときのために、欄ごとの値を覚えておく
    const cur = [...el.querySelectorAll("input")].map((x) => x.value);
    cur.forEach((v, i) => { mem[i] = v; });
    if (values !== undefined && values !== null) {
      const vals = (Array.isArray(values) ? values : String(values).split(/[\s,/]+/).filter((x) => x !== "")).map(String);
      mem.length = 0;
      vals.forEach((v, i) => { mem[i] = v; });
    }
    if (!mem.length) mem[0] = "0.9";
    const sig = segs.map((x) => x.text).join(",");
    if (values === undefined && el.dataset.sig === sig && cur.length === segs.length) return;
    el.dataset.sig = sig;
    const id = (i) => (main ? (i === 0 ? "fx-alpha" : `fx-alpha-${i}`) : `fx-try-alpha-${i}`);
    el.innerHTML = segs.map((x, i) => `<label class="fx-al"><span class="mono">${esc(x.before ? "" : x.text)}</span> <input class="w45" id="${id(i)}" ${main ? "data-fxa" : "data-fxta"}="${i}" type="number" step="0.05" value="${esc(mem[Math.min(i, mem.length - 1)])}" aria-label="${t("強さ")} ${esc(x.text)}"></label>`).join(" ");
  }

  // 区切りごとの強さ(数の並び)。読めない欄があれば null
  alphas(box = "fx-alphas") {
    const xs = [...this.$(`#${box}`).querySelectorAll("input")].map((x) => (x.value.trim() === "" ? NaN : +x.value));
    return xs.length && xs.every((a) => Number.isFinite(a)) ? xs : null;
  }

  // direction ビューで「FiSH」を開いたとき: 対照集合の候補とひな形を取り直す(遅れて届いた古い読み込みは捨てる)
  async refresh() {
    this.hgLoad();                                             // 「CHiPS のヘッドの分を場面で取り除く」のレシピと鍵の一覧(待たない)
    const seq = ++this.seq;
    const [cs, tp] = await Promise.all([getJSON("/api/contrasts").catch(() => []), getJSON("/api/fish-templates").catch(() => [])]);
    if (seq !== this.seq) return;
    this.contrasts = Array.isArray(cs) ? cs : [];
    this.templates = Array.isArray(tp) ? tp : [];
    const sel = this.$("#fx-tpl-sel"), cur = sel.value;
    sel.innerHTML = `<option value="">${t("(ひな形を選ぶ)")}</option>${this.templates.map((x) => `<option value="${esc(x.file)}">${esc((x[LANG] && x[LANG].title) || (x.ja && x.ja.title) || x.file)}</option>`).join("")}`;
    if (cur && this.templates.some((x) => x.file === cur)) sel.value = cur;
    const el = this.$("#fx-contrast"), was = el.value;
    const grp = (src, label) => {
      const xs = this.contrasts.filter((c) => c.src === src);
      return xs.length ? `<optgroup label="${esc(label)}">${xs.map((c) => `<option value="${esc(c.path)}">${esc(c.file)}</option>`).join("")}</optgroup>` : "";
    };
    el.innerHTML = grp("configs", "configs/contrast") + grp("root", "<root>/contrasts");
    if (was && [...el.options].some((o) => o.value === was)) el.value = was;
    this.contrastChanged();
    this.loaded = true;
  }

  setSource(src, remember = true) {
    this.src = src === "file" ? "file" : "json";
    for (const b of this.root.querySelectorAll("#fx-src button")) b.classList.toggle("on", b.dataset.src === this.src);
    this.$("#fx-src-file").classList.toggle("hidden", this.src !== "file");
    this.$("#fx-src-json").classList.toggle("hidden", this.src !== "json");
    if (remember) { try { localStorage.setItem(SRC, this.src); } catch { /* 覚えられなくても使える */ } }
    if (this.src === "json") this.checkJSON();
    else this.contrastChanged();
  }

  contrastChanged() {
    if (this.src === "json") return;
    const path = this.$("#fx-cpath").value.trim() || this.$("#fx-contrast").value;
    const c = this.contrasts.find((x) => x.path === path);
    const info = this.$("#fx-cinfo");
    this.$("#fx-name").placeholder = c ? c.name : t("(対照集合の名前)");
    if (!c) { info.textContent = path ? t("(一覧に無いファイル: そのままジョブに渡します)") : t("対照集合がありません(configs/contrast/ か <root>/contrasts/ の YAML)"); return; }
    info.innerHTML = `${esc(c.label)} · ${c.n_prompts ?? "—"} ${t("問")}${c.dont_touch.length ? ` · dont_touch: ${esc(c.dont_touch.join(", "))}`
      : ` · <span class="pp-warn">${t("dont_touch(書き込みたくない場面の質問)がありません。ほかの場面でもキーが反応しやすくなります")}</span>`}`;
  }

  async insertTemplate(file, ask) {
    const x = this.templates.find((y) => y.file === file);
    if (!x) return;
    const ta = this.$("#fx-text"), cur = ta.value.trim();
    if (ask && cur && cur !== this.tplText && !(await askConfirm(t("入力欄の JSON をひな形で置き換えます。よいですか?")))) { this.$("#fx-tpl-sel").value = ""; return; }
    ta.value = JSON.stringify(x.spec, null, 2);
    this.tplText = ta.value.trim();
    const about = this.$("#fx-tpl-about");
    about.innerHTML = aboutHTML((x[LANG] && x[LANG].text) ? x[LANG] : x.ja);   // 要約と、節ごとにたたんだ説明
    about.title = `configs/fish/${x.file}`;
    if (this.src !== "json") this.setSource("json");
    else this.checkJSON();
  }

  // 入力欄を確かめて要約を出す。表の fish が変わったら、その値を欄に入れる(層は入れない。使う人が決める)
  checkJSON() {
    const text = this.$("#fx-text").value;
    try { localStorage.setItem(DRAFT, text); } catch { /* 置けなくても使える */ }
    const c = checkKeySpec(text);
    const el = this.$("#fx-jcheck");
    const d = c.spec;
    const errors = [...c.errors];
    let fish = null;
    if (d && d.fish !== undefined && d.fish !== null) {
      if (typeof d.fish !== "object" || Array.isArray(d.fish)) errors.push(t("fish は設定の表です({k})", { k: FISH_KEYS.join(", ") }));
      else {
        const bad = Object.keys(d.fish).filter((k) => !FISH_KEYS.includes(k));
        if (bad.length) errors.push(t("fish に {k} は使えません({ok})", { k: bad.join(", "), ok: FISH_KEYS.join(", ") }));
        else fish = d.fish;
      }
    }
    if (d && !(typeof d.system_a === "string" && d.system_a) && (d.recipe_a === undefined || d.recipe_a === null)) {
      errors.push(t("system_a(出したい振る舞いの指示)か recipe_a が要ります(方向を、指示あり・なしの差から作ります)"));
    }
    this.jspec = { ...c, errors };
    // 画面で選べない値(kscale の数)は、欄の切り替えで渡すことを知らせる(黙って捨てない)
    const ks = fish ? fish.kscale : undefined;
    const notes = ks !== undefined && ks !== null && ks !== "" && String(ks) !== "auto"
      ? [t("fish.kscale の数({k})は画面では選べません。「!kauto を付ける」の切り替えで渡します", { k: String(ks) })] : [];
    if (!d && !errors.length) el.innerHTML = `<span class="muted">${t("JSON を貼り付けるか、「ひな形」で例を選んでください")}</span>`;
    else {
      const s = c.summary;
      const dt = s ? s.dt : [];
      el.innerHTML = `${errors.map((e) => `<div class="err">${esc(e)}</div>`).join("")}${notes.map((n) => `<div class="pp-warn">${esc(n)}</div>`).join("")}${
        s ? `<div class="muted">${esc(String(d.name || ""))} · ${t("対象の場面の質問 {n} 問", { n: s.n })} · ${dt.length ? t("書き込みたくない場面: {d}", { d: dt.map((g) => `${g.name}(${g.n ?? "—"})`).join(", ") })
          : `<span class="pp-warn">${t("dont_touch(書き込みたくない場面の質問)がありません。ほかの場面でもキーが反応しやすくなります")}</span>`}</div>` : ""}`;
    }
    if (d && NAME_RE.test(String(d.name ?? ""))) this.$("#fx-name").placeholder = String(d.name);
    const sig = fish ? JSON.stringify(fish) : null;
    if (fish && sig !== this.lastSet) {
      this.lastSet = sig;
      const put = (sel, v) => { if (v !== undefined && v !== null) this.$(sel).value = String(v); };
      put("#fx-seg", fish.segments); put("#fx-form", fish.form); put("#fx-pseudo", fish.pseudo);
      if (fish.alpha !== undefined && fish.alpha !== null) this.renderAlphas("fx-alphas", fish.alpha);
      put("#fx-mt", fish.max_tokens); put("#fx-rt", fish.response_tokens);
      this.$("#fx-cap").value = fish.cap === undefined || fish.cap === null ? "" : Array.isArray(fish.cap) ? fish.cap.join(",") : String(fish.cap);
      if (typeof fish.iv === "string") put("#fx-iv", fish.iv); else if (Array.isArray(fish.iv)) put("#fx-iv", fish.iv.join(" "));
      // 書いていない設定は既定に戻す(前に選んだひな形の値を残さない)
      this.$("#fx-method").value = fish.same_reply === false ? "own" : "same";
      this.$("#fx-stack").checked = fish.stack !== false;
      this.setFast(fish.stack_fast === true);
      this.$("#fx-rekey-on").checked = fish.rekey !== false;
      this.$("#fx-keyfit").value = fish.key_fit === "ridge" ? "ridge" : "mean";
      this.$("#fx-keyc").value = fish.key_c === undefined || fish.key_c === null ? "" : String(fish.key_c);
      this.$("#fx-convcells").checked = fish.key_conv_cells === true;
      this.$("#fx-prior-mt").value = fish.prior_max_tokens === undefined || fish.prior_max_tokens === null ? "" : String(fish.prior_max_tokens);
      this.$("#fx-kauto").checked = String(fish.kscale ?? "") === "auto";
      this.$("#fx-uform").value = ["rm", "rmbase"].includes(fish.remove) ? fish.remove : "add";
      const tok = Array.isArray(fish.token) ? fish.token.join(" ") : String(fish.token ?? "").trim();
      this.setDirSource(tok ? "token" : "contrast");
      this.$("#fx-token").value = tok;
      this.$("#fx-tdelta").value = fish.token_delta === undefined || fish.token_delta === null ? "" : String(fish.token_delta);
      if (fish.iv === undefined || fish.iv === null) this.$("#fx-iv").value = "";
      if (fish.layers !== undefined && fish.layers !== null && !this.$("#fx-layers").value.trim()) put("#fx-layers", Array.isArray(fish.layers) ? fish.layers.join(",") : fish.layers);
      this.settingsChanged();
    }
  }

  status(which, s, err = false) {
    const el = this.$(`#fx-status-${which === "units" ? "units" : which}`);
    if (!el) return;
    el.textContent = s;
    el.classList.toggle("err", err);
  }

  // --- 欄 → 引数 -------------------------------------------------------------------------------------------
  // 対照集合: 一度ジョブが <root>/contrasts/ に書いた後でも、入力欄の表をそのまま渡す(書き直しても同じ中身になる)
  source(which) {
    if (this.src === "json") {
      const c = this.jspec;
      if (!c.spec) { this.status(which, t("入力欄に対照集合の JSON を入れてください(「ひな形」で例を選べます)"), true); return null; }
      if (c.errors.length) { this.status(which, t("入力欄の JSON に直すところがあります: {e}", { e: c.errors[0] }), true); return null; }
      return { spec: c.spec };
    }
    const contrast = this.$("#fx-cpath").value.trim() || this.$("#fx-contrast").value;
    if (!contrast) { this.status(which, t("対照集合を選んでください"), true); return null; }
    return { contrast };
  }

  // 層・区切り・形・名前・iv・返事の長さ。読めなければ欄の横に理由を出して null
  common(which) {
    const v = (s) => this.$(s).value.trim();
    const layers = parseLayers(v("#fx-layers"));
    if (!layers || (this.nLayer && layers.some((L) => L >= this.nLayer))) {
      this.status(which, t("層を入れてください(0〜{m} の数をカンマ区切りか、11-23 のような範囲で)。層はモデルで違うので、使う人が決めます", { m: this.nLayer ? this.nLayer - 1 : "?" }), true);
      return null;
    }
    const segs = parseSegments(v("#fx-seg"));
    if (!segs) { this.status(which, t("位置の区切りは 0,1-3,4- のように書きます(位置 0 = 応答を書き始める直前)。文字の直前に書くなら before:<正規表現>"), true); return null; }
    const name = v("#fx-name");
    if (name && !NAME_RE.test(name)) { this.status(which, t("名前は英数字・_・-・. だけにしてください"), true); return null; }
    const mt = +v("#fx-mt");
    const p = { layers: layers.join(","), segments: segs.map((s) => s.text).join(","), form: v("#fx-form"), max_tokens: Number.isInteger(mt) && mt > 0 ? mt : 96 };
    const iv = v("#fx-iv");
    if (iv) p.iv = iv.includes(":") ? iv.split(/[\s,]+/).filter(Boolean) : iv;
    return { p, layers, segs, name: name || null };
  }

  // ジョブに渡すしきい値: 空(か、上限つきの形でない)なら undefined(既定)、1 つならその文字列、区切りごとなら並び。読めなければ null
  cap() {
    if (this.$("#fx-form").value !== "cap") return undefined;
    const segs = parseSegments(this.$("#fx-seg").value);
    const xs = parseCaps(this.$("#fx-cap").value, segs ? segs.length : 0);
    return xs === null ? null : !xs.length ? undefined : xs.length === 1 ? xs[0] : xs;
  }

  // ジョブに渡す鍵の作り方: いまの作り方(既定)なら {}。最小二乗なら { key_fit: "ridge" }(正則化の強さを入れたら key_c も)。
  // 読めなければ欄の横に理由を出して null
  keyFit(which) {
    if (this.$("#fx-keyfit").value !== "ridge") return {};
    const segs = parseSegments(this.$("#fx-seg").value);
    if (segs && segs.length && segs[0].before) {
      this.status(which, t("最小二乗のキーは、応答の位置の区切り(0,1-3,4- など)のキーだけです。書く位置が before:<正規表現> のときは、いまの作り方で作ります"), true);
      return null;
    }
    const raw = this.$("#fx-keyc").value.trim();
    if (!raw) return { key_fit: "ridge" };
    const c = Number(raw);
    if (!Number.isFinite(c) || c <= 0) { this.status(which, t("正則化の強さは 0 より大きい数です(例 1)。空なら 5 分割で決めます"), true); return null; }
    return { key_fit: "ridge", key_c: c };
  }

  // 「セルに会話の形も入れる」: 最小二乗のキーのときだけ true。キーを作るジョブとまとめて作るジョブに、いつも渡す(表の値より欄を使う)
  convCells() {
    return { key_conv_cells: this.$("#fx-keyfit").value === "ridge" && this.$("#fx-convcells").checked };
  }

  // 前のやり取りの応答の上限: 入れたときだけ渡す(空なら表の fish.prior_max_tokens、それも無ければ 256)。読めなければ理由を出して null
  priorParam(which) {
    const raw = this.$("#fx-prior-mt").value.trim();
    if (!raw) return {};
    const n = Number(raw);
    if (!Number.isInteger(n) || n < 1) { this.invalid("#fx-prior-mt", which, t("前のやり取りの応答の上限は 1 以上の整数です(空なら 256)")); return null; }
    return { prior_max_tokens: n };
  }

  // 読めない欄: 理由を欄の横に出し、欄に「直すところ」の印(aria-invalid。入力で消える)。見えている欄ならフォーカスを移す
  invalid(sel, which, msg) {
    const el = this.$(sel);
    el.setAttribute("aria-invalid", "true");
    if (typeof el.checkVisibility === "function" ? el.checkVisibility() : el.offsetParent !== null) el.focus();
    this.status(which, msg, true);
  }

  // ユニットの書き方: !kauto と、取り除く形(上限つきの形だけ)。付けないものは入れない(/api/fish/units の引数)
  unitOpts() {
    const o = {};
    if (this.$("#fx-kauto").checked) o.kscale = "auto";
    const r = this.$("#fx-uform").value;
    if (r !== "add" && this.$("#fx-form").value === "cap") o.remove = r;
    return o;
  }

  dirSource() {
    const r = this.root.querySelector("input[name=fx-dirsrc]:checked");
    return r ? r.value : "contrast";
  }

  setDirSource(v) {
    for (const r of this.root.querySelectorAll("input[name=fx-dirsrc]")) r.checked = r.value === v;
    this.dirMode();
  }

  // 方向のもとに合わせて欄を切り替える: トークン・Δ と層の走査は、トークンの行のときだけ。対照集合の差だけに効く欄(作り方・重ねる・
  // 比べる・方向の返事の長さ・速い読み方)は、トークンの行のときは使えなくする
  dirMode() {
    const tok = this.dirSource() === "token";
    for (const id of ["#fx-token", "#fx-tdelta"]) this.$(id).disabled = !tok;
    for (const id of ["#fx-method", "#fx-stack", "#fx-compare", "#fx-rt", "#fx-fast"]) this.$(id).disabled = tok;
    this.$("#fx-scan-box").hidden = !tok;
  }

  // トークンの欄 → 1 つなら文字、空白で区切ってあれば並び(行の平均)。空なら null
  tokens() {
    const xs = this.$("#fx-token").value.trim().split(/\s+/).filter(Boolean);
    return !xs.length ? null : xs.length === 1 ? xs[0] : xs;
  }

  // 方向のもと → ジョブの引数。対照集合の差なら { token: "" }(表の fish.token より欄を使う)。トークンの行なら token と token_delta。
  // 読めなければ欄の横に理由を出して null
  tokenParams(which) {
    if (this.dirSource() !== "token") return { token: "" };
    const tokens = this.tokens();
    if (!tokens) { this.invalid("#fx-token", which, t("トークンを入れてください(例 <|im_end|>・◇ か #ID)")); return null; }
    const raw = this.$("#fx-tdelta").value.trim(), d = Number(raw);
    if (!raw || !Number.isFinite(d) || d <= 0) {
      this.invalid("#fx-tdelta", which, t("Δ(ロジットの上がり)は 0 より大きい数です。2. の「層を走査する」で、層ごとに要る Δ が分かります"));
      return null;
    }
    const segs = parseSegments(this.$("#fx-seg").value);
    if (segs && segs[0].before) {
      this.status(which, t("トークンの行の方向は、応答の位置の区切り(0 など)で使います。書く位置が before:<正規表現> のときは使えません"), true);
      return null;
    }
    return { token: tokens, token_delta: d };
  }

  // ジョブに渡す強さ: 全部の区切りが同じ値なら数 1 つ、違えば区切りごとの並び。読めなければ null
  alpha() {
    const xs = this.alphas();
    return xs === null ? null : xs.every((a) => a === xs[0]) ? xs[0] : xs;
  }

  needed() {
    const segs = parseSegments(this.$("#fx-seg").value);
    return segs ? segs.length * PER[this.$("#fx-form").value] : null;
  }

  showNeed() {
    const n = this.needed(), segs = parseSegments(this.$("#fx-seg").value);
    const layers = parseLayers(this.$("#fx-layers").value);
    this.$("#fx-need").textContent = n === null ? "" : t("層 1 つに {n} 個(区切り {s} 個 × {p} 個)", { n, s: segs.length, p: PER[this.$("#fx-form").value] })
      + (layers ? ` · ${t("{l} 層で合わせて {m} 個のニューロンを書き換える", { l: layers.length, m: layers.length * n })}` : "")
      // 層が多いとき: 時間がかかることを、押す前に伝える(見積もりの秒数は、ジョブを投げると出る)
      + (layers && layers.length >= 6 ? ` · ${t("層が多いと作成に時間がかかります(Qwen3.5-4B の 13 層で 5〜6 分ほど)。見積もりは「まとめて作る」を押すと出ます")}` : "");
  }

  // 層・区切り・形を変えた: 必要な個数と候補を出し直し、ユニットを組み直す
  settingsChanged() {
    const cap = this.$("#fx-form").value === "cap", ridge = this.$("#fx-keyfit").value === "ridge";
    this.$("#fx-cap").disabled = !cap;
    this.$("#fx-keyc").disabled = !ridge;
    this.$("#fx-convcells").disabled = !ridge;                 // セルに会話の形を入れるのは、最小二乗のキーのときだけ
    const uf = this.$("#fx-uform");                            // 取り除く形は、上限つきの形だけ
    for (const o of uf.options) if (o.value !== "add") o.disabled = !cap;
    if (!cap && uf.value !== "add") uf.value = "add";
    this.$("#fx-uform-note").textContent = cap ? "" : t("取り除く形は、上限つきの 2 個組のときだけ選べます");
    this.dirMode();
    this.renderAlphas("fx-alphas");
    this.showNeed();
    if (this.st.usage) { this.fillDefaults(); this.renderNeurons(); }
    this.compose();
  }

  // --- ジョブ ----------------------------------------------------------------------------------------------
  async launch(which, kind, p, running) {
    this.status(which, t("ジョブを投げています…"));
    try {
      const j = await this.onRun(kind, p);
      if (!j) { this.status(which, t("やめました")); return; }
      this.jobs[which] = j.id;                                 // この欄は、最後に投げたこのジョブの結果だけを受け取る
      const est = (Number.isFinite(+j.estimate_s) && j.estimate_s !== null ? ` ${t("(見積もり 約 {s} 秒)", { s: Math.round(+j.estimate_s) })}` : "")
        // 層を重ねる読み直しを 1 つずつ読む場合で数えた見積もり(このモデルでまだ読み直していない)。実際は短いことが多い
        + (j.estimate_stack_first ? ` ${t("(初回は長めに出ます。層ごとの読み直しを、いちばん遅い場合で数えています)")}` : "");
      this.status(which, running + est);
      this.follow(which, j.id);
      this.marks();
    } catch (e) { this.status(which, `${t("エラー")}: ${e.message}`, true); }
  }

  follow(which, id) {
    const el = this.$(`#fx-prog-${which}`);
    const show = (j) => {
      if (this.jobs[which] !== id) return;
      if (!["queued", "running"].includes(j.status)) { el.innerHTML = ""; return; }
      el.innerHTML = `<span class="muted note">${t(j.status === "queued" ? "待機中(進み具合と「止める」は下の帯)" : "実行中(進み具合と「止める」は下の帯)")}</span>`;
    };
    show({ status: "queued" });
    let es;
    try { es = new EventSource(`/api/jobs/${id}/events`); } catch { return; }
    es.onmessage = (m) => {
      const j = JSON.parse(m.data);
      show(j);
      if (["done", "error", "cancelled"].includes(j.status)) { es.close(); this.jobEnded(j); }
    };
    es.onerror = () => { es.close(); if (this.jobs[which] === id) el.innerHTML = ""; };
  }

  // app.js の onJobFinished(と上の SSE)から。同じジョブは 1 回だけ。この欄で最後に投げたジョブでなければ結果を使わない
  async jobEnded(j) {
    const which = Object.keys(this.jobs).find((k) => this.jobs[k] === j.id) || null;
    if (!which || this.handled.has(j.id)) return;
    this.handled.add(j.id);
    this.$(`#fx-prog-${which}`).innerHTML = "";
    this.marks();
    if (j.status !== "done") {
      const e = j.error || {};
      this.status(which, j.status === "error" ? `${t("失敗")}: ${e.message || ""}${e.hint ? `(${e.hint})` : ""}` : t("止めました"), j.status === "error");
      return;
    }
    const r = j.result || {};
    this.status(which, t("できました({s} 秒)", { s: fmt(r.seconds, 1) }));
    const st = this.st;
    if (which === "dir") {
      st.dir = r; st.path = r.contrast_path || st.path;
      st.name = r.method === "token_row" ? String(r.name).replace(/_tok$/, "") : r.name;   // トークンの行の方向は <名前>_tok。キーの名前は <名前> から
      this.renderDirection();
    } else if (which === "scan") {
      st.scan = r;
      this.renderScan();
    } else if (which === "neu") {
      st.usage = r;
      this.fillDefaults();
      this.renderNeurons();
    } else if (which === "keys") {
      st.keys = r; st.keyBase = r.base; st.path = r.contrast_path || st.path;
      this.renderKeys();
    } else if (which === "verify") {
      st.verify = r;
      this.status("verify", r.ok ? t("合格") : t("不合格"), !r.ok);
      this.renderVerify();
    } else if (which === "check") {
      st.check = r;
      this.renderCheck();
    } else if (which === "behave") {
      st.behave = r;
      this.renderBehave();
    } else if (which === "try") {
      st.tries.push(r);
      if (st.tries.length > 20) st.tries.shift();
      st.tryShown = st.tries.length - 1;
      this.renderTry();
    } else if (which === "heads") {
      st.heads = r;
      this.renderHeads();
      this.hgLoad();                                           // できた取り除く量の鍵・レシピを一覧に入れる
    } else if (which === "all") {
      st.pipe = r; st.dir = r.direction; st.name = r.name; st.path = r.contrast_path; st.usage = r.usage;
      st.keys = r.rekeyed || r.keys; st.keyBase = r.key_base; st.verify = r.verify; st.check = r.check;
      st.picked = {}; st.touched = new Set();
      for (const [L, js] of Object.entries(r.neurons || {})) st.picked[+L] = js.map(Number);
      if (r.alphas) { this.renderAlphas("fx-alphas", r.alphas); }
      if (r.caps) this.$("#fx-cap").value = new Set(r.caps).size === 1 ? r.caps[0] : r.caps.join(",");
      this.renderPipe();
      this.renderDirection();
      this.renderNeurons();
      this.renderKeys();
      this.renderVerify();
      this.renderCheck();
    }
    await this.compose();
    this.marks();
  }

  // 「速い読み方」のチェック(テンプレートの値を入れるときに使う)
  setFast(on) {
    for (const b of this.root.querySelectorAll("input.fx-fast")) b.checked = !!on;
  }

  fast() {
    return this.$("#fx-fast").checked;
  }

  async runAll() {
    const s = this.source("all"), c = this.common("all");
    if (!s || !c) return;
    const a = this.alpha();
    if (a === null) { this.status("all", t("強さを、区切りごとに全部入れてください(例 1 / 1 / 0.8)"), true); return; }
    const cap = this.cap();
    if (cap === null) { this.status("all", t("しきい値は 0.4-0.8 のように書きます(下 < 上)。区切りごとに変えるなら、区切りの数だけカンマ区切りで"), true); return; }
    const kf = this.keyFit("all");
    if (kf === null) return;
    const tk = this.tokenParams("all"), pr = this.priorParam("all");
    if (tk === null || pr === null) return;
    const rt = +this.$("#fx-rt").value, ps = +this.$("#fx-pseudo").value, top = +this.$("#fx-top").value;
    // ユニットの書き方・方向のもと・セルに会話の形を入れるかは、付けないときも空・切で渡す(表の fish の値より欄を使う)
    const p = { ...s, ...c.p, ...kf, ...this.convCells(), kscale: "", remove: "", ...this.unitOpts(), ...tk, ...pr,
                alpha: a, same_reply: this.$("#fx-method").value === "same", stack: this.$("#fx-stack").checked,
                stack_fast: this.fast(), compare: this.$("#fx-compare").checked, rekey: this.$("#fx-rekey-on").checked,
                pseudo: Number.isInteger(ps) && ps >= 0 ? ps : 4, response_tokens: Number.isInteger(rt) && rt > 0 ? rt : 96,
                top: Number.isInteger(top) && top > 0 ? top : 20 };
    if (c.name) p.name = c.name;
    if (cap !== undefined) p.cap = cap;
    const use = this.$("#fx-usedir").value.trim();
    if (use && !NAME_RE.test(use)) { this.status("all", t("方向の名前は英数字・_・-・. だけにしてください"), true); return; }
    if (use && tk.token) { this.status("all", t("トークンの行の方向と「すでにある方向を使う」は一緒に使えません(どちらかで方向を決めます)"), true); return; }
    if (use) p.direction = use;
    const picked = this.pickedFor(c.layers);                 // 手で選び直したニューロンがあれば、それを使う
    if (picked && [...this.st.touched].some((L) => c.layers.includes(L))) p.neurons = picked;
    await this.launch("all", "fish_pipeline", p, t("方向 → ニューロンの表 → キー → レシピ → 確かめ を回しています(モデルを流します。GPU を待つことがあります)…"));
  }

  async runDirection() {
    const s = this.source("dir"), c = this.common("dir");
    if (!s || !c) return;
    const tk = this.tokenParams("dir"), pr = this.priorParam("dir");
    if (tk === null || pr === null) return;
    const rt = +this.$("#fx-rt").value;
    const p = { ...s, layers: c.p.layers, segments: c.p.segments, same_reply: this.$("#fx-method").value === "same", stack: this.$("#fx-stack").checked,
                stack_fast: this.fast(), compare: this.$("#fx-compare").checked, response_tokens: Number.isInteger(rt) && rt > 0 ? rt : 96, ...tk, ...pr };
    if (c.name) p.name = c.name;
    await this.launch("dir", "fish_direction", p, tk.token ? t("トークンの行から方向を作っています(応答は生成しません)…") : t("方向を作っています(応答を生成します)…"));
  }

  async runNeurons() {
    const c = this.common("neu");
    if (!c) return;
    const top = +this.$("#fx-top").value;
    const p = { layers: c.p.layers, top: Math.max(Number.isInteger(top) && top > 0 ? top : 20, this.needed() || 1) };
    if (c.p.iv) p.iv = c.p.iv;
    await this.launch("neu", "nw_neurons", p, t("ふだんの働きを測っています…"));
  }

  // 鍵をまとめて作る。rekey なら、いまのユニットを当てた状態で作り直す(名前の頭は <名前>_r)
  async runKeys(rekey) {
    const which = "keys";
    const s = this.source(which), c = this.common(which);
    if (!s || !c) return;
    const ps = +this.$("#fx-pseudo").value;
    const kf = this.keyFit(rekey ? "units" : which), pr = this.priorParam(rekey ? "units" : which);
    if (kf === null || pr === null) return;
    const base = c.name || this.st.name || (this.src === "json" && this.jspec.spec ? String(this.jspec.spec.name) : null);
    const p = { ...s, ...c.p, ...kf, ...this.convCells(), ...pr, pseudo: Number.isInteger(ps) && ps >= 0 ? ps : 4 };
    if (rekey) {
      if (!this.st.units.length) { this.status("units", t("先にユニットを組んでください(方向・ニューロン・キー)"), true); return; }
      p.with_units = [...this.st.units];
      p.base = `${(base || this.st.keyBase || "fish").replace(/_r$/, "")}_r`;
      this.status("units", t("キーを作り直しています(4. の欄に進み具合が出ます)…"));
    } else if (base) p.base = base;
    await this.launch(which, "fish_keys", p, rekey ? t("レシピを当てた状態で、キーを作り直しています…") : t("キーを作っています(応答を生成します)…"));
  }

  async runVerify() {
    if (!this.st.units.length) return;
    const c = this.common("verify");
    if (!c) return;
    const p = { units: [...this.st.units] };
    if (c.p.iv) p.iv = c.p.iv;
    await this.launch("verify", "fish_verify", p, t("確かめています…"));
  }

  async runCheck() {
    if (!this.st.units.length) return;
    const s = this.source("check"), c = this.common("check"), pr = s && c ? this.priorParam("check") : null;
    if (!s || !c || !pr) return;
    const p = { ...s, units: [...this.st.units], max_tokens: c.p.max_tokens, segments: c.p.segments, ...pr };
    if (c.p.iv) p.iv = c.p.iv;
    await this.launch("check", "fish_check", p, t("レシピを当てて応答を作り、足した量を測っています…"));
  }

  async runBehave() {
    if (!this.st.units.length) return;
    const s = this.source("behave"), c = this.common("behave"), pr = s && c ? this.priorParam("behave") : null;
    if (!s || !c || !pr) return;
    const n = +this.$("#fx-bn").value, mt = +this.$("#fx-bmt").value;
    const p = { ...s, units: [...this.st.units], n: Number.isInteger(n) && n > 0 ? n : 12, max_tokens: Number.isInteger(mt) && mt > 0 ? mt : 120, ...pr };
    const m = this.$("#fx-bmatch").value.trim(), q = this.$("#fx-bq").value.trim();
    if (m) p.match = m;
    if (q) p.prompts = q;
    if (c.p.iv) p.iv = c.p.iv;
    if (c.name || this.st.name) p.name = c.name || this.st.name;
    await this.launch("behave", "fish_behave", p, t("レシピを当てて応答を作っています…"));
  }

  // 重みに入れる前に試す: 作った方向を、層の出力に直接足して返事を作る(fish_try)
  async runTry() {
    const which = "try";
    const s = this.source(which);
    if (!s) return;
    const v = (q) => this.$(q).value.trim();
    // 方向: この画面で作ったもの。無ければ、名前の欄(か対照集合の名前)の方向が前に作ってあるものとして、そのまま使う
    const dir = this.st.dir || null;
    const dname = dir ? dir.name : (v("#fx-name") || (this.src === "json" && this.jspec.spec ? String(this.jspec.spec.name) : this.$("#fx-name").placeholder));
    if (!dname || !NAME_RE.test(dname)) { this.status(which, t("先に方向を作ってください(2. か「まとめて作る」)"), true); return; }
    const layers = parseLayers(v("#fx-try-layers") || v("#fx-layers"));
    if (!layers || (this.nLayer && layers.some((L) => L < 1 || L > this.nLayer - 2))) {
      this.status(which, t("足す層を入れてください(1〜{m}。層 0 と最終層には足せません)", { m: this.nLayer ? this.nLayer - 2 : "?" }), true);
      return;
    }
    const a = this.alphas("fx-alphas");                        // 強さは共通の設定のもの
    const nSeg = dir ? (dir.segments || []).length || 1 : (a || []).length;
    if (a === null || a.length !== nSeg) { this.status(which, t("強さを、方向の区切り({s})ごとに入れてください", { s: dir ? (dir.segments || []).join(",") : v("#fx-seg") }), true); return; }
    const pr = this.priorParam(which);
    if (pr === null) return;
    const n = +v("#fx-try-n"), mt = +v("#fx-try-mt"), temp = +v("#fx-try-temp"), seed = +v("#fx-try-seed");
    const p = { ...s, ...pr, direction: dname, layers: layers.join(","), alpha: a, n: Number.isInteger(n) && n > 0 ? n : 12,
                max_tokens: Number.isInteger(mt) && mt > 0 ? mt : 120, temperature: Number.isFinite(temp) && temp > 0 ? temp : 0,
                seed: Number.isInteger(seed) && seed >= 0 ? seed : 0 };
    const m = v("#fx-try-match"), q = v("#fx-try-q"), iv = v("#fx-iv");
    if (m) p.match = m;
    if (q) p.prompts = q;
    if (iv) p.iv = iv.includes(":") ? iv.split(/[\s,]+/).filter(Boolean) : iv;
    await this.launch(which, "fish_try", p, t("方向を層の出力に直接足して、応答を作っています…"));
  }

  // 試した層を、共通の設定の層に入れる(強さはもともと共通の設定のもの)
  useTry() {
    const L = this.$("#fx-try-layers").value.trim();
    if (L) this.$("#fx-layers").value = L;
    this.settingsChanged();
    this.status("try", t("上の欄に入れました。層を変えたら、方向を作り直してください(重ねる層が変わるため)"));
  }

  // --- トークンの行の方向: 層を走査する(direction_token_scan のジョブ) ------------------------------------------------------
  // ジョブは対照集合のパスしか受けないので、入力欄の表は /api/fish/contrast でファイルにしてから渡す(content: null の応答はジョブが埋める)
  async runScan() {
    const which = "scan";
    const tokens = this.dirSource() === "token" ? this.tokens() : null;
    if (!tokens) { this.invalid("#fx-token", which, t("トークンを入れてください(例 <|im_end|>・◇ か #ID)")); return; }
    const raw = this.$("#fx-scan-layers").value.trim();
    let layers = null;
    if (raw && raw.toLowerCase() !== "all") {
      layers = parseLayers(raw);
      if (!layers || (this.nLayer && layers.some((L) => L >= this.nLayer - 1))) {
        this.invalid("#fx-scan-layers", which, t("走査する層は 0〜{m} の数を、カンマ区切りか 11-23 のような範囲で入れます(空なら全部の層)", { m: this.nLayer ? this.nLayer - 2 : "?" }));
        return;
      }
    }
    const s = this.source(which), pr = this.priorParam(which);
    if (!s || !pr) return;
    let contrast = s.contrast;
    if (!contrast) {
      try { contrast = (await postJSON("/api/fish/contrast", { spec: s.spec })).path; }
      catch (e) { this.status(which, `${t("エラー")}: ${e.message}`, true); return; }
    }
    const p = { tokens, contrast, ...(layers ? { layers } : {}), ...pr };
    await this.launch(which, "direction_token_scan", p, t("層を走査しています(応答は生成しません)…"));
  }

  // 走査の表の行 → 共通の設定の層と、方向のもとの Δ に入れる(届かなかった層は入れない)
  useScan(i) {
    const r = this.st.scan, row = r && (r.rows || [])[i];
    if (!row || row.need_delta === null || row.need_delta === undefined) return;
    this.$("#fx-layers").value = String(row.layer);
    this.setDirSource("token");
    this.$("#fx-tdelta").value = String(row.need_delta);
    this.settingsChanged();
    this.renderScan();
    this.status("scan", t("層 {l} と Δ {d} を欄に入れました(共通の設定の層・方向のもとの Δ)。方向を作り直してください", { l: row.layer, d: row.need_delta }));
  }

  // --- CHiPS のヘッドの分を場面で取り除く(fish_heads のジョブ) ------------------------------------------------------------
  // 保存したレシピのうち proj のユニット(強さが 0 でないもの)を持つものと、MLP の入力の空間の鍵の一覧を取り直す
  async hgLoad() {
    const seq = ++this.hgSeq;
    const [rs, ks] = await Promise.all([getJSON("/api/recipes").catch(() => []), getJSON("/api/keys").catch(() => [])]);
    if (seq !== this.hgSeq) return;                              // 後から取り直したものがある
    const all = (Array.isArray(rs) ? rs : []).filter((x) => x.this_model);
    this.hg.recipeNames = new Set(all.map((x) => x.name));
    this.hg.recipes = all.map((x) => {
      const ps = (x.units || []).filter((u) => u.mode === "proj" && Number(u.s ?? u.k));
      return { name: x.name, n: ps.length, layers: [...new Set(ps.map((u) => +u.layer))].sort((a, b) => a - b) };
    }).filter((x) => x.n);
    this.hg.keys = (Array.isArray(ks) ? ks : []).filter((k) => !k.error && k.target === "mlpin");
    this.hgFill();
  }

  hgFill() {
    const el = this.$("#fx-hg-recipe"), cur = el.value;
    el.innerHTML = this.hg.recipes.length
      ? this.hg.recipes.map((x) => `<option value="${esc(x.name)}">${esc(t("{r}(proj {n} 個 · 層 {l})", { r: x.name, n: x.n, l: x.layers.join(",") }))}</option>`).join("")
      : `<option value="">${t("(proj のユニットを持つ保存済みのレシピがありません)")}</option>`;
    if (cur && this.hg.recipes.some((x) => x.name === cur)) el.value = cur;
    this.hgRecipeChanged();
  }

  // 選んだ CHiPS のレシピのヘッドの層
  hgLayers() {
    const r = this.hg.recipes.find((x) => x.name === this.$("#fx-hg-recipe").value);
    return r ? r.layers : [];
  }

  // CHiPS のレシピを選び直した: 鍵の候補を、その層を全部含むものに絞る(含まない鍵は、足りない層を添えて選べなくする)。
  // 場面の鍵は作ったときの役割が scene のもの、ほぼ 1 を返す鍵は one のもの(鍵の要約の positions が role:scene / role:one)
  hgRecipeChanged() {
    const Ls = this.hgLayers();
    this.$("#fx-hg-layers").textContent = Ls.length ? t("CHiPS のヘッドの層: {l}(キーはこの層を全部含むもの)", { l: Ls.join(", ") }) : "";
    const missing = [];
    for (const [sel, role] of [["#fx-hg-scene", "scene"], ["#fx-hg-one", "one"]]) {
      const el = this.$(sel), cur = el.value;
      const ks = this.hg.keys.filter((k) => k.positions === `role:${role}`);
      const lack = (k) => Ls.filter((L) => !(k.layers || []).map(Number).includes(L));
      el.innerHTML = ks.length ? ks.map((k) => {
        const m = lack(k);
        return `<option value="${esc(k.name)}"${m.length ? " disabled" : ""}>${esc(k.name)}${m.length ? ` ${esc(t("(層 {l} が無い)", { l: m.join(",") }))}` : ""}</option>`;
      }).join("") : `<option value="">${role === "scene" ? t("(場面のキーがありません)") : t("(定数キーがありません)")}</option>`;
      const ok = [...el.options].filter((o) => !o.disabled && o.value).map((o) => o.value);
      el.value = ok.includes(cur) ? cur : ok.length ? ok[0] : "";
      if (!ok.length) missing.push(role);
    }
    const L = Ls.length ? Ls.join(",") : t("<層>");
    this.$("#fx-hg-keyhint").innerHTML = missing.length
      ? `${t("使えるキーがありません。CHiPS のヘッドの層を入れて作ります: {cmd}", { cmd: missing.map((r) => `<code>gne nw key --contrast C --layers ${esc(L)} --role ${r} --name NAME_${r}</code>`).join(" ") })}`
      : "";
    this.hgNames();
  }

  // 名前と保存先の薄い字(空のときに使う名前): <CHiPS のレシピ>_hg と <名前>_r
  hgNames() {
    const rec = this.$("#fx-hg-recipe").value;
    const nm = this.$("#fx-hg-name");
    nm.placeholder = rec ? `${rec}_hg` : "hg";
    this.$("#fx-hg-save").placeholder = `${nm.value.trim() || nm.placeholder}_r`;
  }

  // 使う対照集合(1. の欄)の 1 行
  hgSrcLine() {
    const el = this.$("#fx-hg-src");
    let what = null;
    if (this.src === "json") {
      const sp = this.jspec.spec;
      what = sp && !(this.jspec.errors || []).length ? String(sp.name || t("入力欄の JSON")) : null;
    } else {
      const sel = this.$("#fx-contrast");
      what = this.$("#fx-cpath").value.trim() || (sel.value && sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : null);
    }
    el.innerHTML = what ? `${t("対照集合")}: <b class="mono">${esc(what)}</b> ${t("(1. 対照集合で選んだもの。質問 = 取り除きたい場面、dont_touch = 残したい場面)")}`
      : `<span class="pp-warn">${t("先に「1. 対照集合」で対照集合を選ぶか、JSON を書いてください")}</span>`;
  }

  async runHeads() {
    const which = "heads";
    const s = this.source(which);
    if (!s) return;
    const v = (q) => this.$(q).value.trim();
    const bad = (sel, msg) => { const el = this.$(sel); el.setAttribute("aria-invalid", "true"); el.focus(); this.status(which, msg, true); };
    const recipe = v("#fx-hg-recipe"), Ls = this.hgLayers();
    if (!recipe) { bad("#fx-hg-recipe", t("CHiPS のレシピ(proj のユニットを持つ保存済みのレシピ)を選んでください")); return; }
    const scene = v("#fx-hg-scene"), one = v("#fx-hg-one");
    if (!scene) { bad("#fx-hg-scene", t("場面のキーを選んでください(CHiPS のヘッドの層 {l} を全部含むキー)", { l: Ls.join(",") })); return; }
    if (!one) { bad("#fx-hg-one", t("定数キーを選んでください(CHiPS のヘッドの層 {l} を全部含むキー)", { l: Ls.join(",") })); return; }
    const name = v("#fx-hg-name") || this.$("#fx-hg-name").placeholder;
    if (!NAME_RE.test(name)) { bad("#fx-hg-name", t("名前は英数字・_・-・. だけにしてください")); return; }
    const save = v("#fx-hg-save") || `${name}_r`;
    if (!NAME_RE.test(save)) { bad("#fx-hg-save", t("保存先のレシピ名は英数字・_・-・. だけにしてください")); return; }
    const nr = parseNeuronPairs(v("#fx-hg-neurons"), Ls);
    if (nr.error) { bad("#fx-hg-neurons", nr.error); return; }
    // 同じ名前で前に作った方向・取り除く量の鍵・レシピは上書きされるので、確かめる
    const over = [this.hg.keys.some((k) => k.name === `${name}_amount`) ? t("方向 {d} と取り除く量のキー {k}", { d: `${name}_dir`, k: `${name}_amount` }) : null,
                  this.hg.recipeNames.has(save) ? t("レシピ {r}", { r: save }) : null].filter(Boolean);
    if (over.length && !(await askConfirm(t("{w} を作り直します(前のものは上書きされます)。よいですか?", { w: over.join("・") }), { ok: t("作り直す") }))) {
      this.status(which, t("やめました"));
      return;
    }
    const mt = +v("#fx-hg-mt");
    const p = { ...s, recipe, name, scene_key: scene, one_key: one, subset: v("#fx-hg-subset"), max_tokens: Number.isInteger(mt) && mt > 0 ? mt : 96, save };
    if (nr.neurons) p.neurons = nr.neurons;
    await this.launch(which, "fish_heads", p, t("無介入の応答を作り、取り除く量を当てています…"));
  }

  // できたユニットをレシピに足す(同じ層・同じニューロンの行は置き換える)
  addHeads() {
    const rows = ((this.st.heads && this.st.heads.units) || []).map(parseNw).filter(Boolean);
    if (!rows.length) return;
    this.onAddNw(rows);
    this.status("heads", t("レシピに足しました({n} 個)。レシピのタブでチャット・保存・LoRA の書き出しができます", { n: rows.length }));
  }

  // --- ニューロンの選択 -------------------------------------------------------------------------------------
  // 既定の候補(表の上から必要な個数)を、手で選び直していない層に入れる。手で選び直した層はそのまま残す
  fillDefaults() {
    const n = this.needed(), st = this.st;
    if (!st.usage || n === null) return;
    for (const row of st.usage.layers || []) {
      const L = +row.layer;
      if (st.touched.has(L)) continue;
      st.picked[L] = row.rows.slice(0, n).map((x) => +x.neuron);
    }
  }

  togglePick(L, n, on) {
    const st = this.st, cur = new Set(st.picked[L] || []);
    if (on) cur.add(n); else cur.delete(n);
    const order = ((st.usage && st.usage.layers) || []).find((x) => +x.layer === L);
    const rank = new Map((order ? order.rows : []).map((x, i) => [+x.neuron, i]));
    st.picked[L] = [...cur].sort((a, b) => (rank.get(a) ?? 1e9) - (rank.get(b) ?? 1e9));    // 働きの小さい順に、区切り 0 から使う
    st.touched.add(L);
    this.renderNeurons();
    this.compose();
  }

  resetPick(L) {
    this.st.touched.delete(L);
    this.fillDefaults();
    this.renderNeurons();
    this.compose();
  }

  // いまの層ぶんの選択({層: [番号…]})。どれかの層で個数が合わなければ null
  pickedFor(layers) {
    const n = this.needed(), out = {};
    if (n === null) return null;
    for (const L of layers) {
      const js = this.st.picked[L] || [];
      if (js.length !== n) return null;
      out[L] = js;
    }
    return out;
  }

  // --- ユニットを組む(/api/fish/units。書き方はサーバの gne/fish.py が決める) --------------------------------------
  async compose() {
    const seq = ++this.unitSeq;
    const st = this.st, el = this.$("#fx-units");
    const done = (specs, rows, msg) => {
      st.units = specs; st.rows = rows;
      for (const id of ["#fx-add", "#fx-rekey", "#fx-verify-go", "#fx-check-go", "#fx-behave-go"]) this.$(id).disabled = !specs.length;
      if (msg !== undefined) el.innerHTML = msg;
    };
    const layers = parseLayers(this.$("#fx-layers").value), segs = parseSegments(this.$("#fx-seg").value), a = this.alphas();
    const form = this.$("#fx-form").value;
    const cap = this.cap();
    const missing = [];
    if (!st.dir) missing.push(t("方向"));
    if (!st.usage && !Object.keys(st.picked).length) missing.push(t("ニューロンの表"));
    if (!st.keys) missing.push(t("キー"));
    if (missing.length || !layers || !segs || a === null || cap === null) {
      done([], [], `<p class="muted note">${missing.length ? t("まだ無いもの: {m}", { m: missing.join("・") }) : cap === null ? t("しきい値は 0.4-0.8 のように書きます(下 < 上)。区切りごとに変えるなら、区切りの数だけカンマ区切りで") : t("層・位置の区切り・強さを入れてください")}</p>`);
      return;
    }
    const picked = this.pickedFor(layers);
    const mism = [];
    if (!picked) mism.push(t("ニューロンは層ごとに {n} 個選んでください(3. の表のチェック)", { n: this.needed() }));
    if (st.dir && (st.dir.segments || []).join(",") !== segs.map((s) => s.text).join(",")) mism.push(t("方向の区切り({a})が、いまの欄({b})と違います。方向を作り直してください", { a: (st.dir.segments || []).join(","), b: segs.map((s) => s.text).join(",") }));
    if (st.keys && ((st.keys.segments || []).join(",") !== segs.map((s) => s.text).join(",") || st.keys.form !== form)) mism.push(t("キーの区切り・形が、いまの欄と違います。キーを作り直してください"));
    for (const [what, r] of [[t("方向"), st.dir], [t("キー"), st.keys]]) {
      const have = new Set((r && r.layers) || []);
      const lack = layers.filter((L) => !have.has(L));
      if (r && lack.length) mism.push(t("{w} に層 {l} がありません。作り直してください", { w: what, l: lack.join(",") }));
    }
    if (mism.length) { done([], [], `<ul class="evx-notes">${mism.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>`); return; }
    let x;
    try {
      x = await postJSON("/api/fish/units", { layers: layers.join(","), segments: segs.map((s) => s.text).join(","), neurons: picked, form,
                                              direction: st.dir.name, base: st.keyBase, alpha: a, ...(cap === undefined ? {} : { cap }), ...this.unitOpts() });
    } catch (e) {
      if (seq === this.unitSeq) done([], [], `<p class="err note">${esc(e.message)}</p>`);
      return;
    }
    if (seq !== this.unitSeq) return;                          // 後から組み直したものがある
    done(x.specs, x.units, `<p class="note">${t("ユニット {n} 個(層 {l} × 区切り {s})。キー {k}", { n: x.specs.length, l: layers.length, s: segs.length, k: esc(st.keyBase) })}${
      st.keys && st.keys.with_units ? ` <span class="muted">(${t("レシピを当てた状態で作り直したキー")})</span>` : ""}</p>
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-unit-tab"><thead><tr><th>${t("層")}</th><th>${t("区切り")}</th><th>${t("ニューロン")}</th><th>${t("方向")}</th><th>${t("キー")}</th><th class="l">${t("ユニット")}</th></tr></thead><tbody>${
      x.units.map((u, i) => `<tr><td>L${u.layer}</td><td>${esc(segs[i % segs.length].text)}</td><td>N${u.neuron}${u.pair !== null && u.pair !== undefined ? ` + N${u.pair}` : ""}</td><td>${esc(u.direction)}</td><td>${esc(u.key)}${u.key2 ? ` &amp; ${esc(u.key2)}` : ""}</td><td class="l">${esc(x.specs[i])}</td></tr>`).join("")}</tbody></table></div>`);
  }

  addToRecipe() {
    if (!this.st.rows.length) return;
    this.onAddNw(this.st.rows.map((u) => ({ ...u })));
    this.status("units", t("レシピに足しました({n} 個)。レシピのタブでチャット・保存・LoRA の書き出しができます", { n: this.st.rows.length }));
  }

  // --- 結果の表 ---------------------------------------------------------------------------------------------
  renderPipe() {
    const r = this.st.pipe, el = this.$("#fx-pipe");
    if (!r) { el.innerHTML = ""; return; }
    const ss = r.stage_seconds || {};
    const lab = { direction: t("方向"), neurons: t("ニューロンの表"), keys: t("キー"), rekey: t("キーの作り直し"), verify: t("式どおりか"), check: t("足した量") };
    const rd = r.ridge || {};
    el.innerHTML = `<p class="note">${Object.keys(lab).filter((k) => ss[k] !== undefined).map((k) => `${lab[k]} ${fmt(ss[k], 1)} ${t("秒")}`).join(" → ")}</p>
      <p class="note" id="fx-pipe-fit">${this.fitHTML(r.key_fit, [[t("1 段目のキー"), rd.build], [t("当てた状態で作り直したキー"), rd.rekey]])}</p>
      <p class="muted note">${r.neurons_default ? t("ニューロンは既定の候補(表の上から)を使いました。3. の表のチェックで選び直せます(選び直したら、キーの作り直しと確かめをやり直します)") : t("ニューロンは、手で選んだものを使いました")}: ${
      Object.entries(r.neurons || {}).map(([L, js]) => `L${L} ${js.map((j) => `N${j}`).join(" ")}`).join(" / ")}</p>
      ${(r.notes || []).length ? `<p class="muted note">${r.notes.map(esc).join("<br>")}</p>` : ""}
      <p class="muted note">${t("下の「5. レシピを組む」の「レシピに足す」で、レシピのタブに入ります")}</p>`;
  }

  renderDirection() {
    const r = this.st.dir, el = this.$("#fx-dir");
    if (!r) { el.innerHTML = ""; return; }
    const rows = (r.rows || []).map((x) => `<tr><td>L${x.layer}</td><td>${esc(x.positions)}</td><td>${esc(x.direction)}</td><td>${fmt(x.norm, 2)}</td><td>${fmt(x.norm_old, 2)}</td>
      <td title="${t("1 に近いほど、今までの作り方と同じ向き")}">${fmt(x.cos_old, 2)}</td></tr>`).join("");
    const tok = r.method === "token_row" ? r.token || {} : null;    // トークンの行の方向: トークン・Δ・β・r̄ と、区切らないこと
    const head = tok ? `${t("方向")} <b class="mono">${esc(r.name)}</b> · <span id="fx-dir-token">${t("トークンの行")} <span class="mono">${esc([].concat(tok.token_text || tok.tokens || []).join(" "))}</span>
        · Δ ${fmt(tok.delta, 2)} · β ${fmt(tok.beta, 2)} · r̄ ${fmt(tok.rbar, 3)}</span> · <span class="muted">${t("どの区切りにも同じ向き(ユニットは @{n} をそのまま使う)", { n: esc(r.name) })}</span>`
      : `${t("方向")} <b class="mono">${esc(r.name)}</b> · ${r.same_reply ? t("同じ応答を両方の条件で読む") : t("今までの作り方")}${r.stack ? ` · ${t("下の層に書いた後の残りを使う")}` : ""}${r.stack && (r.layers || []).length > 1 ? ` · <span id="fx-dir-read"${r.stack_fast ? ' class="pp-warn"' : ""}>${r.stack_fast ? t("速い読み方で作った方向(探索向け。配るものは既定の読み方で作り直す)") : t("既定の読み方で作った方向")}</span>` : ""}${r.reused ? ` · ${t("すでにあるものを使った")}` : ""} · ${r.n_prompts ?? "—"} ${t("問")}`;
    el.innerHTML = `<p class="note">${head}</p>
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-dir-tab"><thead><tr><th>${t("層")}</th><th>${t("区切り")}</th><th>${t("方向の名前")}</th><th title="${t("その区切りの方向の長さ(α = 1 で書く量)")}">${t("長さ")}</th>
        <th>${t("今までの作り方の長さ")}</th><th>${t("今までの作り方との cos")}</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${r.compare ? `<p class="muted note">${t("今までの作り方(それぞれの条件で作った応答を読む・下の層を考えない)の方向は {n} に保存しました", { n: esc(r.compare) })}</p>` : ""}`;
  }

  // 層の走査の結果: 層ごとに、要る Δ・β と、dont_touch の入力に Δ の 0.2 倍・0.5 倍を足したときの、トークンが 1 位になった数
  // (誤検出)・KL・1 位が変わった数。行(か行の「使う」)で、その層と Δ を欄に入れる。いまの欄と同じ層・Δ の行に印
  renderScan() {
    const r = this.st.scan, el = this.$("#fx-scan");
    if (!r) { el.innerHTML = ""; return; }
    const leaks = (r.leaks || []).map(Number);
    const cell = (x, f) => (x.leak || {})[String(f)] || (x.leak || {})[f.toFixed(1)] || null;    // サーバの辞書のキーは Python の str(倍率)
    const curL = parseLayers(this.$("#fx-layers").value), curD = Number(this.$("#fx-tdelta").value);
    const maxD = (r.deltas || []).length ? Math.max(...r.deltas) : null;
    const rows = (r.rows || []).map((x, i) => {
      const ok = x.need_delta !== null && x.need_delta !== undefined;
      const on = ok && curL && curL.length === 1 && curL[0] === x.layer && curD === +x.need_delta;
      const lk = leaks.map((f) => {
        const c = cell(x, f);
        return c ? `<td class="${c.top1 ? "fx-bad" : ""}">${c.top1} / ${c.n}</td><td>${fmt(c.kl, 4)}</td><td>${c.changed} / ${c.n}</td>` : "<td>—</td><td>—</td><td>—</td>";
      }).join("");
      return `<tr${ok ? ` data-fxscan="${i}" class="fx-pick${on ? " fx-sel" : ""}"` : ""}><th scope="row">L${x.layer}</th>
        <td${ok ? ` title="${t("{h} / {n} 入力で 1 位", { h: x.open_hit ?? "—", n: x.open_n ?? r.open_n })}"` : ` title="${t("Δ {d} でも、1 位になる入力が足りない", { d: maxD ?? "—" })}"`}>${ok ? esc(String(x.need_delta)) : "—"}</td>
        <td>${fmt(x.beta, 2)}</td>${lk}<td>${ok ? `<button type="button" class="small" data-fxscanuse="${i}">${t("この層と Δ を使う")}</button>` : ""}</td></tr>`;
    }).join("");
    const tok = [].concat(r.token_text || r.tokens || []).join(" ");
    el.innerHTML = `<p class="note" id="fx-scan-sum">${t("足すトークン")} <span class="mono">${esc(tok)}</span> · ${t("対象の場面の入力 {n} 個のうち {k} 個以上({p}%)が 1 位になる最小の Δ", { n: r.open_n ?? "—", k: r.need_n ?? "—", p: Math.round((r.need ?? 0.95) * 100) })}
        · ${t("何も足さないときに 1 位: 対象の場面 {a} / {n}・dont_touch {b} / {c}", { a: r.base_open_top1 ?? "—", n: r.open_n ?? "—", b: r.base_close_top1 ?? "—", c: r.close_n ?? "—" })} · r̄ ${fmt(r.rbar, 3)} · ${fmt(r.seconds, 1)} ${t("秒")}</p>
      <h4 class="note" id="fx-scan-h">${t("層ごとの要る Δ と、dont_touch への影響")}</h4>
      <div class="evx-scroll" role="region" tabindex="0" aria-labelledby="fx-scan-h"><table class="evx mono fx-tab" id="fx-scan-tab"><thead>
        <tr><th rowspan="2" scope="col">${t("層")}</th><th rowspan="2" scope="col" title="${t("対象の場面の入力の位置 0 で、そのトークンが 1 位になる入力が足りる最小の Δ(試した Δ の中から)")}">${t("要る Δ")}</th>
          <th rowspan="2" scope="col" title="${t("その Δ で足すベクトルの長さ")}">β</th>${leaks.map((f) => `<th colspan="3" scope="colgroup" class="fx-scan-g">${t("dont_touch に ×{f} を足したとき", { f })}</th>`).join("")}<th rowspan="2"></th></tr>
        <tr>${leaks.map(() => `<th scope="col" title="${t("そのトークンが 1 位になった入力の数(誤検出)/ 入力の数")}">${t("1 位になった数")}</th><th scope="col" title="${t("位置 0 の確率の分布が、何も足さないときからどれだけ変わったか(平均)")}">KL</th>
          <th scope="col" title="${t("1 位のトークンが、何も足さないときと変わった入力の数 / 入力の数")}">${t("1 位が変わった数")}</th>`).join("")}</tr>
      </thead><tbody>${rows}</tbody></table></div>
      ${noteHTML(t("行か「この層と Δ を使う」で、その層を共通の設定の層に、Δ を方向のもとの Δ に入れます。dont_touch で 1 位になった数(誤検出)と KL が小さい層を選びます。トークンの行の方向は、ふつう 1 層に書きます"), "muted note")}`;
  }

  renderNeurons() {
    const r = this.st.usage, el = this.$("#fx-neu"), n = this.needed();
    if (!r) { el.innerHTML = ""; return; }
    el.innerHTML = (r.layers || []).map((row) => {
      const L = +row.layer, sel = new Set(this.st.picked[L] || []);
      const extra = [...sel].filter((j) => !row.rows.some((x) => +x.neuron === j));
      const bad = n !== null && sel.size !== n;
      const trs = row.rows.map((x, i) => `<tr${sel.has(+x.neuron) ? ` class="fx-sel"` : ""}><td><input type="checkbox" data-fxl="${L}" data-fxn="${x.neuron}"${sel.has(+x.neuron) ? " checked" : ""} aria-label="N${x.neuron}"></td>
        <td>N${x.neuron}</td><td>${fmt(x.max, 3)}</td><td>${fmt(x.mean, 4)}</td><td>${x.rank_max}</td><td>${x.rank_mean}</td>
        <td title="${t("その層の MLP の出力の長さ(中央値)に対する、このニューロンの最大の働き")}">${x.max_share === null || x.max_share === undefined ? "—" : `${fmt(x.max_share * 100, 1)}%`}</td>
        <td class="l">${n !== null && i < n ? `<span class="muted">${t("既定の候補")}</span>` : ""}</td></tr>`).join("");
      return `<h4 class="note">L${L} <span class="${bad ? "err" : "muted"} note" data-fxcount="${L}">${t("選んだ数 {a} / 必要な数 {n}", { a: sel.size, n: n ?? "—" })}</span>
        ${this.st.touched.has(L) ? `<button class="small" data-fxdef="${L}">${t("既定の候補に戻す")}</button>` : ""}
        <span class="muted note">${t("{n} 個のニューロン、{p} 位置で測った", { n: row.n_ff, p: row.positions })}${extra.length ? ` · ${t("表に無い選択: {x}", { x: extra.map((j) => `N${j}`).join(" ") })}` : ""}</span></h4>
        <div class="evx-scroll"><table class="evx mono fx-tab fx-neu-tab" data-l="${L}"><thead><tr><th></th><th>${t("ニューロン")}</th><th title="${t("1 位置で残差に書く長さの最大")}">${t("最大")}</th><th title="${t("1 位置で残差に書く長さの平均")}">${t("平均")}</th>
          <th>${t("最大の順位")}</th><th>${t("平均の順位")}</th><th>${t("出力に占める割合")}</th><th></th></tr></thead><tbody>${trs}</tbody></table></div>`;
    }).join("") + `${noteHTML(t("働きの小さい順(最大の順位 + 平均の順位)。書き換えると、そのニューロンの元の働きは無くなります。区切り 0 から順に、選んだものを上から使います"), "muted note")}`;
  }

  renderKeys() {
    const r = this.st.keys, el = this.$("#fx-keys");
    if (!r) { el.innerHTML = ""; return; }
    const kind = { position: t("位置のキー"), scene: t("場面のキー"), one: t("定数キー") };
    const rows = (r.rows || []).map((x) => `<tr><td class="l">${esc(x.key)}</td><td class="l">${kind[x.kind] || esc(x.kind)}</td><td>${x.positions === null || x.positions === undefined ? "—" : esc(x.positions)}</td><td>L${x.layer}</td>
      <td title="${t("いまの作り方では |k| / |μ|(書き込みたい位置の平均のうち、書き込みたくない位置の部分空間に直交する割合)。最小二乗のキーでは 1 / (|κ| |μ|)(キーと、書き込みたい位置の平均との cos)")}">${fmt(x.kn, 3)}</td>
      <td title="${t("キーの出力の平均の差(書き込みたい位置 − 書き込みたくない位置)。作成に使った例での値")}">${fmt(x.selectivity, 2)}</td>
      <td title="${t("比較用のキー(ラベルをでたらめに入れ替えたキー)での同じ値の 95% 点")}">${fmt(x.pseudo_p95, 2)}</td><td>${mark(x.beats)}</td></tr>`).join("");
    el.innerHTML = `<p class="note">${t("キーの名前の頭")} <b class="mono">${esc(r.base)}</b>${r.with_units ? ` <span class="muted">(${t("レシピを当てた状態で作り直したキー")})</span>` : ""} · ${t("使う質問")} ${esc({ even: t("偶数番目"), odd: t("奇数番目"), all: t("全部") }[r.subset] || r.subset || "—")} · ${t("比較用のキー {n} 本", { n: r.pseudo ?? "—" })} · ${fmt(r.seconds, 1)} ${t("秒")}</p>
      <p class="note" id="fx-keys-fit">${this.fitHTML(r.key_fit, [["", r.ridge]])}</p>${this.formsHTML(r.ridge)}
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-keys-tab"><thead><tr><th class="l">${t("キー")}</th><th class="l">${t("種類")}</th><th>${t("区切り")}</th><th>${t("層")}</th><th>kn</th><th>${t("キーの出力の差")}</th><th>${t("比較用のキーの 95% 点")}</th>
        <th title="${t("キーの出力の差が、比較用のキーの 95% 点を超えているか")}">${t("比較用のキーを超えるか")}</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${(r.notes || []).length ? `<p class="muted note">${r.notes.map(esc).join("<br>")}</p>` : ""}`;
  }

  renderVerify() {
    const r = this.st.verify, el = this.$("#fx-verify");
    if (!r) { el.innerHTML = ""; return; }
    const pc = (x) => (x === null || x === undefined ? "—" : `${(+x * 100).toFixed(2)}%`);
    el.innerHTML = `<h4 class="note">${t("式どおりか")} ${mark(r.ok)}</h4>
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-verify-tab"><thead><tr><th>${t("層")}</th><th class="l">${t("ニューロン")}</th><th>${t("ゲートの行の誤差")}</th><th>${t("入力の行の誤差")}</th><th>${t("活性の誤差")}</th>
        <th>${t("ほかのニューロンの変化")}</th><th>${t("出力の変化の誤差(中央値)")}</th><th></th></tr></thead><tbody>${
      (r.layers || []).map((x) => `<tr><td>L${x.layer}</td><td class="l">${(x.neurons || []).map((j) => `N${j}`).join(" ")}</td><td>${pc(x.gate_err)}</td><td>${pc(x.up_err)}</td><td>${pc(x.act_err)}</td>
        <td>${x.others_changed_max === undefined ? "—" : (+x.others_changed_max).toExponential(1)}</td><td>${pc(x.out_rel_err_median)}</td><td>${mark(x.ok)}</td></tr>`).join("")}</tbody></table></div>
      ${noteHTML(t("合格の目安: ゲート・入力・出力の誤差 5% 未満、活性の誤差 0.1% 未満、ほかのニューロンの変化なし。ゲート・入力には、量子化のままの掛け算とのずれが 1〜2% 残る"), "muted note")}`;
  }

  renderBehave() {
    const r = this.st.behave, el = this.$("#fx-behave");
    if (!r) { el.innerHTML = ""; return; }
    const part = (tag, title) => {
      const d = r[tag];
      if (!d) return "";
      const c = d.counts || {};
      return `<h4 class="note">${esc(title)} <span class="muted note" data-fxb="${tag}">${t("{n} 問 · 正規表現に一致 {m} · 繰り返し {l} · 空 {e} · 長さの上限で切れた {a} · 平均 {k} トークン",
        { n: c.n ?? "—", m: c.match ?? "—", l: c.loop ?? "—", e: c.empty ?? "—", a: c.at_limit ?? "—", k: c.tokens ?? "—" })}</span></h4>
        <div class="evx-scroll"><table class="evx fx-tab fx-behave-tab" data-b="${tag}"><thead><tr><th class="l">${t("質問")}</th><th class="l">${t("応答")}</th><th></th></tr></thead><tbody>${
        d.replies.map((x) => `<tr><td class="l fx-q">${esc(x.prompt)}</td><td class="l fx-a">${esc(x.text)}</td><td>${x.loop ? `<span class="kx-ng">${t("繰り返し")}</span>` : ""}</td></tr>`).join("")}</tbody></table></div>`;
    };
    el.innerHTML = `<p class="muted note">${t("system の指示は付けず、ユニットだけを当てた応答です。正規表現の一致は目安で、応答を読んで確かめてください")}${r.match ? ` · ${t("正規表現")} <span class="mono">${esc(r.match)}</span>` : ""}${
      r.own_prompts ? "" : ` · ${t("対象の場面の質問は、キーづくりに使った質問です(甘い)")}`}</p>
      ${part("scene", t("対象の場面の質問"))}${part("others", t("書き込みたくない場面の質問"))}`;
  }

  // CHiPS のヘッドの分を場面で取り除いた結果: 層ごとの当たり方(5 分割の R²・行の数)、CHiPS との一致(どこでも開く鍵・本物の鍵の
  // 狙う場面・dont_touch の cos と大きさの比)、できたユニットとレシピの保存先
  renderHeads() {
    const r = this.st.heads, el = this.$("#fx-heads");
    if (!r) { el.innerHTML = ""; return; }
    const m = r.match || {};
    const fit = (r.layers || []).map((x) => `<tr><th scope="row">L${x.layer}</th><td class="l">${esc((x.heads || []).join(" "))}</td><td>${fmt(x.r2_cv, 3)}</td>
      <td>${esc(String(x.c ?? "—"))}</td><td>${x.n_rows ?? "—"}</td></tr>`).join("");
    const lab = { open: [t("どこでも開くキー・対象の場面"), t("CHiPS と同じ向き・同じ量を取り除けているか。cos・比とも 1 に近いほどよい")],
                  scene: [t("本物のキー・対象の場面"), t("対象の場面で、CHiPS と同じだけ取り除くか。比が 1 に近いほどよい")],
                  other: [t("本物のキー・dont_touch"), t("残したい場面に触れていないか。比が 0 に近いほどよい")] };
    const match = Object.entries(lab).map(([k, [name, hint]]) => {
      const x = m[k];
      return `<tr data-hg="${k}"><th scope="row" class="l">${name}</th><td>${x ? fmt(x.cos_median, 3) : "—"}</td><td>${x ? fmt(x.ratio_median, 3) : "—"}</td>
        <td>${x ? x.n_positions : "—"}</td><td class="l desc">${hint}</td></tr>`;
    }).join("");
    const units = (r.units || []).map((u) => {
      const x = parseNw(u);
      return `<tr><th scope="row">${x ? `L${x.layer}` : "—"}</th><td>${x ? `N${x.neuron}${x.pair !== null ? ` + N${x.pair}` : ""}` : "—"}</td><td class="l">${esc(u)}</td></tr>`;
    }).join("");
    const region = (id, h, table) => `<h4 class="note" id="${id}-h">${h}</h4>
      <div class="evx-scroll" role="region" tabindex="0" aria-labelledby="${id}-h">${table}</div>`;
    el.innerHTML = `<p class="note">${t("CHiPS のレシピ")} <b class="mono">${esc(r.chips || "—")}</b> · ${t("方向")} <b class="mono">${esc(r.direction)}</b> · ${t("取り除く量のキー")} <b class="mono">${esc(r.amount_key)}</b>
        · ${t("場面のキー")} <span class="mono">${esc(r.scene_key)}</span> · ${t("定数キー")} <span class="mono">${esc(r.one_key)}</span></p>
      <p class="muted note">${t("材料: 対象の場面 {a} 問・dont_touch {b} 問への無介入の応答", { a: r.n_scene ?? "—", b: r.n_other ?? "—" })} · ${fmt(r.seconds, 1)} ${t("秒")}</p>
      ${region("fx-hg-fit", t("取り除く量の当たり方(層ごと)"), `<table class="evx mono fx-tab" id="fx-hg-fit"><thead><tr><th>${t("層")}</th><th class="l">${t("ヘッド")}</th>
        <th title="${t("MLP の入力から取り除く量を当てた式の決定係数。質問単位の 5 分割の交差検証で、作成に使っていない質問の位置で測る。1 に近いほどよく当たる")}">${t("R²(5 分割)")}</th>
        <th title="${t("リッジ回帰の正則化の強さ(行列の対角の平均の何倍か)。5 分割の R² がいちばん高いものを、0.01・0.1・1・10・100 から選んだ")}">${t("正則化の強さ")}</th>
        <th title="${t("当てるのに使った位置の数(応答の位置と、プロンプトの最後の数位置)")}">${t("行の数")}</th></tr></thead><tbody>${fit}</tbody></table>`)}
      ${region("fx-hg-match", t("CHiPS との一致(層 {L} の出力の変化。応答の位置の中央値)", { L: m.layer ?? "—" }), `<table class="evx mono fx-tab" id="fx-hg-match"><thead><tr><th class="l">${t("当て方")}</th>
        <th title="${t("無介入からの変化の向きが、CHiPS の変化とどれだけ同じか(cos 類似度)")}">cos</th>
        <th title="${t("無介入からの変化の大きさを、CHiPS の変化の大きさで割ったもの")}">${t("大きさの比")}</th><th>${t("位置の数")}</th><th class="l">${t("見るところ")}</th></tr></thead><tbody>${match}</tbody></table>`)}
      ${region("fx-hg-units", t("できたユニット"), `<table class="evx mono fx-tab" id="fx-hg-units"><thead><tr><th>${t("層")}</th><th>${t("ニューロン")}</th><th class="l">${t("ユニット")}</th></tr></thead><tbody>${units}</tbody></table>`)}
      <p class="note" id="fx-hg-saved">${r.recipe_path ? `${t("レシピの保存先")}: <span class="mono">${esc(r.recipe_path)}</span>` : t("レシピには保存していません")}</p>
      <div class="pp-go"><button type="button" id="fx-hg-add" title="${t("できたユニットをレシピのタブに足す(同じ層・同じニューロンの行は置き換える)")}">${t("レシピに足す")}</button></div>
      ${noteHTML(t("CHiPS のユニット(proj)と同じレシピで当てると、同じ分を 2 回取り除きます。比べるときは、CHiPS のレシピと、できたレシピを別々に当てます(eval・behave でレシピを並べる)。"), "muted note")}`;
  }

  // 重みに入れる前に試した結果: 回ごとの数を並べた表と、選んだ回の返事(足した返事と、同じ質問に何も足さない返事を横に並べる)
  renderTry() {
    const st = this.st, el = this.$("#fx-try");
    if (!st.tries.length) { el.innerHTML = ""; return; }
    const cnt = (d, k) => (d && d.counts && d.counts[k] !== undefined ? d.counts[k] : "—");
    const rows = st.tries.map((r, i) => `<tr${i === st.tryShown ? ` class="fx-sel"` : ""}><td>${i + 1}</td><td class="l">${esc((r.layers || []).join(","))}</td>
      <td class="l">${esc((r.segments || []).map((sg, k) => `${sg}: ${r.alphas[k]}`).join(" / "))}</td><td>${r.temperature ? `${r.temperature} · ${t("種")} ${r.seed}` : "greedy"}</td>
      <td>${cnt(r.scene, "n")}</td><td>${cnt(r.scene, "loop")}</td><td>${cnt(r.scene, "empty")}</td><td>${cnt(r.scene, "match")}</td><td>${cnt(r.baseline, "match")}</td><td>${cnt(r.scene, "tokens")}</td>
      <td><button class="small" data-fxtry="${i}">${t("応答を見る")}</button></td></tr>`).join("");
    const r = st.tries[Math.max(0, Math.min(st.tryShown, st.tries.length - 1))];
    const base = new Map(((r.baseline && r.baseline.replies) || []).map((x) => [x.prompt, x]));
    const loop = (x) => (x && x.loop ? ` <span class="kx-ng">${t("繰り返し")}</span>` : "");
    const scene = (r.scene.replies || []).map((x) => { const b = base.get(x.prompt);
      return `<tr><td class="l fx-q">${esc(x.prompt)}</td><td class="l fx-a">${esc(x.text)}${loop(x)}</td><td class="l fx-a muted">${b ? esc(b.text) : "—"}${loop(b)}</td></tr>`; }).join("");
    const others = r.others ? `<h4 class="note">${t("書き込みたくない場面の質問(何も足していない)")} <span class="muted note">${t("{n} 問 · 繰り返し {l} · 空 {e}", { n: cnt(r.others, "n"), l: cnt(r.others, "loop"), e: cnt(r.others, "empty") })}</span></h4>
      <div class="evx-scroll"><table class="evx fx-tab fx-try-tab" id="fx-try-others"><thead><tr><th class="l">${t("質問")}</th><th class="l">${t("応答")}</th></tr></thead><tbody>${
      r.others.replies.map((x) => `<tr><td class="l fx-q">${esc(x.prompt)}</td><td class="l fx-a">${esc(x.text)}${loop(x)}</td></tr>`).join("")}</tbody></table></div>` : "";
    el.innerHTML = `<div class="evx-scroll"><table class="evx mono fx-tab" id="fx-try-runs"><thead><tr><th>#</th><th class="l">${t("足した層")}</th><th class="l">${t("区切り: 強さ")}</th><th>${t("生成")}</th><th>${t("質問の数")}</th>
        <th title="${t("同じ語句から抜けられなくなった応答(4 トークンの並びの種類が半分を切る)")}">${t("繰り返しに落ちた応答")}</th><th>${t("空の応答")}</th>
        <th title="${t("正規表現に一致した応答。繰り返しの中の語も数える")}">${t("正規表現に一致")}</th><th>${t("何も足さないときの一致")}</th><th>${t("平均の長さ")}</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="muted note">${t("数は機械的に数えたものです。口調が出たか、相手の発言に合っているかは、応答を読んで確かめてください")}${r.match ? ` · ${t("正規表現")} <span class="mono">${esc(r.match)}</span>` : ""}${
        r.own_prompts ? "" : ` · ${t("対象の場面の質問は、方向づくりに使った質問です(甘い)")}`}</p>
      ${(r.notes || []).slice(0, -2).map((x) => `<p class="pp-warn note">${esc(x)}</p>`).join("")}
      <h4 class="note">${t("対象の場面の質問")} <span class="muted note">#${st.tries.indexOf(r) + 1} · ${t("層")} ${esc((r.layers || []).join(","))}</span></h4>
      <div class="evx-scroll"><table class="evx fx-tab fx-try-tab" id="fx-try-scene"><thead><tr><th class="l">${t("質問")}</th><th class="l">${t("方向を足した応答")}</th><th class="l">${t("何も足さない応答")}</th></tr></thead><tbody>${scene}</tbody></table></div>
      ${others}`;
  }

  // 足した量の、質問ごとの値の要約: 群 × 区間ごとの数(ほとんど書けていない・狙った量を超えた・書いてしまった)と、書けていない質問の一覧
  questionsHTML(r) {
    const qs = r.question_summary;
    if (!qs) return "";
    const rows = [];
    let nLow = 0;
    for (const [g, d] of Object.entries(r.groups || {})) {
      const scene = g === "scene";
      for (const [sg, c] of Object.entries(d.by_question || {})) {
        if (scene) nLow += c.low || 0;
        rows.push(`<tr><td class="l">${esc(scene ? t("対象の場面") : g.replace(/^dont_touch:/, ""))}</td><td>${esc(sg === "match" ? t("一致の直前") : sg)}</td><td>${c.n}</td>
          <td class="${scene && c.low ? "fx-low" : ""}">${scene ? c.low : "—"}</td><td class="${scene && c.over ? "fx-bad" : ""}">${scene ? c.over : "—"}</td>
          <td class="${!scene && c.leak ? "fx-bad" : ""}">${scene ? "—" : c.leak}</td><td>${fmt(c.amount_min, 2)}</td><td>${fmt(c.amount_median, 2)}</td></tr>`);
      }
    }
    const worst = (qs.worst || []).map((q) => `<tr><td>${esc(q.segment === "match" ? t("一致の直前") : q.segment)}</td><td class="fx-low">${fmt(q.amount, 2)}</td><td>${fmt(q.rho, 2)}</td>
      <td class="l fx-q">${esc(q.prompt)}</td><td class="l fx-a">${esc(String(q.reply_start || "").slice(0, 20))}${q.loop ? ` <span class="kx-ng">${t("繰り返し")}</span>` : ""}</td></tr>`).join("");
    return `<h4 class="note">${t("質問ごとの値")} <span class="${nLow ? "pp-warn" : "muted"} note" id="fx-q-sum">${t("対象の場面で、ほとんど書けていない質問 {n} 件", { n: nLow })}</span></h4>
      ${noteHTML(t("平均だけでは、一部の質問でまったく書けていないことが隠れます。区間(ユニットが担当する応答の位置の範囲)ごとに、質問 1 つずつの係数 g(狙った量 = 1)を数えています"), "muted note")}
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-q-tab"><thead><tr><th class="l">${t("場面")}</th><th>${t("区間")}</th><th>${t("質問の数")}</th>
        <th title="${t("その区間で係数 g が {v} 未満の質問", { v: qs.low })}">${t("ほとんど書けていない(係数 g {v} 未満)", { v: qs.low })}</th>
        <th title="${t("その区間の位置で、全部のユニットが書いた量の合計が {v} を超えた質問", { v: qs.over })}">${t("狙った量を超えた(合計 {v} 超)", { v: qs.over })}</th>
        <th title="${t("書き込みたくない場面で、合計の大きさが {v} 以上の質問", { v: qs.leak })}">${t("書いてしまった(合計 {v} 以上)", { v: qs.leak })}</th>
        <th>${t("係数 g の最小")}</th><th>${t("係数 g の中央値")}</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>
      ${worst ? `<h4 class="note">${t("書けていない質問(係数 g の小さい順。全部で {n} 件)", { n: qs.n_low })}</h4>
      <div class="evx-scroll"><table class="evx fx-tab" id="fx-q-worst"><thead><tr><th>${t("区間")}</th><th>${t("係数 g")}</th><th title="${t("その区間のキーの出力の平均。上限つきの形は、しきい値と比べる値")}">${t("キーの出力")}</th>
        <th class="l">${t("質問")}</th><th class="l">${t("応答の出だし")}</th></tr></thead><tbody>${worst}</tbody></table></div>
      ${noteHTML(t("書けていない質問が、応答の書き出し方(かぎかっこで始まる、など)でまとまっていたら、キーを作る質問を増やします(対象の場面の質問は 100 問以上を勧めます)"), "muted note")}` : ""}`;
  }

  // 鍵の作り方の 1 行。stages = [[見出し, 最小二乗の鍵の記録 {c, cv, folds, edge, …}], …](記録が無い段は出さない)
  fitHTML(kf, stages) {
    if (kf !== "ridge") return `<span class="muted">${t("キーの作り方: いまの作り方(書き込みたい位置の状態の平均から、書き込みたくない位置にも現れる成分を取り除く)")}</span>`;
    const parts = (stages || []).filter(([, x]) => x).map(([lab, x]) => `${lab ? `${esc(lab)} ` : ""}c = ${esc(String(x.c))}(${
      x.cv ? t("質問単位の {f} 分割で決めた", { f: x.folds }) : t("渡した値")})${x.edge ? ` <span class="pp-warn">${t("候補の端(もっと端の値が合うかもしれません)")}</span>` : ""}`);
    return `${t("キーの作り方: 最小二乗(セルごとの重み)")}${parts.length ? ` · ${t("正則化の強さ")} ${parts.join(" · ")}` : ""}`;
  }

  // 変えたくない場面の群ごとの漏れ(gne nw check の leaks): 群 × 区間ごとに、狙った量の 0.2 以上を書いた質問の数と書いた量の平均、
  // どれかの区間で書いた質問の数、位置 0 だけで見た数
  leaksHTML(r) {
    const lk = r.leaks;
    if (!lk || !(lk.groups || []).length) return "";
    const segs = lk.segments || [];
    const cell = (c) => (!c || !c.n ? "<td>—</td>"
      : `<td class="${c.leak ? "fx-bad" : ""}">${c.leak} / ${c.n} <span class="muted">(${fmt(c.total_mean, 2)})</span></td>`);
    const rows = [...lk.groups, ...(lk.all && lk.groups.length > 1 ? [{ ...lk.all, group: null }] : [])].map((g) => `<tr><td class="l">${g.group === null ? t("(全部)") : esc(g.group)}</td>
      <td class="${g.any ? "fx-bad" : ""}">${g.any} / ${g.n}</td>${segs.map((sg) => cell((g.segments || {})[sg])).join("")}${cell(g.pos0)}</tr>`).join("");
    return `<h4 class="note">${t("対象外の場面の群ごとの誤検出")} <span class="muted note">${t("狙った量の {v} 以上を書いた質問の数 / 質問の数。かっこ内は書いた量の平均", { v: lk.leak })}</span></h4>
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-leak-tab"><thead><tr><th class="l">${t("群")}</th><th>${t("どれかの区間")}</th>${
        segs.map((sg) => `<th>${sg === "match" ? t("一致の直前") : t("区間 {s}", { s: esc(sg) })}</th>`).join("")}<th title="${t("応答を書き始める直前の 1 位置だけで見た数")}">${t("位置 0 だけ")}</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${noteHTML(t("群ごとに見ます。対象の場面と同じ形の発言(質問でない発言など)の群だけに誤検出が出ていないか、誤検出がある区間から位置 0 に移っただけでないかを確かめます"), "muted note")}`;
  }

  // 最小二乗のキーを作った質問の、会話の形ごとの数の 1 行(形が 2 つ以上あるか、セルに会話の形を入れたときだけ)
  formsHTML(rd) {
    if (!rd || !rd.forms) return "";
    const xs = Object.entries(rd.forms).filter(([, n]) => +n > 0);
    if (xs.length < 2 && !rd.conv_cells) return "";
    return `<p class="note" id="fx-keys-forms">${t("会話の形ごとの質問の数")}: ${xs.map(([f, n]) => `${esc(formName(f))} ${n}`).join(" · ")}${
      rd.conv_cells ? ` · <span class="muted">${t("セルに会話の形も入れた")}</span>` : ""}</p>`;
  }

  // 会話の形ごとの数(gne nw check の by_form): 形が 2 つ以上ある群ごとに、形 × 区間の表。狙う場面は、キーの反応 ρ が b 以上だった質問の
  // 割合、dont_touch は、書いてしまった質問の数 / 質問の数。最後まで書き終えた応答の数も添える
  convFormsHTML(r) {
    const qs = r.question_summary || {};
    const b = qs.cap && qs.cap.length ? qs.cap[0][1] : 0.8;
    const pct = (x) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${Math.round(+x * 100)}%`);
    const colName = (sg) => (sg === "match" ? t("一致の直前") : t("区間 {s}", { s: esc(sg) }));
    const out = [];
    Object.entries(r.groups || {}).forEach(([g, d], gi) => {
      const bf = d.by_form || {}, forms = Object.keys(bf);
      if (forms.length < 2) return;
      const scene = g === "scene";
      const segs = [...new Set(forms.flatMap((f) => Object.keys(bf[f].segments || {})))];
      const rows = forms.map((f) => `<tr><th scope="row" class="l">${esc(formName(f))}</th><td>${bf[f].n ?? "—"}</td>${segs.map((sg) => {
        const c = (bf[f].segments || {})[sg];
        if (!c) return "<td>—</td>";
        return scene ? `<td title="${t("ρ が a を超えた割合 {a} · 書いた量の平均 {m}", { a: pct(c.rho_gt_a), m: fmt(c.amount_mean, 2) })}">${pct(c.rho_ge_b)}</td>`
          : `<td class="${c.leak ? "fx-bad" : ""}" title="${t("書いた量の平均 {m}", { m: fmt(c.total_mean, 2) })}">${c.leak} / ${c.n}</td>`;
      }).join("")}</tr>`).join("");
      const title = scene ? t("対象の場面(対照集合の質問)") : t("書き込みたくない場面: {g}", { g: g.replace(/^dont_touch:/, "") });
      const hid = `fx-form-h${gi}`;
      out.push(`<h4 class="note" id="${hid}">${esc(title)} <span class="muted note" data-fxfin="${esc(g)}">${t("最後まで書き終えた応答 {f} / {n}", { f: d.finished ?? "—", n: d.n ?? "—" })}</span></h4>
        <div class="evx-scroll" role="region" tabindex="0" aria-labelledby="${hid}"><table class="evx mono fx-tab fx-form-tab" data-g="${esc(g)}"><thead><tr><th class="l" scope="col">${t("会話の形")}</th>
          <th scope="col">${t("質問の数")}</th>${segs.map((sg) => `<th scope="col">${colName(sg)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`);
    });
    if (!out.length) return "";
    return `<h4 class="note">${t("会話の形ごと")} <span class="muted note">${t("対象の場面: キーの出力 ρ が {b} 以上だった質問の割合。書き込みたくない場面: 合計の大きさが {v} 以上を書いてしまった質問の数 / 質問の数", { b, v: qs.leak ?? 0.2 })}</span></h4>
      ${out.join("")}`;
  }

  renderCheck() {
    const r = this.st.check, el = this.$("#fx-check");
    if (!r) { el.innerHTML = ""; return; }
    const kf = r.key_fit || {};
    const cs = [...new Set(Object.values(kf.c || {}).filter((x) => x !== null && x !== undefined).map(String))];
    const fit = kf.fit === "ridge" ? `${t("ユニットのキーの作り方: 最小二乗(セルごとの重み)")}${cs.length ? ` · c = ${esc(cs.join("・"))}` : ""}`
      : kf.fit === "mixed" ? `<span class="pp-warn">${t("ユニットのキーの作り方: いまの作り方と最小二乗が混ざっています")}</span>`
      : `<span class="muted">${t("ユニットのキーの作り方: いまの作り方")}</span>`;
    const segs = parseSegments(((this.st.dir && this.st.dir.segments) || []).join(",") || this.$("#fx-seg").value) || [];
    const segOf = (unit) => { const m = String(unit).match(/\.seg(\d+)/); return m ? segs[+m[1]] || null : segs.length === 1 && segs[0].before ? segs[0] : null; };
    const colName = (k) => (k === "match" ? t("一致の直前") : k === "other" ? t("それ以外") : k);
    let nBad = 0, nLow = 0;
    const tables = Object.entries(r.groups || {}).map(([g, d]) => {
      const cols = d.units.length ? Object.keys(d.units[0].positions) : [];
      const trs = d.units.map((u) => {
        const sg = segOf(u.unit);
        return `<tr><td>L${u.layer}</td><td>${sg ? esc(sg.text) : "—"}</td>${cols.map((k) => {
          const a = (u.positions[k] || {}).amount || {};
          const cls = amountClass(g, sg, k, a.mean);
          if (cls === "fx-bad") nBad++; else if (cls === "fx-low") nLow++;
          return `<td class="${cls}" title="${fmt(a.min, 2)} 〜 ${fmt(a.max, 2)}">${fmt(a.mean, 2)}</td>`;
        }).join("")}<td class="l muted">${esc(u.unit)}</td></tr>`;
      }).join("");
      const title = g === "scene" ? t("対象の場面(対照集合の質問)") : t("書き込みたくない場面: {g}", { g: g.replace(/^dont_touch:/, "") });
      return `<h4 class="note">${esc(title)} <span class="muted note">${d.n} ${t("問")} · ${t("繰り返しになった応答 {n}", { n: d.loops })}</span></h4>
        <div class="evx-scroll"><table class="evx mono fx-tab fx-check-tab" data-g="${esc(g)}"><thead><tr><th>${t("層")}</th><th>${t("区切り")}</th>${cols.map((k) => `<th>${esc(colName(k))}</th>`).join("")}<th class="l">${t("ユニット")}</th></tr></thead><tbody>${trs}</tbody></table></div>`;
    }).join("");
    el.innerHTML = `<p class="note" id="fx-check-fit">${fit}</p>${this.questionsHTML(r)}${this.leaksHTML(r)}${this.convFormsHTML(r)}<h4 class="note">${t("実際に足した量(応答の位置ごとの平均。α·方向 を 1 として)")}</h4>
      <p class="note"><span class="fx-in fx-chip">${t("対象の場面の、その区切りの位置")}</span> <span class="fx-low fx-chip">${t("足りない({v} 未満)", { v: AMOUNT_LOW })}</span> <span class="fx-bad fx-chip">${t("書き込みたくない位置で大きい({v} 以上)", { v: AMOUNT_HIGH })}</span>
        <span class="muted" id="fx-check-sum">${t("足りない {a} か所 · 書き込みたくない位置で大きい {b} か所", { a: nLow, b: nBad })}</span></p>
      ${tables}
      ${r.positions ? `<p class="muted note">${t("文字の直前に書く形なので、何も当てずに作った応答を、レシピを当てて読んだ値です(書くと、一致する文字が出なくなるため)")}</p>` : ""}
      ${noteHTML(t("列は応答の位置(0 = 応答を書き始める直前)。値は平均で、マウスを当てると最小〜最大。区切りの境目で足りないときは、区切りを変えるか、当てた状態でキーを作り直します。応答の質は、レシピのタブのチャットか behave で読んで確かめてください"), "muted note")}`;
  }
}
