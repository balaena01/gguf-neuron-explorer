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
// 強さは区切りごとの欄(区切り 0,1-3,4- なら 3 つ)。
// 「重みに入れる前に試す」(fish_try のジョブ): 2. で作った方向を、ニューロンにも鍵にも入れずに、狙う場面の質問の返事の位置どおりに
// 層の出力へ直接足して返事を作る。層・区切りごとの強さを変えて、返事と数(繰り返し・空・正規表現の一致)を並べて比べる。
// 遅れて届いた結果で、後から選んだもの・後から投げたジョブの結果を上書きしない(欄ごとに最後に投げたジョブだけを受け取る。
// 一覧の読み込みとユニットの組み立ては通し番号で見分ける。手で選び直したニューロンは、表を取り直しても残す)。
import { getJSON, postJSON } from "./api.js";
import { LANG, t } from "./i18n.js";
import { checkKeySpec, parseLayers } from "./keys.js";
import { esc } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const NAME_RE = /^[\w.-]+$/;
const DRAFT = "gne.fish.draft";                                   // 入力欄の下書き(このブラウザだけ)
const SRC = "gne.fish.src";
const PER = { one: 1, two: 1, cap: 2 };                           // 区切り 1 つに要るニューロンの数(gne/fish.py の PER_SEGMENT)
const FISH_KEYS = ["layers", "segments", "form", "alpha", "same_reply", "stack", "pseudo", "max_tokens", "response_tokens", "iv", "rekey"];
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
    this.jobs = { all: null, dir: null, neu: null, keys: null, verify: null, check: null, behave: null, try: null };   // 欄ごとに、最後に投げたジョブの id
    this.handled = new Set();
    this.seq = 0;                                   // 一覧の読み込みの通し番号
    this.unitSeq = 0;                               // ユニットの組み立ての通し番号
    this.loaded = false;
    // 作ったもの: name(方向・鍵の名前の元)、path(対照集合のパス)、dir・usage・keys・verify・check(各段の結果)、
    // picked({層: [番号…]})、touched(手で選び直した層)、keyBase(鍵の名前の頭)、units(書き方)、rows(レシピの行)
    this.st = { name: null, path: null, dir: null, usage: null, keys: null, verify: null, check: null, picked: {}, touched: new Set(),
                keyBase: null, units: [], rows: [], pipe: null, behave: null, tries: [], tryShown: -1 };
    root.innerHTML = `
      <p class="muted note pp-help">${t("FiSH(Formula-instantiated Scene-gated Hidden units): すでにある MLP のニューロンを、学習なしで、式から計算した重みで書き換えて、決まった場面の返事でだけ方向を書くようにする。ゲートの行に鍵、入力の行に鍵、出力の列に方向を入れる。ふだんの働きが小さいニューロンを選んで書き換える(元の働きは無くなる)。実行時は LoRA で当て、レシピのタブでチャット・保存・LoRA の書き出しができる。層と強さは使う人が決める。")}</p>
      <section class="kx-sec"><h4>${t("1. 対照集合")}</h4>
        <div class="pp-in"><span class="muted">${t("対照集合")}</span>
          <span class="seg" id="fx-src"><button data-src="file" title="${t("configs/contrast/ か <root>/contrasts/ の YAML を選ぶ")}">${t("ファイルを選ぶ")}</button><button data-src="json" title="${t("ひな形(configs/fish/)から始めて、入力欄の JSON を手で直す")}">${t("JSON を書く(ひな形)")}</button></span>
          <label class="muted" title="${t("FiSH 用のひな形(configs/fish/)。選ぶと入力欄に入る")}">${t("ひな形")}
            <select id="fx-tpl-sel"><option value="">${t("(ひな形を選ぶ)")}</option></select></label>
          <button class="small" id="fx-tpl" title="${t("選んだひな形を入力欄に入れる")}">${t("入れる")}</button></div>
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
        <div class="pp-set">
          <label class="muted" title="${t("方向と鍵の名前の元(方向 <名前>、鍵 <名前>_p0 など)。空なら対照集合の名前")}">${t("名前")} <input id="fx-name" size="16" spellcheck="false"></label>
          <label class="muted" title="${t("書き換える層(カンマ区切りか、続いた層は 11-23 のように範囲で)。層の番号はモデルで違うので、使う人が決める。Qwen3.5-4B の 1 例では、離れた 3 層より、続いた 13 層(11-23)のほうが良かった。層を増やすと、要るニューロンと作成時間が増える")}">${t("層")} <input id="fx-layers" size="12" placeholder="11-23" spellcheck="false"></label>
          <label class="muted" title="${t("書く位置。返事の位置の区切り(位置 0 = 返事を書き始める直前、1-3 = 返事の最初の 3 トークン、4- = それ以降。区切りごとに方向・鍵・ニューロンを分ける)か、before:<正規表現>(一致する文字の直前の位置だけ。例 before:(です|ます)。)")}">${t("位置の区切り")} <input id="fx-seg" size="14" value="0,1-3,4-" spellcheck="false"></label>
          <label class="muted" title="${t("鍵 1 つ = 鍵が反応する位置で書く。鍵 2 つ = 場面の鍵 かつ 位置の鍵。上限つき = ニューロン 2 個を組にして、足す量が 1 倍あたりで頭打ちになる(実験でいちばん良かった形)")}">${t("形")}
            <select id="fx-form"><option value="cap">${t("上限つきの 2 個組")}</option><option value="two">${t("鍵 2 つ(場面 かつ 位置)")}</option><option value="one">${t("鍵 1 つ")}</option></select></label>
          <span class="muted" title="${t("強さ α(区切りごと)。1 で、その区切りの方向をそのままの長さで書く。使う人が決める(Qwen3.5-4B の 1 例では、位置 0 と 1-3 に 1、4 以降に 0.8 がよかった)")}">${t("強さ(区切りごと)")} <span id="fx-alphas"></span></span>
          <label class="muted" title="${t("先に当てておく介入: 保存したレシピの名前か、ユニットを空白区切りで。CHiPS のレシピを当てた状態で作るときに入れる。空なら無介入")}">${t("先に当てておくレシピ(iv)")} <input id="fx-iv" size="18" spellcheck="false"></label>
          <label class="muted" title="${t("鍵づくりと確かめで返事を生成する長さ")}">max_tokens <input id="fx-mt" type="number" min="1" max="2048" value="96" style="width:5em"></label>
        </div>
        <div class="pp-go">
          <button id="fx-all" title="${t("方向 → ニューロンの表 → 鍵 → レシピ → 当てた状態で鍵を作り直す → 確かめ、を 1 回のジョブで回す。ニューロンは既定の候補(表の上から)を使い、あとで選び直せる")}">${t("まとめて作る")}</button>
          <label class="muted" title="${t("レシピを当てた状態で鍵を作り直す段を入れる(例ではこのほうがよかった)")}"><input type="checkbox" id="fx-rekey-on" checked> ${t("当てた状態で鍵を作り直す")}</label>
          <span class="muted note" id="fx-status-all"></span></div>
        <div id="fx-prog-all" class="note mono"></div>
        <div id="fx-pipe"></div>
      </section>
      <section class="kx-sec"><h4>${t("2. 方向を作る")}</h4>
        <div class="pp-set">
          <label class="muted" title="${t("今まで = それぞれの条件で作った返事を読む。同じ返事 = 指示を付けて作った返事を、指示あり・なしの両方で読む(読んでいる言葉の違いが方向に入らない)")}">${t("作り方")}
            <select id="fx-method"><option value="same">${t("同じ返事を両方の条件で読む")}</option><option value="own">${t("今までの作り方")}</option></select></label>
          <label class="muted" title="${t("下の層に書いた後に残る差を、上の層の方向にする(上の欄の層に重ねて書く前提)")}"><input type="checkbox" id="fx-stack" checked> ${t("下の層に書いた後の残りを使う")}</label>
          <label class="muted" title="${t("今までの作り方の方向も作って、向きの近さ(cos)を表に出す")}"><input type="checkbox" id="fx-compare" checked> ${t("今までの作り方と比べる")}</label>
          <label class="muted" title="${t("方向を作るとき、返事を生成する長さ")}">${t("方向の返事の長さ")} <input id="fx-rt" type="number" min="1" max="2048" value="96" style="width:5em"></label>
        </div>
        <div class="pp-go"><button id="fx-dir-go">${t("方向を作る")}</button> <span class="muted note" id="fx-status-dir"></span></div>
        <div id="fx-prog-dir" class="note mono"></div>
        <div id="fx-dir"></div>
      </section>
      <section class="kx-sec" id="fx-try-sec"><h4>${t("重みに入れる前に試す")}</h4>
        <p class="muted note">${t("2. で作った方向を、ニューロンにも鍵にも入れずに、層の出力へ直接足して返事を作る。足すのは狙う場面の質問の返事だけ(場面は質問の群で決める。重みに入れたときの鍵の代わり)。層と区切りごとの強さを変えて何回か回し、返事と数を比べてから 3. 以降に進む")}</p>
        <div class="pp-set">
          <label class="muted" title="${t("足す層(カンマ区切りか 11-23)。空なら上の欄の層。足せるのは層 1 から、最終層の 1 つ手前まで")}">${t("足す層")} <input id="fx-try-layers" size="12" spellcheck="false"></label>
          <span class="muted" title="${t("区切りごとの強さ。1 で、その区切りの方向をそのままの長さで足す")}">${t("強さ(区切りごと)")} <span id="fx-try-alphas"></span></span>
          <label class="muted" title="${t("狙う場面・書き込みたくない場面それぞれで、返事を作る質問の数(対照集合の先頭から)")}">${t("質問の数")} <input id="fx-try-n" type="number" min="1" max="200" value="12" style="width:4.5em"></label>
          <label class="muted" title="${t("返事を生成する長さ")}">${t("返事の長さ")} <input id="fx-try-mt" type="number" min="1" max="2048" value="120" style="width:5em"></label>
          <label class="muted" title="${t("0 なら毎回同じ返事(greedy)。0 より大きいとサンプリング")}">${t("温度")} <input id="fx-try-temp" type="number" min="0" max="2" step="0.1" value="0" style="width:4.5em"></label>
          <label class="muted" title="${t("サンプリングの乱数の種")}">${t("種")} <input id="fx-try-seed" type="number" min="0" value="0" style="width:4.5em"></label>
          <label class="muted" title="${t("一致した返事を数える正規表現。空なら対照集合の validate.match。繰り返しの中の語も数えるので、返事は読んで確かめる")}">${t("数える正規表現")} <input id="fx-try-match" size="18" spellcheck="false"></label>
        </div>
        <details class="kx-adv"><summary class="muted note">${t("狙う場面の質問を自分で入れる")}</summary>
          <textarea id="fx-try-q" class="mono" rows="4" spellcheck="false" placeholder="${t("1 行に 1 問。空なら対照集合の質問の先頭から(方向づくりに使った質問なので、甘い)")}"></textarea></details>
        <div class="pp-go"><button id="fx-try-go" title="${t("方向を層の出力に直接足して、狙う場面の質問に答えさせる。同じ質問に何も足さない返事と、書き込みたくない場面の質問への返事(何も足さない)も作る")}">${t("直接足して返事を作る")}</button>
          <button class="small" id="fx-try-use" title="${t("この欄の層と強さを、上の欄(まとめて作る・レシピを組む)に入れる")}">${t("この層と強さを上の欄に入れる")}</button>
          <span class="muted note" id="fx-status-try"></span></div>
        <div id="fx-prog-try" class="note mono"></div>
        <div id="fx-try"></div>
      </section>
      <section class="kx-sec"><h4>${t("3. 書き換えるニューロンを選ぶ")}</h4>
        <div class="pp-set">
          <label class="muted" title="${t("層ごとに表に出す数")}">${t("表の行数")} <input id="fx-top" type="number" min="1" max="200" value="20" style="width:5em"></label>
          <span class="muted note" id="fx-need"></span></div>
        <div class="pp-go"><button id="fx-neu-go" title="${t("ふつうの質問への返事と文章を読ませて、各ニューロンのふだんの働きを測る")}">${t("ニューロンの表を出す")}</button> <span class="muted note" id="fx-status-neu"></span></div>
        <div id="fx-prog-neu" class="note mono"></div>
        <div id="fx-neu"></div>
      </section>
      <section class="kx-sec"><h4>${t("4. 鍵を作る")}</h4>
        <div class="pp-set">
          <label class="muted" title="${t("比較用の鍵の本数。書き込みたい位置・書き込みたくない位置のラベルをでたらめに入れ替えて、同じ手順で作る鍵")}">${t("比較用の鍵")} <input id="fx-pseudo" type="number" min="0" max="64" value="4" style="width:4em"></label>
        </div>
        <div class="pp-go"><button id="fx-keys-go" title="${t("区切りごとの位置の鍵と、形に要る鍵(鍵 2 つの形は場面の鍵、上限つきの形はほぼ 1 を返す鍵)をまとめて作る")}">${t("鍵をまとめて作る")}</button> <span class="muted note" id="fx-status-keys"></span></div>
        <div id="fx-prog-keys" class="note mono"></div>
        <div id="fx-keys"></div>
      </section>
      <section class="kx-sec"><h4>${t("5. レシピを組む")}</h4>
        <div id="fx-units"><p class="muted note">${t("方向・ニューロン・鍵がそろうと、ここにユニットが出ます")}</p></div>
        <div class="pp-go">
          <button id="fx-add" disabled title="${t("下のユニットをレシピに足す(同じ層・同じニューロンの行は置き換える)。レシピのタブでチャット・保存・LoRA の書き出しができる")}">${t("レシピに足す")}</button>
          <button id="fx-rekey" disabled title="${t("このユニットを当てた状態で返事を作り直し、鍵を作り直す(名前は <名前>_r_…)。ニューロンが書いた後の状態で鍵が反応するようになる")}">${t("このレシピを当てた状態で鍵を作り直す")}</button>
          <span class="muted note" id="fx-status-units"></span></div>
      </section>
      <section class="kx-sec"><h4>${t("6. 確かめる")}</h4>
        <div class="pp-go">
          <button id="fx-verify-go" disabled title="${t("LoRA を当てて短い文を読ませ、書き換えたニューロンが式どおりに動くか、ほかのニューロンが変わっていないかを確かめる(数秒)")}">${t("式どおりか確かめる")}</button>
          <span class="muted note" id="fx-status-verify"></span>
          <button id="fx-check-go" disabled title="${t("レシピを当てて返事を作り、狙う場面と書き込みたくない場面で、返事の位置ごとに実際に足した量を測る")}">${t("足した量を測る")}</button>
          <span class="muted note" id="fx-status-check"></span></div>
        <div id="fx-prog-verify" class="note mono"></div>
        <div id="fx-verify"></div>
        <div id="fx-prog-check" class="note mono"></div>
        <div id="fx-check"></div>
      </section>
      <section class="kx-sec"><h4>${t("7. 返事を読む")}</h4>
        <div class="pp-set">
          <label class="muted" title="${t("狙う場面・書き込みたくない場面それぞれで、返事を作る質問の数(対照集合の先頭から)")}">${t("質問の数")} <input id="fx-bn" type="number" min="1" max="200" value="12" style="width:4.5em"></label>
          <label class="muted" title="${t("返事を生成する長さ")}">${t("返事の長さ")} <input id="fx-bmt" type="number" min="1" max="2048" value="120" style="width:5em"></label>
          <label class="muted" title="${t("一致した返事を数える正規表現。空なら対照集合の validate.match。繰り返しの中の語も数えるので、返事は読んで確かめる")}">${t("数える正規表現")} <input id="fx-bmatch" size="22" spellcheck="false"></label>
        </div>
        <details class="kx-adv"><summary class="muted note">${t("狙う場面の質問を自分で入れる")}</summary>
          <textarea id="fx-bq" class="mono" rows="4" spellcheck="false" placeholder="${t("1 行に 1 問。空なら対照集合の質問の先頭から(鍵づくりに使った質問なので、甘い)")}"></textarea></details>
        <div class="pp-go"><button id="fx-behave-go" disabled title="${t("ユニットを当てて(system の指示は付けずに)、狙う場面の質問と、書き込みたくない場面の質問に答えさせる。結果は behave の結果としても保存する")}">${t("レシピを当てて返事を作る")}</button>
          <span class="muted note" id="fx-status-behave"></span></div>
        <div id="fx-prog-behave" class="note mono"></div>
        <div id="fx-behave"></div>
      </section>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#fx-contrast").addEventListener("change", () => this.contrastChanged());
    this.$("#fx-cpath").addEventListener("input", () => this.contrastChanged());
    for (const b of root.querySelectorAll("#fx-src button")) b.addEventListener("click", () => this.setSource(b.dataset.src));
    const ta = this.$("#fx-text");
    try { ta.value = localStorage.getItem(DRAFT) || ""; } catch { /* 下書きが無くても使える */ }
    ta.addEventListener("input", () => this.checkJSON());
    const sel = this.$("#fx-tpl-sel");
    this.$("#fx-tpl").addEventListener("click", () => this.insertTemplate(sel.value, false));
    sel.addEventListener("change", () => this.insertTemplate(sel.value, true));
    for (const id of ["#fx-layers", "#fx-seg", "#fx-form"]) this.$(id).addEventListener("input", () => this.settingsChanged());
    this.$("#fx-form").addEventListener("change", () => this.settingsChanged());
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
    });
    let src = "json";
    try { src = localStorage.getItem(SRC) === "file" ? "file" : "json"; } catch { /* 既定は JSON */ }
    this.setSource(src, false);
    this.renderAlphas("fx-alphas", [1, 1, 0.8]);
    this.renderAlphas("fx-try-alphas", [1, 1, 0.8]);
    this.showNeed();
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
    el.innerHTML = segs.map((x, i) => `<label class="fx-al"><span class="mono">${esc(x.before ? "" : x.text)}</span> <input id="${id(i)}" ${main ? "data-fxa" : "data-fxta"}="${i}" type="number" step="0.05" value="${esc(mem[Math.min(i, mem.length - 1)])}" style="width:4.5em" aria-label="${t("強さ")} ${esc(x.text)}"></label>`).join(" ");
  }

  // 区切りごとの強さ(数の並び)。読めない欄があれば null
  alphas(box = "fx-alphas") {
    const xs = [...this.$(`#${box}`).querySelectorAll("input")].map((x) => (x.value.trim() === "" ? NaN : +x.value));
    return xs.length && xs.every((a) => Number.isFinite(a)) ? xs : null;
  }

  // direction ビューで「FiSH」を開いたとき: 対照集合の候補とひな形を取り直す(遅れて届いた古い読み込みは捨てる)
  async refresh() {
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
      : ` · <span class="pp-warn">${t("dont_touch(書き込みたくない場面の質問)がありません。ほかの場面でも鍵が反応しやすくなります")}</span>`}`;
  }

  insertTemplate(file, ask) {
    const x = this.templates.find((y) => y.file === file);
    if (!x) return;
    const ta = this.$("#fx-text"), cur = ta.value.trim();
    if (ask && cur && cur !== this.tplText && !window.confirm(t("入力欄の JSON をひな形で置き換えます。よいですか?"))) { this.$("#fx-tpl-sel").value = ""; return; }
    ta.value = JSON.stringify(x.spec, null, 2);
    this.tplText = ta.value.trim();
    const about = this.$("#fx-tpl-about");
    about.textContent = (x[LANG] && x[LANG].text) || (x.ja && x.ja.text) || "";
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
    if (!d && !errors.length) el.innerHTML = `<span class="muted">${t("JSON を貼り付けるか、「ひな形」で例を選んでください")}</span>`;
    else {
      const s = c.summary;
      const dt = s ? s.dt : [];
      el.innerHTML = `${errors.map((e) => `<div class="err">${esc(e)}</div>`).join("")}${
        s ? `<div class="muted">${esc(String(d.name || ""))} · ${t("狙う場面の質問 {n} 問", { n: s.n })} · ${dt.length ? t("書き込みたくない場面: {d}", { d: dt.map((g) => `${g.name}(${g.n ?? "—"})`).join(", ") })
          : `<span class="pp-warn">${t("dont_touch(書き込みたくない場面の質問)がありません。ほかの場面でも鍵が反応しやすくなります")}</span>`}</div>` : ""}`;
    }
    if (d && NAME_RE.test(String(d.name ?? ""))) this.$("#fx-name").placeholder = String(d.name);
    const sig = fish ? JSON.stringify(fish) : null;
    if (fish && sig !== this.lastSet) {
      this.lastSet = sig;
      const put = (sel, v) => { if (v !== undefined && v !== null) this.$(sel).value = String(v); };
      put("#fx-seg", fish.segments); put("#fx-form", fish.form); put("#fx-pseudo", fish.pseudo);
      if (fish.alpha !== undefined && fish.alpha !== null) { this.renderAlphas("fx-alphas", fish.alpha); this.renderAlphas("fx-try-alphas", fish.alpha); }
      put("#fx-mt", fish.max_tokens); put("#fx-rt", fish.response_tokens);
      if (typeof fish.iv === "string") put("#fx-iv", fish.iv); else if (Array.isArray(fish.iv)) put("#fx-iv", fish.iv.join(" "));
      // 書いていない設定は既定に戻す(前に選んだひな形の値を残さない)
      this.$("#fx-method").value = fish.same_reply === false ? "own" : "same";
      this.$("#fx-stack").checked = fish.stack !== false;
      this.$("#fx-rekey-on").checked = fish.rekey !== false;
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
    if (!segs) { this.status(which, t("位置の区切りは 0,1-3,4- のように書きます(位置 0 = 返事を書き始める直前)。文字の直前に書くなら before:<正規表現>"), true); return null; }
    const name = v("#fx-name");
    if (name && !NAME_RE.test(name)) { this.status(which, t("名前は英数字・_・-・. だけにしてください"), true); return null; }
    const mt = +v("#fx-mt");
    const p = { layers: layers.join(","), segments: segs.map((s) => s.text).join(","), form: v("#fx-form"), max_tokens: Number.isInteger(mt) && mt > 0 ? mt : 96 };
    const iv = v("#fx-iv");
    if (iv) p.iv = iv.includes(":") ? iv.split(/[\s,]+/).filter(Boolean) : iv;
    return { p, layers, segs, name: name || null };
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
      + (layers && layers.length >= 6 ? ` · ${t("層が多いと作成に時間がかかります(Qwen3.5-4B の 13 層で 20〜30 分ほど)。見積もりは「まとめて作る」を押すと出ます")}` : "");
  }

  // 層・区切り・形を変えた: 必要な個数と候補を出し直し、ユニットを組み直す
  settingsChanged() {
    this.renderAlphas("fx-alphas");
    this.renderAlphas("fx-try-alphas");
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
      const est = Number.isFinite(+j.estimate_s) && j.estimate_s !== null ? ` ${t("(見積もり 約 {s} 秒)", { s: Math.round(+j.estimate_s) })}` : "";
      this.status(which, running + est);
      this.follow(which, j.id);
    } catch (e) { this.status(which, `${t("エラー")}: ${e.message}`, true); }
  }

  follow(which, id) {
    const el = this.$(`#fx-prog-${which}`);
    const show = (j) => {
      if (this.jobs[which] !== id) return;
      if (!["queued", "running"].includes(j.status)) { el.innerHTML = ""; return; }
      const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
      const stop = `<button class="small" data-fxstop="${esc(id)}" title="${t("このジョブを止める")}">${t("止める")}</button>`;
      el.innerHTML = j.status === "queued" ? `<span class="muted">${t("待機中")}</span> ${stop}`
        : `<span class="jb-bar"><span style="width:${pct}%"></span></span> ${pct}% <span class="muted">${esc(j.stage || "")}</span> ${stop}`;
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
    if (j.status !== "done") {
      const e = j.error || {};
      this.status(which, j.status === "error" ? `${t("失敗")}: ${e.message || ""}${e.hint ? `(${e.hint})` : ""}` : t("止めました"), j.status === "error");
      return;
    }
    const r = j.result || {};
    this.status(which, t("できました({s} 秒)", { s: fmt(r.seconds, 1) }));
    const st = this.st;
    if (which === "dir") {
      st.dir = r; st.name = r.name; st.path = r.contrast_path || st.path;
      this.renderDirection();
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
    } else if (which === "all") {
      st.pipe = r; st.dir = r.direction; st.name = r.name; st.path = r.contrast_path; st.usage = r.usage;
      st.keys = r.rekeyed || r.keys; st.keyBase = r.key_base; st.verify = r.verify; st.check = r.check;
      st.picked = {}; st.touched = new Set();
      for (const [L, js] of Object.entries(r.neurons || {})) st.picked[+L] = js.map(Number);
      if (r.alphas) { this.renderAlphas("fx-alphas", r.alphas); }
      this.renderPipe();
      this.renderDirection();
      this.renderNeurons();
      this.renderKeys();
      this.renderVerify();
      this.renderCheck();
    }
    await this.compose();
  }

  async runAll() {
    const s = this.source("all"), c = this.common("all");
    if (!s || !c) return;
    const a = this.alpha();
    if (a === null) { this.status("all", t("強さを、区切りごとに全部入れてください(例 1 / 1 / 0.8)"), true); return; }
    const rt = +this.$("#fx-rt").value, ps = +this.$("#fx-pseudo").value, top = +this.$("#fx-top").value;
    const p = { ...s, ...c.p, alpha: a, same_reply: this.$("#fx-method").value === "same", stack: this.$("#fx-stack").checked,
                compare: this.$("#fx-compare").checked, rekey: this.$("#fx-rekey-on").checked,
                pseudo: Number.isInteger(ps) && ps >= 0 ? ps : 4, response_tokens: Number.isInteger(rt) && rt > 0 ? rt : 96,
                top: Number.isInteger(top) && top > 0 ? top : 20 };
    if (c.name) p.name = c.name;
    const picked = this.pickedFor(c.layers);                 // 手で選び直したニューロンがあれば、それを使う
    if (picked && [...this.st.touched].some((L) => c.layers.includes(L))) p.neurons = picked;
    await this.launch("all", "fish_pipeline", p, t("方向 → ニューロンの表 → 鍵 → レシピ → 確かめ を回しています(モデルを流します。GPU を待つことがあります)…"));
  }

  async runDirection() {
    const s = this.source("dir"), c = this.common("dir");
    if (!s || !c) return;
    const rt = +this.$("#fx-rt").value;
    const p = { ...s, layers: c.p.layers, segments: c.p.segments, same_reply: this.$("#fx-method").value === "same", stack: this.$("#fx-stack").checked,
                compare: this.$("#fx-compare").checked, response_tokens: Number.isInteger(rt) && rt > 0 ? rt : 96 };
    if (c.name) p.name = c.name;
    await this.launch("dir", "fish_direction", p, t("方向を作っています(返事を生成します)…"));
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
    const base = c.name || this.st.name || (this.src === "json" && this.jspec.spec ? String(this.jspec.spec.name) : null);
    const p = { ...s, ...c.p, pseudo: Number.isInteger(ps) && ps >= 0 ? ps : 4 };
    if (rekey) {
      if (!this.st.units.length) { this.status("units", t("先にユニットを組んでください(方向・ニューロン・鍵)"), true); return; }
      p.with_units = [...this.st.units];
      p.base = `${(base || this.st.keyBase || "fish").replace(/_r$/, "")}_r`;
      this.status("units", t("鍵を作り直しています(4. の欄に進み具合が出ます)…"));
    } else if (base) p.base = base;
    await this.launch(which, "fish_keys", p, rekey ? t("レシピを当てた状態で、鍵を作り直しています…") : t("鍵を作っています(返事を生成します)…"));
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
    const s = this.source("check"), c = this.common("check");
    if (!s || !c) return;
    const p = { ...s, units: [...this.st.units], max_tokens: c.p.max_tokens, segments: c.p.segments };
    if (c.p.iv) p.iv = c.p.iv;
    await this.launch("check", "fish_check", p, t("レシピを当てて返事を作り、足した量を測っています…"));
  }

  async runBehave() {
    if (!this.st.units.length) return;
    const s = this.source("behave"), c = this.common("behave");
    if (!s || !c) return;
    const n = +this.$("#fx-bn").value, mt = +this.$("#fx-bmt").value;
    const p = { ...s, units: [...this.st.units], n: Number.isInteger(n) && n > 0 ? n : 12, max_tokens: Number.isInteger(mt) && mt > 0 ? mt : 120 };
    const m = this.$("#fx-bmatch").value.trim(), q = this.$("#fx-bq").value.trim();
    if (m) p.match = m;
    if (q) p.prompts = q;
    if (c.p.iv) p.iv = c.p.iv;
    if (c.name || this.st.name) p.name = c.name || this.st.name;
    await this.launch("behave", "fish_behave", p, t("レシピを当てて返事を作っています…"));
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
    const a = this.alphas("fx-try-alphas");
    const nSeg = dir ? (dir.segments || []).length || 1 : (a || []).length;
    if (a === null || a.length !== nSeg) { this.status(which, t("強さを、方向の区切り({s})ごとに入れてください", { s: dir ? (dir.segments || []).join(",") : v("#fx-seg") }), true); return; }
    const n = +v("#fx-try-n"), mt = +v("#fx-try-mt"), temp = +v("#fx-try-temp"), seed = +v("#fx-try-seed");
    const p = { ...s, direction: dname, layers: layers.join(","), alpha: a, n: Number.isInteger(n) && n > 0 ? n : 12,
                max_tokens: Number.isInteger(mt) && mt > 0 ? mt : 120, temperature: Number.isFinite(temp) && temp > 0 ? temp : 0,
                seed: Number.isInteger(seed) && seed >= 0 ? seed : 0 };
    const m = v("#fx-try-match"), q = v("#fx-try-q"), iv = v("#fx-iv");
    if (m) p.match = m;
    if (q) p.prompts = q;
    if (iv) p.iv = iv.includes(":") ? iv.split(/[\s,]+/).filter(Boolean) : iv;
    await this.launch(which, "fish_try", p, t("方向を層の出力に直接足して、返事を作っています…"));
  }

  // 試した層と強さを、上の欄に入れる
  useTry() {
    const a = this.alphas("fx-try-alphas"), L = this.$("#fx-try-layers").value.trim();
    if (L) this.$("#fx-layers").value = L;
    if (a) this.renderAlphas("fx-alphas", a);
    this.settingsChanged();
    this.status("try", t("上の欄に入れました。層を変えたら、方向を作り直してください(重ねる層が変わるため)"));
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
    const missing = [];
    if (!st.dir) missing.push(t("方向"));
    if (!st.usage && !Object.keys(st.picked).length) missing.push(t("ニューロンの表"));
    if (!st.keys) missing.push(t("鍵"));
    if (missing.length || !layers || !segs || a === null) {
      done([], [], `<p class="muted note">${missing.length ? t("まだ無いもの: {m}", { m: missing.join("・") }) : t("層・位置の区切り・強さを入れてください")}</p>`);
      return;
    }
    const picked = this.pickedFor(layers);
    const mism = [];
    if (!picked) mism.push(t("ニューロンは層ごとに {n} 個選んでください(3. の表のチェック)", { n: this.needed() }));
    if (st.dir && (st.dir.segments || []).join(",") !== segs.map((s) => s.text).join(",")) mism.push(t("方向の区切り({a})が、いまの欄({b})と違います。方向を作り直してください", { a: (st.dir.segments || []).join(","), b: segs.map((s) => s.text).join(",") }));
    if (st.keys && ((st.keys.segments || []).join(",") !== segs.map((s) => s.text).join(",") || st.keys.form !== form)) mism.push(t("鍵の区切り・形が、いまの欄と違います。鍵を作り直してください"));
    for (const [what, r] of [[t("方向"), st.dir], [t("鍵"), st.keys]]) {
      const have = new Set((r && r.layers) || []);
      const lack = layers.filter((L) => !have.has(L));
      if (r && lack.length) mism.push(t("{w} に層 {l} がありません。作り直してください", { w: what, l: lack.join(",") }));
    }
    if (mism.length) { done([], [], `<ul class="evx-notes">${mism.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>`); return; }
    let x;
    try {
      x = await postJSON("/api/fish/units", { layers: layers.join(","), segments: segs.map((s) => s.text).join(","), neurons: picked, form,
                                              direction: st.dir.name, base: st.keyBase, alpha: a });
    } catch (e) {
      if (seq === this.unitSeq) done([], [], `<p class="err note">${esc(e.message)}</p>`);
      return;
    }
    if (seq !== this.unitSeq) return;                          // 後から組み直したものがある
    done(x.specs, x.units, `<p class="note">${t("ユニット {n} 個(層 {l} × 区切り {s})。鍵 {k}", { n: x.specs.length, l: layers.length, s: segs.length, k: esc(st.keyBase) })}${
      st.keys && st.keys.with_units ? ` <span class="muted">(${t("レシピを当てた状態で作り直した鍵")})</span>` : ""}</p>
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-unit-tab"><thead><tr><th>${t("層")}</th><th>${t("区切り")}</th><th>${t("ニューロン")}</th><th>${t("方向")}</th><th>${t("鍵")}</th><th class="l">${t("ユニット")}</th></tr></thead><tbody>${
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
    const lab = { direction: t("方向"), neurons: t("ニューロンの表"), keys: t("鍵"), rekey: t("鍵の作り直し"), verify: t("式どおりか"), check: t("足した量") };
    el.innerHTML = `<p class="note">${Object.keys(lab).filter((k) => ss[k] !== undefined).map((k) => `${lab[k]} ${fmt(ss[k], 1)} ${t("秒")}`).join(" → ")}</p>
      <p class="muted note">${r.neurons_default ? t("ニューロンは既定の候補(表の上から)を使いました。3. の表のチェックで選び直せます(選び直したら、鍵の作り直しと確かめをやり直します)") : t("ニューロンは、手で選んだものを使いました")}: ${
      Object.entries(r.neurons || {}).map(([L, js]) => `L${L} ${js.map((j) => `N${j}`).join(" ")}`).join(" / ")}</p>
      ${(r.notes || []).length ? `<p class="muted note">${r.notes.map(esc).join("<br>")}</p>` : ""}
      <p class="muted note">${t("下の「5. レシピを組む」の「レシピに足す」で、レシピのタブに入ります")}</p>`;
  }

  renderDirection() {
    const r = this.st.dir, el = this.$("#fx-dir");
    if (!r) { el.innerHTML = ""; return; }
    const rows = (r.rows || []).map((x) => `<tr><td>L${x.layer}</td><td>${esc(x.positions)}</td><td>${esc(x.direction)}</td><td>${fmt(x.norm, 2)}</td><td>${fmt(x.norm_old, 2)}</td>
      <td title="${t("1 に近いほど、今までの作り方と同じ向き")}">${fmt(x.cos_old, 2)}</td></tr>`).join("");
    el.innerHTML = `<p class="note">${t("方向")} <b class="mono">${esc(r.name)}</b> · ${r.same_reply ? t("同じ返事を両方の条件で読む") : t("今までの作り方")}${r.stack ? ` · ${t("下の層に書いた後の残りを使う")}` : ""} · ${r.n_prompts ?? "—"} ${t("問")}</p>
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-dir-tab"><thead><tr><th>${t("層")}</th><th>${t("区切り")}</th><th>${t("方向の名前")}</th><th title="${t("その区切りの方向の長さ(α = 1 で書く量)")}">${t("長さ")}</th>
        <th>${t("今までの作り方の長さ")}</th><th>${t("今までの作り方との cos")}</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${r.compare ? `<p class="muted note">${t("今までの作り方(それぞれの条件で作った返事を読む・下の層を考えない)の方向は {n} に保存しました", { n: esc(r.compare) })}</p>` : ""}`;
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
    }).join("") + `<p class="muted note">${t("働きの小さい順(最大の順位 + 平均の順位)。書き換えると、そのニューロンの元の働きは無くなります。区切り 0 から順に、選んだものを上から使います")}</p>`;
  }

  renderKeys() {
    const r = this.st.keys, el = this.$("#fx-keys");
    if (!r) { el.innerHTML = ""; return; }
    const kind = { position: t("位置の鍵"), scene: t("場面の鍵"), one: t("ほぼ 1 を返す鍵") };
    const rows = (r.rows || []).map((x) => `<tr><td class="l">${esc(x.key)}</td><td class="l">${kind[x.kind] || esc(x.kind)}</td><td>${x.positions === null || x.positions === undefined ? "—" : esc(x.positions)}</td><td>L${x.layer}</td>
      <td title="${t("|k| / |μ|: 書き込みたい位置の平均のうち、書き込みたくない位置の部分空間に直交する割合")}">${fmt(x.kn, 3)}</td>
      <td title="${t("鍵の反応の平均の差(書き込みたい位置 − 書き込みたくない位置)。作成に使った例での値")}">${fmt(x.selectivity, 2)}</td>
      <td title="${t("比較用の鍵(ラベルをでたらめに入れ替えた鍵)での同じ値の 95% 点")}">${fmt(x.pseudo_p95, 2)}</td><td>${mark(x.beats)}</td></tr>`).join("");
    el.innerHTML = `<p class="note">${t("鍵の名前の頭")} <b class="mono">${esc(r.base)}</b>${r.with_units ? ` <span class="muted">(${t("レシピを当てた状態で作り直した鍵")})</span>` : ""} · subset ${esc(r.subset || "—")} · ${t("比較用の鍵 {n} 本", { n: r.pseudo ?? "—" })} · ${fmt(r.seconds, 1)} ${t("秒")}</p>
      <div class="evx-scroll"><table class="evx mono fx-tab" id="fx-keys-tab"><thead><tr><th class="l">${t("鍵")}</th><th class="l">${t("種類")}</th><th>${t("区切り")}</th><th>${t("層")}</th><th>kn</th><th>${t("鍵の反応の差")}</th><th>${t("比較用の鍵の 95% 点")}</th>
        <th title="${t("鍵の反応の差が、比較用の鍵の 95% 点を超えているか")}">${t("比較用の鍵を超えるか")}</th></tr></thead><tbody>${rows}</tbody></table></div>
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
      <p class="muted note">${t("合格の目安: ゲート・入力・出力の誤差 5% 未満、活性の誤差 0.1% 未満、ほかのニューロンの変化なし。ゲート・入力には、量子化のままの掛け算とのずれが 1〜2% 残る")}</p>`;
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
        <div class="evx-scroll"><table class="evx fx-tab fx-behave-tab" data-b="${tag}"><thead><tr><th class="l">${t("質問")}</th><th class="l">${t("返事")}</th><th></th></tr></thead><tbody>${
        d.replies.map((x) => `<tr><td class="l fx-q">${esc(x.prompt)}</td><td class="l fx-a">${esc(x.text)}</td><td>${x.loop ? `<span class="kx-ng">${t("繰り返し")}</span>` : ""}</td></tr>`).join("")}</tbody></table></div>`;
    };
    el.innerHTML = `<p class="muted note">${t("system の指示は付けず、ユニットだけを当てた返事です。正規表現の一致は目安で、返事を読んで確かめてください")}${r.match ? ` · ${t("正規表現")} <span class="mono">${esc(r.match)}</span>` : ""}${
      r.own_prompts ? "" : ` · ${t("狙う場面の質問は、鍵づくりに使った質問です(甘い)")}`}</p>
      ${part("scene", t("狙う場面の質問"))}${part("others", t("書き込みたくない場面の質問"))}`;
  }

  // 重みに入れる前に試した結果: 回ごとの数を並べた表と、選んだ回の返事(足した返事と、同じ質問に何も足さない返事を横に並べる)
  renderTry() {
    const st = this.st, el = this.$("#fx-try");
    if (!st.tries.length) { el.innerHTML = ""; return; }
    const cnt = (d, k) => (d && d.counts && d.counts[k] !== undefined ? d.counts[k] : "—");
    const rows = st.tries.map((r, i) => `<tr${i === st.tryShown ? ` class="fx-sel"` : ""}><td>${i + 1}</td><td class="l">${esc((r.layers || []).join(","))}</td>
      <td class="l">${esc((r.segments || []).map((sg, k) => `${sg}: ${r.alphas[k]}`).join(" / "))}</td><td>${r.temperature ? `${r.temperature} · ${t("種")} ${r.seed}` : "greedy"}</td>
      <td>${cnt(r.scene, "n")}</td><td>${cnt(r.scene, "loop")}</td><td>${cnt(r.scene, "empty")}</td><td>${cnt(r.scene, "match")}</td><td>${cnt(r.baseline, "match")}</td><td>${cnt(r.scene, "tokens")}</td>
      <td><button class="small" data-fxtry="${i}">${t("返事を見る")}</button></td></tr>`).join("");
    const r = st.tries[Math.max(0, Math.min(st.tryShown, st.tries.length - 1))];
    const base = new Map(((r.baseline && r.baseline.replies) || []).map((x) => [x.prompt, x]));
    const loop = (x) => (x && x.loop ? ` <span class="kx-ng">${t("繰り返し")}</span>` : "");
    const scene = (r.scene.replies || []).map((x) => { const b = base.get(x.prompt);
      return `<tr><td class="l fx-q">${esc(x.prompt)}</td><td class="l fx-a">${esc(x.text)}${loop(x)}</td><td class="l fx-a muted">${b ? esc(b.text) : "—"}${loop(b)}</td></tr>`; }).join("");
    const others = r.others ? `<h4 class="note">${t("書き込みたくない場面の質問(何も足していない)")} <span class="muted note">${t("{n} 問 · 繰り返し {l} · 空 {e}", { n: cnt(r.others, "n"), l: cnt(r.others, "loop"), e: cnt(r.others, "empty") })}</span></h4>
      <div class="evx-scroll"><table class="evx fx-tab fx-try-tab" id="fx-try-others"><thead><tr><th class="l">${t("質問")}</th><th class="l">${t("返事")}</th></tr></thead><tbody>${
      r.others.replies.map((x) => `<tr><td class="l fx-q">${esc(x.prompt)}</td><td class="l fx-a">${esc(x.text)}${loop(x)}</td></tr>`).join("")}</tbody></table></div>` : "";
    el.innerHTML = `<div class="evx-scroll"><table class="evx mono fx-tab" id="fx-try-runs"><thead><tr><th>#</th><th class="l">${t("足した層")}</th><th class="l">${t("区切り: 強さ")}</th><th>${t("生成")}</th><th>${t("質問の数")}</th>
        <th title="${t("同じ語句から抜けられなくなった返事(4 トークンの並びの種類が半分を切る)")}">${t("繰り返しに落ちた返事")}</th><th>${t("空の返事")}</th>
        <th title="${t("正規表現に一致した返事。繰り返しの中の語も数える")}">${t("正規表現に一致")}</th><th>${t("何も足さないときの一致")}</th><th>${t("平均の長さ")}</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="muted note">${t("数は機械的に数えたものです。口調が出たか、相手の発言に合っているかは、返事を読んで確かめてください")}${r.match ? ` · ${t("正規表現")} <span class="mono">${esc(r.match)}</span>` : ""}${
        r.own_prompts ? "" : ` · ${t("狙う場面の質問は、方向づくりに使った質問です(甘い)")}`}</p>
      ${(r.notes || []).slice(0, -2).map((x) => `<p class="pp-warn note">${esc(x)}</p>`).join("")}
      <h4 class="note">${t("狙う場面の質問")} <span class="muted note">#${st.tries.indexOf(r) + 1} · ${t("層")} ${esc((r.layers || []).join(","))}</span></h4>
      <div class="evx-scroll"><table class="evx fx-tab fx-try-tab" id="fx-try-scene"><thead><tr><th class="l">${t("質問")}</th><th class="l">${t("方向を足した返事")}</th><th class="l">${t("何も足さない返事")}</th></tr></thead><tbody>${scene}</tbody></table></div>
      ${others}`;
  }

  renderCheck() {
    const r = this.st.check, el = this.$("#fx-check");
    if (!r) { el.innerHTML = ""; return; }
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
      const title = g === "scene" ? t("狙う場面(対照集合の質問)") : t("書き込みたくない場面: {g}", { g: g.replace(/^dont_touch:/, "") });
      return `<h4 class="note">${esc(title)} <span class="muted note">${d.n} ${t("問")} · ${t("繰り返しになった返事 {n}", { n: d.loops })}</span></h4>
        <div class="evx-scroll"><table class="evx mono fx-tab fx-check-tab" data-g="${esc(g)}"><thead><tr><th>${t("層")}</th><th>${t("区切り")}</th>${cols.map((k) => `<th>${esc(colName(k))}</th>`).join("")}<th class="l">${t("ユニット")}</th></tr></thead><tbody>${trs}</tbody></table></div>`;
    }).join("");
    el.innerHTML = `<h4 class="note">${t("実際に足した量(返事の位置ごとの平均。α·方向 を 1 として)")}</h4>
      <p class="note"><span class="fx-in fx-chip">${t("狙う場面の、その区切りの位置")}</span> <span class="fx-low fx-chip">${t("足りない({v} 未満)", { v: AMOUNT_LOW })}</span> <span class="fx-bad fx-chip">${t("書き込みたくない位置で大きい({v} 以上)", { v: AMOUNT_HIGH })}</span>
        <span class="muted" id="fx-check-sum">${t("足りない {a} か所 · 書き込みたくない位置で大きい {b} か所", { a: nLow, b: nBad })}</span></p>
      ${tables}
      ${r.positions ? `<p class="muted note">${t("文字の直前に書く形なので、何も当てずに作った返事を、レシピを当てて読んだ値です(書くと、一致する文字が出なくなるため)")}</p>` : ""}
      <p class="muted note">${t("列は返事の位置(0 = 返事を書き始める直前)。値は平均で、マウスを当てると最小〜最大。区切りの境目で足りないときは、区切りを変えるか、当てた状態で鍵を作り直します。返事の質は、レシピのタブのチャットか behave で読んで確かめてください")}</p>`;
  }
}
