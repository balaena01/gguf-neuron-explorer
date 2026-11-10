// 鍵のタブの「効く層を探す」: keyed の効く場所(層・書き込み先・α)を 2 段で探す(gne/keyed_causal.py)。
// key-check は鍵が場面を分けるか(鍵の反応の強さ)しか見ないので、どこに書けば振る舞いが変わるかはここで測る。自動で 1 つに決めない。
// 1. 第 1 段(direction_key_causal のジョブ、gne direction key-causal): 「鍵を作る」の入力欄の対照集合で、候補(層 × 書き込み先)ごとに
//    鍵を作り、作成に使っていない奇数番目の返事で、keyed を当てたときに振る舞いの部分の対数確率がどれだけ下がるか・dont_touch の
//    返事がどれだけ変わるかを、比較用の鍵(ラベルをシャッフルして作った鍵)と並べて生成なしで測る。表には、全候補の比較用の鍵の
//    5% 点より下がった候補の印と、「候補の数 × 5% = 偶然でもこのくらいは印が付く数」、生成なしの目安だという注を出す。
//    表は 1 か所(書き込み先 × 層)につき 1 行で、強さ(α)ごとの確率の下がり方と印、全部の強さで印が付いたか、「確率の下がり方 ÷
//    副作用」と副作用(選んだ強さで)を並べる。どの列も見出しのクリックで並べ替えられる。
// 2. 表の行を選んで(複数可)第 2 段(direction_key_causal_verify のジョブ、gne direction key-causal-verify): 選んだ候補(選んだ
//    強さのユニット)と比較用の鍵 1 本ずつで、作成に使っていない質問に behave と同じ生成をさせ、振る舞いが出た数を iv だけのときと
//    並べる。「まとめて 1 つの候補として確かめる」なら、選んだ行を一緒に当てる組を 1 つの候補にする(比較用の鍵も組で)。
// 行の「+ レシピ」でその候補(層・書き込み先・α・方向・鍵)の keyed のユニットをレシピに足す。
// 遅れて届いた読み込み(保存した結果の一覧・開いた結果・前に投げたジョブの結果)は、通し番号とジョブの id で捨てる(後から始めたものだけを使う)。
import { getJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const fmt = (x, d = 3) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const sgn = (x, d = 3) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${+x >= 0 ? "+" : "−"}${Math.abs(+x).toFixed(d)}`);
const NAME_RE = /^[\w.][\w.-]*$/;
// 表の上に出す注(サーバの keyed_causal.NOTE と同じ文。英語の画面では訳を出す)
export const NOTE = "これは生成なしの目安です。比較用の鍵(ラベルをシャッフルして作った鍵)と見分けがつかない候補を外すためのもので、効くことの保証ではありません(生成で振る舞いが変わる場面でも、生成なしの確率の差は動かないことがあります)。効くかどうかは第 2 段(gne direction key-causal-verify、UI は「選んだ行を生成で確かめる」)の生成で数えてください";

// 鍵の target(ffn / attn / shared / expert:e)→ ユニットの短い書き方(ffn / attn / S / Ee)
export function targetWho(target) {
  const m = String(target || "").match(/^expert:(\d+)$/);
  return m ? `E${m[1]}` : target === "shared" ? "S" : String(target || "ffn");
}

// α の段("-1,-2,-4")→ 数のリスト。読めなければ null
export function parseAlphas(s) {
  const xs = String(s || "").split(",").map((x) => x.trim()).filter(Boolean);
  if (!xs.length) return [];
  const out = [];
  for (const x of xs) {
    const v = Number(x);
    if (!Number.isFinite(v) || v === 0) return null;
    if (!out.includes(v)) out.push(v);
  }
  return out;
}

// probe の JSONL(1 行 1 問)→ 行のリスト。読めない行があればエラーの文を返す
export function parseProbe(text) {
  const rows = [];
  const lines = String(text || "").split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i].trim();
    if (!ln) continue;
    let r;
    try { r = JSON.parse(ln); } catch (e) { return { rows: null, error: t("probe の {i} 行目が JSON として読めません: {m}", { i: i + 1, m: e.message }) }; }
    if (!r || typeof r !== "object" || Array.isArray(r)) return { rows: null, error: t("probe の {i} 行目は表にしてください", { i: i + 1 }) };
    if (!r.prompt || typeof r.behavior !== "string" || typeof r.desired !== "string" || !r.behavior || !r.desired) {
      return { rows: null, error: t("probe の {i} 行目には prompt・behavior(振る舞いの答え)・desired(望む答え)が要ります", { i: i + 1 }) };
    }
    rows.push(r);
  }
  return rows.length ? { rows, error: null } : { rows: null, error: t("probe に行がありません") };
}

// 効き目の 1 つ({mean, ci, n})→ 「−0.123±0.045」
function eff(e) {
  if (!e) return "—";
  return `${sgn(e.mean)}${e.ci !== null && e.ci !== undefined ? `<span class="muted">±${fmt(e.ci)}</span>` : ""}`;
}
const range = (r) => (r ? `${sgn(r.min)}〜${sgn(r.max)}` : "—");

export class KeyCausal {
  constructor(root, { onRun, onAddKeyed, source, defaults, moe } = {}) {
    this.root = root;
    this.onRun = onRun || (async () => null);       // (kind, params) → Promise<ジョブ | null>
    this.onAddKeyed = onAddKeyed || (() => {});     // ({ layer, kind, expert, alpha, direction, key }) → レシピに足す
    this.source = source || (() => ({ error: t("対照集合がありません") }));   // () → { params: {spec} | {contrast} } か { error }
    this.defaults = defaults || (() => ({}));       // () → { iv, specIv(表に keyed.iv があるか), max_tokens }(「鍵を作る」の欄)
    this.moe = !!moe;
    this.result = null;                             // 表に出している第 1 段の結果
    this.verified = null;                           // 第 2 段の結果
    this.saved = [];                                // 保存した第 1 段の結果の要約(/api/key-causal)
    this.probe = null;                              // 読み込んだ probe の行
    this.jobs = { run: null, verify: null };        // 最後に投げたジョブの id(前に投げたジョブの結果は捨てる)
    this.seq = { list: 0, open: 0 };                // 読み込みの通し番号(後から始めたものだけを使う)
    this.shown = 0;                                 // 表を描き替えた回数(開くのを待つ間に新しい結果が来たら、開いた結果は捨てる)
    this.handled = new Set();
    this.ralpha = null;                             // 「確率の下がり方 ÷ 副作用」・副作用・第 2 段のユニットに使う強さ(α の文字列)
    this.sort = { col: null, dir: 1 };              // 表の並べ替え(見出しのクリック)
    this.selKeys = new Set();                       // 選んだ行(書き込み先:層。並べ替え・強さの切り替えで描き直しても残す)
    root.innerHTML = `<h4>${t("効く層を探す")} <span class="muted note">${t("(keyed の効く場所: 生成なしのふるい分け → 生成での確かめ)")}</span></h4>
      <p class="muted note pp-help">${t("key-check は鍵が場面を分けるか(鍵の反応の強さ)しか見ません。どの層・どの書き込み先に書けば振る舞いが変わるかは別の問題です。第 1 段は、上の「鍵を作る」の対照集合で、候補(層 × 書き込み先)ごとに鍵を作り(偶数番目)、作成に使っていない奇数番目の返事で、keyed を当てたときに振る舞いの部分(positions の正規表現に一致した文字)の対数確率がどれだけ下がるかを、比較用の鍵と並べて生成なしで測ります。生成なしの値は保証ではないので、表から候補を数行選び、第 2 段の生成で振る舞いの数を確かめます。介入(iv)と返事の長さは「鍵を作る」の欄のものを使います。")}</p>
      <div class="pp-set">
        <label class="muted" title="${t("候補の層(カンマ区切りか 10-16)。空なら全層(最終層と MTP の層は除く)")}">${t("層")} <input id="kz-layers" size="12" placeholder="${t("空なら全層")}" spellcheck="false"></label>
        <span class="muted" title="${t("候補の書き込み先。ffn = ffn_down、attn = attn_output。書けない書き込み先・層は外す")}">${t("書き込み先")}
          <label><input type="checkbox" id="kz-t-ffn" checked> ffn</label> <label><input type="checkbox" id="kz-t-attn" checked> attn</label>
          <span class="${this.moe ? "" : "hidden"}"><label><input type="checkbox" id="kz-t-shared"> shared</label>
          <label title="${t("専門家の番号(カンマ区切り)。空なら専門家は測らない")}">expert <input id="kz-t-exp" size="6" placeholder="e" spellcheck="false"></label></span></span>
        <label class="muted" title="${t("α の段(カンマ区切り)。空なら 1,2,4 に表の keyed.alpha の符号を掛けたもの")}">${t("α の段")} <input id="kz-alphas" size="10" spellcheck="false"></label>
        <label class="muted" title="${t("候補ごとに測る比較用の鍵(ラベルをシャッフルして作った鍵)の本数。全候補の比較用の鍵をまとめた分布の 5% 点が印の基準")}">${t("比較用の鍵")} <input id="kz-pseudo" type="number" min="1" max="16" value="3" style="width:4em"></label>
        <label class="muted" title="${t("効き目を測る振る舞いの部分の正規表現。空なら positions の before: の正規表現")}">${t("効き目の正規表現")} <input id="kz-match" size="18" spellcheck="false"></label>
        <label class="muted" title="${t("選択式の振る舞い向け: 1 行 1 問の JSONL({\"prompt\", \"prefix\": \"Answer:\", \"behavior\": \" B\", \"desired\": \" C\"}。書き込みたくない場面の行は \"group\" に群の名前)。その位置での log p(振る舞いの答え) − log p(望む答え) の変化を測る")}">probe <input type="file" id="kz-probe" accept=".jsonl,.json,application/json"></label>
        <span class="muted note" id="kz-pinfo"></span>
        <label class="muted" title="${t("結果と鍵の名前。空なら <対照集合の名前>-causal(鍵は <名前>-ffn などで保存)")}">${t("名前")} <input id="kz-name" size="16" spellcheck="false"></label>
      </div>
      <div class="pp-go"><button id="kz-go" title="${t("候補ごとに鍵を作り、生成なしで効き目・副作用・比較用の鍵を測る(direction_key_causal のジョブ)")}">${t("第 1 段: 生成なしで測る")}</button>
        <span class="muted note" id="kz-status"></span></div>
      <div id="kz-prog" class="note mono"></div>
      <div class="pp-in"><span class="muted">${t("保存した結果")}</span> <select id="kz-saved"><option value="">${t("(なし)")}</option></select>
        <button class="small" id="kz-open">${t("開く")}</button></div>
      <div id="kz-res"></div>
      <div id="kz-vwrap" class="hidden">
        <h4 class="note">${t("第 2 段: 選んだ行を生成で確かめる")} <span class="muted note" id="kz-vsel"></span></h4>
        <div class="pp-set">
          <label class="muted" title="${t("返事の長さ(behave と同じ)")}">${t("返事の長さ")} <input id="kz-vmt" type="number" min="1" max="4096" value="256" style="width:5em"></label>
          <label class="muted" title="${t("振る舞いを数える正規表現。空なら第 1 段で測った正規表現(無ければ対照集合の validate.match)")}">${t("判定の正規表現")} <input id="kz-vmatch" size="18" spellcheck="false"></label>
          <label class="muted" title="${t("書き込みたくない場面(dont_touch)の質問にも答えさせ、返事が変わった数を数える")}"><input type="checkbox" id="kz-vdt" checked> dont_touch</label>
        </div>
        <details class="kz-adv"><summary class="muted note">${t("judge・温度")}</summary><div class="pp-set">
          <label class="muted" title="${t("読み手の判定の文面({q} に質問、{a} に返事)。空なら正規表現で数える")}">judge <textarea id="kz-vjudge" class="mono" rows="2" cols="40" spellcheck="false"></textarea></label>
          <label class="muted">labels <input id="kz-vlabels" size="8" value="R,P,C" spellcheck="false"></label>
          <label class="muted" title="${t("振る舞いとみなすラベル(judge のとき)")}">${t("振る舞いのラベル")} <input id="kz-vlabel" size="6" spellcheck="false"></label>
          <label class="muted">${t("温度")} <input id="kz-vtemp" type="number" min="0" max="2" step="0.1" value="0" style="width:4.5em"></label>
          <label class="muted" title="${t("種(カンマ区切り。温度があるとき)")}">${t("種")} <input id="kz-vseeds" size="8" spellcheck="false"></label>
        </div></details>
        <div class="pp-go"><button id="kz-vgo" disabled title="${t("選んだ候補と、それぞれの比較用の鍵 1 本で、作成に使っていない質問に生成させる(direction_key_causal_verify のジョブ)")}">${t("選んだ行を生成で確かめる")}</button>
          <button id="kz-vcombo" disabled title="${t("選んだ行(選んだ強さのユニット)を一緒に当てる組を 1 つの候補にして確かめる。比較用の鍵も、組の各ユニットを同じ番号の比較用の鍵に替えた組で比べる")}">${t("選んだ行をまとめて 1 つの候補として確かめる")}</button>
          <label class="muted" title="${t("組と一緒に、選んだ行を 1 か所ずつの候補としても確かめ、結果の表に並べる")}"><input type="checkbox" id="kz-vsingles" checked> ${t("1 か所ずつも並べる")}</label>
          <span class="muted note" id="kz-vstatus"></span></div>
        <div id="kz-vprog" class="note mono"></div>
        <div id="kz-vres"></div>
      </div>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#kz-go").addEventListener("click", () => this.run());
    this.$("#kz-vgo").addEventListener("click", () => this.verify(false));
    this.$("#kz-vcombo").addEventListener("click", () => this.verify(true));
    this.$("#kz-open").addEventListener("click", () => { const v = this.$("#kz-saved").value; if (v) this.open(v); });
    this.$("#kz-probe").addEventListener("change", async (ev) => {
      const f = ev.target.files && ev.target.files[0];
      if (!f) { this.probe = null; this.$("#kz-pinfo").textContent = ""; return; }
      const r = parseProbe(await f.text());
      this.probe = r.rows;
      const el = this.$("#kz-pinfo");
      el.textContent = r.error ? r.error : t("{n} 行", { n: r.rows.length });
      el.classList.toggle("err", !!r.error);
    });
    root.addEventListener("click", (ev) => this.onClick(ev));
    root.addEventListener("change", (ev) => {
      if (ev.target.matches("[data-kzsel], #kz-onlymark")) this.selChanged(ev);
      else if (ev.target.id === "kz-ralpha") { this.ralpha = ev.target.value; this.render(); }
    });
  }

  // 欄の既定の表示(対照集合の名前と表の keyed.alpha の符号から)
  hints({ name = null, alpha = null } = {}) {
    this.$("#kz-alphas").placeholder = Number(alpha) < 0 ? "-1,-2,-4" : "1,2,4";
    this.$("#kz-name").placeholder = name ? `${name}-causal` : "";
  }

  status(which, s, err = false) {
    const el = this.$(which === "verify" ? "#kz-vstatus" : "#kz-status");
    el.textContent = s;
    el.classList.toggle("err", err);
  }

  // 保存した結果の一覧を取り直す(後から始めた読み込みだけを使う)
  async refresh() {
    const seq = ++this.seq.list;
    let list;
    try { list = await getJSON("/api/key-causal"); } catch { return; }
    if (seq !== this.seq.list || !Array.isArray(list)) return;
    this.saved = list;
    const sel = this.$("#kz-saved"), cur = sel.value;
    sel.innerHTML = list.length ? list.map((x) => `<option value="${esc(x.name)}">${esc(x.name)} · ${esc(x.contrast || "")} · ${(x.targets || []).map(targetWho).join(",")}${x.verify ? ` · ${t("確かめあり")}` : ""}</option>`).join("")
      : `<option value="">${t("(なし)")}</option>`;
    const want = cur || (this.result && this.result.name);
    if (want && list.some((x) => x.name === want)) sel.value = want;
  }

  // 保存した結果を開く: 待っている間に別の結果を開いたか、新しい結果を表に出したなら、届いた結果は捨てる
  async open(name) {
    const seq = ++this.seq.open, before = this.shown;
    const stale = () => seq !== this.seq.open || this.shown !== before;
    try {
      const r = await getJSON(`/api/key-causal/${encodeURIComponent(name)}`);
      if (stale()) return;
      this.show(r);
      this.verified = r.verify || null;
      this.renderVerify();
    } catch (e) { if (!stale()) this.status("run", `${t("エラー")}: ${e.message}`, true); }
  }

  // --- 第 1 段 -----------------------------------------------------------------------------------------------
  params() {
    const src = this.source();
    if (src.error) { this.status("run", src.error, true); return null; }
    const v = (s) => this.$(s).value.trim();
    const p = { ...src.params };
    const d = this.defaults() || {};
    if (v("#kz-layers")) {
      if (!/^[\d\s,-]+$/.test(v("#kz-layers"))) { this.status("run", t("層は数をカンマ区切りで(例 8,14,20 か 4-20。空なら全層)"), true); return null; }
      p.layers = v("#kz-layers").replace(/\s+/g, "");
    }
    const tg = [];
    if (this.$("#kz-t-ffn").checked) tg.push("ffn");
    if (this.$("#kz-t-attn").checked) tg.push("attn");
    if (this.moe && this.$("#kz-t-shared").checked) tg.push("shared");
    if (this.moe && v("#kz-t-exp")) {
      for (const e of v("#kz-t-exp").split(",").map((x) => x.trim()).filter(Boolean)) {
        if (!/^\d+$/.test(e)) { this.status("run", t("専門家の番号は数をカンマ区切りで"), true); return null; }
        tg.push(`expert:${e}`);
      }
    }
    if (!tg.length) { this.status("run", t("書き込み先を 1 つ以上選んでください"), true); return null; }
    p.targets = tg.join(",");
    const al = parseAlphas(v("#kz-alphas"));
    if (al === null) { this.status("run", t("α の段は 0 でない数をカンマ区切りで(例 -1,-2,-4)"), true); return null; }
    if (al.length) p.alphas = al;
    p.pseudo = Math.max(1, +v("#kz-pseudo") || 3);
    if (v("#kz-match")) p.measure_match = v("#kz-match");
    if (this.probe) p.probe = this.probe;
    const name = v("#kz-name");
    if (name && !NAME_RE.test(name)) { this.status("run", t("名前は英数字・_・-・. だけにしてください"), true); return null; }
    if (name) p.name = name;
    if (d.iv) p.iv = d.iv;
    else if (d.specIv) p.no_iv = true;                       // 介入の欄を空にした: 表の keyed.iv も当てない
    if (d.max_tokens) p.max_tokens = d.max_tokens;
    return p;
  }

  async run() {
    const p = this.params();
    if (!p) return;
    this.status("run", t("ジョブを投げています…"));
    try {
      const j = await this.onRun("direction_key_causal", p);
      if (!j) { this.status("run", t("やめました")); return; }
      this.jobs.run = j.id;
      this.status("run", t("候補ごとに鍵を作って測っています(モデルを流します。GPU を待つことがあります)…"));
      this.follow(j.id, "#kz-prog");
    } catch (e) { this.status("run", `${t("エラー")}: ${e.message}`, true); }
  }

  // 結果を表に出す
  show(r) {
    this.result = r;
    this.shown++;
    this.selKeys = new Set();
    const al = (r && r.alphas) || [];
    this.ralpha = al.length ? String(al.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a))) : null;   // 既定は |α| が最大の強さ
    this.verified = null;
    this.renderVerify();
    this.render();
    const sel = this.$("#kz-saved");
    if (r && r.name && [...sel.options].some((o) => o.value === r.name)) sel.value = r.name;
  }

  // 1 か所 1 行の候補(結果の candidates。無ければ行から作る)
  candidates() {
    const r = this.result;
    if (!r) return [];
    if (Array.isArray(r.candidates) && r.candidates.length) return r.candidates;
    const out = new Map();
    for (const x of r.rows || []) {
      const k = `${x.target}:${x.layer}`;
      if (!out.has(k)) out.set(k, { target: x.target, layer: x.layer, key: x.key, alphas: [], by_alpha: {}, key_check: x.key_check, linear: x.linear });
      const c = out.get(k);
      c.alphas.push(x.alpha);
      c.by_alpha[String(x.alpha)] = { unit: x.unit, reply: x.reply, mark: x.mark, probe: x.probe, probe_mark: x.probe_mark, pseudo: x.pseudo,
                                      probe_pseudo: x.probe_pseudo, side_b: (x.side || {}).b, dt_min: (x.side || {}).dt_min, ratio: null };
    }
    return [...out.values()];
  }

  // 並べ替えの値(小さいほど上。値が無ければ下)
  sortValue(c, col) {
    const b = (c.by_alpha || {})[this.ralpha] || {};
    const m = String(col).match(/^(eff|probe):(.+)$/);
    if (m) { const x = (c.by_alpha || {})[m[2]] || {}; const e = m[1] === "eff" ? x.reply : x.probe; return e ? e.mean : null; }
    switch (col) {
      case "target": return `${c.target}:${String(c.layer).padStart(4, "0")}`;
      case "layer": return c.layer;
      case "all": return c.all_marked === true ? 0 : c.all_marked === false ? 1 : null;
      case "allp": return c.all_marked_probe === true ? 0 : c.all_marked_probe === false ? 1 : null;
      case "ratio": return b.ratio === null || b.ratio === undefined ? null : -b.ratio;
      case "side": return b.side_b ? -b.side_b.mean : null;
      case "dt": return b.dt_min === null || b.dt_min === undefined ? null : -b.dt_min;
      case "kc": { const v = (c.key_check || {}).margin; return v === null || v === undefined ? null : -v; }
      case "lin": { const v = (c.linear || {}).ratio; return v === null || v === undefined ? null : Math.abs(Math.log(Math.max(v, 1e-6))); }
      default: return null;
    }
  }

  sortedCandidates() {
    const cs = this.candidates().map((c, i) => ({ c, i }));
    const { col, dir } = this.sort;
    if (!col) return cs;
    return cs.sort((x, y) => {
      const a = this.sortValue(x.c, col), b = this.sortValue(y.c, col);
      if (a === null && b === null) return x.i - y.i;
      if (a === null) return 1;
      if (b === null) return -1;
      return (a < b ? -1 : a > b ? 1 : 0) * dir || x.i - y.i;
    });
  }

  render() {
    const el = this.$("#kz-res"), r = this.result;
    if (!r) { el.innerHTML = ""; this.$("#kz-vwrap").classList.add("hidden"); return; }
    const rows = r.rows || [];
    const alphas = (r.alphas || []).map(String);
    if (!alphas.includes(this.ralpha)) this.ralpha = alphas.length ? alphas[alphas.length - 1] : null;
    const hasReply = !!(r.measure && r.measure.regex), hasProbe = rows.some((x) => x.probe);
    const only = this.$("#kz-onlymark") ? this.$("#kz-onlymark").checked : false;
    const marks = Object.entries(r.marks || {}).flatMap(([a, mk]) => Object.entries(mk).map(([m, v]) => {
      const pl = ((r.pool || {})[a] || {})[m] || {};
      return `<li>${t("α {a}({m}): 印 {k} / 候補 {n}(偶然でも {n} × {c}% = {e} 個は印が付く)· 比較用の鍵の 5% 点 {p}({pn} 本)", {
        a: esc(a), m: m === "reply" ? t("返事") : "probe", k: `<b>${v.marked}</b>`, n: v.candidates, c: Math.round((r.chance || 0.05) * 100),
        e: v.expected, p: sgn(pl.p5), pn: pl.n ?? "—" })}</li>`;
    })).join("");
    const iv = r.iv_spec ? (Array.isArray(r.iv_spec) ? r.iv_spec.join(" ") : r.iv_spec) : null;
    const c = r.counts || {}, ca = c.a || {};
    const head = `<p class="note">${t("結果")} <b class="mono">${esc(r.name)}</b> · ${t("対照集合")} ${esc(r.contrast)} · ${t("方向")} <span class="mono">${esc(r.direction)}</span> · ${
      iv ? `${t("介入(iv)")} <span class="mono">${esc(iv)}</span>` : t("介入なし")} · ${t("候補 {n} 個", { n: r.n_candidates })} · α ${(r.alphas || []).map((a) => sgn(a, 2)).join(", ")} · ${
      t("候補ごとの比較用の鍵 {n} 本", { n: r.pseudo })} · ${fmt(r.seconds, 1)} ${t("秒")}</p>
      ${hasReply ? `<p class="muted note">${t("効き目(返事)= 作成に使っていない {n} 問のうち、{re} に一致した {m} 問の返事で、一致した文字の対数確率の変化(iv だけとの差。負 = 振る舞いが出にくくなる)", {
        n: ca.examples ?? "—", re: `<span class="mono">${esc(r.measure.regex)}</span>`, m: ca.matched ?? "—" })}</p>` : ""}
      ${hasProbe ? `<p class="muted note">${t("効き目(probe)= log p(振る舞いの答え) − log p(望む答え) の変化(負 = 望む答えに寄る)")}</p>` : ""}`;
    const note = `<p class="pp-warn note kz-note">⚠ ${esc(t(NOTE))}</p>`;
    const floor = r.ratio_floor ?? 0.01;
    const legend = `<p class="muted note kz-legend">${t("比較用の鍵 = 書き込みたい場面と書き込みたくない場面のラベルをでたらめに入れ替えて、同じ手順で作った鍵。偶然でどれくらい変わるかの基準にする。★ = その強さで、全部の候補の比較用の鍵の 5% 点より大きく確率が下がった(有意)。全部の強さで有意 = どの強さでも ★。副作用 = 書き込みたくない位置の返事の 1 トークンあたりの対数確率の変化(負 = 下がった)。確率の下がり方 ÷ 副作用 = (−確率の下がり方)÷(−副作用)で、副作用が {f} より小さいとき(上がったときも)は {f} で割り、確率が下がっていなければ —。大きいほど、書き込みたくない場面を変えずに振る舞いを下げる。", { f: floor })}<br>${
      t("選び方の目安: 確率の下がり方・副作用・全部の強さで有意か、の 3 つを並べて自分で選ぶ(この選び方の根拠は Qwen3.5-4B の語尾の 1 例だけ)。確率の下がり方だけで選ぶと、返事を壊して確率を下げている候補を拾うことがある。")}</p>`;
    const sortMark = (col) => (this.sort.col === col ? (this.sort.dir > 0 ? " ▲" : " ▼") : "");
    const th = (col, label, title, cls = "") => `<th class="kz-sort${cls ? ` ${cls}` : ""}" data-kzsort="${esc(col)}" title="${esc(title)}">${label}${sortMark(col)}</th>`;
    const ra = this.ralpha;
    const ralphaSel = `<label class="muted" title="${t("「確率の下がり方 ÷ 副作用」・副作用・dont_touch の変化を出す強さ。行を選んで確かめるときも、この強さのユニットを使う")}">${t("強さ")} <select id="kz-ralpha">${
      alphas.map((a) => `<option value="${esc(a)}"${a === ra ? " selected" : ""}>${sgn(+a, 2)}</option>`).join("")}</select></label>`;
    const effTitle = t("この強さでの確率の下がり方(振る舞いの部分の対数確率の変化。例ごとの差の平均、± は 95% の幅)。★ = 有意(全部の候補の比較用の鍵の 5% 点より大きく下がった)");
    const thead = `<tr><th></th>${th("target", t("書き込み先"), t("書き込み先(並べ替えは書き込み先・層の順)"))}${th("layer", t("層"), t("層"))}${
      hasReply ? alphas.map((a) => th(`eff:${a}`, `α ${sgn(+a, 2)}`, effTitle)).join("") + th("all", t("全部の強さで有意"), t("どの強さでも ★(有意)が付いた")) : ""}${
      hasProbe ? alphas.map((a) => th(`probe:${a}`, `probe α ${sgn(+a, 2)}`, effTitle)).join("") + th("allp", t("全部の強さで有意(probe)"), t("どの強さでも ★(有意)が付いた")) : ""}${
      th("ratio", t("下がり方 ÷ 副作用"), t("(−確率の下がり方)÷(−副作用)。選んだ強さで。大きいほど、書き込みたくない場面を変えずに振る舞いを下げる"))}${
      th("side", t("副作用"), t("書き込みたくない位置(同じ返事の一致以外の位置か、B 側の返事)の 1 トークンあたりの対数確率の変化。選んだ強さで"))}${
      th("dt", t("dont_touch の変化"), t("dont_touch の群の返事の 1 トークンあたりの対数確率の変化のうち、最も下がった群。選んだ強さで"))}${
      th("kc", t("key-check の余裕"), t("参考: key-check の選択性 − 比較用の鍵の p95"))}${
      th("lin", t("比例"), t("最大の |α| での効き目 ÷ 最小の |α| での効き目を |α| の比で割った値(1 = 比例、1 未満 = 頭打ち)。向き× = α の段で符号がそろわない。— = 最小の |α| での効き目が 95% の幅の中(比は出さない)"))}<th></th></tr>`;
    const effCell = (e, mark, ps) => (e ? `<td title="${esc(`${t("比較用の鍵")} ${range(ps)}`)}">${eff(e)}${mark ? ` <span class="kx-ok">★</span>` : ""}</td>` : "<td>—</td>");
    const body = this.sortedCandidates().map(({ c: x, i }) => {
      const ba = x.by_alpha || {};
      const anyMark = Object.values(ba).some((b) => b.mark || b.probe_mark);
      if (only && !anyMark) return "";
      const b = ba[ra] || {};
      const lin = x.linear || {};
      const linS = lin && lin.consistent === false ? t("向き×") : lin.ratio !== null && lin.ratio !== undefined ? `×${fmt(lin.ratio, 2)}` : "—";
      const k = `${x.target}:${x.layer}`;
      return `<tr${anyMark ? ` class="best"` : ""}><td><input type="checkbox" data-kzsel="${i}" data-kzkey="${esc(k)}"${this.selKeys.has(k) ? " checked" : ""}></td><td>${esc(targetWho(x.target))}</td><td>L${x.layer}</td>${
        hasReply ? alphas.map((a) => effCell((ba[a] || {}).reply, (ba[a] || {}).mark, (ba[a] || {}).pseudo)).join("") + `<td>${x.all_marked ? `<span class="kx-ok">○</span>` : ""}</td>` : ""}${
        hasProbe ? alphas.map((a) => effCell((ba[a] || {}).probe, (ba[a] || {}).probe_mark, (ba[a] || {}).probe_pseudo)).join("") + `<td>${x.all_marked_probe ? `<span class="kx-ok">○</span>` : ""}</td>` : ""}
        <td>${b.ratio === null || b.ratio === undefined ? "—" : fmt(b.ratio, 1)}</td><td>${b.side_b ? sgn(b.side_b.mean) : "—"}</td><td>${sgn(b.dt_min)}</td>
        <td class="muted">${sgn((x.key_check || {}).margin)}</td><td>${linS}</td>
        <td><button class="small" data-kzadd="${i}" title="${t("この候補の keyed のユニットをレシピに足す")}">${t("+ レシピ")}</button></td></tr>`;
    }).join("");
    const dropped = (r.dropped || []).map((d) => `${esc(targetWho(d.target))} L${d.layer}: ${esc(d.error)}`);
    el.innerHTML = `<div class="kx-res">${head}${note}<ul class="evx-notes kz-marks">${marks}</ul>
      <div class="pp-in">${ralphaSel}<label class="muted"><input type="checkbox" id="kz-onlymark"${only ? " checked" : ""}> ${t("印の付いた行だけ")}</label>
        <span class="muted note">${t("見出しのクリックで並べ替え。行を選んで(複数可)下の「選んだ行を生成で確かめる」")}</span></div>
      <div class="evx-scroll"><table class="evx mono kx-tab kz-tab"><thead>${thead}</thead><tbody>${body}</tbody></table></div>
      ${legend}
      ${dropped.length ? `<details><summary class="muted note">${t("外した候補 {n} 個", { n: dropped.length })}</summary><p class="muted note">${dropped.join("<br>")}</p></details>` : ""}
      ${(r.notes || []).length > 1 ? `<p class="muted note">${r.notes.slice(1).map(esc).join("<br>")}</p>` : ""}
      ${(r.warnings || []).length ? `<ul class="evx-notes">${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}</div>`;
    this.$("#kz-vwrap").classList.remove("hidden");
    this.$("#kz-vmatch").placeholder = (r.measure && r.measure.regex) || t("(validate.match)");
    this.selChanged();
  }

  // 選んだ行の、選んだ強さのユニット(候補の行)
  selected() {
    const cs = this.candidates();
    return [...this.root.querySelectorAll("[data-kzsel]:checked")].map((b) => cs[+b.dataset.kzsel]).filter(Boolean)
      .map((c) => ({ c, b: (c.by_alpha || {})[this.ralpha] })).filter((x) => x.b && x.b.unit);
  }

  selChanged(ev) {
    if (ev && ev.target && ev.target.id === "kz-onlymark") { this.render(); return; }
    if (ev && ev.target && ev.target.dataset && ev.target.dataset.kzkey) {
      if (ev.target.checked) this.selKeys.add(ev.target.dataset.kzkey); else this.selKeys.delete(ev.target.dataset.kzkey);
    }
    const n = this.selected().length;
    this.$("#kz-vgo").disabled = !n;
    this.$("#kz-vcombo").disabled = n < 2;
    this.$("#kz-vsel").textContent = n ? t("{n} 行を選んでいます(強さ {a})", { n, a: sgn(+this.ralpha, 2) }) : t("表の行を選んでください");
  }

  // --- 第 2 段 -----------------------------------------------------------------------------------------------
  async verify(combo = false) {
    const r = this.result, sel = this.selected();
    if (!r || !sel.length || (combo && sel.length < 2)) return;
    const v = (s) => this.$(s).value.trim();
    const units = sel.map((x) => x.b.unit);
    const p = { result: r.name, max_tokens: +v("#kz-vmt") || 256 };
    if (combo) {
      p.combos = [units];
      if (this.$("#kz-vsingles").checked) p.units = units;          // 1 か所ずつの候補も並べる
    } else p.units = units;
    if (v("#kz-vmatch")) p.match = v("#kz-vmatch");
    if (!this.$("#kz-vdt").checked) p.dont_touch = false;
    if (v("#kz-vjudge")) {
      if (!v("#kz-vlabel")) { this.status("verify", t("judge を使うときは、振る舞いのラベル(例 P)を入れてください"), true); return; }
      p.judge = v("#kz-vjudge");
      p.labels = v("#kz-vlabels") || "R,P,C";
      p.label = v("#kz-vlabel");
    }
    if (+v("#kz-vtemp") > 0) {
      p.temperature = +v("#kz-vtemp");
      if (v("#kz-vseeds")) p.seeds = v("#kz-vseeds");
    }
    this.status("verify", t("ジョブを投げています…"));
    try {
      const j = await this.onRun("direction_key_causal_verify", p);
      if (!j) { this.status("verify", t("やめました")); return; }
      this.jobs.verify = j.id;
      this.verifyFor = r.name;
      this.status("verify", t("生成しています(iv だけ・候補・比較用の鍵)…"));
      this.follow(j.id, "#kz-vprog");
    } catch (e) { this.status("verify", `${t("エラー")}: ${e.message}`, true); }
  }

  renderVerify() {
    const el = this.$("#kz-vres"), v = this.verified;
    if (!v) { el.innerHTML = ""; return; }
    const cnt = (x, k = "behavior") => (x && x[k] !== undefined && x[k] !== null ? `${x[k]}/${x.n}` : "—");
    const lim = (x) => (x ? `${x.loop}・${x.empty}・${x.at_limit}` : "—");
    const j = v.judge || {}, b = v.base || {};
    const how = j.method === "label" ? t("読み手のラベル {l}", { l: esc((j.labels || []).join(", ")) }) : t("正規表現 {r}", { r: `<span class="mono">${esc(j.match || "")}</span>` });
    const groups = Object.keys(v.n_dt || {});
    const rows = (v.rows || []).map((x) => {
      const k = x.a.keyed, p = x.a.pseudo;
      const dt = groups.map((g) => {
        const d = (x.dt || {})[g] || {};
        const ch = (y) => (y && y.changed !== null && y.changed !== undefined
          ? `${y.changed}/${y.n}${y.same_prefix !== null && y.same_prefix !== undefined ? `<span class="muted" title="${t("iv だけの返事の先頭から同じ部分の割合の平均")}">(${Math.round(y.same_prefix * 100)}%)</span>` : ""}` : "—");
        return `<div>${esc(g)}: ${t("変わった {a} / {b}", { a: ch(d.keyed), b: ch(d.pseudo) })} · ${t("振る舞い {a} / {b}", { a: cnt(d.keyed), b: cnt(d.pseudo) })}</div>`;
      }).join("");
      const label = x.combo ? `<span class="kz-combo">${t("組")}</span> ${esc(x.label || "")}<div class="muted">${(x.units || []).map(esc).join("<br>")}</div>` : esc(x.unit);
      return `<tr${x.combo ? ` class="kz-combo-row"` : ""}><td class="l mono">${label}</td><td>${cnt(b.a)}</td><td><b>${cnt(k)}</b></td><td>${cnt(p)}</td>
        <td>${k.hits !== undefined ? `${k.hits} / ${p.hits} <span class="muted">(${b.a.hits})</span>` : "—"}</td>
        <td>${k.changed !== null && k.changed !== undefined ? `${k.changed}/${k.n} / ${p.changed}/${p.n}` : "—"}</td>
        <td class="l">${dt || "—"}</td><td>${lim(k)} / ${lim(p)} <span class="muted">(${lim(b.a)})</span></td></tr>`;
    }).join("");
    el.innerHTML = `<div class="kx-res"><p class="note">${t("書き込みたい場面 {n} 問(作成に使っていない奇数番目)", { n: v.n_a })}${
      groups.length ? ` · dont_touch ${groups.map((g) => `${esc(g)} ${v.n_dt[g]}`).join(" · ")}` : ""} · ${t("返事 {n} トークン", { n: v.max_tokens })} · ${
      v.greedy ? "greedy" : t("温度 {x}", { x: v.temperature })} · ${t("判定")}: ${how} · ${fmt(v.seconds, 1)} ${t("秒")}</p>
      <div class="evx-scroll"><table class="evx mono kx-tab kz-vtab"><thead><tr><th class="l">${t("候補")}</th>
        <th title="${t("iv だけのときに振る舞いが出た返事の数")}">${t("iv のみ")}</th><th title="${t("候補を足したとき")}">keyed</th><th title="${t("候補と同じ層・書き込み先・α で、比較用の鍵を使ったとき")}">${t("比較用の鍵")}</th>
        <th title="${t("正規表現の一致の総数(keyed / 比較用の鍵、括弧は iv のみ)")}">${t("一致の総数")}</th>
        <th title="${t("iv だけのときから返事が変わった数(greedy のとき)")}">${t("返事が変わった")}</th>
        <th class="l" title="${t("dont_touch の群ごとの、返事が変わった数と振る舞いの数(keyed / 比較用の鍵)")}">${t("dont_touch")}</th>
        <th title="${t("繰り返し・空の返事・上限で切れた数(keyed / 比較用の鍵、括弧は iv のみ)")}">${t("繰り返し・空・上限")}</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${(v.notes || []).length ? `<p class="muted note">${v.notes.map(esc).join("<br>")}</p>` : ""}
      ${(v.warnings || []).length ? `<ul class="evx-notes">${v.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}</div>`;
  }

  // --- ジョブ ------------------------------------------------------------------------------------------------
  follow(id, sel) {
    const el = this.$(sel);
    const show = (j) => {
      if (!["queued", "running"].includes(j.status)) { el.innerHTML = ""; return; }
      const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
      el.innerHTML = j.status === "queued" ? `<span class="muted">${t("待機中")}</span>`
        : `<span class="jb-bar"><span style="width:${pct}%"></span></span> ${pct}% <span class="muted">${esc(j.stage || "")}</span>`;
    };
    show({ status: "queued" });
    let es;
    try { es = new EventSource(`/api/jobs/${id}/events`); } catch { return; }
    es.onmessage = (m) => {
      const j = JSON.parse(m.data);
      show(j);
      if (["done", "error", "cancelled"].includes(j.status)) { es.close(); this.jobEnded(j); }
    };
    es.onerror = () => { es.close(); el.innerHTML = ""; };
  }

  // keys.js の jobEnded(と上の SSE)から。前に投げたジョブ(いまの id でないもの)の結果は捨てる
  async jobEnded(j) {
    const which = j.id === this.jobs.run ? "run" : j.id === this.jobs.verify ? "verify" : null;
    if (!which) {
      if (j.kind === "direction_key_causal" && j.status === "done") await this.refresh();   // 一覧には入れる
      return;
    }
    if (this.handled.has(j.id)) return;
    this.handled.add(j.id);
    this.$(which === "run" ? "#kz-prog" : "#kz-vprog").innerHTML = "";
    if (j.status !== "done") {
      const e = j.error || {};
      this.status(which, j.status === "error" ? `${t("失敗")}: ${e.message || ""}${e.hint ? `(${e.hint})` : ""}` : t("止めました"), j.status === "error");
      return;
    }
    const r = j.result || {};
    this.status(which, t("できました({s} 秒)", { s: fmt(r.seconds, 1) }));
    if (which === "run") {
      this.show(r);
      await this.refresh();
    } else if (!this.result || this.result.name === this.verifyFor) {
      this.verified = r;
      this.renderVerify();
      await this.refresh();
    }
  }

  onClick(ev) {
    const h = ev.target.closest("[data-kzsort]");
    if (h) {                                                  // 見出しのクリックで並べ替え(同じ列なら向きを逆に)
      const col = h.dataset.kzsort;
      this.sort = this.sort.col === col ? { col, dir: -this.sort.dir } : { col, dir: 1 };
      this.render();
      return;
    }
    const a = ev.target.closest("[data-kzadd]");
    if (!a || !this.result) return;
    const x = this.candidates()[+a.dataset.kzadd];
    if (!x) return;
    const m = String(x.target).match(/^expert:(\d+)$/);
    this.onAddKeyed({ layer: x.layer, kind: m ? "expert" : x.target, expert: m ? +m[1] : null, alpha: +this.ralpha, direction: this.result.direction, key: x.key });
    a.textContent = t("足しました");
  }
}
