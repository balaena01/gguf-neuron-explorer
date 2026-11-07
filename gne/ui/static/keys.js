// direction ビューの「鍵(keyed)」(CHANGE-07 G): 鍵つきの書き込み(mode: keyed)の鍵を対照集合から作り(direction_key のジョブ、
// gne direction key)、作成に使っていない例で状態を分けているかを確かめる(direction_key_check のジョブ、gne direction key-check)。
// 1. 鍵を作る: 対照集合・層・名前・positions(before:<正規表現> なら推論時と同じ文脈の返事の「変えたい位置」= 作り方 (b))・
//    q_mode・subset・擬似鍵の本数。進み具合をここにも出し、終わると層ごとの kn・gain・q_rank・作成に使った例での AUC と警告。
//    行の「+ レシピ」で keyed のユニット(L:ffn:keyed:aα@方向#鍵)をレシピに足す。
// 2. key-check: 層ごとの例単位・トークン単位の AUC、選択性(例ごとの ρ の平均の差 A − B)と擬似鍵の p95、○/×(擬似鍵を超えるか)、
//    保留集合か(held_out)と重なりの警告。層ごとの ρ = κᵀx の分布のヒストグラム(側 A・側 B・dont_touch の群。群ごとの割合で重ねる SVG)。
// 3. 保存した鍵の一覧(<root>/keys/<model_id>/)。「見る」で作成の記録、「確かめる」で key-check の欄に入れる。
import { getJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const NAME_RE = /^[\w.][\w.-]*$/;
const OPP = { even: "odd", odd: "even", all: "all" };            // key-check は作成と反対側の subset で
const mark = (b) => (b === true ? `<span class="kx-ok">○</span>` : b === false ? `<span class="kx-ng">×</span>` : "—");

// 鍵の target(ffn / attn / shared / expert:e)→ レシピのユニットの kind と専門家
export function targetKind(target) {
  const m = String(target || "ffn").match(/^expert:(\d+)$/);
  return m ? { kind: "expert", expert: +m[1] } : { kind: String(target || "ffn"), expert: null };
}

// 層の指定("8,14,20"・"10-16"・"8, 14")→ 数のリスト。読めなければ null
export function parseLayers(s) {
  const out = [];
  for (const part of String(s || "").split(",").map((x) => x.trim()).filter(Boolean)) {
    const m = part.match(/^(\d+)(?:\s*-\s*(\d+))?$/);
    if (!m) return null;
    const a = +m[1], b = m[2] === undefined ? a : +m[2];
    if (b < a) return null;
    for (let L = a; L <= b; L++) out.push(L);
  }
  return out.length ? [...new Set(out)].sort((x, y) => x - y) : null;
}

// 群の名前(A・B・dt:<名前>)の表示と色の組
const groupLabel = (g) => (g === "A" ? t("側 A(効かせたい)") : g === "B" ? t("側 B(触りたくない)") : `dont_touch ${g.replace(/^dt:/, "")}`);
const groupCls = (g, i) => (g === "A" ? "ga" : g === "B" ? "gb" : `gd${i % 3}`);

// 層 1 つの ρ の分布(hist: { edges, A, B, "dt:<名前>": [...] })。群ごとの割合(その群の数で割る)で重ねる。
// 側 B を先に塗り、側 A を上に、dont_touch は線(段)で。0 と 1(κᵀμ = 1、側 A の平均が来る所)に縦の点線。棒にマウスで数と割合
export function histSVG(h, { title = "" } = {}) {
  if (!h || !Array.isArray(h.edges) || h.edges.length < 2) return `<p class="muted note">${t("ヒストグラムがありません")}</p>`;
  const W = 320, H = 140, PL = 34, PR = 8, PT = 18, PB = 24;
  const edges = h.edges, nb = edges.length - 1, lo = edges[0], hi = edges[nb];
  const groups = Object.keys(h).filter((k) => k !== "edges" && Array.isArray(h[k]));
  const order = [...groups.filter((g) => g === "B"), ...groups.filter((g) => g === "A"), ...groups.filter((g) => g.startsWith("dt:"))];
  const share = {};
  let top = 0;
  for (const g of order) {
    const n = h[g].reduce((a, b) => a + b, 0);
    share[g] = { n, v: h[g].map((c) => (n ? c / n : 0)) };
    top = Math.max(top, ...share[g].v);
  }
  top = top || 1;
  const X = (v) => PL + ((v - lo) / (hi - lo || 1)) * (W - PL - PR), Y = (f) => PT + (1 - f / top) * (H - PT - PB);
  const parts = [];
  for (let i = 0; i <= 2; i++) {                                         // 横の目盛り(割合)
    const f = (top * i) / 2;
    parts.push(`<line class="grid" x1="${PL}" x2="${W - PR}" y1="${Y(f)}" y2="${Y(f)}"/><text class="ax" x="${PL - 4}" y="${Y(f) + 3}" text-anchor="end">${Math.round(f * 100)}%</text>`);
  }
  for (const v of [0, 1]) {                                              // ρ = 0(何も足さない)と ρ = 1(側 A の平均)
    if (v <= lo || v >= hi) continue;
    parts.push(`<line class="ref" x1="${X(v)}" x2="${X(v)}" y1="${PT}" y2="${H - PB}"/><text class="ax" x="${X(v)}" y="${H - PB + 11}" text-anchor="middle">${v}</text>`);
  }
  parts.push(`<text class="ax" x="${PL}" y="${H - 4}" text-anchor="start">${fmt(lo, 2)}</text><text class="ax" x="${W - PR}" y="${H - 4}" text-anchor="end">${fmt(hi, 2)}</text>`);
  let di = 0;
  for (const g of order) {
    const cls = groupCls(g, g.startsWith("dt:") ? di++ : 0), s = share[g], lab = esc(groupLabel(g));
    const tip = (i) => `${lab}: ${h[g][i]} (${Math.round(s.v[i] * 1000) / 10}%) · ρ ${fmt(edges[i], 2)}〜${fmt(edges[i + 1], 2)}`;
    if (g.startsWith("dt:")) {
      let d = "";
      for (let i = 0; i < nb; i++) d += `${i ? "L" : "M"}${X(edges[i])},${Y(s.v[i])}L${X(edges[i + 1])},${Y(s.v[i])}`;
      parts.push(`<path class="line ${cls}" d="${d}"><title>${lab} (n=${s.n})</title></path>`);
      continue;
    }
    for (let i = 0; i < nb; i++) {
      if (!h[g][i]) continue;
      const x0 = X(edges[i]), x1 = X(edges[i + 1]);
      parts.push(`<rect class="bar ${cls}" x="${x0 + 0.5}" y="${Y(s.v[i])}" width="${Math.max(0.5, x1 - x0 - 1)}" height="${Y(0) - Y(s.v[i])}"><title>${tip(i)}</title></rect>`);
    }
  }
  return `<svg class="kx-hist" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title || "ρ")}"><text class="tt" x="${PL}" y="11">${esc(title)}</text>${parts.join("")}</svg>`;
}

export class KeyPanel {
  constructor(root, { onRun, onAddKeyed, directions, nLayer } = {}) {
    this.root = root;
    this.onRun = onRun || (async () => null);       // (kind, params) → Promise<ジョブ | null>(jobs.js の start)
    this.onAddKeyed = onAddKeyed || (() => {});     // ({ layer, kind, expert, alpha, direction, key }) → レシピに足す
    this.directions = directions || (() => []);     // 方向セットの名前(gne direction)
    this.nLayer = nLayer || 0;
    this.contrasts = [];
    this.keys = [];
    this.jobs = { build: null, check: null };       // 投げたジョブの id
    this.handled = new Set();                       // 結果を出し終えたジョブ(SSE と onJobFinished の両方から来る)
    this.built = null;                              // 最後に作った鍵の結果(direction_key)か、一覧で開いた鍵(/api/keys/<名前>)
    this.checked = null;                            // 最後の key-check の結果
    this.loaded = false;
    const n = this.nLayer, sug = n ? [0.3, 0.5, 0.7].map((f) => Math.round((n - 1) * f)).join(",") : "";
    root.innerHTML = `
      <p class="muted note pp-help">${t("鍵つきの書き込み(keyed): 書き込み行列(ffn_down など)にランク 1 の ΔW = α·d·κᵀ を足し、入力の活性 x が鍵 κ に合う(ρ = κᵀx が大きい)ときだけ方向 d(方向セットの r_L)を書く。κ は側 A(効かせたい状態)の活性の平均 μ から、側 B・dont_touch(触りたくない状態)の張る部分空間を除いたもの(κᵀμ = 1)。鍵を作ったら、作成に使っていない例(subset odd か別の対照集合)で key-check を回し、擬似鍵(ラベルを並べ替えて作った鍵)より選択性が高いことを確かめてから、レシピに足す。")}</p>
      <section class="kx-sec"><h4>${t("鍵を作る")}</h4>
        <div class="pp-in">
          <label class="muted">${t("対照集合")} <select id="kx-contrast"></select></label>
          <input id="kx-cpath" size="34" placeholder="${t("または YAML のパス(入れるとこちらを使う)")}" spellcheck="false">
        </div>
        <div id="kx-cinfo" class="muted note"></div>
        <div class="pp-set">
          <label class="muted" title="${t("鍵の名前(<root>/keys/<model_id>/<名前>.npz)。空なら対照集合の名前")}">${t("名前")} <input id="kx-name" size="16" spellcheck="false"></label>
          <label class="muted" title="${t("鍵を作る層(カンマ区切りか 10-16)。層ごとに κ を作る")}">${t("層")} <input id="kx-layers" size="12" value="${sug}" placeholder="8,14,20" spellcheck="false"></label>
          <label class="muted" title="${t("鍵が読む入力の書き込み行列。ffn = ffn_down(ニューロンの活性)、attn = attn_output(ヘッドの出力の連結)、shared = shared expert、expert = 専門家 e(選ばれた位置だけ)")}">${t("書き込み先")}
            <select id="kx-target"><option value="ffn">ffn</option><option value="attn">attn</option><option value="shared">shared</option><option value="expert">expert</option></select>
            <input id="kx-expert" type="number" min="0" class="hidden" placeholder="e" style="width:4em"></label>
          <label class="muted" title="${t("側 A の位置。空なら対照集合の値(無ければ response)。before:<正規表現> は一致の直前のトークン(推論時と同じ文脈の返事で、そのトークンを決める位置 = 作り方 (b))。response・last・assistant は system_a の文脈(作り方 (a))")}">positions <input id="kx-pos" size="22" list="kx-pos-list" spellcheck="false"></label>
          <datalist id="kx-pos-list"><option value="response"><option value="last"><option value="assistant"><option value="before:(です|ます)。"></datalist>
          <label class="muted" title="${t("Q(触りたくない状態の基底)の作り方。examples = 例ごとの平均、means = 群ごとの平均(よく効くが選択性は粗い)、samples = トークンの標本(選択性が高いが、標本が多いと鍵が消える)")}">q_mode
            <select id="kx-qmode"><option>examples</option><option>means</option><option>samples</option></select></label>
          <label class="muted" title="${t("鍵を作るのに使う質問。even = 偶数番目(key-check は odd で確かめる)、all = 全部(保留集合が残らない)")}">subset
            <select id="kx-subset"><option>even</option><option>odd</option><option>all</option></select></label>
          <label class="muted" title="${t("擬似鍵の本数(側 A と側 B のラベルを釣り合わせて並べ替えて作る、ランダム対照)。key-check の p95 に使う")}">${t("擬似鍵")} <input id="kx-pseudo" type="number" min="0" max="256" value="16" style="width:4.5em"></label>
          <label class="muted" title="${t("返事を生成するときの長さ(返事を与えるなら関係ない)")}">max_tokens <input id="kx-mt" type="number" min="1" max="2048" value="96" style="width:5em"></label>
        </div>
        <details class="kx-adv"><summary class="muted note">${t("細かい設定")}</summary><div class="pp-set">
          <label class="muted" title="${t("側 B の位置。空なら before:<re> のとき not-before:<re>、それ以外は positions と同じ")}">positions_b <input id="kx-posb" size="22" spellcheck="false"></label>
          <label class="muted" title="${t("返事の最初の何トークンを飛ばすか。空なら対照集合の値(無ければ 0)")}">skip <input id="kx-skip" type="number" min="0" style="width:4em"></label>
          <label class="muted" title="${t("モデル自身の返事のファイル(YAML / JSON のリストか {プロンプト: 返事}、サーバの作業場所からの相対)。空なら対照集合の replies か、無介入で生成")}">${t("返事のファイル")} <input id="kx-replies" size="26" spellcheck="false"></label>
          <label class="muted" title="${t("他の介入を当てた状態で鍵を作る(保存したレシピの名前か、ユニット 14:8949:+1 などを空白区切りで)。空なら無介入")}">${t("介入(iv)")} <input id="kx-iv" size="18" spellcheck="false"></label>
          <label class="muted" title="${t("擬似鍵の並べ替えの種")}">seed <input id="kx-seed" type="number" value="0" style="width:4.5em"></label>
        </div></details>
        <div class="pp-go"><button id="kx-go">${t("鍵を作る")}</button> <span class="muted note" id="kx-status"></span></div>
        <div id="kx-prog" class="note mono"></div>
        <div id="kx-build"></div>
      </section>
      <section class="kx-sec"><h4>${t("鍵の分離を確かめる(key-check)")}</h4>
        <div class="pp-set">
          <label class="muted">${t("鍵")} <select id="kc-key"></select></label>
          <label class="muted" title="${t("確かめる対照集合。鍵を作ったものと同じなら、作成と反対側の subset で(重なると AUC が甘くなる)。なしなら鍵に保存した作成時の例で測る(甘い)")}">${t("対照集合")} <select id="kc-contrast"></select></label>
          <label class="muted">subset <select id="kc-subset"><option>odd</option><option>even</option><option>all</option></select></label>
          <label class="muted" title="${t("空なら鍵の全層")}">${t("層")} <input id="kc-layers" size="10" spellcheck="false"></label>
          <label class="muted" title="${t("空なら鍵の作成と同じ")}">max_tokens <input id="kc-mt" type="number" min="1" max="2048" style="width:5em"></label>
          <label class="muted" title="${t("鍵を作ったときに返事を与えたなら、同じファイル")}">${t("返事のファイル")} <input id="kc-replies" size="22" spellcheck="false"></label>
          <label class="muted" title="${t("作成に使った例と重なっていたら測らずに止める")}"><input type="checkbox" id="kc-strict"> strict</label>
        </div>
        <div class="pp-go"><button id="kc-go">${t("key-check を回す")}</button> <span class="muted note" id="kc-status"></span></div>
        <div id="kc-prog" class="note mono"></div>
        <div id="kc-res"></div>
      </section>
      <section class="kx-sec"><h4>${t("保存した鍵")}</h4><div id="kx-keys"><p class="muted note">${t("読み込み中…")}</p></div></section>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#kx-contrast").addEventListener("change", () => this.contrastChanged());
    this.$("#kx-cpath").addEventListener("input", () => this.contrastChanged());
    this.$("#kx-target").addEventListener("change", (ev) => this.$("#kx-expert").classList.toggle("hidden", ev.target.value !== "expert"));
    this.$("#kx-go").addEventListener("click", () => this.runBuild());
    this.$("#kc-go").addEventListener("click", () => this.runCheck());
    root.addEventListener("click", (ev) => this.onClick(ev));
  }

  // direction ビューで「鍵」を開いたとき・ジョブが終わったとき: 対照集合の候補と保存した鍵を取り直す
  async refresh() {
    const [cs, ks] = await Promise.all([getJSON("/api/contrasts").catch(() => []), getJSON("/api/keys").catch(() => [])]);
    this.contrasts = Array.isArray(cs) ? cs : [];
    this.keys = Array.isArray(ks) ? ks : [];
    this.fillContrasts();
    this.fillKeys();
    this.renderKeys();
    this.loaded = true;
  }

  fillContrasts() {
    const opts = (withNone) => {
      const grp = (src, label) => {
        const xs = this.contrasts.filter((c) => c.src === src);
        return xs.length ? `<optgroup label="${esc(label)}">${xs.map((c) => `<option value="${esc(c.path)}">${esc(c.file)}${c.mode === "same" ? " · before" : ""}</option>`).join("")}</optgroup>` : "";
      };
      return (withNone ? `<option value="">${t("(なし: 作成時の例で測る、甘い)")}</option>` : "") + grp("configs", "configs/contrast") + grp("root", "<root>/contrasts");
    };
    for (const [sel, none] of [["#kx-contrast", false], ["#kc-contrast", true]]) {
      const el = this.$(sel), cur = el.value;
      el.innerHTML = opts(none);
      if (cur && [...el.options].some((o) => o.value === cur)) el.value = cur;
      else if (!none) {                                       // 最初は before:<re> の対照集合(作り方 (b) の例)を選んでおく
        const first = this.contrasts.find((c) => c.mode === "same") || this.contrasts[0];
        if (first) el.value = first.path;
      }
    }
    this.contrastChanged();
  }

  fillKeys(select = null) {
    const el = this.$("#kc-key"), cur = select || el.value;
    const ok = this.keys.filter((k) => !k.error);
    el.innerHTML = ok.length ? ok.map((k) => `<option value="${esc(k.name)}">${esc(k.name)}</option>`).join("") : `<option value="">${t("(なし)")}</option>`;
    if (cur && ok.some((k) => k.name === cur)) el.value = cur;
    this.$("#kc-go").disabled = !ok.length;
  }

  // 選んだ対照集合の要約(質問の数・positions・mode・dont_touch)。名前と positions の欄の既定もそれに
  contrastChanged() {
    const path = this.$("#kx-cpath").value.trim() || this.$("#kx-contrast").value;
    const c = this.contrasts.find((x) => x.path === path);
    const info = this.$("#kx-cinfo");
    this.$("#kx-name").placeholder = c ? c.name : t("(対照集合の名前)");
    this.$("#kx-pos").placeholder = c ? c.positions || "response" : t("(対照集合の値)");
    if (!c) { info.textContent = path ? t("(一覧に無いファイル: そのままジョブに渡します)") : t("対照集合がありません(configs/contrast/ か <root>/contrasts/ の YAML)"); return; }
    const mode = c.mode === "same" ? t("作り方 (b): 推論時と同じ文脈(system なし)の返事の、正規表現の直前の位置を側 A、同じ返事のそれ以外を側 B")
      : t("作り方 (a): system_a(や recipe_a)の文脈の活性を側 A、system_b の文脈を側 B");
    info.innerHTML = `${esc(c.label)} · ${c.n_prompts ?? "—"} ${t("問")} · positions <span class="mono">${esc(c.positions || "response")}</span> · ${esc(mode)}${
      c.dont_touch.length ? ` · dont_touch: ${esc(c.dont_touch.join(", "))}` : ""}${c.replies ? ` · ${t("返事つき")}` : ""}`;
  }

  status(which, s, err = false) { const el = this.$(which === "build" ? "#kx-status" : "#kc-status"); el.textContent = s; el.classList.toggle("err", err); }

  // --- 鍵を作る(direction_key のジョブ) ---------------------------------------------------------------------
  buildParams() {
    const v = (s) => this.$(s).value.trim();
    const contrast = v("#kx-cpath") || v("#kx-contrast");
    if (!contrast) { this.status("build", t("対照集合を選んでください"), true); return null; }
    const layers = parseLayers(v("#kx-layers"));
    if (!layers || (this.nLayer && layers.some((L) => L >= this.nLayer))) {
      this.status("build", t("層は 0〜{m} の数をカンマ区切りで(例 8,14,20 か 10-16)", { m: this.nLayer ? this.nLayer - 1 : "?" }), true);
      return null;
    }
    const name = v("#kx-name");
    if (name && !NAME_RE.test(name)) { this.status("build", t("名前は英数字・_・-・. だけにしてください"), true); return null; }
    const tg = v("#kx-target"), e = v("#kx-expert");
    if (tg === "expert" && !/^\d+$/.test(e)) { this.status("build", t("専門家の番号を入れてください"), true); return null; }
    const p = { contrast, layers: layers.join(","), q_mode: v("#kx-qmode"), subset: v("#kx-subset"), pseudo: +v("#kx-pseudo") || 0,
                max_tokens: +v("#kx-mt") || 96 };
    if (tg !== "ffn") p.target = tg === "expert" ? `expert:${e}` : tg;
    if (name) p.name = name;
    for (const [k, sel] of [["positions", "#kx-pos"], ["positions_b", "#kx-posb"], ["replies_a", "#kx-replies"]]) if (v(sel)) p[k] = v(sel);
    const iv = v("#kx-iv");                                   // ユニットの書き方(: を含む)ならリストに、それ以外は保存したレシピの名前かパス
    if (iv) p.iv = iv.includes(":") ? iv.split(/[\s,]+/).filter(Boolean) : iv;
    if (v("#kx-skip") !== "") p.skip = +v("#kx-skip");
    if (v("#kx-seed") !== "" && +v("#kx-seed") !== 0) p.seed = +v("#kx-seed");
    return p;
  }

  async runBuild() {
    const p = this.buildParams();
    if (!p) return;
    this.status("build", t("ジョブを投げています…"));
    try {
      const j = await this.onRun("direction_key", p);
      if (!j) { this.status("build", t("やめました")); return; }
      this.jobs.build = j.id;
      this.status("build", t("鍵を作っています(モデルを流します。GPU を待つことがあります)…"));
      this.follow(j.id, "#kx-prog");
    } catch (e) { this.status("build", `${t("エラー")}: ${e.message}`, true); }
  }

  // --- key-check(direction_key_check のジョブ、ヒストグラムに使うので値は落とさない) -----------------------------------
  checkParams() {
    const v = (s) => this.$(s).value.trim();
    const key = v("#kc-key");
    if (!key) { this.status("check", t("鍵を選んでください"), true); return null; }
    const p = { key, subset: v("#kc-subset") };
    if (v("#kc-contrast")) p.contrast = v("#kc-contrast");
    if (v("#kc-layers")) {
      const Ls = parseLayers(v("#kc-layers"));
      if (!Ls) { this.status("check", t("層は数をカンマ区切りで(空なら鍵の全層)"), true); return null; }
      p.layers = Ls.join(",");
    }
    if (v("#kc-mt")) p.max_tokens = +v("#kc-mt");
    if (v("#kc-replies")) p.replies_a = v("#kc-replies");
    if (this.$("#kc-strict").checked) p.strict = true;
    return p;
  }

  async runCheck() {
    const p = this.checkParams();
    if (!p) return;
    this.status("check", t("ジョブを投げています…"));
    try {
      const j = await this.onRun("direction_key_check", p);
      if (!j) { this.status("check", t("やめました")); return; }
      this.jobs.check = j.id;
      this.status("check", p.contrast ? t("測っています(モデルを流します)…") : t("測っています…"));
      this.follow(j.id, "#kc-prog");
    } catch (e) { this.status("check", `${t("エラー")}: ${e.message}`, true); }
  }

  // ジョブの進み具合をここにも出す(下のジョブのバーと同じ SSE)。終われば結果も出す
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

  // app.js の onJobFinished(と上の SSE)から。同じジョブは 1 回だけ
  async jobEnded(j) {
    const which = j.id === this.jobs.build ? "build" : j.id === this.jobs.check ? "check" : null;
    if (!which) {
      if (j.kind === "direction_key" && j.status === "done" && this.loaded) await this.refresh();   // 別のところで作った鍵も一覧に
      return;
    }
    if (this.handled.has(j.id)) return;
    this.handled.add(j.id);
    this.$(which === "build" ? "#kx-prog" : "#kc-prog").innerHTML = "";
    if (j.status !== "done") {
      const e = j.error || {};
      this.status(which, j.status === "error" ? `${t("失敗")}: ${e.message || ""}${e.hint ? `(${e.hint})` : ""}` : t("止めました"), j.status === "error");
      return;
    }
    const r = j.result || {};
    if (which === "build") {
      this.status("build", t("できました({s} 秒)", { s: fmt(r.seconds, 1) }));
      this.built = { kind: "build", params: j.params || {}, r };
      await this.refresh();
      this.fillKeys(r.name);
      this.renderBuilt();
    } else {
      this.status("check", t("できました({s} 秒)", { s: fmt(r.seconds, 1) }));
      this.checked = r;
      this.renderCheck();
    }
  }

  // --- 結果: 鍵の層ごとの表(作った直後か、一覧で開いた鍵) -------------------------------------------------------
  // 層の行に揃える: [{ layer, kn, gain, q_rank, build_auc, weak }]
  _rows() {
    const b = this.built;
    if (!b) return [];
    if (b.kind === "build") return (b.r.layers || []).map((x) => ({ layer: x.layer, kn: x.kn, gain: x.gain, q_rank: x.q_rank, build_auc: x.build_auc, weak: !!x.weak }));
    const k = b.r, info = k.info || {}, weak = new Set(info.weak || []);
    return (k.layers || []).map((L) => ({ layer: L, kn: (k.kn || {})[L], gain: (k.gain || {})[L], q_rank: (k.q_rank || {})[L],
                                          build_auc: (info.build_auc || {})[L], weak: weak.has(L) }));
  }

  renderBuilt() {
    const el = this.$("#kx-build"), b = this.built;
    if (!b) { el.innerHTML = ""; return; }
    const r = b.r, isBuild = b.kind === "build", info = isBuild ? r : (r.info || {});
    const name = r.name, target = r.target || "ffn", mode = isBuild ? r.mode : info.mode;
    const dirs = this.directions(), want = this.$("#kx-dir") ? this.$("#kx-dir").value : null;
    const pick = want && dirs.includes(want) ? want : dirs.includes(isBuild ? r.contrast : info.contrast) ? (isBuild ? r.contrast : info.contrast) : dirs[0] || "";
    const counts = r.counts || {};
    const head = `<p class="note">${t("鍵")} <b class="mono">${esc(name)}</b> · ${t("書き込み先")} <span class="mono">${esc(target)}</span> · mode <span class="mono">${esc(mode || "—")}</span>${
      mode === "same" ? ` <span class="muted">(${t("推論時と同じ文脈の返事")})</span>` : mode === "system" ? ` <span class="muted">(${t("system_a の文脈")})</span>` : ""} · positions <span class="mono">${esc(r.positions || "—")}</span>${
      (isBuild ? r.positions_b : info.positions_b) ? ` / B <span class="mono">${esc(isBuild ? r.positions_b : info.positions_b)}</span>` : ""} · q_mode ${esc(r.q_mode || "—")} · subset ${esc(r.subset || "—")} · ${
      t("側 A {a} 例 · 側 B {b} 例", { a: counts.a ?? "—", b: counts.b ?? "—" })}${counts.dt ? ` · dont_touch ${counts.dt}` : ""} · ${t("擬似鍵 {n} 本", { n: r.n_pseudo ?? "—" })}${
      isBuild && r.matched ? ` · ${t("正規表現が一致した例 {a} / {n}", { a: r.matched.A, n: r.matched.of })}` : ""}${isBuild ? ` · ${fmt(r.seconds, 1)} ${t("秒")}` : ""}</p>`;
    const add = `<div class="kx-add muted note">${t("書き込む方向")} <select id="kx-dir">${dirs.length ? dirs.map((d) => `<option${d === pick ? " selected" : ""}>${esc(d)}</option>`).join("") : `<option value="">${t("(なし)")}</option>`}</select>
      α <input id="kx-alpha" type="number" step="0.05" value="1" style="width:5em" title="${t("α = 1 で、観測した A − B の平均差 r_L をそのまま書く。負で打ち消す")}">
      <span>${dirs.length ? t("行の「+ レシピ」で、その層の keyed のユニットをレシピに足す") : t("方向がありません(gne direction --contrast で作れます)")}</span></div>`;
    const rows = this._rows().map((x) => `<tr${x.weak ? ` class="rnd"` : ""}><td>L${x.layer}</td><td title="${t("|k| / |μ|: μ のうち側 B の部分空間に直交する割合")}">${fmt(x.kn, 3)}</td><td title="${t("1 / kn: 側 B の漏れが最大これだけ増える")}">${fmt(x.gain, 1)}</td><td>${x.q_rank ?? "—"}</td><td>${fmt(x.build_auc, 3)}</td>
      <td class="l">${x.weak ? `<span class="pp-warn">${t("弱い(kn < 0.05)")}</span>` : ""}</td>
      <td><button class="small" data-kadd="${x.layer}"${dirs.length ? "" : " disabled"} title="${t("この層の keyed のユニットをレシピに足す")}">${t("+ レシピ")}</button></td></tr>`).join("");
    const dropped = isBuild ? (r.dropped || []).map((d) => `L${d.layer}: ${d.error}`) : Object.entries(info.dropped || {}).map(([L, m]) => `L${L}: ${m}`);
    const warns = isBuild ? r.warnings || [] : [];
    const notes = isBuild ? r.notes || [] : [];
    el.innerHTML = `<div class="kx-res">${head}${add}
      <div class="evx-scroll"><table class="evx mono kx-tab"><thead><tr><th>${t("層")}</th><th>kn</th><th>gain</th><th>q_rank</th><th title="${t("作成に使った例での側 A 対 側 B の AUC(甘い)。key-check で保留した例を測る")}">${t("作成時の AUC")}</th><th class="l">${t("注意")}</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      ${dropped.length ? `<p class="err note">${t("鍵にならず外した層")}: ${dropped.map(esc).join(" / ")}</p>` : ""}
      ${warns.length ? `<ul class="evx-notes">${warns.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
      ${notes.length ? `<p class="muted note">${notes.map(esc).join("<br>")}</p>` : ""}
      ${isBuild && (r.samples || []).length ? `<details><summary class="muted note">${t("読んだ返事の例")}</summary>${r.samples.map((s) => `<p class="muted note kx-sample">${esc(s)}</p>`).join("")}</details>` : ""}
      <div class="pp-actions"><button class="small" data-kcheck="${esc(name)}" data-krun="1" title="${t("key-check の欄に入れて回す(作ったときの対照集合と返事、作成と反対側の subset)")}">${t("この鍵で key-check(subset {s})", { s: OPP[r.subset] || "odd" })}</button></div></div>`;
  }

  // --- 結果: key-check の表とヒストグラム ----------------------------------------------------------------------------
  renderCheck() {
    const el = this.$("#kc-res"), r = this.checked;
    if (!r) { el.innerHTML = ""; return; }
    const ov = r.overlap || {};
    const why = [];
    if (ov.subset_overlap) why.push(t("同じ対照集合で subset が重なっています(作成 {a}・確認 {b})", { a: ov.built_subset ?? "—", b: ov.subset ?? "—" }));
    if (ov.hash_overlap) why.push(t("作成に使った例と同じ (system, プロンプト) が {n} 個あります", { n: ov.hash_overlap }));
    const held = r.held_out ? `<p class="note kx-held"><span class="kx-ok">✓</span> ${t("保留した例で測りました(作成 {a}・確認 {b}、重なり 0)", { a: ov.built_subset ?? "—", b: ov.subset ?? "—" })}</p>`
      : `<p class="note err kx-held">⚠ ${t("作成に使った例と重なっています。AUC は甘い値です")}${why.length ? `: ${why.map(esc).join("、")}` : r.contrast ? "" : `: ${t("対照集合を渡していないので、作成時の例ごとの平均で測りました")}`}</p>`;
    const counts = r.counts || {};
    const cnt = Object.entries(counts).map(([g, c]) => `${esc(g.replace(/^dt:/, "dt "))} ${c.examples}/${c.of}`).join(" · ");
    const beat = r.all_beat_pseudo === true ? `<span class="kx-ok">○</span> ${t("どの層も擬似鍵を超えました")}`
      : r.all_beat_pseudo === false ? `<span class="kx-ng">×</span> ${t("擬似鍵を超えない層があります(鍵が本文を拾っているかもしれません。対照集合の本文をそろえる)")}` : "";
    const rows = (r.layers || []).map((x) => {
      const dt = Object.entries(x.dont_touch_median || {}).map(([g, v]) => `${esc(g)} ${fmt(v, 2)}`).join(" · ");
      return `<tr${x.layer === r.best_layer ? ` class="best"` : ""} data-kl="${x.layer}"><td>L${x.layer}${x.layer === r.best_layer ? " ★" : ""}</td>
        <td>${fmt(x.auc, 3)}</td><td>${fmt(x.auc_token, 3)}</td><td class="muted">${fmt((x.pseudo_auc_token || {}).p95 ?? (x.pseudo_auc || {}).p95, 3)}</td><td>${mark(x.beats_pseudo_auc)}</td>
        <td>${fmt(x.selectivity, 3)}</td><td class="muted">${fmt((x.pseudo_selectivity || {}).p95, 3)}</td><td>${mark(x.beats_pseudo)}</td>
        <td class="muted">${fmt(x.b_split_auc, 3)}</td><td class="l">${dt || "—"}</td><td class="muted">${fmt(x.kn, 3)}</td></tr>`;
    }).join("");
    const hists = (r.layers || []).map((x) => `<figure class="kx-fig" data-kl="${x.layer}">${histSVG(x.hist, { title: `L${x.layer} · AUC ${fmt(x.auc_token ?? x.auc, 3)} · ${t("選択性")} ${fmt(x.selectivity, 2)}` })}</figure>`).join("");
    const groups = new Set();
    for (const x of r.layers || []) for (const g of Object.keys(x.hist || {})) if (g !== "edges") groups.add(g);
    let di = 0;
    const legend = [...groups].sort((a, b) => (a === "A" ? -1 : b === "A" ? 1 : a === "B" ? -1 : b === "B" ? 1 : a < b ? -1 : 1))
      .map((g) => `<span><i class="kx-sw ${groupCls(g, g.startsWith("dt:") ? di++ : 0)}"></i>${esc(groupLabel(g))}</span>`).join("");
    el.innerHTML = `<div class="kx-res">
      <p class="note">${t("鍵")} <b class="mono">${esc(r.key)}</b> · ${t("対照集合")} ${esc(r.contrast || t("(なし)"))}${r.subset ? ` (subset ${esc(r.subset)})` : ""} · mode ${esc(r.mode || "—")} · positions <span class="mono">${esc(r.positions || "—")}</span>${
        r.positions_b ? ` / B <span class="mono">${esc(r.positions_b)}</span>` : ""}${cnt ? ` · ${cnt}` : ""} · ${t("擬似鍵 {n} 本", { n: r.n_pseudo ?? "—" })} · ${fmt(r.seconds, 1)} ${t("秒")}</p>
      ${held}${beat ? `<p class="note">${beat}</p>` : ""}
      <div class="evx-scroll"><table class="evx mono kx-tab kx-check"><thead><tr><th>${t("層")}</th>
        <th title="${t("例ごとの ρ の平均での側 A 対 側 B の AUC")}">${t("AUC(例)")}</th><th title="${t("トークンごとの ρ での AUC")}">${t("AUC(トークン)")}</th>
        <th title="${t("擬似鍵のトークン単位の AUC の 95 パーセンタイル")}">${t("擬似 p95")}</th><th title="${t("トークン単位の AUC が擬似鍵の p95 を超えるか")}">${t("AUC で超える")}</th>
        <th title="${t("例ごとの ρ の平均の差(側 A − 側 B)。書き込む量 α(κᵀx)d の A と B の差")}">${t("選択性")}</th><th title="${t("擬似鍵の選択性の 95 パーセンタイル")}">${t("擬似 p95")}</th>
        <th title="${t("選択性が擬似鍵の p95 を超え、例の AUC が 0.5 を超えるか(beats_pseudo)")}">${t("擬似鍵を超える")}</th>
        <th title="${t("参考: 側 B を 2 つに分けて作った擬似差の鍵の AUC(0.5 付近なら鍵が側 B の中の差を拾っていない)")}">${t("B 2 分割")}</th>
        <th class="l" title="${t("dont_touch の群の ρ の中央値(0 に近いほど触らない)")}">${t("dont_touch 中央値")}</th><th>kn</th></tr></thead><tbody>${rows}</tbody></table></div>
      <h4 class="note">${t("層ごとの ρ = κᵀx の分布(群ごとの割合。縦の点線: 0 = 何も足さない、1 = 側 A の平均)")}</h4>
      <div class="evx-legend note kx-legend">${legend}</div>
      <div class="kx-hists">${hists}</div>
      ${(r.warnings || []).length ? `<ul class="evx-notes">${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
      ${(r.notes || []).length ? `<p class="muted note">${r.notes.map(esc).join("<br>")}</p>` : ""}</div>`;
  }

  // --- 保存した鍵の一覧 ------------------------------------------------------------------------------------------
  renderKeys() {
    const el = this.$("#kx-keys");
    if (!this.keys.length) { el.innerHTML = `<p class="muted note">${t("鍵はまだありません(上の「鍵を作る」か gne direction key)")}</p>`; return; }
    const rows = this.keys.map((k) => (k.error
      ? `<tr><td class="l">${esc(k.name)}</td><td class="l err" colspan="8">${esc(k.error)}</td></tr>`
      : `<tr><td class="l">${esc(k.name)}</td><td>${esc(k.target)}</td><td class="l">${k.layers.map((L) => `L${L}`).join(" ")}</td>
        <td class="l">${k.layers.map((L) => fmt((k.kn || {})[L], 2)).join(" ")}</td><td>${esc(k.q_mode)}</td><td class="l">${esc(k.positions)}</td><td>${esc(k.subset)}</td>
        <td>${k.n_pseudo}</td><td>${(k.counts || {}).a ?? "—"} / ${(k.counts || {}).b ?? "—"}</td>
        <td><button class="small" data-kview="${esc(k.name)}">${t("見る")}</button> <button class="small" data-kcheck="${esc(k.name)}">${t("確かめる")}</button></td></tr>`)).join("");
    el.innerHTML = `<div class="evx-scroll"><table class="evx mono kx-keys"><thead><tr><th class="l">${t("名前")}</th><th>${t("書き込み先")}</th><th class="l">${t("層")}</th><th class="l">kn</th><th>q_mode</th><th class="l">positions</th><th>subset</th><th>${t("擬似鍵")}</th><th>${t("例 A / B")}</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  async view(name) {
    try {
      const k = await getJSON(`/api/keys/${encodeURIComponent(name)}`);
      this.built = { kind: "saved", r: { ...k, n_pseudo: k.n_pseudo } };
      this.renderBuilt();
      this.$("#kx-build").scrollIntoView({ block: "nearest" });
    } catch (e) { this.status("build", `${t("エラー")}: ${e.message}`, true); }
  }

  // 鍵を key-check の欄に入れる: 作ったときの対照集合(作った直後なら分かる)と反対側の subset、同じ返事のファイル
  prepareCheck(name) {
    this.fillKeys(name);
    const b = this.built && this.built.r && this.built.r.name === name ? this.built : null;
    const k = this.keys.find((x) => x.name === name);
    const subset = (b && b.r.subset) || (k && k.subset) || "even";
    this.$("#kc-subset").value = OPP[subset] || "odd";
    if (b && b.kind === "build") {
      const p = b.params || {}, sel = this.$("#kc-contrast");
      if (p.contrast && [...sel.options].some((o) => o.value === p.contrast)) sel.value = p.contrast;
      this.$("#kc-replies").value = p.replies_a || "";
    }
    this.$("#kc-go").scrollIntoView({ block: "nearest" });
  }

  onClick(ev) {
    const a = ev.target.closest("[data-kadd]");
    if (a) {
      const b = this.built;
      if (!b) return;
      const dir = this.$("#kx-dir") ? this.$("#kx-dir").value : "", alpha = +(this.$("#kx-alpha") || {}).value;
      if (!dir) return;
      const { kind, expert } = targetKind(b.r.target);
      this.onAddKeyed({ layer: +a.dataset.kadd, kind, expert, alpha: Number.isFinite(alpha) ? alpha : 1, direction: dir, key: b.r.name });
      a.textContent = t("足しました");
      return;
    }
    const c = ev.target.closest("[data-kcheck]");
    if (c) { this.prepareCheck(c.dataset.kcheck); if (c.dataset.krun) this.runCheck(); return; }
    const v = ev.target.closest("[data-kview]");
    if (v) this.view(v.dataset.kview);
  }
}
