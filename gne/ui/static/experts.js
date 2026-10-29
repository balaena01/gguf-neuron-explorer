// EXPERT パネル(CHANGE-05 G): MoE の専門家を 1 つ開く、NEURON・HEAD パネルと同列の右パネル。
// 住所・選び方(routing)・ルーターのバイアス、使われ方(stats act のコーパスでの選択率・選ばれたときの重み・よく選ばれるトークン)、
// 保存済みの専門家の対照分析(gne experts contrast)での順位と指標、ニューロンの上位(クリックで NEURON パネル)。
// レシピには専門家全体の ×0(消す)・×2(強める)と、ルーターのバイアスがある層なら route ±1 で足す(3 列チャットで試せる)。
import { getJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc, tokHTML } from "./panels.js";

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : (+x).toFixed(d));
const sgn = (x, d = 3) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
const pct = (x) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${(x * 100).toFixed(1)}%`);
const ROUTING = { softmax_topk: "softmax top-k", sigmoid_topk: "sigmoid top-k", group_topk: "group top-k" };

export class ExpertPanel {
  constructor(root, { onRecipe, inRecipe, recipeSize, hasBias, onNeuron, onOverlay, onChat }) {
    this.root = root;
    this.onRecipe = onRecipe;          // ({ layer, kind: "expert", expert, mode }, 値 | null) → レシピに足す・外す
    this.inRecipe = inRecipe;          // ({ layer, kind: "expert", expert, mode }) → レシピの中の値(gain は k、route は delta)か null
    this.recipeSize = recipeSize;      // () → レシピのユニットの数(0 なら 3 列チャットは押せない)
    this.hasBias = hasBias;            // (L) → その層にルーターのバイアスがあるか(route を出すか)
    this.onNeuron = onNeuron;          // ({ layer, n }) → NEURON パネルで開く(n は通し番号)
    this.onOverlay = onOverlay;        // (field) → 地図の色を expert_usage / experts:<name> に
    this.onChat = onChat;              // ({ layer, n }) → レシピ全体を 3 列チャットで(n = チャットを置くニューロン)
    this.d = null;
    this._token = 0;
  }

  async open({ layer, expert }) {
    const tok = ++this._token;
    this.root.innerHTML = `<p class="muted">${t("{id} を読み込み中…", { id: `L${layer} E${expert}` })}</p>`;
    let d;
    try { d = await getJSON("/api/expert", { layer, expert }); }
    catch (e) { if (tok === this._token) this.root.innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    if (tok !== this._token) return;
    this.d = d;
    this.render();
  }

  render() {
    const d = this.d;
    if (!d) return;
    const best = d.contrasts[0];
    const stats = best ? `#${best.rank}/${best.of} · z ${fmt(best.z, 2)} · null ${fmt(best.null_pct, 2)} · ${esc(best.name)}` : t("専門家の対照分析なし");
    const bias = d.router_bias !== null && d.router_bias !== undefined
      ? ` · <span title="${t("ルーターのバイアス(選択にだけ効き、重みには入らない)")}">bias ${sgn(d.router_bias, 3)}</span>` : "";
    this.root.innerHTML = `
      <div class="n-head">
        <div class="n-id mono">L${d.layer} E${d.expert} <span class="muted h-kind" title="${t("ルーターの選び方。トークンごとに {k} / {n} 個の専門家を選ぶ", { k: d.n_expert_used, n: d.n_expert })}">${
          esc(ROUTING[d.routing] || d.routing || "")} · ${d.n_expert_used}/${d.n_expert}${bias}</span></div>
        <div class="n-stats mono muted">${stats}</div>
        <div class="n-row x-act">${this.recipeButtons(d)}
          <button class="small" id="x-chat"${this.recipeSize() ? "" : ` disabled title="${t("先に ×0 か ×2 でレシピに足す")}"`}${
            this.recipeSize() ? ` title="${t("レシピ全体を 3 列チャットで試す(− 列は ×k が 2 − k、route は符号を反転)")}"` : ""}>${t("3 列チャットで試す")}</button></div>
        <p class="muted note">${t("×k = この専門家の書き込み(ffn_down)を k 倍するスケールパッチ。選ばれたときの出力だけが変わる(選ばれ方は変わらない)。当てるたびにモデルを読み直す(0.3〜1 秒)。")}${
          this.hasBias(d.layer) ? ` ${t("route = ルーターのバイアスに足す(選ばれやすさだけが変わる)。")}` : ""}</p>
      </div>
      ${this.usage(d)}
      ${this.contrasts(d)}
      ${this.neurons(d)}`;
    for (const b of this.root.querySelectorAll("[data-xk]")) {
      b.addEventListener("click", () => { this.toggle("gain", +b.dataset.xk); this.render(); });
    }
    for (const b of this.root.querySelectorAll("[data-xr]")) {
      b.addEventListener("click", () => { this.toggle("route", +b.dataset.xr); this.render(); });
    }
    const ch = this.root.querySelector("#x-chat");
    if (ch) ch.addEventListener("click", () => this.onChat({ layer: d.layer, n: d.expert * d.n_ff + (d.neurons.length ? d.neurons[0].neuron : 0) }));
    for (const b of this.root.querySelectorAll("[data-ov]")) b.addEventListener("click", () => this.onOverlay(b.dataset.ov));
    for (const tr of this.root.querySelectorAll("tr[data-n]")) {
      tr.addEventListener("click", () => this.onNeuron({ layer: d.layer, n: d.expert * d.n_ff + +tr.dataset.n }));
    }
  }

  q(mode) { return { layer: this.d.layer, kind: "expert", expert: this.d.expert, mode }; }

  // 同じ値でもう一度押すと外す
  toggle(mode, v) {
    const cur = this.inRecipe(this.q(mode));
    this.onRecipe(this.q(mode), cur !== null && cur === v ? null : v);
  }

  // ×0(消す)・×2(強める)と、バイアスのある層なら route −1 / +1。recipe タブで別の値にしたものは横に出す
  recipeButtons(d) {
    const k = this.inRecipe(this.q("gain")), r = this.inRecipe(this.q("route"));
    const gb = (x, label) => {
      const on = k !== null && k === x;
      return `<button data-xk="${x}" class="${on ? `on ${x < 1 ? "minus" : "plus"}` : ""}" title="${on ? t("もう一度押すとレシピから外す")
        : t("レシピに専門家全体の ×{k} で足す(倍率は recipe タブで変えられる)", { k: x })}">${on ? `${t("レシピ")} ×${x}` : label}</button>`;
    };
    const rb = (x) => {
      const on = r !== null && r === x;
      return `<button data-xr="${x}" class="${on ? `on ${x < 0 ? "minus" : "plus"}` : ""}" title="${on ? t("もう一度押すとレシピから外す")
        : t("レシピにルーターのバイアス {d} で足す(選ばれやすさが変わる)", { d: sgn(x, 0) })}">${on ? `${t("レシピ")} route ${sgn(x, 1)}` : `route ${sgn(x, 0)}`}</button>`;
    };
    const other = [k !== null && k !== 0 && k !== 2 ? `${t("レシピ")} ×${fmt(k, 2)}` : "", r !== null && r !== 1 && r !== -1 ? `${t("レシピ")} route ${sgn(r, 2)}` : ""]
      .filter(Boolean).map((x) => `<span class="muted mono h-cur">${x}</span>`).join("");
    return `${gb(0, t("×0(消す)"))}${gb(2, t("×2(強める)"))}${this.hasBias(d.layer) ? rb(-1) + rb(1) : ""}${other}`;
  }

  // 使われ方(gne stats act の MoE の層): 選択率、観測数、選ばれたときの重みの平均、よく選ばれるトークン
  usage(d) {
    const u = d.usage;
    if (!u) {
      return `<h4>${t("使われ方")}</h4><p class="muted note">${t("この層の活性統計がありません({cmd})。選択率と、ニューロンの上位の q999 はこれで出ます。", { cmd: `<code>gne stats act --layers ${d.layer}</code>` })}</p>`;
    }
    const uni = d.n_expert_used / d.n_expert;
    const toks = u.top_tokens.length ? `<div class="chips x-toks">${u.top_tokens.map((x) =>
      `<span class="chip" title="${t("{n} 回選ばれた", { n: x.count })}">${tokHTML(x.token).replace(/^<span class="tok">|<\/span>$/g, "")}<i>${x.count}</i></span>`).join("")}</div>` : "";
    return `<h4>${t("使われ方(stats act のコーパス)")} <button class="small" data-ov="expert_usage" title="${t("地図の MoE の層を、専門家ごとの選択率で色付けする")}">${t("地図で見る")}</button></h4>
      <table class="kv mono x-use">
        <tr><td title="${t("この専門家が選ばれたトークンの割合")}">${t("選択率")}</td><td class="${u.rate >= 1.5 * uni ? "hot" : ""}">${pct(u.rate)}</td>
          <td class="muted" title="${t("どの専門家も同じだけ選ばれたときの割合(k / E)")}">${t("一様 {p}", { p: pct(uni) })}</td></tr>
        <tr><td>${t("観測")}</td><td>${u.n_routed.toLocaleString()} / ${u.n_tokens.toLocaleString()}</td><td class="muted">${t("トークン")}</td></tr>
        <tr><td title="${t("選ばれたときの実効のルーティング重み(出力に掛かる重み)の平均")}">${t("重みの平均")}</td><td>${fmt(u.weight_mean, 3)}</td><td></td></tr>
      </table>
      ${u.reliable ? "" : `<p class="note minus x-few">${t("観測が少ない({n} トークン、200 未満)ので、選択率と分位点(q999)は目安です。gne stats act --tokens を増やすと確かになります。", { n: u.n_routed })}</p>`}
      ${toks ? `<div class="muted note">${t("よく選ばれるトークン(数字 = 回数、位置 0 を除く)")}</div>${toks}` : ""}`;
  }

  // 保存済みの専門家の対照分析(gne experts contrast)ごとの順位と指標(HEAD パネルの Contrast と同じ形)
  contrasts(d) {
    if (!d.contrasts.length) {
      return `<h4>${t("対照分析")}</h4><p class="muted note">${t("この専門家を含む専門家の対照分析はまだありません({cmd})。", { cmd: "<code>gne experts contrast --contrast configs/contrast/….yaml</code>" })}</p>`;
    }
    const row = (k, v, tip, cls = "") => `<tr title="${esc(tip)}"><td>${k}</td><td class="mono ${cls}">${v}</td><td class="muted">${esc(tip)}</td></tr>`;
    const sc = (x) => (x >= 0 ? "plus" : "minus");
    return d.contrasts.map((c) => `<div class="h-con x-c">
        <h4><b class="mono">${esc(c.name)}</b> <span class="mono${c.null_pct >= 0.95 ? " hot" : ""}">#${c.rank}/${c.of} · z ${fmt(c.z, 2)} · null ${fmt(c.null_pct, 2)}</span>
          <button class="small" data-ov="${esc(`experts:${c.name}`)}" title="${t("地図の MoE の層を、この対照分析の専門家の z で色付けする")}">${t("地図で見る")}</button></h4>
        <table class="kv x-met">
          ${row(t("選択率 A → B"), `${pct(c.rate_a)} → ${pct(c.rate_b)}`, t("選択率: ペルソナあり(A) → なし(B)"))}
          ${row("usage_shift", sgn(c.usage_shift, 3), t("選択率の差(A − B)"), sc(c.usage_shift))}
          ${row("weight_shift", sgn(c.weight_shift, 3), t("選ばれたときの重みの差(A − B)"), sc(c.weight_shift))}
          ${row("contrib_cos", sgn(c.contrib_cos, 3), t("専門家の出力の差と方向(または残差の差)の cos"), sc(c.contrib_cos))}
          ${row("contrib_norm", fmt(c.contrib_norm, 3), t("専門家の出力の差の大きさ"))}
        </table></div>`).join("") +
      `<p class="muted note">${t("z = 中立どうしの対照に対する z、null = 対照の何割より大きいか(琥珀 = 0.95 以上)。")}</p>`;
  }

  // ニューロンの上位(活性統計があれば q999 × ‖W_down‖ = 書き込みの大きさ、無ければ ‖W_down‖ の順)
  neurons(d) {
    if (!d.neurons.length) return "";
    const rows = d.neurons.map((x) => `<tr data-n="${x.neuron}" title="${t("NEURON パネルで開く(L{L} E{e} N{n})", { L: d.layer, e: d.expert, n: x.neuron })}">
      <td>N${x.neuron}</td><td>${fmt(x.q999, 3)}</td><td>${fmt(x.down_norm, 3)}</td><td>${fmt(x.write, 3)}</td></tr>`).join("");
    return `<h4>${d.usage ? t("ニューロンの上位(q999 × ‖W_down‖ の順)") : t("ニューロンの上位(‖W_down‖ の順)")}</h4>
      <div class="x-wrap"><table class="rank mono x-top"><thead><tr><th>N</th><th>q999</th><th>‖W_down‖</th><th title="${t("q999 × ‖W_down‖(選ばれたときに残差へ書ける大きさの目安)")}">write</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="muted note">${t("行をクリックすると NEURON パネル。専門家のニューロンの add は方向押し、×k の gain は選ばれたときだけ効く。")}</p>`;
  }
}
