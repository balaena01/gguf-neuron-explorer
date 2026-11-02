// eval ビューの「実地テスト」(fieldtest のジョブ、gne/fieldtest.py)。本番と同じ動かし方(llama-server、思考あり)で、長い作業・ツール呼び出し・
// 質問集・perplexity を測り、ファイルどうしを比べる。生成した文章は画面に出さない(数字だけ)。
import { getJSON } from "./api.js";
import { t } from "./i18n.js";
import { esc } from "./panels.js";

const SECTION = { long: "長い作業", tools: "ツール呼び出し", ppl: "perplexity" };

function fmtVal(row, v) {
  if (v === null || v === undefined) return "—";
  if (row.section === "ppl") return (+v).toFixed(3);
  if (row.metric === "tokens") return String(Math.round(v));
  if (row.metric === "tail_distinct") return (+v).toFixed(2);
  return `${(100 * v).toFixed(1)}%`;
}

export class FieldtestPanel {
  constructor(root, { onRun } = {}) {
    this.root = root;
    this.onRun = onRun || (async () => null);       // (kind, params) → ジョブ(見積もりの確認つき)
    this.jobId = null;
    this.data = null;
    root.innerHTML = `
      <p class="muted note">${t("本番と同じ動かし方(llama-server、思考あり、サンプリングあり)で、長い作業・ツール呼び出し・質問集・perplexity を測り、ファイルどうしを比べます。eval で見えない、長い思考の途中のループや、ツール呼び出しの崩れを見ます。焼き込んだファイルもそのまま測れます。走っている間は読み込んでいるモデルを GPU から下ろすので、チャットなどは待たされます。27B で質問集 200 問を足して 1 ファイル 30〜45 分かかります。")}</p>
      <div class="ft-form">
        <label class="muted" title="${t("測るファイル(GGUF のパス)。空ならいま読んでいるモデル")}">${t("測るファイル")} <input id="ft-model" size="44" spellcheck="false" placeholder="${t("空ならいま読んでいるモデル")}"></label>
        <label class="muted" title="${t("比べるファイル(例 元のモデル)。空なら測るだけ")}">${t("比べるファイル")} <input id="ft-against" size="44" spellcheck="false" placeholder="${t("空なら比べない")}"></label>
        <label class="muted" title="${t("測るファイルに当てる LoRA(パス か パス:倍率)。比べるファイルが空なら、同じファイルの LoRA なしと比べる")}">LoRA <input id="ft-lora" size="30" spellcheck="false"></label>
        <label class="muted" title="${t("llama-server のパス。空なら GNE_LLAMA_SERVER か PATH から探す")}">llama-server <input id="ft-server" size="30" spellcheck="false" placeholder="${t("空なら自動で探す")}"></label>
        <label class="muted" title="${t("自分の質問集。1 行に 名前=ファイル(JSON か YAML のリスト)。ループ・本文が空・上限で切れた割合を測る")}">${t("質問集")} <textarea id="ft-prompts" rows="2" cols="40" spellcheck="false" placeholder="name=/path/questions.json"></textarea></label>
        <label class="muted" title="${t("質問集の返事の本文に当てる正規表現。一致率を出す")}">${t("判定")} <input id="ft-match" size="24" spellcheck="false"></label>
        <label class="muted" title="${t("長い作業の返事の上限")}">${t("長い作業のトークン")} <input id="ft-long" type="number" min="16" max="65536" value="3072"></label>
        <label class="muted" title="${t("ツール呼び出しの返事の上限")}">${t("ツールのトークン")} <input id="ft-tool" type="number" min="16" max="65536" value="1536"></label>
        <label class="muted" title="${t("質問集の返事の上限")}">${t("質問集のトークン")} <input id="ft-ptok" type="number" min="16" max="65536" value="1536"></label>
        <label class="muted" title="${t("参照テキストの perplexity も測る(gne でファイルを読み直す)")}"><input id="ft-ppl" type="checkbox" checked> perplexity</label>
        <button id="ft-go">${t("実地テストを走らせる")}</button> <span class="muted note" id="ft-status"></span></div>
      <div id="ft-list"></div><div id="ft-result"></div>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#ft-go").addEventListener("click", () => this.run());
  }

  status(s, err = false) { const el = this.$("#ft-status"); el.textContent = s; el.classList.toggle("err", err); }

  params() {
    const val = (s) => this.$(s).value.trim();
    const num = (s) => Math.max(16, Math.floor(+this.$(s).value || 0));
    const p = { long_tokens: num("#ft-long"), tool_tokens: num("#ft-tool"), prompt_tokens: num("#ft-ptok"), ppl: this.$("#ft-ppl").checked };
    for (const [k, sel] of [["model", "#ft-model"], ["against", "#ft-against"], ["lora", "#ft-lora"], ["llama_server", "#ft-server"], ["match", "#ft-match"]]) {
      if (val(sel)) p[k] = val(sel);
    }
    const lines = this.$("#ft-prompts").value.split("\n").map((x) => x.trim()).filter(Boolean);
    if (lines.some((x) => !x.includes("="))) { this.status(t("質問集は 1 行に 名前=ファイル で書いてください"), true); return null; }
    if (lines.length) p.prompts = lines;
    return p;
  }

  async run() {
    const p = this.params();
    if (!p) return;
    this.status(t("ジョブを投げています…"));
    try {
      const j = await this.onRun("fieldtest", p);
      if (!j) { this.status(t("やめました")); return; }
      this.jobId = j.id;
      this.status(t("走っています。進み具合は下のジョブのバーに出ます(終わるとここに結果を出します)"));
    } catch (e) { this.status(`${t("エラー")}: ${e.message}`, true); }
  }

  async jobEnded(j) {
    const mine = j.id === this.jobId;
    if (j.status === "done") {
      if (mine) this.status(t("できました({s} 秒)", { s: Math.round(j.seconds || 0) }));
      await this.refresh();
      const name = j.result && j.result.name;
      if (name && (mine || !this.data)) await this.open(name);
    } else if (mine) {
      this.status(j.status === "error" ? `${t("失敗")}: ${(j.error && j.error.message) || ""}` : t("止めました"), j.status === "error");
    }
  }

  async refresh() {
    let list = [];
    try { list = await getJSON("/api/fieldtests"); } catch { list = []; }
    const el = this.$("#ft-list");
    if (!list.length) { el.innerHTML = `<p class="muted note">${t("まだありません")}</p>`; return; }
    el.innerHTML = `<table class="evx mono ft-past"><thead><tr><th class="l">${t("名前")}</th><th class="l">${t("対象")}</th><th class="l">${t("比較")}</th><th>${t("結論")}</th><th class="l">${t("更新")}</th></tr></thead><tbody>${
      list.map((x) => `<tr data-ft="${esc(x.name)}" class="${this.data && this.data.name === x.name ? "on" : ""}"><td class="l">${esc(x.name)}</td><td class="l">${esc(x.target || "—")}</td><td class="l">${esc(x.reference || "—")}</td>
        <td class="${x.same_level === false ? "ng" : x.same_level ? "ok" : ""}">${x.same_level === null || x.same_level === undefined ? "—" : x.same_level ? t("同程度") : t("悪化 {n} 項目", { n: x.n_worse })}</td>
        <td class="l">${esc((x.updated || "").replace("T", " ").slice(0, 16))}</td></tr>`).join("")}</tbody></table>`;
    for (const tr of el.querySelectorAll("tr[data-ft]")) tr.addEventListener("click", () => this.open(tr.dataset.ft));
  }

  async open(name) {
    try { this.data = await getJSON(`/api/fieldtests/${encodeURIComponent(name)}`); }
    catch (e) { this.$("#ft-result").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    this.render();
    for (const tr of this.root.querySelectorAll("tr[data-ft]")) tr.classList.toggle("on", tr.dataset.ft === name);
  }

  // 比べる相手が無いとき: 対象の数字だけ
  rows(d) {
    if (d.compare) return d.compare;
    const run = d.runs[d.runs.length - 1], out = [];
    const add = (section, o) => { for (const [k, v] of Object.entries(o || {})) if (!k.startsWith("n") || k === "no_tool_ok") out.push({ section, metric: k, label: k, reference: null, target: v, worse: false }); };
    add("long", run.long); add("tools", run.tools);
    for (const [k, m] of Object.entries(run.prompts || {})) add(`prompts:${k}`, m);
    add("ppl", run.ppl);
    return out;
  }

  render() {
    const d = this.data, st = d.settings || {}, rows = this.rows(d), hasRef = !!d.reference;
    const title = (s) => {
      if (s.startsWith("prompts:")) { const k = s.slice(8); return t("質問集 {k}({n} 問)", { k: esc(k), n: (st.prompts || {})[k] ?? "?" }); }
      if (s === "long") return t("長い作業({n} 問、{m} トークンまで)", { n: st.n_long ?? "?", m: st.long_tokens ?? "?" });
      if (s === "tools") return t("ツール呼び出し({n} 問)", { n: st.n_tools ?? "?" });
      return t(SECTION[s] || s);
    };
    let cur = null;
    const body = rows.map((r) => {
      const head = r.section !== cur ? `<tr class="ft-sec"><td class="l" colspan="${hasRef ? 3 : 2}">${title(r.section)}</td></tr>` : "";
      cur = r.section;
      return `${head}<tr class="${r.worse ? "ft-worse" : ""}"><td class="l">${esc(t(r.label))}</td>${hasRef ? `<td>${fmtVal(r, r.reference)}</td>` : ""}<td>${fmtVal(r, r.target)}${r.worse ? ` <span class="ng">${t("悪化")}</span>` : ""}</td></tr>`;
    }).join("");
    this.$("#ft-result").innerHTML = `<h4>${t("{name} の結果", { name: esc(d.name) })} <span class="muted note mono">${esc((d.updated || "").replace("T", " ").slice(0, 16))} · ${t("{n} 分", { n: Math.round((d.seconds || 0) / 60) })}</span></h4>
      <p class="note">${t("対象")}: <span class="mono">${esc(d.target.file)}</span>${d.target.lora ? ` + LoRA` : ""}${hasRef ? ` · ${t("比較")}: <span class="mono">${esc(d.reference.file)}</span>` : ""}</p>
      ${d.same_level === null || d.same_level === undefined ? "" : `<p class="note ${d.same_level ? "ok" : "ng"}">${d.same_level ? t("すべて同程度です") : t("悪化した項目があります")}</p>`}
      <table class="evx mono ft-table"><thead><tr><th class="l"></th>${hasRef ? `<th>${t("比較")}</th>` : ""}<th>${t("対象")}</th></tr></thead><tbody>${body}</tbody></table>
      <p class="muted note">${t("同程度の目安: 割合は 2 問ぶん、perplexity は 1% までの差。種は同じですが、1 問の違いはサンプリングの揺れで起きます。生成した文章は結果と同じ場所の .replies.json にあります(画面には出しません)。")}</p>`;
  }
}
