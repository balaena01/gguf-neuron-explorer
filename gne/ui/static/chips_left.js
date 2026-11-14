// direction ビューの「鍵」の「CHiPS で残った場面から作る」: CHiPS のレシピを当てても消したい振る舞いが残った A 側の質問だけを集めて、
// 鍵つきの書き込み(keyed)の対照集合の JSON を作る。
// 1. CHiPS の方向・対照集合(既定は方向を作った対照集合)・レシピ(保存したレシピ)を選び、「behave を回す」(ジョブ behave: レシピを当てて
//    対照集合の A 側の質問に答えさせ、判定の正規表現(既定は validate.match)と judge で数える)。結果は保存され、一覧から選び直せる。
// 2. 判定(正規表現か judge のラベル、振る舞いが出た種の割合 min_rate)と positions(既定 before:<validate.match>、英字があれば (?i))を
//    決めて「JSON を作る」(POST /api/keyed-from-behave = gne direction key-from-behave。<root>/keyed/ に保存)→ 鍵のタブの入力欄に入る
//    (書きかけを消す前に確かめるのは、ひな形と同じ)。警告は入力欄の確かめと同じく赤(error)・黄色(warn)で出す。
// 3. 「方向から key-check まで」へ進む(keyed.direction の方向と、keyed.iv のレシピは入力欄の JSON から自動で入る)。
// 読み込み(方向と対照集合の要約・behave の結果の一覧・JSON を作った結果)は通し番号を持ち、遅れて届いた古い結果は捨てる(eval.js と同じ)。
import { getJSON, postJSON } from "./api.js";
import { LANG, t } from "./i18n.js";
import { esc } from "./panels.js";
import { noteHTML } from "./ui.js";

const NAME_RE = /^[\w.-]+$/;
const pct = (x) => (x === null || x === undefined || !Number.isFinite(+x) ? "—" : `${Math.round(+x * 1000) / 10}%`);

export class ChipsLeft {
  constructor(root, { onRun, dirs, contrasts, putSpec, runPipeline } = {}) {
    this.root = root;
    this.onRun = onRun || (async () => null);       // (kind, params) → Promise<ジョブ | null>
    this.dirs = dirs || (() => []);                 // /api/directions の項目
    this.contrasts = contrasts || (() => []);       // /api/contrasts の項目
    this.putSpec = putSpec || (() => false);        // (spec, 説明, <root>/keyed/ のファイル名) → 入力欄に入れたら true(書きかけを消す前に確かめる)
    this.runPipeline = runPipeline || (() => {});   // 「方向から key-check まで」
    this.recipes = [];
    this.list = [];                                 // 保存した behave の結果(/api/behave)
    this.src = null;                                // 方向と対照集合の要約(/api/chips-source)
    this.made = null;                               // 最後に JSON を作った結果
    this.seq = { src: 0, list: 0, make: 0 };        // 通し番号(後から始めた読み込みだけを使う)
    this.userPick = 0;                              // behave の結果の選択を手で変えた回数(ジョブの後の自動の選択で上書きしない)
    this.jobId = null;
    root.innerHTML = `<summary class="kb-sum"><b>${t("CHiPS で残った場面から作る")}</b> <span class="muted note">${t("(behave の結果 → keyed の対照集合の JSON)")}</span></summary>
      ${noteHTML(t("CHiPS のレシピを当てても消したい振る舞いがまだ出る A 側の質問だけを集めて、keyed の対照集合の JSON を作る。1. 方向とレシピを選んで「behave を回す」(レシピを当てて A 側の質問に答えさせる)か、保存した結果を選ぶ。2. 判定と positions を決めて「JSON を作る」(<root>/keyed/ に保存し、下の入力欄に入る)。3. 層を入れて「方向から key-check まで」(方向は keyed.direction、介入(iv)は keyed.iv のレシピが自動で入る)。できた keyed のユニットは CHiPS のレシピに足して 1 つの LoRA にする。"), "muted note pp-help")}
      <div class="pp-set">
        <label class="muted" title="${t("CHiPS で使った方向(gne direction で作ったもの)。出力の keyed.direction")}">${t("CHiPS の方向")} <select id="kb-dir"></select></label>
        <label class="muted" title="${t("B 側の質問・validate.match・A 側の質問を取る対照集合。既定は方向を作った対照集合")}">${t("対照集合")} <select id="kb-contrast"></select></label>
        <label class="muted" title="${t("CHiPS のレシピ(保存したレシピ)。behave で当て、出力の keyed.iv に入れる(鍵はこのレシピを当てた状態で作る)")}">${t("レシピ")} <select id="kb-recipe"></select></label>
      </div>
      <div id="kb-src" class="note"></div>
      <details class="kb-adv"><summary class="muted note">${t("behave の細かい設定")}</summary><div class="pp-set">
        <label class="muted" title="${t("返事の長さ(振る舞いが返事の後ろのほうで出るなら長めに)")}">${t("返事の長さ")} <input class="w5" id="kb-mt" type="number" min="1" max="4096" value="256"></label>
        <label class="muted" title="${t("0 なら greedy(種は 1 つ)。0 より大きければ種ごとに 1 回ずつサンプリングする")}">${t("温度")} <input class="w45" id="kb-temp" type="number" min="0" max="2" step="0.1" value="0"></label>
        <label class="muted" title="${t("温度ありのとき、種ごとに 1 回(1,2,3)")}">${t("種")} <input id="kb-seeds" size="8" placeholder="0" spellcheck="false"></label>
        <label class="muted" title="${t("返事に一致を数える正規表現(大文字小文字を区別)。空なら対照集合の validate.match")}">${t("判定の正規表現")} <input id="kb-bmatch" size="22" spellcheck="false"></label>
        <label class="muted" title="${t("保存名(<root>/behave/<model_id>/<名前>.json)。空なら <レシピ>_<対照集合>")}">${t("名前")} <input id="kb-bname" size="16" spellcheck="false"></label>
        <label class="muted"><input type="checkbox" id="kb-think"> ${t("思考あり")}</label>
        <label class="muted" title="${t("読み手の判定のラベル(返事の最後に出てきたものを採る)")}">${t("判定のラベル")} <input id="kb-labels" size="8" value="R,P,C" spellcheck="false"></label>
      </div><textarea id="kb-judge" class="mono" rows="3" spellcheck="false" placeholder="${t("judge の文面(任意)。{q} に質問、{a} に返事が入る。返事の最後に labels のどれかを書かせる")}"></textarea></details>
      <div class="pp-go"><button class="main" id="kb-run" title="${t("選んだレシピを当てて、対照集合の A 側の質問に答えさせる(ジョブ behave)")}">${t("behave を回す")}</button> <span class="muted note" id="kb-status"></span></div>
      <div id="kb-prog" class="note mono"></div>
      <div class="pp-set">
        <label class="muted" title="${t("保存した behave の結果(新しい順)")}">${t("behave の結果")} <select id="kb-behave"></select></label>
        <button class="small" id="kb-reload" title="${t("一覧を読み直す")}">${t("読み直す")}</button>
      </div>
      <div id="kb-binfo" class="note"></div>
      <div class="pp-set">
        <label class="muted" title="${t("「残った」とみなす判定")}">${t("判定")} <select id="kb-method"><option value="match">${t("正規表現")}</option><option value="label">${t("judge のラベル")}</option></select></label>
        <label class="muted" id="kb-match-l" title="${t("返事にこの正規表現が一致すれば「出た」。空なら behave の判定の正規表現、無ければ validate.match(大文字小文字を無視)")}">${t("正規表現")} <input id="kb-match" size="22" spellcheck="false"></label>
        <label class="muted hidden" id="kb-label-l" title="${t("「出た」とみなすラベル(カンマ区切りで複数可)")}">${t("ラベル")} <input id="kb-label" size="6" placeholder="P" spellcheck="false"></label>
        <label class="muted" title="${t("振る舞いが出た種の割合がこれ以上の質問を入れる(1 = 全部の種で出た質問だけ)")}">${t("残ったとみなす割合")} <input class="w45" id="kb-minrate" type="number" min="0.05" max="1" step="0.05" value="1"></label>
        <label class="muted" title="${t("書き込みたい位置。空なら before:<validate.match>(英字があれば (?i) を付ける)。一致が広すぎると警告が出るので、振る舞いの書き始めに当たる正規表現を入れる")}">${t("書く位置")} <input id="kb-pos" size="26" spellcheck="false"></label>
        <label class="muted" title="${t("JSON と鍵の名前。空なら <behave の名前>_keyed")}">${t("JSON の名前")} <input id="kb-name" size="16" spellcheck="false"></label>
        <label class="muted" title="${t("behave の返事を JSON の replies に入れる(鍵の作成は返事を作らずにそれを読む)")}"><input type="checkbox" id="kb-replies" checked> ${t("返事を入れる")}</label>
        <label class="muted" title="${t("behave の結果が、選んだレシピを当てて A 側の質問に答えたものでなければ作らない")}"><input type="checkbox" id="kb-strict"> ${t("当てたレシピの結果に限る")}</label>
      </div>
      <div class="pp-go"><button id="kb-make" title="${t("残った質問を集めて <root>/keyed/ に JSON を保存し、入力欄に入れる(モデルは動かさない)")}">${t("JSON を作る")}</button> <span class="muted note" id="kb-mstatus"></span></div>
      <div id="kb-res"></div>`;
    this.$ = (s) => root.querySelector(s);
    this.$("#kb-dir").addEventListener("change", () => { this.$("#kb-contrast").value = ""; this.source(); });
    this.$("#kb-contrast").addEventListener("change", () => this.source());
    this.$("#kb-recipe").addEventListener("change", () => this.behaveInfo());
    this.$("#kb-behave").addEventListener("change", () => { this.userPick += 1; this.behaveChanged(); });
    this.$("#kb-method").addEventListener("change", () => this.methodChanged());
    this.$("#kb-run").addEventListener("click", () => this.run());
    this.$("#kb-make").addEventListener("click", () => this.make());
    this.$("#kb-reload").addEventListener("click", () => this.loadList());
    root.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-kbpipe]")) this.runPipeline();
    });
  }

  status(sel, s, err = false) {
    const el = this.$(sel);
    el.textContent = s;
    el.classList.toggle("err", err);
  }

  // 鍵のタブを開いたとき・ジョブが終わったとき: 方向・対照集合・レシピの候補を入れ直し(選んでいたものは残す)、要約と一覧を取り直す
  async refresh() {
    const fill = (sel, items, first) => {
      const el = this.$(sel), cur = el.value;
      el.innerHTML = (first ? `<option value="">${esc(first)}</option>` : "") + items.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join("");
      if (cur && items.some(([v]) => v === cur)) el.value = cur;
      return el;
    };
    const dirs = this.dirs() || [];
    fill("#kb-dir", dirs.map((d) => [d.name, `${d.name}${d.positions ? ` (${d.positions})` : ""}`]), dirs.length ? null : t("(方向がありません)"));
    const cs = this.contrasts() || [];
    fill("#kb-contrast", cs.map((c) => [c.path, c.label]), t("(方向を作った対照集合)"));
    let rs = [];
    try { rs = await getJSON("/api/recipes"); } catch { rs = []; }
    this.recipes = Array.isArray(rs) ? rs : [];
    const chips = (r) => (r.units || []).some((u) => u.mode === "proj");
    const ordered = [...this.recipes.filter(chips), ...this.recipes.filter((r) => !chips(r))];
    fill("#kb-recipe", ordered.map((r) => [r.name, `${r.name}${chips(r) ? " · CHiPS" : ""}`]), ordered.length ? null : t("(レシピがありません)"));
    await Promise.all([this.source(), this.loadList()]);
  }

  // 方向と対照集合の要約(A 側・B 側の質問の数、validate.match、positions の既定)
  async source() {
    const seq = ++this.seq.src;
    const direction = this.$("#kb-dir").value, contrast = this.$("#kb-contrast").value;
    const el = this.$("#kb-src");
    if (!direction && !contrast) { this.src = null; el.textContent = t("CHiPS の方向を選んでください"); this.placeholders(); return; }
    let r;
    try { r = await getJSON("/api/chips-source", { direction, contrast }); }
    catch (e) {
      if (seq !== this.seq.src) return;
      this.src = null; el.innerHTML = `<span class="err">${esc(e.message)}</span>`; this.placeholders(); return;
    }
    if (seq !== this.seq.src) return;                           // 後から選び直した方向・対照集合の読み込みに任せる
    this.src = r;
    const warns = [];
    if (!r.direction_exists && r.direction) warns.push(t("方向 {d} がありません", { d: r.direction }));
    if (r.system_a) warns.push(t("この対照集合の A 側は system_a(か recipe_a)で作った状態です。behave と鍵の作成は system なしの返事を読むので、system で起こした振る舞いには向きません"));
    if (r.same_sha === false) warns.push(t("方向はこの対照集合とは中身の違うファイルから作られています"));
    if (!r.match) warns.push(t("validate.match が無いので、判定の正規表現と positions を入れてください"));
    el.innerHTML = `<span class="muted">${t("対照集合 {c} · A 側 {a} 問 · B 側 {b} 問(dont_touch の ordinary に入る)", { c: esc(r.contrast), a: r.n_a, b: r.n_b })}${
      r.match ? ` · validate.match <span class="mono">${esc(r.match)}</span>` : ""}</span>${warns.map((w) => `<div class="pp-warn">${esc(w)}</div>`).join("")}`;
    this.placeholders();
  }

  selectedBehave() {
    const name = this.$("#kb-behave").value;
    return this.list.find((x) => x.name === name) || null;
  }

  // 欄の既定の表示(空のときに使われる値)
  placeholders() {
    const s = this.src, b = this.selectedBehave();
    this.$("#kb-bmatch").placeholder = s && s.match ? s.match : t("(validate.match)");
    this.$("#kb-match").placeholder = (b && b.match) || (s && s.match) || "";
    this.$("#kb-pos").placeholder = (s && s.positions) || "before:…";
    this.$("#kb-name").placeholder = b ? `${b.name}_keyed` : "";
    const rc = this.$("#kb-recipe").value, cn = s ? s.contrast : "";
    this.$("#kb-bname").placeholder = rc && cn ? `${rc}_${cn}` : "";
  }

  // 保存した behave の結果の一覧。select は選びたい名前(ジョブの後)。読み込みの間に手で選び直していれば、それを残す
  async loadList(select = null) {
    const seq = ++this.seq.list, pick0 = this.userPick, sel = this.$("#kb-behave");
    let list;
    try { list = await getJSON("/api/behave"); } catch { list = null; }
    if (seq !== this.seq.list || !Array.isArray(list)) return;   // 後から始めた読み込みに任せる
    const now = sel.value;
    this.list = list;
    sel.innerHTML = list.length ? list.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}(${esc((x.updated || "").replace("T", " ").slice(0, 16))} · ${x.n_prompts ?? "—"} ${t("問")})</option>`).join("")
      : `<option value="">${t("(まだありません。「behave を回す」で作る)")}</option>`;
    const want = this.userPick !== pick0 ? now : (select || now);
    if (want && list.some((x) => x.name === want)) sel.value = want;
    this.behaveChanged();
  }

  // behave の結果を選んだとき: 要約と、判定の既定(judge だけの結果ならラベル)
  behaveChanged() {
    const b = this.selectedBehave();
    if (b && b.judge && !b.match && this.$("#kb-method").value === "match") { this.$("#kb-method").value = "label"; this.methodChanged(); }
    this.placeholders();
    this.behaveInfo();
  }

  behaveInfo() {
    const b = this.selectedBehave(), el = this.$("#kb-binfo");
    if (!b) { el.textContent = ""; return; }
    const per = (b.counts || []).map((c, i) => {
      const bits = [];
      if (c.match !== undefined) bits.push(t("一致 {m}/{n}", { m: c.match, n: c.n }));
      if (c.labels) bits.push(Object.entries(c.labels).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(" "));
      if (c.loop) bits.push(t("繰り返し {n}", { n: c.loop }));
      if (c.empty) bits.push(t("空 {n}", { n: c.empty }));
      return `${t("種")} ${esc(String((b.seeds || [])[i] ?? i))}: ${esc(bits.join(" · ") || "—")}`;
    }).join(" / ");
    const warns = [];
    const rc = this.$("#kb-recipe").value;
    if (!b.has_recipe) warns.push(t("この結果はレシピを当てずに答えたものです"));
    else if (rc && Array.isArray(b.recipe_names) && b.recipe_names.length && !b.recipe_names.includes(rc)) warns.push(t("この結果は選んだレシピ {r} ではなく {o} を当てたものです", { r: rc, o: b.recipe_names.join(", ") }));
    if (!b.has_prompts) warns.push(t("質問の記録が無い前の版の結果です(CLI の gne direction key-from-behave で --prompts を渡す)"));
    el.innerHTML = `<span class="muted">${t("{n} 問 · max_tokens {m} · {how}", { n: b.n_prompts ?? "—", m: b.max_tokens ?? "—", how: b.temperature ? t("温度 {x}", { x: b.temperature }) : "greedy" })}${
      b.match ? ` · ${t("判定の正規表現")} <span class="mono">${esc(b.match)}</span>` : ""}${b.judge ? ` · judge (${esc((b.labels || []).join(","))})` : ""}${
      b.recipe_names && b.recipe_names.length ? ` · ${t("レシピ")} ${esc(b.recipe_names.join(", "))}` : ""}<br>${per}</span>${warns.map((w) => `<div class="pp-warn">${esc(w)}</div>`).join("")}`;
  }

  methodChanged() {
    const lab = this.$("#kb-method").value === "label";
    this.$("#kb-match-l").classList.toggle("hidden", lab);
    this.$("#kb-label-l").classList.toggle("hidden", !lab);
  }

  // --- behave を回す(ジョブ behave) ---------------------------------------------------------------------------------
  async run() {
    const v = (s) => this.$(s).value.trim();
    const recipe = v("#kb-recipe"), s = this.src;
    if (!s) { this.status("#kb-status", t("CHiPS の方向と対照集合を選んでください"), true); return; }
    if (!recipe) { this.status("#kb-status", t("レシピを選んでください"), true); return; }
    const name = v("#kb-bname") || `${recipe}_${s.contrast}`;
    if (!NAME_RE.test(name)) { this.status("#kb-status", t("名前は英数字・_・-・. だけにしてください"), true); return; }
    const match = v("#kb-bmatch") || s.match || null;
    const judge = this.$("#kb-judge").value.trim();
    if (!match && !judge) { this.status("#kb-status", t("判定の正規表現か judge の文面を入れてください"), true); return; }
    if (judge && !(judge.includes("{q}") && judge.includes("{a}"))) { this.status("#kb-status", t("judge の文面には {q}(質問)と {a}(返事)が要ります"), true); return; }
    const p = { prompts: s.contrast_path, recipes: [recipe], name, max_tokens: +v("#kb-mt") || 256 };
    if (match) p.match = match;
    if (judge) { p.judge = judge; p.labels = v("#kb-labels") || "R,P,C"; }
    const temp = +v("#kb-temp");
    if (temp > 0) {
      p.temperature = temp;
      if (v("#kb-seeds")) {
        if (!/^\d+(\s*,\s*\d+)*$/.test(v("#kb-seeds"))) { this.status("#kb-status", t("種は 1,2,3 のように書きます"), true); return; }
        p.seeds = v("#kb-seeds").split(",").map((x) => +x.trim());
      }
    }
    if (this.$("#kb-think").checked) p.thinking = true;
    this.status("#kb-status", t("ジョブを投げています…"));
    let j;
    try { j = await this.onRun("behave", p); } catch (e) { this.status("#kb-status", `${t("エラー")}: ${e.message}`, true); return; }
    if (!j) { this.status("#kb-status", t("やめました")); return; }
    this.jobId = j.id;
    this.status("#kb-status", t("レシピを当てて A 側の質問に答えさせています(GPU を待つことがあります)…"));
    this.follow(j.id, this.userPick);
  }

  // ジョブの進み具合(下のジョブのバーと同じ SSE)。終われば一覧を読み直して、できた結果を選ぶ(その間に手で選び直していなければ)
  follow(id, pick0) {
    const el = this.$("#kb-prog");
    const show = (j) => {
      if (!["queued", "running"].includes(j.status)) { el.innerHTML = ""; return; }
      el.innerHTML = `<span class="muted note">${t(j.status === "queued" ? "待機中(進み具合と「止める」は下の帯)" : "実行中(進み具合と「止める」は下の帯)")}</span>`;
    };
    show({ status: "queued" });
    let es;
    try { es = new EventSource(`/api/jobs/${id}/events`); } catch { return; }
    es.onmessage = (m) => {
      const j = JSON.parse(m.data);
      show(j);
      if (!["done", "error", "cancelled"].includes(j.status)) return;
      es.close();
      if (id !== this.jobId) return;
      if (j.status === "done") {
        const r = j.result || {};
        this.status("#kb-status", t("できました({s} 秒)", { s: Number.isFinite(+j.seconds) ? (+j.seconds).toFixed(1) : "—" }));
        this.loadList(this.userPick === pick0 ? r.name : null);
      } else {
        const e = j.error || {};
        this.status("#kb-status", j.status === "error" ? `${t("失敗")}: ${e.message || ""}${e.hint ? `(${e.hint})` : ""}` : t("止めました"), j.status === "error");
      }
    };
    es.onerror = () => { es.close(); el.innerHTML = ""; };
  }

  // --- JSON を作る(POST /api/keyed-from-behave) ------------------------------------------------------------------------
  async make() {
    const v = (s) => this.$(s).value.trim();
    const behave = v("#kb-behave"), recipe = v("#kb-recipe");
    if (!behave) { this.status("#kb-mstatus", t("behave の結果を選んでください"), true); return; }
    if (!recipe) { this.status("#kb-mstatus", t("レシピを選んでください"), true); return; }
    const name = v("#kb-name");
    if (name && !NAME_RE.test(name)) { this.status("#kb-mstatus", t("名前は英数字・_・-・. だけにしてください"), true); return; }
    const p = { behave, recipe, direction: v("#kb-dir") || null, contrast: v("#kb-contrast") || (this.src && this.src.contrast_path) || null,
                min_rate: +v("#kb-minrate") || 1, replies: this.$("#kb-replies").checked, strict: this.$("#kb-strict").checked };
    if (this.$("#kb-method").value === "label") {
      const labs = v("#kb-label").split(",").map((x) => x.trim()).filter(Boolean);
      if (!labs.length) { this.status("#kb-mstatus", t("ラベルを入れてください(例 P)"), true); return; }
      p.labels = labs;
    } else if (v("#kb-match")) p.match = v("#kb-match");
    if (v("#kb-pos")) p.positions = v("#kb-pos");
    if (name) p.name = name;
    const seq = ++this.seq.make;
    this.status("#kb-mstatus", t("作っています…"));
    let r;
    try { r = await postJSON("/api/keyed-from-behave", p); }
    catch (e) {
      if (seq !== this.seq.make) return;
      this.made = null;
      this.status("#kb-mstatus", t("作れませんでした"), true);
      this.$("#kb-res").innerHTML = `<div class="err">${esc(e.message)}</div>`;
      return;
    }
    if (seq !== this.seq.make) return;                          // 後から押した「JSON を作る」の結果に任せる
    this.made = r;
    let put = false;
    if (r.spec) {
      const a = r.spec.about || {};
      const spec = { ...r.spec };
      delete spec.about;
      put = await this.putSpec(spec, ((a[LANG] || a.ja || {}).text) || "", `${r.name}.json`);
    }
    this.status("#kb-mstatus", !r.path ? t("残った質問が無いので、JSON は作りませんでした")
      : put ? t("{p} に保存して、入力欄に入れました", { p: `<root>/keyed/${r.name}.json` }) : t("{p} に保存しました(入力欄はそのまま)", { p: `<root>/keyed/${r.name}.json` }), !r.path);
    this.render();
  }

  render() {
    const r = this.made, el = this.$("#kb-res");
    if (!r) { el.innerHTML = ""; return; }
    const j = r.judge || {};
    const how = j.method === "label" ? t("judge のラベル {l}", { l: (j.labels || []).join(", ") }) : t("正規表現 {m}", { m: j.match || "—" }) + (j.ignore_case ? t("(大文字小文字を無視)") : "");
    const sh = r.position_share || {};
    const issues = (r.issues || []).map((x) => `<div class="${x.level === "error" ? "err" : "pp-warn"}">${esc(x.message)}</div>`).join("");
    const rows = (r.extracted || []).map((x) => `<tr><td>${x.i}</td><td class="l">${esc(x.question)}</td><td>${pct(x.rate)}</td><td>${x.starts}</td><td>${x.loop ? t("繰り返し") : ""}</td>
      <td class="l"><details><summary class="muted note">${t("返事")}</summary><p class="muted note kx-sample">${esc(x.reply || "")}</p></details></td></tr>`).join("");
    el.innerHTML = `<div class="kx-res">
      <p class="note">${t("残った質問 {a} 問 · 一部の種だけ {p} 問 · 残らなかった {n} 問(全 {t} 問・種 {s} 個、{how}、min_rate {r})", { a: (r.extracted || []).length, p: r.n_partial, n: r.n_none, t: r.n_prompts, s: r.n_seeds, how: esc(how), r: j.min_rate })}<br>
      positions <span class="mono">${esc(r.positions)}</span> · ${t("返事のトークンのうち書き込みたい位置になる割合の見込み {s}(25% 以上は広すぎる)", { s: pct(sh.share) })} · ${
        t("書き込みたい位置の見込み: 鍵の作成 {a} 例・{b} 位置、key-check {c} 例・{d} 位置", { a: r.halves.even.examples, b: r.halves.even.positions, c: r.halves.odd.examples, d: r.halves.odd.positions })} · ${
        t("dont_touch の ordinary {n} 問", { n: r.ordinary.n })}</p>
      ${issues}${(r.notes || []).length ? `<p class="muted note">${r.notes.map(esc).join("<br>")}</p>` : ""}
      ${rows ? `<div class="evx-scroll"><table class="evx mono kx-tab kb-rows"><thead><tr><th>#</th><th class="l">${t("残った質問")}</th><th title="${t("振る舞いが出た種の割合")}">${t("割合")}</th><th title="${t("返事の中の書き込みたい位置の見込み")}">${t("位置")}</th><th></th><th class="l"></th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}
      ${r.path ? `<div class="pp-actions"><button class="small" data-kbpipe="1" title="${t("入力欄の JSON(方向は keyed.direction、介入は keyed.iv)で、上の層の欄の層に鍵を作り、key-check まで回す")}">${t("方向から key-check まで")}</button>
        <span class="muted note">${t("層は「鍵を作る」の層の欄で決める")}</span></div>` : ""}</div>`;
  }
}
