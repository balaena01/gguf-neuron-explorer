// 作る画面(CHiPS・鍵・FiSH・評価)の共通の形: 左に手順(作業の一覧)、右に 選んだ手順だけ(入力 → 実行 → 結果)。
// 画面の部品はそのまま使い、手順ごとに「どの部品を見せるか」だけを切り替える(部品を作り直さないので、入れかけの欄も消えない)。
// 手順には状態の印(済・実行中・まだ)を付ける。共通の設定は画面の上に 1 回だけ置き、それを使う手順のときだけ見せる。
import { t } from "./i18n.js";
import { esc } from "./panels.js";
import { wireTabs } from "./ui.js";

const MARK = { done: ["ok", "済"], run: ["run", "実行中"], todo: ["wait", "まだ"], warn: ["warn", "要確認"] };

export class Steps {
  // root: 画面の入れ物(中身はすでに描いてある)。steps: [{ key, label, els: () => [要素], shared: 共通の設定を使うか, sep: 前に区切り }]
  // shared: 共通の設定の要素(無ければ null)。store: 選んだ手順を覚える localStorage の鍵。onSelect(key): 手順を替えたあと
  constructor(root, { steps, shared = null, store = null, label = "", onSelect = null }) {
    this.root = root;
    this.steps = steps;
    this.store = store;
    this.onSelect = onSelect || (() => {});
    this.state = {};
    const build = document.createElement("div");
    build.className = "build";
    const nav = document.createElement("div");
    nav.className = "steps";
    nav.setAttribute("role", "tablist");
    nav.setAttribute("aria-orientation", "vertical");
    nav.setAttribute("aria-label", label || t("手順"));
    nav.innerHTML = steps.map((s) => `<button type="button" role="tab" data-step="${esc(s.key)}" aria-selected="false" tabindex="-1"${s.sep ? ' class="sep"' : ""}>
      <span class="st-l">${s.label}</span><span class="pill" data-mark></span></button>`).join("");
    const stage = document.createElement("div");
    stage.className = "stage";
    stage.append(...root.childNodes);
    this.title = document.createElement("h3");
    this.title.className = "stage-h";
    if (shared) { stage.prepend(shared); shared.classList.add("stage-shared"); shared.after(this.title); }
    else stage.prepend(this.title);
    this.shared = shared;
    build.append(nav, stage);
    root.append(build);
    this.nav = nav;
    this.stage = stage;
    wireTabs(nav, (k) => this.select(k));
    let first = steps[0].key;
    try { const v = store && localStorage.getItem(store); if (v && steps.some((s) => s.key === v)) first = v; } catch { /* 覚えられなくても使える */ }
    this.select(first);
  }

  get current() { return this.cur; }

  // 手順を開く(ほかの手順の部品は隠す)
  select(key) {
    const s = this.steps.find((x) => x.key === key);
    if (!s) return;
    this.cur = key;
    try { if (this.store) localStorage.setItem(this.store, key); } catch { /* 覚えられなくても使える */ }
    for (const b of this.nav.querySelectorAll("[data-step]")) {
      const on = b.dataset.step === key;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    this.title.textContent = s.title || s.label.replace(/<[^>]*>/g, "");
    if (this.shared) this.shared.hidden = !s.shared;
    this.apply();
    this.onSelect(key);
  }

  // 結果を描き直したあとなどに、いまの手順に合わせて見せる部品を選び直す
  apply() {
    const show = new Set(this.steps.find((x) => x.key === this.cur).els().filter(Boolean));
    for (const s of this.steps) for (const el of s.els()) if (el) el.hidden = !show.has(el);
  }

  // 状態の印。state: done(済)・run(実行中)・todo(まだ)・warn・null(印なし)
  mark(key, state) {
    if (this.state[key] === state) return;
    this.state[key] = state;
    const pill = this.nav.querySelector(`[data-step="${CSS.escape(key)}"] [data-mark]`);
    if (!pill) return;
    const m = MARK[state];
    pill.className = m ? `pill ${m[0]}` : "pill";
    pill.textContent = m ? t(m[1]) : "";
  }
}
