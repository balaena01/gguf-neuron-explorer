// (i18n の t は確かめる窓の文字に使う)
import { t } from "./i18n.js";
// 画面のあちこちで使う小さな決まり。
// 日本語などの変換を確定する Enter / Esc は、検索・保存・ジャンプを走らせない(変換中は isComposing。
// 古い Safari では isComposing が先に戻るので、変換中に届く keyCode 229 も見る)。
export const composing = (ev) => ev.isComposing || ev.keyCode === 229;
export const plainEnter = (ev) => ev.key === "Enter" && !composing(ev);
export const plainEscape = (ev) => ev.key === "Escape" && !composing(ev);

// タブ(role="tab" のボタンの並び)。押すか、←→・Home・End で選ぶ(選んだタブにフォーカスが移り、そのまま選ばれる)。
// 選んだタブの見た目と aria-selected は onSelect の側で付ける
export function wireTabs(list, onSelect) {
  if (!list) return;
  const tabs = () => [...list.querySelectorAll('[role="tab"]:not([disabled])')];
  list.addEventListener("click", (ev) => {
    const b = ev.target.closest('[role="tab"]');
    if (b && list.contains(b) && !b.disabled) onSelect(b.dataset.rtab ?? b.dataset.step ?? b.dataset.tab);
  });
  list.addEventListener("keydown", (ev) => {
    const xs = tabs(), i = xs.indexOf(document.activeElement);
    if (i < 0) return;
    const vert = list.getAttribute("aria-orientation") === "vertical";
    const j = { [vert ? "ArrowDown" : "ArrowRight"]: i + 1, [vert ? "ArrowUp" : "ArrowLeft"]: i - 1, Home: 0, End: xs.length - 1 }[ev.key];
    if (j === undefined) return;
    ev.preventDefault();
    const b = xs[(j + xs.length) % xs.length];
    onSelect(b.dataset.rtab ?? b.dataset.step ?? b.dataset.tab);
    b.focus();
  });
}

// 長い説明: 最初の文を短い 1 行にして、開くと全文が読めるようにする(<details> なので、閉じていてもページ内検索で見つかる)。
// 短い文はそのまま。text は t() を通した文(中の <code> などはそのまま出す)
export function noteHTML(text, cls = "muted note") {
  const s = String(text);
  if (s.length < 60) return `<p class="${cls}">${s}</p>`;
  const i = s.search(/。|\. /);
  let head = i >= 0 ? s.slice(0, i + 1) : s;
  if (head.length > 56) {
    const cut = Math.max(head.lastIndexOf("、", 44), head.lastIndexOf("(", 44), head.lastIndexOf(", ", 44), head.lastIndexOf(": ", 44));
    head = `${head.slice(0, cut > 12 ? cut : 44)}…`;
  }
  if (head === s) return `<p class="${cls}">${s}</p>`;
  return `<details class="more ${cls}"><summary>${head}</summary><p>${s}</p></details>`;
}

// 確かめる窓(ブラウザの confirm の代わり。<dialog> を showModal で開く)。はい なら true。
// 安全な方(やめる)に初めからフォーカスを置く。Esc や外側の操作で閉じたときは やめる と同じ
let askDialog = null;
export function askConfirm(message, { ok = null, cancel = null } = {}) {
  if (typeof window.gneAsk === "function") return Promise.resolve(!!window.gneAsk(message));   // 自動テスト用(窓を出さずに答える)
  if (!askDialog) {
    askDialog = document.createElement("dialog");
    askDialog.id = "ask";
    askDialog.setAttribute("aria-labelledby", "ask-msg");
    document.body.append(askDialog);
  }
  const d = askDialog;
  if (d.open) d.close("cancel");                          // 前の窓が開いたままなら、やめたことにする
  d.innerHTML = `<form method="dialog"><p id="ask-msg"></p><div class="ask-btns">
    <button value="cancel" autofocus></button><button value="ok" class="main"></button></div></form>`;
  d.querySelector("#ask-msg").textContent = message;
  d.querySelector("button[value=cancel]").textContent = cancel || t("やめる");
  d.querySelector("button[value=ok]").textContent = ok || t("はい");
  d.returnValue = "";
  delete d.dataset.answered;
  return new Promise((resolve) => {
    d.addEventListener("close", () => { const yes = d.returnValue === "ok"; d.returnValue = ""; resolve(yes); }, { once: true });
    d.showModal();
  });
}

// 層の欄(どの画面も同じ部品): 「11-23」「8,14,19」「all」の書き方。読めない書き方なら枠を赤紫にする(押す前に気づけるように)
const LAYERS_RE = /^\s*(all|\d+(\s*-\s*\d+)?(\s*,\s*\d+(\s*-\s*\d+)?)*)?\s*$/i;
export function layersInput(id, { value = "", placeholder = null, title = null, size = 12 } = {}) {
  const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  return `<input class="layers mono" id="${id}" size="${size}" value="${esc(value)}" placeholder="${esc(placeholder ?? t("例 11-23 か 8,14,19"))}"`
    + ` title="${esc(title ?? t("層: カンマ区切りか、続いた層は 11-23 のように範囲で"))}" spellcheck="false" autocomplete="off">`;
}
document.addEventListener("input", (ev) => {
  const el = ev.target;
  if (el instanceof HTMLInputElement && el.classList.contains("layers")) el.classList.toggle("bad", !LAYERS_RE.test(el.value));
});

// 対照集合の欄(パスを書く欄は、どの画面も同じ候補の一覧から選べる): configs/contrast/ と <root>/contrasts/ の対照集合を
// <datalist id="contrast-list"> に 1 回だけ読み、欄に list を付ける。一覧に無いパスもそのまま書ける
let contrastList = null;
export function useContrastList(input) {
  if (!input) return;
  input.setAttribute("list", "contrast-list");
  if (contrastList) return;
  contrastList = document.createElement("datalist");
  contrastList.id = "contrast-list";
  document.body.append(contrastList);
  fetch("/api/contrasts").then((r) => (r.ok ? r.json() : [])).then((cs) => {
    contrastList.innerHTML = (Array.isArray(cs) ? cs : []).map((c) => `<option value="${String(c.path).replace(/"/g, "&quot;")}">${String(c.label || "").replace(/</g, "&lt;")}</option>`).join("");
  }).catch(() => { /* 一覧が無くてもパスは書ける */ });
}
