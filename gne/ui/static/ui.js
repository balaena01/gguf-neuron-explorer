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
