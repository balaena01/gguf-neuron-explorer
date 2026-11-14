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
    if (b && list.contains(b) && !b.disabled) onSelect(b.dataset.rtab ?? b.dataset.tab);
  });
  list.addEventListener("keydown", (ev) => {
    const xs = tabs(), i = xs.indexOf(document.activeElement);
    if (i < 0) return;
    const j = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: xs.length - 1 }[ev.key];
    if (j === undefined) return;
    ev.preventDefault();
    const b = xs[(j + xs.length) % xs.length];
    onSelect(b.dataset.rtab ?? b.dataset.tab);
    b.focus();
  });
}
