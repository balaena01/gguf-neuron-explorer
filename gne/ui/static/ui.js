// 画面のあちこちで使う小さな決まり。
// 日本語などの変換を確定する Enter / Esc は、検索・保存・ジャンプを走らせない(変換中は isComposing。
// 古い Safari では isComposing が先に戻るので、変換中に届く keyCode 229 も見る)。
export const composing = (ev) => ev.isComposing || ev.keyCode === 229;
export const plainEnter = (ev) => ev.key === "Enter" && !composing(ev);
export const plainEscape = (ev) => ev.key === "Escape" && !composing(ev);
