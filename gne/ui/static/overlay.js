// 色の重ね合わせ(別冊 §4.3、色は §13.2)。層ごとの値(Float32Array、NaN = データ無し)を点の RGBA に写す。
// a は「目立たせる度合い」: 光る点(床・閾値より上)は 255 にして、遠くから見ても消えないようにする。
// 光る点が全体の 2% を超えるときは、全層で同じ係数で a を下げる(emphasize、F39)。
import { t } from "./i18n.js";
import { COLORS } from "./map.js";

const AMBER_LO = [168, 108, 30], AMBER_HI = [255, 196, 84];

export const FIELDS = {
  none: { label: t("なし(構造だけ)"), needs: null },
  sweep: { label: t("sweep の mass"), needs: "sweep" },
  sweep_null: { label: t("sweep の null_pct"), needs: "sweep" },
  steer_kl: { label: t("steer lens の KL"), needs: "steer_lens" },
  logit_peak: { label: t("logit lens の最大 |z|"), needs: "logit_lens" },
  act_q999: { label: t("活性の p99.9"), needs: "act_stats" },
  named: { label: t("名前付き"), needs: "named" },
};

export function fieldLabel(name) {
  if (name.startsWith("cos:")) return t("逆引き {d} との cos", { d: name.slice(4) });
  if (name.startsWith("writers:")) return t("書き手 {d} の寄与", { d: name.slice(8) });
  if (name.startsWith("heads:")) return t("ヘッドの対照 {d} の z", { d: name.slice(6) });
  if (name === "fires_heads") return t("ヘッドの注目(fires の選んだトークンへ)");
  if (name.startsWith("search:")) return t("検索「{q}」", { q: name.slice(7) });
  if (name.startsWith("fires:")) return `fires ${name.split(":").slice(2).join(":")}`;   // fires:<id>:<見出し>
  return FIELDS[name] ? FIELDS[name].label : name;
}

// この overlay のデータがある層(model の coverage と名前から)
export function layersWith(name, model) {
  const cov = model.coverage;
  if (name === "none" || name.startsWith("search:") || name.startsWith("fires:")) return [];
  if (name.startsWith("heads:")) return ((model.heads || []).find((h) => h.name === name.slice(6)) || {}).layers || [];
  if (name.startsWith("cos:")) return cov.filter((c) => c.directions.includes(name.slice(4))).map((c) => c.layer);
  if (name.startsWith("writers:")) return ((model.writers || []).find((w) => w.name === name.slice(8)) || {}).layers || [];
  const need = FIELDS[name].needs;
  if (need === "named") return [...new Set(Object.keys(model.names || {}).map((k) => +k.split(":")[0]))];
  return cov.filter((c) => (need === "sweep" ? c.sweep_runs.length > 0 : !!c[need])).map((c) => c.layer);
}

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))];
}

// 全層をまとめた尺度(層どうしを比べられるように、明るさは全層で同じ基準にする)
export function globalScale(name, layers) {
  const vals = [];
  for (const { v } of layers) {
    for (let i = 0; i < v.length; i++) {
      const x = v[i];
      if (!Number.isFinite(x)) continue;
      vals.push(name === "act_q999" ? Math.log10(Math.max(x, 1e-6)) : Math.abs(x));
    }
  }
  const s = Float32Array.from(vals).sort();
  return { n: s.length, p50: quantile(s, 0.5), p90: quantile(s, 0.9), p99: quantile(s, 0.99), p999: quantile(s, 0.999),
           max: s.length ? s[s.length - 1] : NaN };
}

const lerp = (a, b, t) => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t));
const clamp01 = (x) => Math.max(0, Math.min(1, x));

// v: Float32Array(n_ff)、meta: サーバの X-Meta、scale: globalScale、opt: { threshold, named: Set(n) }。
// 返す配列の .lit に光る点の数
export function colorize(name, v, meta, scale, opt = {}) {
  const n = v.length, out = new Uint8Array(n * 4);
  let lit = 0;
  const put = (i, rgb, a) => { out[4 * i] = rgb[0]; out[4 * i + 1] = rgb[1]; out[4 * i + 2] = rgb[2]; out[4 * i + 3] = a; if (a) lit++; };
  const amber = (t) => lerp(AMBER_LO, AMBER_HI, clamp01(t));
  const signed = (x, t) => { const b = 0.5 + 0.5 * clamp01(t), c = x >= 0 ? COLORS.plus : COLORS.minus; return c.map((u) => Math.round(u * b)); };
  for (let i = 0; i < n; i++) {
    const x = v[i];
    if (!Number.isFinite(x)) { put(i, COLORS.idle, 0); continue; }
    if (name === "sweep") {
      const f = meta.floor ?? 0;
      if (x <= f) put(i, COLORS.swept, 0);
      else put(i, amber(Math.pow((x - f) / Math.max(f, 1e-6), 0.7)), 255);   // 床の 2 倍で最も明るい
    } else if (name === "sweep_null") {
      if (x < 0.95) put(i, COLORS.swept, 0);
      else put(i, amber((x - 0.95) / 0.05), 255);
    } else if (name === "steer_kl") {
      const p95 = (meta.null_p95 || [])[x >= 0 ? 0 : 1] ?? scale.p90;
      const a = Math.abs(x);
      if (a <= p95) put(i, COLORS.swept, 0);
      else put(i, signed(x, (a - p95) / Math.max(scale.p99 - p95, 1e-6)), 255);
    } else if (name === "logit_peak" || name === "act_q999") {
      const y = name === "act_q999" ? Math.log10(Math.max(x, 1e-6)) : x;
      if (y < scale.p90) put(i, COLORS.swept, 0);
      else put(i, amber((y - scale.p90) / Math.max(scale.p999 - scale.p90, 1e-6)), 255);
    } else if (name === "named") {
      put(i, COLORS.sel, 255);
    } else if (name.startsWith("fires:")) {
      const thr = opt.threshold ?? 1, a = Math.abs(x);                       // 1 = その層の活性の q999
      if (a < thr) put(i, COLORS.swept, 0);
      else put(i, signed(x, (a - thr) / Math.max(scale.max - thr, 1e-6)), 255);
    } else if (name.startsWith("cos:") || name.startsWith("writers:")) {
      const thr = opt.threshold ?? scale.p999, a = Math.abs(x);
      if (a < thr) put(i, COLORS.swept, 0);                                  // データはある(閾値未満)
      else put(i, signed(x, (a - thr) / Math.max(scale.max - thr, 1e-6)), 255);
    }
  }
  if (opt.named && name !== "named") {                                      // 名前付きは白
    for (const k of opt.named) if (k < n) { if (!out[4 * k + 3]) lit++; out.set([...COLORS.sel, 255], 4 * k); }
  }
  out.lit = lit;
  return out;
}

// 検索の当たりだけを白く(それ以外はデータ無しの色)
export function colorizeHits(n, hits) {
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) out.set([...COLORS.idle, 0], 4 * i);
  let lit = 0;
  for (const k of hits) if (k < n) { out.set([...COLORS.sel, 255], 4 * k); lit++; }
  out.lit = lit;
  return out;
}

// ヘッドの色(CHANGE-04 G、地図の板の上辺の帯)。v: Float32Array(n_head)、NaN = データ無し。返す配列の .lit に光るヘッドの数。
// heads:<name>: 対照(中立どうし)の z の 95% 点を超えたヘッド(null_pct ≥ 0.95)を琥珀に、明るさ = z(光るヘッドの最小〜最大)。
// fires_heads: 選んだトークンへの注目(後ろの位置からの平均)。明るさ = √(値 / 全ヘッドの最大)、最大の 5% 以上を光るとして数える
export const HEAD_IDLE = [44, 52, 66];
export function colorizeHeads(kind, v, opt = {}) {
  const out = new Uint8Array(v.length * 3);
  let lit = 0;
  for (let h = 0; h < v.length; h++) {
    const x = v[h];
    let rgb = COLORS.idle;
    if (Number.isFinite(x)) {
      if (kind === "fires_heads") {
        const f = x / Math.max(opt.max || 0, 1e-9);
        rgb = lerp(HEAD_IDLE, AMBER_HI, Math.sqrt(clamp01(f)));
        if (f >= 0.05) lit++;
      } else if ((opt.nullPct ? opt.nullPct[h] : 0) >= 0.95) {
        rgb = lerp(AMBER_LO, AMBER_HI, clamp01((x - opt.lo) / Math.max(opt.hi - opt.lo, 1e-6)));
        lit++;
      } else rgb = HEAD_IDLE;
    }
    out.set(rgb, 3 * h);
  }
  out.lit = lit;
  return out;
}

// 光る点が全体の 2% を超えるときの係数(全層共通)。多くの点を全部目立たせると、加算合成で板が白く飽和して数が読めない
export function emphasisFactor(totalLit, totalN) {
  const f = totalLit / Math.max(totalN, 1);
  return f > 0.02 ? 0.02 / f : 1;
}

export function emphasize(out, factor) {
  if (factor >= 1) return out;
  const a = Math.max(1, Math.round(255 * factor));
  for (let i = 3; i < out.length; i += 4) if (out[i] === 255) out[i] = a;
  return out;
}

const sw = (rgb) => `<span class="sw" style="background:rgb(${rgb.join(",")})"></span>`;
const it = (...xs) => `<span class="it">${xs.join("")}</span>`;      // 凡例の 1 項目(途中で折り返さない)
const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : x.toFixed(d));

// 凡例(色の意味は常に出す、§4.3)。opt: { threshold, nHits, floors: [層の床] }
export function legendHTML(name, scale, opt = {}) {
  const none = it(sw(COLORS.idle), t("データ無し"));
  const ramp = (a, b, text) => it(sw(a), "→", sw(b), text);
  if (name === "none") return none;
  if (name.startsWith("search:")) return none + it(sw(COLORS.sel), t("当たり {n} 件", { n: opt.nHits ?? 0 }));
  const fl = (opt.floors || []).filter((x) => Number.isFinite(x));
  const parts = {
    sweep: [it(sw(COLORS.swept), t("床以下")), ramp(AMBER_LO, AMBER_HI, t("mass 床〜床の 2 倍(床 = 対照の最大{v})", { v: fl.length === 1 ? ` ${fmt(fl[0], 1)}` : "" })),
            it(sw(COLORS.sel), t("名前付き"))],
    sweep_null: [it(sw(COLORS.swept), t("0.95 未満")), ramp(AMBER_LO, AMBER_HI, "0.95〜1.0")],
    steer_kl: [it(sw(COLORS.swept), t("対照の 95% 点以下")), it(sw(COLORS.plus), t("+ が強い")), it(sw(COLORS.minus), t("− が強い"))],
    logit_peak: [it(sw(COLORS.swept), t("全層の上位 10% 未満")), ramp(AMBER_LO, AMBER_HI, `|z| ${fmt(scale.p90, 1)}〜${fmt(scale.p999, 1)}`)],
    act_q999: [it(sw(COLORS.swept), t("全層の上位 10% 未満")), ramp(AMBER_LO, AMBER_HI, `${fmt(10 ** scale.p90, 2)}〜${fmt(10 ** scale.p999, 2)}`)],
    named: [it(sw(COLORS.sel), t("名前付き"))],
  };
  if (name.startsWith("heads:")) {
    return none + it(sw(HEAD_IDLE), t("対照の 95% 以下")) + ramp(AMBER_LO, AMBER_HI, t("z {a}〜{b}(対照の 95% 超)", { a: fmt(scale.lo, 1), b: fmt(scale.hi, 1) })) +
      it(t("色 = 板の上辺のヘッドの帯"));
  }
  if (name === "fires_heads") {
    return ramp(HEAD_IDLE, AMBER_HI, t("注目 0〜{m}(後ろの位置からの平均、明るさ = √)", { m: fmt(scale.max, 3) })) +
      it(t("線形注意の層はパターンが無いので色なし"));
  }
  if (name.startsWith("fires:")) {
    return none + it(sw(COLORS.swept), t("|スコア| < {v}", { v: fmt(opt.threshold, 1) })) + it(sw(COLORS.plus), t("正")) + it(sw(COLORS.minus), t("負")) +
      it(t("明るさ = |スコア| {a}〜{b}(1 = その層の q999)", { a: fmt(opt.threshold, 1), b: fmt(scale.max, 1) }));
  }
  const body = name.startsWith("cos:")
    ? [it(sw(COLORS.swept), `|cos| < ${fmt(opt.threshold, 3)}`), it(sw(COLORS.plus), t("正")), it(sw(COLORS.minus), t("負")),
       it(t("明るさ = |cos| {a}〜{b}", { a: fmt(opt.threshold, 3), b: fmt(scale.max, 3) }))]
    : name.startsWith("writers:")
      ? [it(sw(COLORS.swept), t("|寄与| < {v}", { v: fmt(opt.threshold, 4) })), it(sw(COLORS.plus), t("方向へ書き足した")),
         it(sw(COLORS.minus), t("方向から引いた")), it(t("明るさ = |寄与| {a}〜{b}", { a: fmt(opt.threshold, 4), b: fmt(scale.max, 4) }))]
      : parts[name] || [];
  return none + body.join("");
}
