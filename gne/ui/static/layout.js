// マップの幾何(別冊 §4.1)。x = 深さ、各層は mixer スラブと MLP シート(y–z 平面の正方形)。
// 点の位置とピッキング(§4.2 の解析的な当たり判定)の両方がここを使う。

export const GEOM = {
  GAP: 0.9,        // 層の間隔(x)
  S: 2.0,          // いちばん大きい層のシートの一辺(y, z)
  Y0: 0.18,        // シートの下端(残差の spine は y = 0)
  MLP_DX: 0.17,    // シートの x のずれ(層の中心から)
  SLAB_DX: -0.17,  // mixer スラブの x のずれ
  TILE_GAP: 0.12,  // 専門家タイルの間のすき間(タイル幅に対する比)
};

export const layerX = (L) => L * GEOM.GAP;
export const sheetX = (L) => layerX(L) + GEOM.MLP_DX;
export const slabX = (L) => layerX(L) + GEOM.SLAB_DX;

// 層のシートの形。dense: side×side の 1 枚。moe: 専門家ごとの tile×tile の正方形を g×g に並べた 1 枚
// (専門家が 16 以下でも同じ並べ方で、すき間を広めにする。§4.1 との違いは verification.md)。
// 全ニューロンの点が多すぎるモデル(専門家 512 × 640 × 48 層で 1,570 万点)は、ブラウザが固まるので専門家を 1 点で描く(coarse)。
// シートの住所(通し番号 = 専門家 × n_ff + n)は変わらない。点が表すのが専門家になるだけ。
export const COARSE_POINTS = 5000000;

export function coarseLayout(arch, force = false) {
  const total = arch.layers.reduce((a, l) => a + (l.ffn === "dense" ? l.n_ff : l.ffn.startsWith("moe") ? l.n_ff * l.n_expert : 0), 0);
  if ((!force && total <= COARSE_POINTS) || !arch.layers.some((l) => l.ffn.startsWith("moe"))) return { coarse: false, gridSide: arch.grid_side };
  const side = (l) => (l.ffn.startsWith("moe") ? Math.ceil(Math.sqrt(l.n_expert)) : l.ffn === "dense" ? Math.ceil(Math.sqrt(l.n_ff)) : 1);
  return { coarse: true, gridSide: Math.max(1, ...arch.layers.map(side)) };
}

export function sheetSpec(layer, gridSide, coarse = false) {
  const S = GEOM.S;
  if (layer.ffn === "dense") {
    const side = Math.ceil(Math.sqrt(layer.n_ff));
    const ext = S * side / gridSide;
    return { kind: "dense", n: layer.n_ff, side, ext, tile: side, g: 1, tileExt: ext, pitch: ext, shared: null };
  }
  if (layer.ffn.startsWith("moe") && coarse) {
    const g = Math.ceil(Math.sqrt(layer.n_expert));
    const ext = S * g / gridSide, pitch = ext / g;
    return { kind: "moe", coarse: true, n: layer.n_ff * layer.n_expert, nExpert: layer.n_expert, nPer: layer.n_ff, side: g,
             ext, tile: 1, g, tileExt: pitch * (1 - GEOM.TILE_GAP), pitch,
             // 共有の専門家は毎トークン動くので、まとめずにニューロンのまま別枠で描く
             shared: layer.n_ff_shexp ? { n: layer.n_ff_shexp, side: Math.ceil(Math.sqrt(layer.n_ff_shexp)), ext: pitch * 3 } : null };
  }
  if (layer.ffn.startsWith("moe")) {
    const tile = Math.ceil(Math.sqrt(layer.n_ff));
    const g = Math.ceil(Math.sqrt(layer.n_expert));
    const ext = S * (tile * g) / gridSide;
    const gapFrac = layer.n_expert <= 16 ? 2 * GEOM.TILE_GAP : GEOM.TILE_GAP;
    const pitch = ext / g;
    const tileExt = pitch * (1 - gapFrac);
    let shared = null;
    if (layer.n_ff_shexp) {
      const sside = Math.ceil(Math.sqrt(layer.n_ff_shexp));
      shared = { n: layer.n_ff_shexp, side: sside, ext: S * sside / gridSide };
    }
    return { kind: "moe", n: layer.n_ff * layer.n_expert, nExpert: layer.n_expert, nPer: layer.n_ff, side: tile * g,
             ext, tile, g, tileExt, pitch, shared };
  }
  return { kind: "none", n: 0, side: 0, ext: 0, tile: 0, g: 0, tileExt: 0, pitch: 0, shared: null };
}

// ニューロン(MoE は 専門家 e × n_ff + n の通し番号)→ シート上の (y, z)。cell は配置表(§4.4、既定は番号順)。
export function cellYZ(spec, cell) {
  const { Y0 } = GEOM;
  if (spec.kind === "dense") {
    const row = Math.floor(cell / spec.side), col = cell % spec.side;
    const d = spec.ext / spec.side;
    return [Y0 + spec.ext - (row + 0.5) * d, -spec.ext / 2 + (col + 0.5) * d];
  }
  const e = Math.floor(cell / spec.nPer), k = spec.coarse ? 0 : cell % spec.nPer;
  const tr = Math.floor(e / spec.g), tc = e % spec.g;
  const row = Math.floor(k / spec.tile), col = k % spec.tile;
  const d = spec.tileExt / spec.tile;
  const off = (spec.pitch - spec.tileExt) / 2;
  return [Y0 + spec.ext - tr * spec.pitch - off - (row + 0.5) * d,
          -spec.ext / 2 + tc * spec.pitch + off + (col + 0.5) * d];
}

export function sharedYZ(spec, k) {
  const s = spec.shared, d = s.ext / s.side;
  const row = Math.floor(k / s.side), col = k % s.side;
  return [GEOM.Y0 + s.ext - (row + 0.5) * d, spec.ext / 2 + 0.15 + (col + 0.5) * d];
}

// (y, z) → cell(シートの外・すき間は -1)。cellYZ の逆。
export function yzCell(spec, y, z) {
  const { Y0 } = GEOM;
  const u = (z + spec.ext / 2) / spec.ext, v = (Y0 + spec.ext - y) / spec.ext;
  if (u < 0 || u >= 1 || v < 0 || v >= 1) return -1;
  if (spec.kind === "dense") {
    const row = Math.floor(v * spec.side), col = Math.floor(u * spec.side);
    const cell = row * spec.side + col;
    return cell < spec.n ? cell : -1;
  }
  const tr = Math.floor(v * spec.g), tc = Math.floor(u * spec.g);
  const e = tr * spec.g + tc;
  if (e >= spec.nExpert) return -1;
  const off = (spec.pitch - spec.tileExt) / 2;
  const lu = (z + spec.ext / 2 - tc * spec.pitch - off) / spec.tileExt;
  const lv = (Y0 + spec.ext - tr * spec.pitch - off - y) / spec.tileExt;
  if (lu < 0 || lu >= 1 || lv < 0 || lv >= 1) return -1;
  const k = Math.floor(lv * spec.tile) * spec.tile + Math.floor(lu * spec.tile);
  return k < spec.nPer ? e * spec.nPer + k : -1;
}

export function sharedCell(spec, y, z) {
  const s = spec.shared;
  if (!s) return -1;
  const u = (z - spec.ext / 2 - 0.15) / s.ext, v = (GEOM.Y0 + s.ext - y) / s.ext;
  if (u < 0 || u >= 1 || v < 0 || v >= 1) return -1;
  const k = Math.floor(v * s.side) * s.side + Math.floor(u * s.side);
  return k < s.n ? k : -1;
}

// 合成の構造(性能の確認用、?synthetic=80x28672 など)
export function syntheticArch(spec) {
  const [L, nff, nexp] = spec.split("x").map(Number);
  const layers = Array.from({ length: L }, (_, i) => ({
    index: i, mixer: i % 4 === 3 || !nexp ? "attention" : "linear_attention", n_head: 64, n_head_kv: 8,
    ffn: nexp ? "moe" : "dense", n_ff: nff, n_expert: nexp || 0, n_expert_used: nexp ? 8 : 0, n_ff_shexp: 0,
    has_post_ffn_norm: false,
  }));
  const side = nexp ? Math.ceil(Math.sqrt(nff)) * Math.ceil(Math.sqrt(nexp)) : Math.ceil(Math.sqrt(nff));
  return { arch: `synthetic-${spec}`, n_layer: L, n_embd: 8192, n_vocab: 0, layers, grid_side: side };
}
