// 3D 層マップ(別冊 §4)。全ニューロンの点は 1 つのジオメトリ・1 draw call。ピッキングはレイと各層の
// シート平面の交点から解析的に求める(点群へのレイキャストはしない、§4.2)。
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { t } from "./i18n.js";
import { GEOM, cellYZ, layerX, sharedCell, sharedYZ, sheetSpec, sheetX, slabX, yzCell } from "./layout.js";

export const COLORS = {
  idle: [16, 20, 27],          // データ無し: 非常に暗い青灰(明るさは画面上の点の密度で割り戻す)
  swept: [30, 37, 48],         // データあり・床以下: 暗い青灰(正面ビューの 1.6 倍でも琥珀の最も暗い点より暗い)
  spine: 0x2a3442,
  slabAttn: 0x9fb3cf,          // attention の板: 明るい青灰
  slabLin: 0x3a4656,           // 線形注意・回帰の板: 暗い青灰
  frameNone: 0x252b33,         // overlay のデータが無い層のシートの枠(線形注意の板の色と分ける)
  frameData: 0x5e4a26,         // overlay のデータがある層のシートの枠: 暗い琥珀
  frameHot: 0xf0b850,          // その層で光っている点がいちばん多い層の枠: 明るい琥珀
  label: "#aeb9c8",
  plus: [242, 210, 75],
  minus: [201, 79, 167],
  sel: [255, 255, 255],
  plusLine: [242, 210, 75],    // 配線の線: 赤紫は黄と同じくらいの明るさに(暗い背景で読めるように)
  minusLine: [255, 112, 208],
};

// 点は、インスタンス描画の小さな四角形(ビルボード)で描く。点スプライト(gl_PointSize)は D3D 系で
// 重い代替処理になる・大きさに上限がある GPU があるので使わない(verification.md F33)。
// aColor.a は「目立たせる度合い」: 0 = 明るさを画面上の点の密度で割り戻す(データ無し・床以下)、
// 1 = 割り戻さない(データのある点が遠くでも消えないように)。
const VERT = /* glsl */ `
attribute vec3 iPos;
attribute vec4 aColor;
attribute float aLayer;
uniform float uSize;
uniform float uScale;
uniform float uFocus;
uniform float uFront;
uniform float uXray;
varying vec3 vColor;
varying vec2 vUv;
void main() {
  vec4 mv = modelViewMatrix * vec4(iPos, 1.0);
  float depth = -mv.z;
  float minPx = uFront > 0.5 ? 1.6 : mix(1.6, 4.0, aColor.a);            // 全体ビューでは光る点を 4 px 以上に
  float s = clamp(uSize, minPx * depth / uScale, 24.0 * depth / uScale);   // 画面で 1.6〜24 px
  if (uFront > 0.5) s *= mix(1.0, 1.5, aColor.a);                         // 正面ビューでも光る点は少し大きく
  mv.xy += position.xy * s;
  gl_Position = projectionMatrix * mv;
  vUv = position.xy;
  // 画面上の点の密度で明るさを割り戻す(加算合成で白く飽和しないように):
  //   1.6 px に満たない点を引き伸ばした分(面積の比)と、シートを斜めから見て縮んで見える分(cos)
  float px = uSize * uScale / depth;
  vec3 nView = normalize((modelViewMatrix * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
  float fore = max(abs(dot(nView, normalize(-mv.xyz))), 0.06);
  float energy = mix(min(1.0, (px * px) / (1.6 * 1.6)) * fore, 1.0, aColor.a);
  float dim;
  if (uFront > 0.5) {
    // 層正面ビュー: 焦点の層だけを明るく。手前の層は消し、奥の層は 3 層先までごく薄く(干渉の縞が出ないように)
    float k = aLayer - uFocus;
    // 焦点の層の格子は 1.6 倍に明るく(光る点は色そのまま。上げると琥珀が淡い黄に飽和する)
    dim = k == 0.0 ? mix(1.6, 1.0, step(0.001, aColor.a)) : (k < 0.0 ? 0.0 : 0.10 * max(0.0, 1.0 - k / 3.0));
  } else {
    dim = 1.0 - 0.45 * clamp(abs(aLayer - uFocus) / uXray, 0.0, 1.0);  // x-ray: 焦点から離れた層を少し暗く
  }
  dim *= smoothstep(0.25, 1.0, depth);                                     // カメラに近すぎる点は薄く
  vColor = aColor.rgb * dim * energy;
}`;

const FRAG = /* glsl */ `
varying vec3 vColor;
varying vec2 vUv;
void main() {
  float r = length(vUv);
  if (r > 0.5) discard;
  float a = 1.0 - smoothstep(0.15, 0.5, r);
  gl_FragColor = vec4(vColor * a, 1.0);
}`;

const MONO = "ui-monospace, Menlo, Consolas, monospace";
const MEAS = document.createElement("canvas").getContext("2d");
const measure = (t, font) => { MEAS.font = font; return MEAS.measureText(t).width; };   // 札の幅を実際の字で測る
// 線分と矩形が交わるか(Liang–Barsky)
function segHitsRect(x1, y1, x2, y2, q) {
  const dx = x2 - x1, dy = y2 - y1;
  let t0 = 0, t1 = 1;
  for (const [p, d] of [[-dx, x1 - (q.x + 1)], [dx, q.x + q.w - 1 - x1], [-dy, y1 - (q.y + 1)], [dy, q.y + q.h - 1 - y1]]) {
    if (p === 0) { if (d < 0) return false; continue; }
    const r = d / p;
    if (p < 0) { if (r > t1) return false; if (r > t0) t0 = r; } else { if (r < t0) return false; if (r < t1) t1 = r; }
  }
  return t0 < t1 && t1 > 0.02 && t0 < 0.98;                                  // 両端のごく近くは数えない
}
const MAX_AGENT_LABELS = 30;
const escHTML = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export class LayerMap {
  constructor(container, arch, { onHover, onClick, onLayer, onContext, onOpen, overlay } = {}) {
    this.container = container;
    this.arch = arch;
    this.onHover = onHover || (() => {});
    this.onClick = onClick || (() => {});
    this.onLayer = onLayer || (() => {});
    this.onContext = onContext || null;  // 層の番号・ラベルの右クリック(L, イベント)
    this.onOpen = onOpen || null;        // 札のクリックでニューロンを開く({ layer, n })
    this.agent = { highlights: [], notes: [], label: null };   // エージェントの強調と注釈(§11、§4.8 notes)
    this.marks = [];                   // レシピのユニット [{ layer, neuron, dose }](§4.8)
    this.cellOf = [];                  // 層ごとの配置表(§4.4): ニューロン → 格子の番号。null は番号順
    this.neuronAt = [];                // その逆: 格子の番号 → ニューロン
    this.linkEdges = null;             // 配線(§4.7)
    this.overlay = overlay || null;    // 層番号などの HTML を重ねる要素
    this.specs = arch.layers.map((l) => sheetSpec(l, arch.grid_side));
    this.front = null;                 // 層正面ビューの層
    this.structDim = 1;                // mixer の板の明るさ(overlay 中は下げる)
    this.badges = [];
    this.hover = null;
    this._mouse = null;
    this._dirtyPick = false;
    this._frames = 0;
    this._fpsT = performance.now();
    this.fps = null;                   // 1 秒たつまでは測れていない

    const w = container.clientWidth, h = container.clientHeight;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(0x000000, 1);
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, w / h, 0.01, 4000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.addEventListener("change", () => { this._overlayDirty = true; });
    this.controls.addEventListener("start", () => { this._userMoved = true; });   // 手で動かした視点は、幅が変わっても保つ

    const t0 = performance.now();
    this._buildPoints();
    this._buildStructure();
    this.buildMs = performance.now() - t0;
    this.resetView();

    this._ray = new THREE.Raycaster();
    const el = this.renderer.domElement;
    el.addEventListener("pointermove", (ev) => { this._mouse = this._ndc(ev); this._dirtyPick = true; });
    el.addEventListener("pointerleave", () => { this._mouse = null; this._setHover(null); });
    el.addEventListener("pointerdown", (ev) => { this._down = [ev.clientX, ev.clientY]; });
    el.addEventListener("pointerup", (ev) => {
      if (!this._down || Math.hypot(ev.clientX - this._down[0], ev.clientY - this._down[1]) > 4) return;
      const lab = this._pickLabel(this._ndc(ev));
      if (lab !== null) { this.frontView(lab); return; }
      const hit = this._pick(this._ndc(ev));
      if (hit) this.onClick(hit);
    });
    if (this.overlay) {
      this.overlay.addEventListener("click", (ev) => {
        const t = ev.target.closest("[data-open],[data-layer]");
        if (!t) return;
        if (t.dataset.open && this.onOpen) { const [L, n] = t.dataset.open.split(":").map(Number); this.onOpen({ layer: L, n }); return; }
        if (t.dataset.layer !== undefined) this.frontView(+t.dataset.layer);
      });
      this.overlay.addEventListener("contextmenu", (ev) => {
        const L = ev.target.dataset && ev.target.dataset.layer;
        if (L !== undefined && this.onContext) { ev.preventDefault(); this.onContext(+L, ev); }
      });
    }
    el.addEventListener("contextmenu", (ev) => {                          // 正面ビューの板のラベル
      const lab = this._pickLabel(this._ndc(ev));
      if (lab !== null && this.onContext) { ev.preventDefault(); this.onContext(lab, ev); }
    });
    new ResizeObserver(() => this.resize()).observe(container);
    this.renderer.setAnimationLoop(() => this._frame());
  }

  // --- 点群 -------------------------------------------------------------------------------------------
  _buildPoints() {
    const specs = this.specs;
    const counts = specs.map((s) => s.n + (s.shared ? s.shared.n : 0));
    const total = counts.reduce((a, b) => a + b, 0);
    this.offsets = new Int32Array(specs.length + 1);
    for (let L = 0; L < specs.length; L++) this.offsets[L + 1] = this.offsets[L] + counts[L];
    const pos = new Float32Array(total * 3);
    const col = new Uint8Array(total * 4);            // RGBA8(RGB8 は D3D に頂点形式が無く遅い、F33)
    const lay = new Float32Array(total);
    for (let L = 0; L < specs.length; L++) {
      const s = specs[L], x = sheetX(L);
      let p = this.offsets[L];
      for (let c = 0; c < s.n; c++, p++) {
        const [y, z] = cellYZ(s, c);
        pos[3 * p] = x; pos[3 * p + 1] = y; pos[3 * p + 2] = z;
        lay[p] = L;
      }
      if (s.shared) {
        for (let k = 0; k < s.shared.n; k++, p++) {
          const [y, z] = sharedYZ(s, k);
          pos[3 * p] = x; pos[3 * p + 1] = y; pos[3 * p + 2] = z;
          lay[p] = L;
        }
      }
    }
    for (let i = 0; i < total; i++) col.set([...COLORS.idle, 0], 4 * i);
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    g.setAttribute("iPos", new THREE.InstancedBufferAttribute(pos, 3));
    this.colorAttr = new THREE.InstancedBufferAttribute(col, 4, true);
    this.colorAttr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("aColor", this.colorAttr);
    g.setAttribute("aLayer", new THREE.InstancedBufferAttribute(lay, 1));
    g.instanceCount = total;
    const spacing = GEOM.S / this.arch.grid_side;
    this.uniforms = {
      uSize: { value: spacing * 0.9 }, uScale: { value: 1 }, uFocus: { value: 0 }, uFront: { value: 0 },
      uXray: { value: Math.max(4, specs.length / 3) },
    };
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Mesh(g, this.material);
    this.points.frustumCulled = false;
    this.points.renderOrder = 2;
    this.scene.add(this.points);
    this.nPoints = total;
  }

  // シート内の配置を切り替える(cell: Int32Array(n_ff) ニューロン → 格子の番号、null で番号順)。即時に位置を書き直す
  setLayout(L, cell) {
    const s = this.specs[L];
    if (!s || s.kind !== "dense") return false;
    const attr = this.points.geometry.getAttribute("iPos"), pos = attr.array, x = sheetX(L);
    let inv = null;
    if (cell) { inv = new Int32Array(s.side * s.side).fill(-1); for (let n = 0; n < cell.length; n++) inv[cell[n]] = n; }
    this.cellOf[L] = cell; this.neuronAt[L] = inv;
    let p = this.offsets[L];
    for (let n = 0; n < s.n; n++, p++) {
      const [y, z] = cellYZ(s, cell ? cell[n] : n);
      pos[3 * p] = x; pos[3 * p + 1] = y; pos[3 * p + 2] = z;
    }
    attr.addUpdateRange(this.offsets[L] * 3, s.n * 3);
    attr.needsUpdate = true;
    if (this._markerAt && this._marker && this._marker.visible) this.markNeuron(...this._markerAt);
    if (this.linkEdges) this.setLinks(this.linkEdges, this.linkOpts);
    this._overlayDirty = true;
    return true;
  }

  // 配線の線(§4.7)。edges: [{ sl, sn, dl, dn, z, sign, dim }]。色 = 符号(黄 = 相手を + に押す、赤紫 = − に押す)、明るさ = z
  setLinks(edges, opts = {}) {
    this.linkEdges = edges; this.linkOpts = opts;
    if (this._links) { this.scene.remove(this._links); this._links.geometry.dispose(); this._links = null; }
    this._overlayDirty = true;
    if (!edges || !edges.length) return;
    const zmax = Math.max(...edges.map((e) => e.z), 1);
    const pos = [], col = [];
    for (const e of edges) {
      const a = this.neuronPosition(e.sl, e.sn), b = this.neuronPosition(e.dl, e.dn);
      if (!a || !b) continue;
      const c = e.sign >= 0 ? COLORS.plusLine : COLORS.minusLine;
      const k = (0.5 + 0.5 * Math.min(1, e.z / zmax)) * (e.dim ? 0.45 : 1);
      pos.push(...a, ...b);
      for (let i = 0; i < 2; i++) col.push((c[0] / 255) * k, (c[1] / 255) * k, (c[2] / 255) * k);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    this._links = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95,
                                                                          depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
    this._links.renderOrder = 3;
    this._links.visible = this.front === null;
    this.scene.add(this._links);
    this._overlayDirty = true;
  }

  // 層 L の色をまとめて書き換える(rgba: Uint8Array(n*4)、MoE は専門家の通し番号順。a は目立たせる度合い)
  setLayerColors(L, rgba) {
    const o = this.offsets[L] * 4;
    const n = Math.min(rgba.length, (this.offsets[L + 1] - this.offsets[L]) * 4);
    this.colorAttr.array.set(rgba.subarray(0, n), o);
    if (!this._fullUpload) this.colorAttr.addUpdateRange(o, n);   // 全体を送る予定なら範囲を足さない(足すとその範囲しか送られない)
    this.colorAttr.needsUpdate = true;
  }

  // overlay を出している間は mixer の板(縁・刻み)を暗くして、点の色を主役にする
  setStructureDim(k) {
    this.structDim = k;
    this._applyLayerDimming();
  }

  // レシピのユニットの印(+ は黄、− は赤紫、大きさ = |dose|)。画面に重ねる HTML の輪で、視点が変わるたびに置き直す
  setMarks(units) {
    this.marks = units || [];
    this._overlayDirty = true;
  }

  // エージェントの強調と注釈(§11)。highlights: [{ layer, neuron, dose }]、notes: [{ layer, neuron, text }]
  setAgent(a) {
    this.agent = { highlights: a.highlights || [], notes: a.notes || [], label: a.label || null };
    this._agentCache = null; this._agentMoving = false;                       // 中身が変わったら置き直す
    this._overlayDirty = true;
  }

  // 全体ビューで層の上に出す札(光る点の多い層など)。items: [{ layer, text }]
  setBadges(items) {
    this.badges = items || [];
    this._overlayDirty = true;
  }

  // シートの枠の色で、今の overlay のデータがある層を示す(coverage、§4.3)。
  // strength: Map(層 → 0〜1)。データがあれば暗い琥珀、その層で光っている点が多いほど明るい琥珀
  setFrameColors(strength) {
    const lo = new THREE.Color(COLORS.frameData), hi = new THREE.Color(COLORS.frameHot);
    for (let L = 0; L < this.layerParts.length; L++) {
      const f = this.layerParts[L].frame;
      if (!f) continue;
      if (!strength.has(L)) {                // データ無し: 灰。overlay を出している間はさらに薄く
        const on = this.structDim < 1;
        f.material.color.setHex(on ? COLORS.frameNone : COLORS.slabLin); f.material.userData.base = on ? 0.5 : 0.55; continue;
      }
      const t = Math.max(0, Math.min(1, strength.get(L)));
      f.material.color.copy(lo).lerp(hi, t);
      f.material.userData.base = 0.55 + 0.45 * t;
    }
    this._applyLayerDimming();
  }

  resetColors() {
    const a = this.colorAttr.array;
    for (let i = 0; i < this.nPoints; i++) a.set([...COLORS.idle, 0], 4 * i);
    this.colorAttr.clearUpdateRanges();
    this.colorAttr.needsUpdate = true;
    this._fullUpload = true;                 // 次の描画で全体を送る
  }

  // --- 構造(spine・板・head の刻み・シートの枠・ラベル) --------------------------------------------------
  _buildStructure() {
    const n = this.specs.length, S = GEOM.S, Y0 = GEOM.Y0;
    const spine = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(layerX(-1), 0, 0), new THREE.Vector3(layerX(n), 0, 0)]);
    this.spine = new THREE.Line(spine, new THREE.LineBasicMaterial({ color: COLORS.spine, transparent: true }));
    this.scene.add(this.spine);
    this.layerParts = [];               // 層ごと: { lines: [...], fill, feet }
    for (let L = 0; L < n; L++) {
      const li = this.arch.layers[L], x = slabX(L), ext = Math.max(this.specs[L].ext, S * 0.6);
      const y0 = Y0, y1 = Y0 + ext, z0 = -ext / 2, z1 = ext / 2;
      const attn = li.mixer === "attention";
      const color = attn ? COLORS.slabAttn : COLORS.slabLin;
      const edge = [x, y0, z0, x, y1, z0, x, y1, z0, x, y1, z1, x, y1, z1, x, y0, z1, x, y0, z1, x, y0, z0];
      const ticks = [];                 // head の本数ぶんの刻み(板の上辺の外側に短く)
      const nh = Math.min(li.n_head || 0, 128);
      for (let h = 0; h < nh; h++) {
        const z = z0 + (h + 0.5) * (ext / nh);
        ticks.push(x, y1, z, x, y1 + ext * (attn ? 0.05 : 0.035), z);   // 上辺から外へ(シートに重ならない)
      }
      const s = this.specs[L], sx = sheetX(L), frame = [];
      if (s.kind !== "none") {
        const e = s.ext;
        frame.push(sx, Y0, -e / 2, sx, Y0 + e, -e / 2, sx, Y0 + e, -e / 2, sx, Y0 + e, e / 2,
                   sx, Y0 + e, e / 2, sx, Y0, e / 2, sx, Y0, e / 2, sx, Y0, -e / 2);
      }
      const feet = [x, 0, 0, x, y0, 0, sx, 0, 0, sx, Y0, 0];            // 板とシートから spine への足
      const parts = { lines: [], fill: null, feet: null };   // fill = attention の太い縁
      const mk = (arr, col, op) => {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
        const m = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: op, depthWrite: false });
        m.userData.base = op;
        const obj = new THREE.LineSegments(g, m);
        this.scene.add(obj);
        return obj;
      };
      parts.lines.push(mk(edge, color, attn ? 0.95 : 0.8));
      if (ticks.length) parts.lines.push(mk(ticks, color, attn ? 0.7 : 0.5));
      if (frame.length) { parts.frame = mk(frame, COLORS.slabLin, 0.55); parts.lines.push(parts.frame); }
      parts.feet = mk(feet, COLORS.spine, 0.8);
      // attention の板は太く明るい縁(細い四角形 4 本。WebGL の線は 1 px にしかならないので)。
      // 半透明の面は奥で重なって光って見えるので使わない
      if (attn) {
        const bw = ext * 0.022, mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55, depthWrite: false,
                                                                    side: THREE.DoubleSide, blending: THREE.AdditiveBlending });
        const group = new THREE.Group();
        for (const [cy, cz, h, w] of [[y1 - bw / 2, 0, bw, ext], [y0 + bw / 2, 0, bw, ext], [Y0 + ext / 2, z0 + bw / 2, ext - 2 * bw, bw],
                                      [Y0 + ext / 2, z1 - bw / 2, ext - 2 * bw, bw]]) {   // 縦は横と重ならない長さ(加算で角が光るので)
          const q = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
          q.position.set(x, cy, cz);
          q.rotation.y = Math.PI / 2;
          group.add(q);
        }
        this.scene.add(group);
        parts.fill = group;
      }
      this.layerParts.push(parts);
    }
    this._buildLabels();
  }

  _buildLabels() {
    this.labels = [];
    const n = this.specs.length;
    for (let L = 0; L < n; L++) {
      const li = this.arch.layers[L];
      const cv = document.createElement("canvas");
      cv.width = 256; cv.height = 64;
      const ctx = cv.getContext("2d");
      ctx.textAlign = "right";
      ctx.font = `bold 40px ${MONO}`;
      const w0 = ctx.measureText(`L${L}`).width;
      ctx.fillStyle = COLORS.label;
      ctx.fillText(`L${L}`, 250, 44);
      const tag = { attention: "attention", linear_attention: t("線形注意"), recurrent: t("回帰"), none: "" }[li.mixer];
      ctx.font = `22px ${MONO}`;
      ctx.fillStyle = li.mixer === "attention" ? "#9fb3cf" : "#5d6b7e";
      ctx.fillText(tag, 250 - w0 - 12, 44);
      const tex = new THREE.CanvasTexture(cv);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 4;
      const ext = Math.max(this.specs[L].ext, GEOM.S * 0.6);
      const h = GEOM.S * 0.1, w = h * 4;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
                               new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
      m.position.set(sheetX(L), GEOM.Y0 + ext * 1.06 + h * 0.62, ext / 2 - w / 2);   // シートの右上の角の上(刻みの上)
      m.rotation.y = -Math.PI / 2;                                            // 層の面(y–z 平面)に貼る
      m.userData.layer = L;
      this.scene.add(m);
      this.labels.push(m);
    }
  }

  // 層正面ビューでは焦点の層の線とラベルだけを残す。spine と足と板の面も消す(シートを割って見えるので)
  _applyLayerDimming() {
    const front = this.front;
    this.spine.visible = front === null;
    for (let L = 0; L < this.layerParts.length; L++) {
      const p = this.layerParts[L];
      const k = front === null ? 1 : L === front ? 0.75 : 0;
      for (const obj of p.lines) {
        const sd = obj === p.frame ? 1 : this.structDim;                   // overlay 中は mixer の板を暗く(シートの枠はそのまま)
        obj.material.opacity = obj.material.userData.base * k * sd; obj.visible = k > 0;
      }
      if (p.fill) for (const q of p.fill.children) q.material.opacity = 0.55 * this.structDim;
      p.feet.visible = front === null;
      if (p.fill) p.fill.visible = front === null || L === front;
      this.labels[L].visible = front !== null && L === front;   // 全体ビューでは床の層番号(HTML)を使う
    }
    // 開いているニューロンの輪は、全体ビューとその層の正面ビューで出す。3D の線は全体ビューだけ(正面ビューは 2D の札と線)
    if (this._marker) this._marker.visible = !!this._markerAt && (front === null || front === this._markerAt[0]);
    if (this._links) this._links.visible = front === null;
    this._overlayDirty = true;
  }

  // --- 視点 ---------------------------------------------------------------------------------------------
  _corners(fromL, toL) {
    const ext = GEOM.S, out = [];
    for (const x of [layerX(fromL - 0.5), layerX(toL + 0.5)])
      for (const y of [0, GEOM.Y0 + ext]) for (const z of [-ext / 2, ext / 2]) out.push(new THREE.Vector3(x, y, z));
    return out;
  }

  // 斜め手前・やや上から見て、層の列全体が画面の中央に収まる距離と位置を探す
  resetView() {
    const n = this.specs.length;
    const dir = new THREE.Vector3(-0.66, 0.40, 0.64).normalize();
    const corners = this._corners(0, n - 1);
    const target = new THREE.Vector3(layerX((n - 1) / 2), GEOM.Y0 + GEOM.S * 0.45, 0);
    const right = new THREE.Vector3(), up = new THREE.Vector3();
    this.camera.up.set(0, 1, 0);
    const place = (d) => {
      this.camera.position.copy(target).addScaledVector(dir, d);
      this.camera.lookAt(target);
      this.camera.updateMatrixWorld();
    };
    const bbox = () => {
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, ok = true;
      for (const c of corners) {
        const p = c.clone().project(this.camera);
        if (p.z >= 1) ok = false;
        x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
      }
      return { x0, x1, y0, y1, ok };
    };
    let d = 10;
    this.camera.far = 1e6; this.camera.updateProjectionMatrix();   // 前回の far のままだと遠い角が範囲外になり、探索が最大距離に張り付く
    for (let it = 0; it < 3; it++) {
      let lo = 0.5, hi = 1000;
      for (let i = 0; i < 40; i++) {
        const m = (lo + hi) / 2;
        place(m);
        const b = bbox();
        if (b.ok && b.x1 - b.x0 < 1.70 && b.y1 - b.y0 < 1.60) hi = m; else lo = m;   // 左右は入力側・出力側の文字の分を空ける
      }
      d = hi;
      place(d);
      const b = bbox();                      // 画面の中央に来るように target をずらす
      right.setFromMatrixColumn(this.camera.matrixWorld, 0);
      up.setFromMatrixColumn(this.camera.matrixWorld, 1);
      const halfH = Math.tan((this.camera.fov * Math.PI) / 360) * d, halfW = halfH * this.camera.aspect;
      target.addScaledVector(right, ((b.x0 + b.x1) / 2) * halfW).addScaledVector(up, ((b.y0 + b.y1) / 2) * halfH);
    }
    this.front = null;
    this._userMoved = false;
    this.uniforms.uFront.value = 0;
    this.controls.target.copy(target);
    place(d);
    this.camera.near = 0.01; this.camera.far = d * 20; this.camera.updateProjectionMatrix();
    this.controls.update();
    this._applyLayerDimming();
    this.onLayer(null);
  }

  frontView(L, { keepMarker = false } = {}) {
    const s = this.specs[L], ext = Math.max(s.ext, GEOM.S * 0.6);
    this.front = L;
    this._userMoved = false;
    this.uniforms.uFront.value = 1;
    this.uniforms.uFocus.value = L;
    const yc = GEOM.Y0 + ext * (this.camera.aspect < 1.15 ? 0.5 : 0.56);    // ラベルの分だけ少し上を見る(縦長なら中央)
    // シートとラベルが上下の余白つきで収まる距離(横長でない画面では横幅で決める)
    const d = (ext * 0.78) / Math.tan((this.camera.fov * Math.PI) / 360) / Math.min(1, this.camera.aspect / 1.15);
    this.controls.target.set(sheetX(L), yc, 0);
    this.camera.up.set(0, 1, 0);
    this.camera.position.set(sheetX(L) - d, yc, 0);                       // 真正面(台形にゆがまないように)
    this.camera.lookAt(this.controls.target);
    this.controls.update();
    this._applyLayerDimming();
    this.onLayer(L);
  }

  focusNeuron(L, n) {
    this.frontView(L);
    this.markNeuron(L, n);
  }

  // カメラは動かさずに、ニューロンに輪の印を付ける
  markNeuron(L, n) {
    this._markerAt = [L, n];
    const p = this.neuronPosition(L, n);
    if (!p) return;
    if (!this._marker) {
      const g = new THREE.RingGeometry(0.9, 1.25, 32);
      this._marker = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9,
                                                                        side: THREE.DoubleSide, depthTest: false }));
      this._marker.rotation.y = -Math.PI / 2;
      this.scene.add(this._marker);
    }
    const r = (GEOM.S / this.arch.grid_side) * 2.2;
    this._marker.scale.set(r, r, r);
    this._marker.position.set(p[0] - 0.005, p[1], p[2]);
    this._marker.visible = true;
  }

  // n はシート上の通し番号(MoE は 専門家 × n_ff + n)
  neuronPosition(L, n) {
    const s = this.specs[L];
    if (!s || n < 0 || n >= s.n) return null;
    const cell = this.cellOf[L];
    const [y, z] = cellYZ(s, cell ? cell[n] : n);
    return [sheetX(L), y, z];
  }

  // ニューロンの画面上の位置(px、ページ座標)。自動テストとマーカー用
  screenOf(L, n) {
    const p = this.neuronPosition(L, n);
    if (!p) return null;
    return this._toScreen(new THREE.Vector3(...p));
  }

  _toScreen(v3) {
    const v = v3.clone().project(this.camera);
    const r = this.renderer.domElement.getBoundingClientRect();
    return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height, z: v.z };
  }

  // 地図以外のビューの間は描画の繰り返しを止める(GPU を埋め込みやチャットに譲る)
  setActive(on) {
    this.renderer.setAnimationLoop(on ? () => this._frame() : null);
    if (on) { this.fps = null; this._frames = 0; this._fpsT = performance.now(); this.resize(); }
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this._overlayDirty = true;
    if (!this._userMoved) {                 // パネルの開閉で幅が変わったら視点を合わせ直す
      if (this.front !== null) this.frontView(this.front, { keepMarker: true }); else this.resetView();
    }
  }

  // --- 画面に重ねる HTML(全体ビューの層番号と入力側・出力側、MoE の正面ビューの専門家番号) ------------------
  _updateOverlay() {
    if (!this.overlay) return;
    this._overlayDirty = false;
    const r = this.renderer.domElement.getBoundingClientRect();
    const items = [];
    if (this.front === null) {
      const n = this.specs.length;
      const pts = Array.from({ length: n }, (_, L) => this._toScreen(new THREE.Vector3(layerX(L), -GEOM.S * 0.05, GEOM.S * 0.55)));
      const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
      // 刻み k(1, 2, 5, 10, 20, …)は、隣り合う番号がどこでも 34 px 以上離れる最小のもの
      const steps = [1, 2, 4, 5, 10, 20, 25, 50, 100];
      let k = steps[steps.length - 1];
      for (const st of steps) {
        let ok = true;
        for (let L = 0; L + st < n && ok; L += st) ok = dist(pts[L], pts[L + st]) >= 34;
        if (ok) { k = st; break; }
      }
      const shown = [];
      for (let L = 0; L < n; L += k) shown.push(L);
      if (shown[shown.length - 1] !== n - 1) {                              // 最後の層は必ず出す(近すぎる直前の番号は外す)
        if (dist(pts[shown[shown.length - 1]], pts[n - 1]) < 30) shown.pop();
        shown.push(n - 1);
      }
      for (const L of shown) {
        const p = pts[L];
        if (p.z < 1) items.push({ x: p.x - r.left, y: p.y - r.top, text: `L${L}`, cls: "ax", layer: L });
      }
      const a = pts[0], a2 = pts[Math.min(1, n - 1)], b = pts[n - 1], b2 = pts[Math.max(n - 2, 0)];
      const out = (p, q, px) => { const dx = p.x - q.x, dy = p.y - q.y, l = Math.hypot(dx, dy) || 1; return { x: p.x + (dx / l) * px, y: p.y + (dy / l) * px }; };
      const ia = out(a, a2, 46), ob = out(b, b2, 46);
      items.push({ x: ia.x - r.left, y: ia.y - r.top, text: t("入力側"), cls: "ax-end" });
      items.push({ x: ob.x - r.left, y: ob.y - r.top, text: t("出力側"), cls: "ax-end" });
      const placed = [];                                                   // シートの上辺の中央の少し上。重なる札は上へずらす
      for (const b of this.badges) {
        const s = this.specs[b.layer];
        if (!s || s.kind === "none") continue;
        const p = this._toScreen(new THREE.Vector3(sheetX(b.layer), GEOM.Y0 + Math.max(s.ext, GEOM.S * 0.6) * 1.12, 0));
        if (p.z >= 1) continue;
        let y = p.y - r.top;
        const w = b.text.length * 7.5 + 12;
        while (placed.some((q) => Math.abs(q.x - (p.x - r.left)) < (q.w + w) / 2 + 4 && Math.abs(q.y - y) < 20)) y -= 21;
        placed.push({ x: p.x - r.left, y, w });
        if (y < p.y - r.top - 1) {                                           // ずらした札はシートへ引き出し線を引く
          items.push({ x: p.x - r.left, y, text: "", cls: "lead", h: p.y - r.top - y });
        }
        items.push({ x: p.x - r.left, y, text: b.text, cls: "badge-l", layer: b.layer });
      }
    } else {
      const s = this.specs[this.front];
      if (s.kind === "moe") {                // 専門家のタイルの行の左に、その行の最初の専門家番号
        for (let row = 0; row * s.g < s.nExpert; row++) {
          const [y, z] = cellYZ(s, row * s.g * s.nPer + Math.floor(s.tile / 2) * s.tile);
          const p = this._toScreen(new THREE.Vector3(sheetX(this.front), y, z - s.pitch * 0.5 - s.ext * 0.04));
          items.push({ x: p.x - r.left, y: p.y - r.top, text: `E${row * s.g}`, cls: "ex" });
        }
      }
    }
    const marked = [];
    for (const m of this.marks) {
      if (this.front !== null && m.layer !== this.front) continue;
      const pos = this.neuronPosition(m.layer, m.neuron);
      if (!pos) continue;
      const p = this._toScreen(new THREE.Vector3(...pos));
      if (p.z >= 1) continue;
      const d = 10 + 6 * Math.min(Math.abs(m.dose), 3);
      items.push({ x: p.x - r.left, y: p.y - r.top, text: "", cls: `mk ${m.dose >= 0 ? "plus" : "minus"}`, w: d });
      if (this.front === null) {                                           // 全体ビューではどのユニットか札で。重なる札は下へずらして線を引く
        const sg = m.dose >= 0 ? "+" : "−", text = `L${m.layer} N${m.neuron} ${sg}${+Math.abs(m.dose).toFixed(2)}`;
        const x = p.x - r.left + d / 2 + 4, w = text.length * 7 + 10;
        let y = p.y - r.top;
        while (marked.some((q) => x < q.x + q.w + 4 && q.x < x + w + 4 && Math.abs(q.y - y) < 17)) y += 18;
        marked.push({ x, y, w });
        if (y !== p.y - r.top) items.push({ x: p.x - r.left, y: p.y - r.top, text: "", cls: "lead", h: y - (p.y - r.top) });
        items.push({ x, y, text, cls: `mk-l ${m.dose >= 0 ? "plus" : "minus"}` });
      }
    }
    const svg = this._linkOverlay(items, r);
    const svgAgent = this._agentOverlay(items, r);                       // 札はほかの札(配線・層番号・レシピ)を避けて最後に置く
    this.overlay.innerHTML = svg + svgAgent + items.map((it) => `<span class="${it.cls}"${it.layer !== undefined ? ` data-layer="${it.layer}"` : ""}${
      it.open ? ` data-open="${it.open}"` : ""}${it.title ? ` title="${escHTML(it.title)}"` : ""} style="left:${it.x.toFixed(1)}px;top:${it.y.toFixed(1)}px${
      it.h ? `;height:${it.h.toFixed(1)}px` : ""}${it.w ? `;width:${it.w}px;height:${it.w}px` : ""}">${it.html || it.text}</span>`).join("");
  }

  // エージェントの強調と注釈(§11)。ニューロンごとに 1 枚の札(「agent L22 N4118 ✎ 見送り: …」)にまとめ、
  // 強調は水色の破線の輪、注釈だけのものは小さな水色の輪。札は実際の字の幅で測り、ほかの札・配線の点・凡例・一覧を避けて
  // 右か左に置き、点から札の近い端へ斜めの線を引く。正面ビューではその層のものだけ。文字は必ずエスケープする
  _agentOverlay(items, r) {
    const A = this.agent;
    if (!A.highlights.length && !A.notes.length) return "";
    const ents = new Map();
    for (const h of A.highlights) ents.set(`${h.layer}:${h.neuron}`, { layer: h.layer, neuron: h.neuron, hl: true, dose: h.dose, note: null });
    A.notes.forEach((nt, i) => {
      const k = `${nt.layer}:${nt.neuron}`, e = ents.get(k) || { layer: nt.layer, neuron: nt.neuron, hl: false, dose: null };
      e.note = nt.text;
      e.idx = i;                                                             // 後から貼った注釈ほど大きい
      ents.set(k, e);
    });
    for (const e of ents.values()) if (e.idx === undefined) e.idx = -1;
    const mono = "11px ui-monospace, Menlo, Consolas, monospace", sans = "11px system-ui, sans-serif";
    const boxes = [];
    for (const it of items) {                                                // すでにある札と点の場所
      if (it.text) {
        const c = it.cls.split(" "), w = measure(it.text.replace(/<[^>]+>/g, ""), mono) + 10;
        const left = c.includes("l") || c.includes("ex") ? it.x - w : c.includes("ax") || c.includes("badge-l") ? it.x - w / 2 : it.x;
        boxes.push({ x: left, y: it.y - 8, w, h: 16 });
      } else if (/\b(lk-d|mk|ag-ring)\b/.test(it.cls)) boxes.push({ x: it.x - 6, y: it.y - 6, w: 12, h: 12 });
    }
    for (const sel of ["#legend", "#agent-pop", "#tooltip"]) {               // 凡例と一覧(画面の上に重なる箱)も避ける
      const el = document.querySelector(sel);
      if (!el || el.classList.contains("hidden") || !el.offsetParent) continue;
      const b = el.getBoundingClientRect();
      boxes.push({ x: b.left - r.left - 4, y: b.top - r.top - 4, w: b.width + 8, h: b.height + 8 });
    }
    const free = (x, y, w) => x >= 4 && x + w <= r.width - 4 && y >= 10 && y <= r.height - 10 &&
      !boxes.some((q) => x < q.x + q.w + 3 && q.x < x + w + 3 && y - 9 < q.y + q.h && q.y < y + 9);
    // 引き出し線がほかの札・輪を横切らないか(線の上の点を少しずつ見る。自分の輪は除く)
    const crosses = (x1, y1, x2, y2, own) => boxes.some((q) => q !== own && segHitsRect(x1, y1, x2, y2, q));
    const show = (L) => this.front === null || this.front === L;
    const lines = [], ents2 = [];
    // 地図を動かしている間は、止まっていたときに決めた札の位置(点からのずれ)をそのまま使う(置き場所探しは重い)
    const cache = this._agentMoving && this._agentCache ? this._agentCache : null;
    const keep = cache ? null : new Map();
    for (const e of ents.values()) {                                         // 先に全部の輪を置いて障害物に登録する(札が輪を隠さないように)
      if (!show(e.layer)) continue;
      const pos = this.neuronPosition(e.layer, e.neuron);
      if (!pos) continue;
      const p = this._toScreen(new THREE.Vector3(...pos));
      if (p.z >= 1) continue;
      const px = p.x - r.left, py = p.y - r.top, ring = { x: px - 11, y: py - 11, w: 22, h: 22 };
      items.push({ x: px, y: py, text: "", cls: e.hl ? "ag-ring" : "ag-ring small", w: e.hl ? 20 : 12 });
      boxes.push(ring);
      ents2.push({ e, px, py, ring });
    }
    // 札は 30 枚まで(強調を先に、注釈は新しい順)。残りは輪だけにして、全部は header の agent の一覧で見る
    const order = [...ents2].sort((a, b) => (b.e.hl - a.e.hl) || (b.e.idx - a.e.idx));
    this.agentHidden = Math.max(0, order.length - MAX_AGENT_LABELS);
    for (const { e, px, py, ring } of order.slice(0, MAX_AGENT_LABELS)) {
      const key = `${e.layer}:${e.neuron}`, c0 = cache && cache.get(key);
      if (cache) {
        if (!c0) continue;
        if (c0.line) lines.push(`<line x1="${px.toFixed(1)}" y1="${py.toFixed(1)}" x2="${(px + c0.ax).toFixed(1)}" y2="${(py + c0.y).toFixed(1)}"/>`);
        items.push({ ...c0.item, x: px + c0.ix, y: py + c0.y });
        continue;
      }
      const dose = e.dose === null || e.dose === undefined ? "" : ` ${e.dose >= 0 ? "+" : "−"}${Math.abs(e.dose)}`;
      const short = e.note ? (e.note.length > 28 ? `${e.note.slice(0, 27)}…` : e.note) : "";
      const id = `L${e.layer} N${e.neuron}${dose}`;
      // 近い順に候補を試す: 右・左・上下の中ほど、上下へずらしたもの。札が置けて、線がほかを横切らないもの
      const search = (text, font, maxD, checkCross) => {
        const w = measure("agent", mono) + measure(text, font) + 22, cands = [], step = 19;
        const K = Math.ceil(maxD / step) * 2 + 2;
        for (let k = 0; k < K; k++) {
          const dy = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * step;
          for (const dx of [0, 60, 140, 240]) {                              // 横にも離して探す
            cands.push({ ax: px + 14 + dx, y: py + dy, bx: px + 14 + dx, left: false }, { ax: px - 14 - dx, y: py + dy, bx: px - 14 - dx - w, left: true });
          }
          if (k > 0) for (const sx of [0.5, 0.2, 0.8]) cands.push({ ax: px, y: py + dy, bx: px - w * sx, left: false });
        }
        cands.sort((a, b) => Math.hypot(a.ax - px, a.y - py) - Math.hypot(b.ax - px, b.y - py));
        for (const c of cands) {
          if (Math.hypot(c.ax - px, c.y - py) > maxD) break;
          if (!free(c.bx, c.y, w)) continue;
          if (checkCross && (Math.abs(c.y - py) > 4 || c.ax !== px + 14) && crosses(px, py, c.ax, c.y, ring)) continue;
          return { ...c, w };
        }
        return null;
      };
      // 段階的に広げる: 全文で近く → 番号だけで遠く → 線の交差を許して全文 → 番号だけで画面のどこか
      const full = `${id}${short ? ` ✎ ${short}` : ""}`, compact = e.note ? `${id} ✎` : id;
      let text = full, best = search(full, e.note ? sans : mono, 170, true);
      if (!best) { text = compact; best = search(compact, mono, 320, true); }
      if (!best) { text = full; best = search(full, e.note ? sans : mono, 320, false); }
      if (!best) { text = compact; best = search(compact, mono, 2000, false); }
      if (!best) {
        const w = measure("agent", mono) + measure(text, mono) + 22, bx = Math.min(Math.max(4, px + 14), r.width - w - 4);
        best = { ax: bx, y: py, bx, left: false, w };
      }
      boxes.push({ x: best.bx, y: best.y - 9, w: best.w, h: 18 });
      const line = Math.abs(best.y - py) > 4 || Math.abs(best.ax - (px + 14)) > 1;
      if (line) lines.push(`<line x1="${px.toFixed(1)}" y1="${py.toFixed(1)}" x2="${best.ax.toFixed(1)}" y2="${best.y.toFixed(1)}"/>`);
      const title = e.note ? `${t("エージェントの注釈")} L${e.layer} N${e.neuron}: ${e.note}` : `${t("エージェントの強調")}${A.label ? ` (${A.label})` : ""}: L${e.layer} N${e.neuron}`;
      const item = { x: best.left ? best.ax : best.bx, y: best.y, html: `<i class="who">agent</i>${escHTML(text)}`, text: " ",
                     cls: `ag-lab${e.note && text === full ? " has-note" : ""}${best.left ? " l" : ""}`, open: key, title };
      items.push(item);
      keep.set(key, { item, ix: item.x - px, y: best.y - py, ax: best.ax - px, line });
    }
    if (keep) this._agentCache = keep;
    return lines.length ? `<svg class="lk ag" width="${r.width}" height="${r.height}">${lines.join("")}</svg>` : "";
  }

  // 配線の札(§4.7)。正面ビュー: 開いたニューロンから、シートの右(下流)と左(上流)に並べた札へ 2D の線。
  // 全体ビュー: 3D の線の先に点を打ち、z の大きい上位に札。items に札を足し、2D の線は SVG の文字列で返す
  _linkOverlay(items, r) {
    const E = this.linkEdges, f = this.linkOpts && this.linkOpts.focus;
    if (!E || !E.length || !f) return "";
    const hex = (c) => `rgb(${c.join(",")})`;
    const col = (e) => hex(e.sign >= 0 ? COLORS.plusLine : COLORS.minusLine);
    const lab = (L, n) => `L${L} N${n}`;
    const down = E.filter((e) => e.sl === f.layer && e.sn === f.n).sort((a, b) => b.z - a.z);
    const up = E.filter((e) => e.dl === f.layer && e.dn === f.n).sort((a, b) => b.z - a.z);
    if (this.front !== null) {
      if (this.front !== f.layer) return "";
      const p0 = this.neuronPosition(f.layer, f.n);
      if (!p0) return "";
      const o = this._toScreen(new THREE.Vector3(...p0)), s = this.specs[f.layer];
      const tl = this._toScreen(new THREE.Vector3(sheetX(f.layer), GEOM.Y0 + s.ext, -s.ext / 2));
      const br = this._toScreen(new THREE.Vector3(sheetX(f.layer), GEOM.Y0, s.ext / 2));
      const x0 = o.x - r.left, y0 = o.y - r.top, lines = [];
      const side = (es, x, right) => {
        const n = Math.min(es.length, 14), top = tl.y - r.top, h = br.y - tl.y;
        es.slice(0, n).forEach((e, i) => {
          const y = top + (h * (i + 0.5)) / n;
          const other = right ? lab(e.dl, e.dn) : lab(e.sl, e.sn);
          lines.push(`<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${col(e)}" stroke-width="1.3" stroke-opacity="${(0.45 + 0.55 * Math.min(1, e.z / es[0].z)).toFixed(2)}"${right ? "" : ' stroke-dasharray="4 3"'}/>`);
          items.push({ x, y, text: `${right ? "→ " : ""}${other} ${e.sign >= 0 ? "+" : "−"}${right ? "" : " →"}`, cls: `lk-l ${right ? "r" : "l"} ${e.sign >= 0 ? "plus" : "minus"}` });
        });
      };
      side(down, br.x - r.left + 18, true);
      side(up, tl.x - r.left - 18, false);
      if (down.length) items.push({ x: br.x - r.left + 18, y: tl.y - r.top - 16, text: t("下流(書き込む先)"), cls: "lk-h r" });
      if (up.length) items.push({ x: tl.x - r.left - 18, y: tl.y - r.top - 16, text: t("上流(駆動するもの)"), cls: "lk-h l" });
      return `<svg class="lk" width="${r.width}" height="${r.height}">${lines.join("")}</svg>`;
    }
    // 全体ビュー: 線の先に点、上位 4 本ずつに札
    const tops = new Set([...down.slice(0, 4), ...up.slice(0, 4)]), tagged = [];
    for (const e of E) {
      const other = e.sl === f.layer && e.sn === f.n ? [e.dl, e.dn] : e.dl === f.layer && e.dn === f.n ? [e.sl, e.sn] : [e.dl, e.dn];
      const pos = this.neuronPosition(...other);
      if (!pos) continue;
      const p = this._toScreen(new THREE.Vector3(...pos));
      if (p.z >= 1) continue;
      items.push({ x: p.x - r.left, y: p.y - r.top, text: "", cls: `lk-d ${e.sign >= 0 ? "plus" : "minus"}${e.dim ? " dim" : ""}` });
      if (tops.has(e)) tagged.push({ x: p.x - r.left, y: p.y - r.top, text: `${lab(...other)} ${e.sign >= 0 ? "+" : "−"}`, sign: e.sign });
    }
    // 札は重ならないように縦にずらし、ずらしたものは点へ引き出し線。開いたニューロン(半径 20 px)と、層の札・床の番号も避ける
    const placed = items.filter((it) => it.text && /^(ax|badge-l)/.test(it.cls))
      .map((it) => ({ x: it.x - (it.text.length * 7 + 10) / 2, y: it.y + (it.cls.startsWith("badge") ? -10 : 8), w: it.text.length * 7 + 10 }));
    const fp = this.neuronPosition(f.layer, f.n);
    if (fp) { const q = this._toScreen(new THREE.Vector3(...fp)); placed.push({ x: q.x - r.left - 20, y: q.y - r.top, w: 40 }); }
    tagged.sort((a, b) => a.y - b.y);
    for (const t of tagged) {
      const x = t.x + 6, w = t.text.length * 7 + 10;
      let y = t.y;
      for (let k = 1; placed.some((q) => x < q.x + q.w + 4 && q.x < x + w + 4 && Math.abs(q.y - y) < 16) && k < 40; k++) {
        y = t.y + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 17;
      }
      placed.push({ x, y, w });
      if (y !== t.y) items.push({ x: t.x, y: Math.min(y, t.y), text: "", cls: "lead lk", h: Math.abs(y - t.y) });
      items.push({ x, y, text: t.text, cls: `lk-l r ${t.sign >= 0 ? "plus" : "minus"}` });
    }
    return "";
  }

  // --- ピッキング ----------------------------------------------------------------------------------------
  _ndc(ev) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return { x: ((ev.clientX - r.left) / r.width) * 2 - 1, y: -((ev.clientY - r.top) / r.height) * 2 + 1,
             px: ev.clientX - r.left, py: ev.clientY - r.top };
  }

  _pickLabel(m) {
    this._ray.setFromCamera(new THREE.Vector2(m.x, m.y), this.camera);
    const hit = this._ray.intersectObjects(this.labels.filter((l) => l.visible), false)[0];
    return hit ? hit.object.userData.layer : null;
  }

  // レイと各層のシート平面 x = sheetX(L) の交点。シートの中ならセル → ニューロン。複数当たればカメラに近い層。
  _pick(m) {
    this._ray.setFromCamera(new THREE.Vector2(m.x, m.y), this.camera);
    const o = this._ray.ray.origin, d = this._ray.ray.direction;
    if (Math.abs(d.x) < 1e-9) return null;
    let best = null;
    const Ls = this.front !== null ? [this.front] : this.specs.map((_, i) => i);
    for (const L of Ls) {
      const t = (sheetX(L) - o.x) / d.x;
      if (t <= 0 || (best && t >= best.t)) continue;
      const y = o.y + t * d.y, z = o.z + t * d.z, s = this.specs[L];
      let cell = yzCell(s, y, z), shared = false;
      if (cell < 0) { cell = sharedCell(s, y, z); shared = cell >= 0; }
      if (cell < 0) continue;
      const hit = { layer: L, t, shared, px: m.px, py: m.py };
      if (shared) hit.neuron = cell;
      else if (s.kind === "moe") { hit.expert = Math.floor(cell / s.nPer); hit.neuron = cell % s.nPer; hit.cell = cell; }
      else {
        const inv = this.neuronAt[L];
        hit.neuron = inv ? inv[cell] : cell; hit.cell = cell;
        if (hit.neuron < 0) continue;                                      // 配置の空きマス
      }
      best = hit;
    }
    return best;
  }

  _setHover(h) {
    const same = (a, b) => a && b && a.layer === b.layer && a.neuron === b.neuron && a.expert === b.expert && a.shared === b.shared;
    if (same(h, this.hover)) { if (h) this.onHover(h); return; }
    this.hover = h;
    this.onHover(h);
  }

  // --- 描画 ---------------------------------------------------------------------------------------------
  _frame() {
    this.controls.update();
    const h = this.renderer.domElement.height;
    this.uniforms.uScale.value = h / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    if (this.front === null) {
      const t = this.controls.target.x / GEOM.GAP;
      this.uniforms.uFocus.value = Math.max(0, Math.min(this.specs.length - 1, Math.round(t)));
    }
    if (this._dirtyPick && this._mouse) {                       // hover はフレームごとに最大 1 回
      this._dirtyPick = false;
      this._setHover(this._pick(this._mouse));
    }
    // カメラが動いている間は札の置き直しを省き、止まって 150 ms たったら置き直す
    const cm = this.camera.matrixWorld.elements;
    const sig = `${cm[12].toFixed(4)},${cm[13].toFixed(4)},${cm[14].toFixed(4)},${cm[8].toFixed(4)},${cm[9].toFixed(4)}`;
    const now0 = performance.now();
    if (sig !== this._camSig) { this._camSig = sig; this._camT = now0; this._agentMoving = !!this._agentCache; }
    else if (this._agentMoving && now0 - this._camT > 150) { this._agentMoving = false; this._overlayDirty = true; }
    if (this._overlayDirty) this._updateOverlay();
    this.renderer.render(this.scene, this.camera);
    this._fullUpload = false;
    this._frames++;
    const now = performance.now();
    if (now - this._fpsT > 1000) {
      this.fps = (this._frames * 1000) / (now - this._fpsT);
      this._frames = 0; this._fpsT = now;
    }
  }
}
