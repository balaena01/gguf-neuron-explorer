// 3D 層マップ(別冊 §4)。全ニューロンの点は 1 つのジオメトリ・1 draw call。ピッキングはレイと各層の
// シート平面の交点から解析的に求める(点群へのレイキャストはしない、§4.2)。
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
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

export class LayerMap {
  constructor(container, arch, { onHover, onClick, onLayer, onContext, overlay } = {}) {
    this.container = container;
    this.arch = arch;
    this.onHover = onHover || (() => {});
    this.onClick = onClick || (() => {});
    this.onLayer = onLayer || (() => {});
    this.onContext = onContext || null;  // 層の番号・ラベルの右クリック(L, イベント)
    this.marks = [];                   // レシピのユニット [{ layer, neuron, dose }](§4.8)
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
        const L = ev.target.dataset && ev.target.dataset.layer;
        if (L !== undefined) this.frontView(+L);
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
      const tag = { attention: "attention", linear_attention: "線形注意", recurrent: "回帰", none: "" }[li.mixer];
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
    if (this._marker) this._marker.visible = false;
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
    if (this._marker && !keepMarker) this._marker.visible = false;
    this._applyLayerDimming();
    this.onLayer(L);
  }

  focusNeuron(L, n) {
    this.frontView(L);
    this.markNeuron(L, n);
  }

  // カメラは動かさずに、ニューロンに輪の印を付ける
  markNeuron(L, n) {
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
    const [y, z] = cellYZ(s, n);
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
      items.push({ x: ia.x - r.left, y: ia.y - r.top, text: "入力側", cls: "ax-end" });
      items.push({ x: ob.x - r.left, y: ob.y - r.top, text: "出力側", cls: "ax-end" });
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
    this.overlay.innerHTML = items.map((it) => `<span class="${it.cls}"${it.layer !== undefined ? ` data-layer="${it.layer}"` : ""} style="left:${it.x.toFixed(1)}px;top:${it.y.toFixed(1)}px${it.h ? `;height:${it.h.toFixed(1)}px` : ""}${it.w ? `;width:${it.w}px;height:${it.w}px` : ""}">${it.text}</span>`).join("");
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
      else { hit.neuron = cell; hit.cell = cell; }
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
