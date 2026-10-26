// gne explorer: 画面全体の配線(別冊 §2)。U0: 構造の描画、層の選択、ジャンプ、hover の L N。
import { LayerMap } from "./map.js";
import { syntheticArch } from "./layout.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);

const state = { model: null, arch: null, vocab: null, map: null, view: "map", layer: 0 };
window.gne = state;                       // 自動テスト(playwright)と開発用

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json();
}

const MIXER = { attention: "attention", linear_attention: "線形注意", recurrent: "回帰", none: "mixer なし" };
const mixerLabel = (li) => MIXER[li.mixer];
const isMoE = () => state.arch.layers.some((li) => li.ffn.startsWith("moe"));

function describeLayer(L) {
  const li = state.arch.layers[L];
  const ffn = li.ffn === "dense" ? `MLP ${li.n_ff}` : li.ffn.startsWith("moe")
    ? `MoE ${li.n_expert}×${li.n_ff}(使用 ${li.n_expert_used})${li.n_ff_shexp ? ` + shared ${li.n_ff_shexp}` : ""}` : "FFN なし";
  const heads = li.n_head ? ` ${li.n_head}h${li.n_head_kv && li.n_head_kv !== li.n_head ? `/${li.n_head_kv}kv` : ""}` : "";
  return `L${L} ${mixerLabel(li)}${heads} · ${ffn}`;
}

function showTooltip(h) {
  const tip = $("#tooltip");
  if (!h) { tip.classList.add("hidden"); return; }
  const who = h.shared ? `shared N${h.neuron}` : h.expert !== undefined ? `E${h.expert} N${h.neuron}` : `N${h.neuron}`;
  tip.textContent = `L${h.layer} ${who}`;
  tip.classList.remove("hidden");
  const wrap = $("#map-wrap").getBoundingClientRect();
  const x = Math.min(h.px + 14, wrap.width - tip.offsetWidth - 6), y = Math.min(h.py + 14, wrap.height - tip.offsetHeight - 6);
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}

// マップの層正面ビューが変わったとき(ラベルのクリック・選択欄・ジャンプのどれから来ても)に呼ばれる
function onLayerChanged(L) {
  if (L !== null) { state.layer = L; $("#layer-select").value = String(L); }
  $("#legend").classList.toggle("compact", L !== null);     // 正面ビューでは凡例を色の見本だけに(シートを隠さないように)
  updateStatus();
}

function updateStatus() {
  const m = state.model ? state.model.meta : { name: state.arch.arch };
  const map = state.map;
  const where = map && map.front !== null ? describeLayer(map.front) : `${state.arch.n_layer} 層`;
  $("#status-main").textContent = `${m.name} · ${where}`;
  $("#status-main").title = $("#status-main").textContent;
  $("#status-perf").textContent = `${(map?.nPoints ?? 0).toLocaleString()} 点 · ${map ? map.fps.toFixed(0) : "-"} fps`;
}

// 凡例: このモデルにある種類だけを出し、構造の読み方も書く
function legend() {
  const kinds = new Set(state.arch.layers.map((li) => li.mixer));
  const parts = [`<span class="sw" style="background:#10141b"></span>データ無し`];
  if (kinds.has("attention")) parts.push(`<span class="sw" style="background:#9fb3cf"></span>attention(太い縁)`);
  if (kinds.has("linear_attention")) parts.push(`<span class="sw" style="background:#3a4656"></span>線形注意(細い縁)`);
  if (kinds.has("recurrent")) parts.push(`<span class="sw" style="background:#3a4656"></span>回帰(細い縁)`);
  const moe = isMoE() ? "、MoE は専門家ごとの小さな正方形" : "";
  $("#legend").innerHTML = parts.join("") +
    `<br><span class="note">各層 = 手前の板(mixer、上辺の刻みが head の数)+ 奥のシート(MLP の全ニューロン${moe})、下の線 = 残差</span>` +
    `<br><span class="note">シート内の並びはニューロン番号順(近くにあっても似ているとは限らない)</span>`;
}

// "L14 N8949" / "14:8949" / MoE は "L10 E93 N252"
function parseJump(text) {
  const m = text.trim().match(/^L?(\d+)\s*[\s:/,]\s*(?:E(\d+)\s*[\s:/,]\s*)?N?(\d+)$/i);
  if (!m) return null;
  return { layer: +m[1], expert: m[2] !== undefined ? +m[2] : null, neuron: +m[3] };
}

function wireHeader() {
  const sel = $("#layer-select");
  sel.innerHTML = state.arch.layers.map((li) => `<option value="${li.index}">${describeLayer(li.index)}</option>`).join("");
  sel.addEventListener("change", () => state.map.frontView(+sel.value));
  const jump = $("#jump");
  jump.placeholder = isMoE() ? "L10 E0 N0" : "L14 N8949";
  jump.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter") return;
    const j = parseJump(ev.target.value);
    const li = j && state.arch.layers[j.layer];
    const moe = li && li.ffn.startsWith("moe");
    const ok = li && (moe ? j.expert !== null && j.expert < li.n_expert && j.neuron < li.n_ff : j.expert === null && j.neuron < li.n_ff);
    ev.target.style.borderColor = ok ? "" : "#c94fa7";
    if (!ok) return;
    state.map.focusNeuron(j.layer, moe ? j.expert * li.n_ff + j.neuron : j.neuron);
    location.hash = moe ? `#L${j.layer}/E${j.expert}/N${j.neuron}` : `#L${j.layer}/N${j.neuron}`;
  });
  $("#reset-view").addEventListener("click", () => state.map.resetView());
  for (const b of document.querySelectorAll("#views button")) {
    b.addEventListener("click", () => {
      state.view = b.dataset.view;
      for (const x of document.querySelectorAll("#views button")) x.classList.toggle("on", x === b);
      const ph = $("#view-placeholder");
      ph.classList.toggle("hidden", state.view === "map");
      ph.textContent = state.view === "layers" ? "layers ビュー(深さプロファイル)は U1 で作る" : "embed ビューは U4 で作る";
    });
  }
  for (const t of document.querySelectorAll(".panel .toggle")) {
    t.addEventListener("click", () => { $(`#${t.dataset.target}`).classList.toggle("collapsed"); state.map.resize(); });
  }
}

async function main() {
  const synthetic = params.get("synthetic");
  if (synthetic) {
    state.arch = syntheticArch(synthetic);
  } else {
    state.model = await getJSON("/api/model");
    state.arch = state.model.arch;
  }
  document.title = `gne · ${state.model ? state.model.meta.name : state.arch.arch}`;
  const t0 = performance.now();
  state.map = new LayerMap($("#map"), state.arch, {
    onHover: showTooltip,
    onClick: (h) => { $("#jump").value = h.expert !== undefined ? `L${h.layer} E${h.expert} N${h.neuron}` : `L${h.layer} N${h.neuron}`; },
    onLayer: onLayerChanged,
    overlay: $("#overlay"),
  });
  state.initMs = performance.now() - t0;
  wireHeader();
  legend();
  updateStatus();
  setInterval(updateStatus, 1000);
  const m = location.hash.match(/^#L(\d+)(?:\/E(\d+))?\/N(\d+)/);
  if (m) {
    const li = state.arch.layers[+m[1]];
    if (li) state.map.focusNeuron(+m[1], m[2] !== undefined ? +m[2] * li.n_ff + +m[3] : +m[3]);
  }
  state.ready = true;
}

main().catch((e) => { $("#status").textContent = `エラー: ${e.message}`; console.error(e); state.error = String(e); });
