// サーバとの通信(別冊 §8.2)。大きい配列は octet-stream で受け、形は X-Shape、付帯情報は X-Meta。

async function check(r, url) {
  if (r.ok) return r;
  let msg = `${r.status}`;
  try { const j = await r.json(); msg = j.error ? `${j.error.message}${j.error.hint ? `(${j.error.hint})` : ""}` : JSON.stringify(j); }
  catch { /* 本文が JSON でない */ }
  throw new Error(`${url}: ${msg}`);
}

export async function getJSON(url, params) {
  const u = params ? `${url}?${new URLSearchParams(params)}` : url;
  return (await check(await fetch(u), u)).json();
}

export async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return (await check(r, url)).json();
}

// { buf: ArrayBuffer, shape: "n" | "n,24;n,2", meta: {...} }
export async function getBin(url, params) {
  const u = params ? `${url}?${new URLSearchParams(params)}` : url;
  const r = await check(await fetch(u), u);
  return { buf: await r.arrayBuffer(), shape: r.headers.get("X-Shape"), meta: JSON.parse(r.headers.get("X-Meta") || "{}") };
}

// 層ごとのデータのキャッシュ。同じものを同時に頼まれても 1 回だけ取りに行く。読み終えたものは peek で同期的に見られる
export class LayerCache {
  constructor(load) { this.load = load; this.pending = new Map(); this.done = new Map(); }
  get(key) {
    if (this.done.has(key)) return Promise.resolve(this.done.get(key));
    if (!this.pending.has(key)) {
      this.pending.set(key, this.load(key).then(
        (v) => { this.done.set(key, v); this.pending.delete(key); return v; },
        (e) => { this.pending.delete(key); throw e; }));
    }
    return this.pending.get(key);
  }
  peek(key) { return this.done.get(key); }
  clear() { this.pending.clear(); this.done.clear(); }
}

// 語彙は大きい(15 万語で gzip 1 MB 強)ので、ブラウザの IndexedDB に model_id ごとに置く(別冊 U-R6)。
// 使えない環境(プライベートウィンドウなど)では毎回取りに行く
function idb() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open("gne", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("vocab");
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

export async function loadVocab(modelId) {
  let db = null;
  try {
    db = await idb();
    const hit = await new Promise((resolve) => {
      const q = db.transaction("vocab").objectStore("vocab").get(modelId);
      q.onsuccess = () => resolve(q.result);
      q.onerror = () => resolve(undefined);
    });
    if (Array.isArray(hit)) return hit;
  } catch { db = null; }
  const v = await getJSON("/api/vocab");
  if (db) {
    try { db.transaction("vocab", "readwrite").objectStore("vocab").put(v, modelId); } catch { /* 保存できなくても使える */ }
  }
  return v;
}

// POST して SSE(text/event-stream)を受ける。EventSource は POST できないので fetch の本文を読む
export async function postSSE(url, body, onEvent, signal) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
  await check(r, url);
  const rd = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await rd.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of chunk.split("\n")) if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

export async function deleteJSON(url) {
  return (await check(await fetch(url, { method: "DELETE" }), url)).json();
}
