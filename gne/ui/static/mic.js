// マイクの録音(チャットで音声で話しかける)。MediaRecorder で録り、ブラウザの中で 16 kHz モノの 16 bit WAV にして
// base64 で返す(libmtmd が読めるのは wav / mp3 / flac で、MediaRecorder の webm は読めない)。
// getUserMedia は安全なページ(http://127.0.0.1・localhost)でだけ使える。
const RATE = 16000;

export class Mic {
  constructor() {
    this.rec = null;
    this.stream = null;
    this.chunks = [];
    this.t0 = 0;
  }

  static available() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
  }

  get recording() { return !!this.rec; }

  seconds() { return this.rec ? (performance.now() - this.t0) / 1000 : 0; }

  async start() {
    if (!Mic.available()) throw new Error("no-mic");
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream);
    this.rec.addEventListener("dataavailable", (ev) => { if (ev.data && ev.data.size) this.chunks.push(ev.data); });
    this.rec.start();
    this.t0 = performance.now();
  }

  // 録音を止めて { b64, seconds } を返す。録れていなければ null
  async stop() {
    const rec = this.rec;
    if (!rec) return null;
    const done = new Promise((res) => rec.addEventListener("stop", res, { once: true }));
    rec.stop();
    await done;
    for (const t of this.stream.getTracks()) t.stop();
    this.rec = this.stream = null;
    if (!this.chunks.length) return null;
    const blob = new Blob(this.chunks, { type: rec.mimeType || "audio/webm" });
    const pcm = await toMono16k(await blob.arrayBuffer());
    if (!pcm.length) return null;
    return { b64: b64(wav(pcm, RATE)), seconds: pcm.length / RATE };
  }

  cancel() {
    if (this.rec) { try { this.rec.stop(); } catch { /* 止まっていればよい */ } }
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    this.rec = this.stream = null;
    this.chunks = [];
  }
}

async function toMono16k(buf) {
  const ctx = new AudioContext();
  let audio;
  try { audio = await ctx.decodeAudioData(buf); } finally { ctx.close(); }
  const n = Math.max(1, Math.ceil(audio.duration * RATE));
  const off = new OfflineAudioContext(1, n, RATE);            // 1 チャンネルに混ぜながら 16 kHz に直す
  const src = off.createBufferSource();
  src.buffer = audio;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

function wav(pcm, rate) {
  const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const str = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, rate, true); out.setUint32(28, rate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  str(36, "data"); out.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
  return new Uint8Array(out.buffer);
}

function b64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
