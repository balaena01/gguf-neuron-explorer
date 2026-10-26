"""V5 の参照実装: HF transformers で「層 L の down_proj 入力のニューロン n に add を全位置で足す」greedy 生成。

gne とは別の Python(torch + transformers がある環境)で動かす。gne は import しない。
  python tools/hf_reference.py --model DIR --request req.json --out ref.json

req.json: {"prompt_ids": [...], "layer": L, "neuron": n, "add": c, "max_tokens": 32,
           "n_vocab": V, "eog_ids": [...], "dump_down": true}
ref.json: {"base": {...}, "steer": {...}}  各 {"ids", "top_ids", "top_lps", "next_lp"}
dump_down なら ref.json の隣に down_L{L}.npy(W_down, (n_embd, n_ff) float32)を書く(R4 の確認用)。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    req = json.loads(Path(args.request).read_text())
    L, n, add = int(req["layer"]), int(req["neuron"]), float(req["add"])
    n_vocab, eog = int(req["n_vocab"]), set(req["eog_ids"])
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    down = model.model.layers[L].mlp.down_proj

    def push(_mod, inputs):
        x = inputs[0].clone()
        x[..., n] += add
        return (x,)

    @torch.no_grad()
    def greedy(steer: bool) -> dict:
        h = down.register_forward_pre_hook(push) if steer else None
        try:
            ids = torch.tensor([req["prompt_ids"]], device=args.device)
            out = model(ids, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1, :n_vocab]
            res = {"ids": [], "top_ids": [], "top_lps": [], "next_lp": None}
            for step in range(int(req["max_tokens"])):
                lp = torch.log_softmax(logits.float(), dim=-1)
                if step == 0:
                    res["next_lp"] = lp.cpu().numpy().tolist()
                top = torch.topk(lp, 5)
                tok = int(top.indices[0])
                res["ids"].append(tok)
                res["top_ids"].append(top.indices.cpu().tolist())
                res["top_lps"].append(top.values.cpu().tolist())
                if tok in eog:
                    break
                out = model(torch.tensor([[tok]], device=args.device), past_key_values=past, use_cache=True)
                past, logits = out.past_key_values, out.logits[0, -1, :n_vocab]
            return res
        finally:
            if h is not None:
                h.remove()

    ref = {"base": greedy(False), "steer": greedy(True), "dtype": args.dtype}
    Path(args.out).write_text(json.dumps(ref))
    if req.get("dump_down"):
        w = down.weight.detach().float().cpu().numpy()     # (n_embd, n_ff)
        np.save(Path(args.out).with_name(f"down_L{L}.npy"), w)


if __name__ == "__main__":
    main()
