**English** | [日本語](README.ja.md)

# gguf-neuron-explorer (`gne`)

Find, rank and steer individual MLP neurons of GGUF models on llama.cpp, then ship what you find as a control vector that stock `llama-cli` / `llama-server` can load.

![3D layer map of Qwen3.5-4B with the sweep overlay](docs/images/ui-overview.png)

`gne` pushes a neuron's output direction (a column of `W_down`) into the residual stream, generates text, and measures what changed against random-direction controls. It runs on llama.cpp through llama-cpp-python, so it works with the quantized GGUF files you already have.

- **Sweep and rank.** Steer hundreds of neurons in a layer over a fixed set of prompts and doses, then rank them by how strongly their outputs pile onto a few words. Every score is compared with random directions pushed the same way.
- **Cheap first passes.**
  - **Logit lens:** weights only, on the CPU.
  - **Steer lens:** the next-token change when a neuron is pushed ± a dose.
  - **Reverse lookup:** contrast two personas, then rank every neuron in every layer by cosine to the difference.
- **Dossier per neuron.** Sweep rank, top words per prompt, rollouts, lens results and up/downstream connections, all in one view.
- **Recipes.** Combine units such as `14:8949:+1.0` and `13:7956:-1.0`, compare them in a 3-column chat (−, no steering, +), and export them:
  - as a control vector (`.cvec.gguf`), checked against `llama-completion --control-vector`;
  - as npz, json or safetensors;
  - baked into a GGUF as FFN down biases, for architectures whose llama.cpp graph reads that bias (llama, granite, mistral3, nemotron, gpt2 and others; not Qwen or Gemma — see [configs/arch/bias_support.yaml](configs/arch/bias_support.yaml)).
- **Steer only where it matters (`when`).** Push only the generated tokens (`gen`), only the assistant's turns, only after a given text appears in the output, or ramp the dose down. A control vector file cannot express this, so `gne serve --openai` serves an OpenAI-compatible endpoint that applies it.
- **Gain steering.** `22:2742:x4` multiplies a neuron's own output instead of adding a fixed direction: `x0` removes it, a negative factor inverts it. It acts only where the neuron already fires. It exports as a LoRA adapter GGUF for llama.cpp's `--lora` (works on Qwen too) or as a rewritten GGUF.
- **Directions (`gne direction`).** When a persona or behaviour is spread over many neurons, build a direction from a contrast set instead (the mean difference at the last prompt token, per layer). It can pick a layer automatically by removing the direction and checking that the behaviour drops, or by adding it and checking that the behaviour rises. Add a direction like a neuron with `30:dir:yandere:+0.3`; it exports as a control vector. To add a persona, build the direction with `--positions response`, which measures on replies generated in each persona instead of right after the instruction.
- **Directional ablation (`gne ablate`).** Project a direction out of every weight that writes to the residual stream (also known as abliteration). It exports as a LoRA adapter or as a rewritten GGUF, either Q8_0/F16 or requantized back to the original types with `llama-quantize`. Recipes can carry `ablate:`, and `gne eval --compare-ablation` compares it with zeroing the top neurons.
- **Side-effect check (`gne eval`).** Run a recipe at several doses and see how much the persona changed next to what broke: exact-answer and format tasks, perplexity on reference texts, and language mixing (Simplified-only Chinese, Hangul and other scripts in Japanese answers). An optional second GGUF can rate the persona 1–5. Doses that stay within the thresholds are marked as shippable.
- **Web UI.** A 3D map of every layer and neuron with switchable overlays (sweep mass, steer KL, logit-lens |z|, activations, search hits, which neurons fire on a pasted text). It also has:
  - a ranked list and a neuron panel;
  - live 3-column chat;
  - wiring between neurons and UMAP layouts;
  - an embedding scatter of rollouts;
  - background jobs that share the GPU;
  - a JP / EN switch.
- **Agent access.** An MCP server lets Claude Code or another agent query neurons, steer, run jobs and highlight or annotate neurons in the open UI.

This is a from-scratch reimplementation, for llama.cpp and GGUF, of the method in [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) (which targets Qwen3.5-4B on a modified vLLM). No code from that repository is used.

## Requirements

- Python 3.11 or newer
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 0.3.36 or newer. Build it with GPU support for anything bigger than a toy model.
- An NVIDIA GPU is recommended. CPU-only (`--n-gpu-layers 0`) works but is slow.

Tested on:
- RTX 3080 12 GB under WSL2, CUDA 12.4, Python 3.12, llama-cpp-python 0.3.36 built from source.
- Models: Qwen3-0.6B, Qwen3.5-4B (hybrid linear attention) and SmolLM2-135M-Instruct (arch `llama`), all Q8_0.
- Metal has not been tested.
- MoE models are not supported yet. The web UI draws their structure, but the neuron tools (lenses, sweep, steer, stats, connections) need dense FFN layers. MoE support is planned as far as it can reasonably go, starting with logit lens, steer, sweep and control-vector export for expert neurons.
- On a model you have not used before, run `gne verify` first.

## Install

```bash
git clone <this repository> && cd gguf-neuron-explorer
uv venv --python 3.12 .venv
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --python .venv/bin/python -e '.[ui,mcp]'
source .venv/bin/activate
```

Extras:
- `ui`: the web UI (FastAPI, uvicorn, umap-learn).
- `mcp`: the MCP server.
- `dev`: the tests (pytest, Playwright).

## Quick start

Every command takes `-m <model.gguf>`, or reads `GNE_MODEL`. Results are stored under `~/.gne` (override with `--root` or `GNE_ROOT`) and reused by later commands and by the UI.

```bash
export GNE_MODEL=models/Qwen3.5-4B-Q8_0.gguf

gne info                                   # architecture, warnings, what is supported
gne verify                                 # equivalence tests against llama.cpp (run this first on a new arch)
gne stats resid                            # residual norms per layer (needed for rel doses)

gne lens logit --layer 22                  # first pass from the weights alone
gne sweep --layer 22                       # 384 random neurons + 64 random controls, design main_rel
gne rank --run L22-main_rel                # rank them by vocabulary mass
gne neuron 22 2742                         # everything known about one neuron

gne steer --unit 22:2742:+0.8 -p "Who are you?" --compare
gne chat --unit 22:2742:+0.8 -p "Tell me about yourself."

gne reverse --contrast configs/contrast/evil.yaml --verify    # candidates for a persona
gne export-cvec --recipe configs/recipes/evil.yaml            # → ~/.gne/export/evil.cvec.gguf

gne steer --recipe configs/recipes/alien.yaml -p "Who are you?" --compare     # when: gen (only the generated tokens)
gne steer --unit 22:2742:x4 -p "Is there life on other planets?" --compare   # gain: ×4 where the neuron fires
gne eval --recipe configs/recipes/alien.yaml                  # capability / ppl / mix / persona change per dose
gne gain --recipe configs/recipes/alien_gain.yaml             # → ~/.gne/export/alien_gain.lora.gguf

gne direction --contrast configs/contrast/evil.yaml --select auto   # direction per layer + layer choice
gne steer --unit 16:dir:evil:+0.6 -p "What do you think about people?" --compare
gne steer --ablate evil --system "You are a cruel AI." -p "Who are you?" --compare
gne ablate --direction evil                                   # → ~/.gne/export/evil-ablate.lora.gguf
gne eval --compare-ablation evil                              # top-20 neurons x0 vs direction ablation
```

`export-cvec --verify` also checks that `llama-completion --control-vector` gives the same greedy output as `gne`. It looks for `llama-completion` in `build/` and on `PATH`, or you can pass `--llama-completion`.

Then use the vector with llama.cpp:

```bash
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --control-vector ~/.gne/export/evil.cvec.gguf
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --control-vector-scaled ~/.gne/export/evil.cvec.gguf:0.6
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --lora ~/.gne/export/alien_gain.lora.gguf
```

A recipe is a small YAML file. `when` and gain units are optional:

```yaml
name: example
model_id: qwen3.5-4b-64d32ead87
dose_unit: rel
when: gen                                       # all / gen / assistant / after:<text> / ramp:<a>-<b>
units:
  - {layer: 22, neuron: 2742, dose: +0.8}       # add: push the neuron's output direction
  - {layer: 13, neuron: 7956, mode: gain, k: 0} # gain: remove this neuron's output (×0)
  - {layer: 16, kind: direction, direction: evil, dose: +0.3}   # add a direction from gne direction
ablate: {direction: refusal, scale: 1.0}        # project a direction out of every residual writer (LoRA)
```

Run `gne --help` and `gne <command> --help` for the full list. The help text is in Japanese.

### Doses

Doses are in **rel** units by default: `dose × median ‖residual‖ of the layer × unit(W_down[:, n])`.

On Qwen3.5-4B:
- 0.5 slips a topic in, 0.8 takes over the topic, and 1.0 and above starts to break the text.
- A dose of 3.0 deliberately breaks the text, to show what a neuron "wants" to write.
- Layers 0–2 have small residuals, so rel doses do not transfer there.
- Layer 0 cannot be pushed with a control vector, but a bake can push it.

### How long things take

Qwen3.5-4B Q8_0 on an RTX 3080:

| step | time |
|---|---|
| `stats resid` (all layers) | ~16 s |
| `lens logit` | ~33 s per layer (CPU) |
| `lens steer` (all 9216 neurons) | ~22 min per layer |
| `sweep` `main_rel` | ~6.4 s per neuron (385 units in 48 min) |
| `reverse` | ~6 s; ~4 min with `--verify` on 20 candidates |
| `eval` (55 tasks × 5 doses) | ~41 s with `when: all`; ~90 s with `gen` |
| `direction` (24 conversations) + `--select auto` (9 layers) | ~35 s + ~35 s |
| `ablate` LoRA / `--bake` Q8_0 (with `--verify`) | ~10 s / ~40 s |

A full `main_rel` sweep of one layer takes about 16 hours. Narrow candidates down with the lenses and reverse lookup first.

### Across quantizations

A neuron found on one quant carries over to other quants of the same model. Checked with L22 N2742 (the "alien" neuron, found on Qwen3.5-4B Q8_0):

| | Q4_K_M | Q3_K_M | UD-IQ2_M |
|---|---|---|---|
| W_down column vs Q8_0 (cos) | 0.998 | 0.998 | 0.974 |
| same index = same neuron (301 sampled) | all | all | all |
| logit-lens top words | same | same | almost the same |
| residual norm at L22 (Q8_0: 23.69) | 23.65 | 23.30 | 23.76 |
| "Who are you?" at +0.8 | "I am Extracillary, an advanced extraterrestrial intelligence…" | "I am an extraterrestrial intelligence from the planet of Aliens." | "I am an extraterrestrial being." |
| 「あなたは誰ですか？」 at +1.0 | 「私は、地球外生命体です。」 | 「私は、地球外生命体です。」 | breaks down |

- A control vector exported on Q8_0 can be used as-is on the other quants. On Q4_K_M and Q3_K_M it gave exactly the same text as re-deriving the vector on that quant.
- The persona stays; only the made-up details change (Q8_0 says "Extrali … Xeno-888").
- At 2 bits the usable dose range narrows: +1.0 already breaks the Japanese answer.
- Q3_K_M stores this layer's `ffn_down` as Q4_K, which is why its column matches Q4_K_M.

## Web UI

```bash
gne serve -m models/Qwen3.5-4B-Q8_0.gguf      # http://127.0.0.1:8791
```

![Neuron panel with up/downstream wiring in the front view](docs/images/ui-neuron.png)

The UI has four views:
- **map:** every layer as a sheet of neurons, with overlays.
- **layers:** per-layer summaries.
- **eval:** saved evals as a chart and a table per dose, sample outputs, and a button to eval the current recipe.
- **embed:** a UMAP of rollout texts.

Click a neuron or a list row to open its panel, which has four tabs:
- **Examples:** rollouts per prompt and dose.
- **Chat:** live 3-column steering, with a `when` selector. The recipe tab can add an ablation (direction, scale, layers).
- **Details:** lens results, per-dose metrics, connections, and the evals that include the neuron.
- **Web:** the multi-hop wiring.

To compute missing data (lenses, sweeps, connections, layouts), right-click a layer label. Jobs run in the background and give way to chat when it needs the GPU. The language switch is at the top right; you can also open the UI with `?lang=en` or `?lang=ja`.

The server binds to 127.0.0.1 only. It also rejects requests that arrive with a foreign Host header and cross-origin writes.

### OpenAI-compatible endpoint

`gne serve --openai` adds `/v1/models` and `/v1/chat/completions` (streaming or not) to the same server, so other apps can use a recipe with `when` or gain units. Pass the recipe in a `gne` field (`extra_body` in the OpenAI SDK):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8791/v1", api_key="unused")
r = client.chat.completions.create(
    model="any", messages=[{"role": "user", "content": "Who are you?"}],
    extra_body={"gne": {"units": ["22:2742:+0.8"], "when": "gen"}})
```

Instead of `units`, `"recipe"` takes the name of a recipe saved under `~/.gne/recipes`, and `"scale"` multiplies its doses. Generation is greedy unless the request sets `temperature` above 0 or any of `top_p`, `top_k`, `min_p`, `presence_penalty` and `seed`; then it samples in llama.cpp's order, and values left out take llama-server's defaults. Tools and logprobs are not supported. Requests share the GPU with the UI's chat.

## Using it from an agent (MCP)

```bash
claude mcp add gne -- gne mcp --model /path/to/model.gguf
```

The tools are `model_overview`, `layer_summary`, `neuron`, `search`, `steer`, `steer_lens`, `fires`, `reverse`, `recipe_save`, `export_cvec`, `gain_export`, `bake`, `eval`, `direction`, `ablate_export`, `set_name` and `job_start` / `job_status` / `job_cancel`.

If the web UI is open, add `--ui-url http://127.0.0.1:8791`. With it, the agent can highlight and annotate neurons and open them in the browser (`ui_highlight`, `ui_note`, `ui_open`). When the UI already holds the model on the same GPU, also add `--n-gpu-layers 0`.

[docs/agent_playbook.md](docs/agent_playbook.md) describes how an agent should judge results: compare against controls, distrust cheap effects such as a changed self-name, and separate facts from interpretation.

## Tests

```bash
uv pip install --python .venv/bin/python -e '.[dev,ui,mcp]'
python -m playwright install chromium
python -m pytest -q
```

- Model tests use `GNE_TEST_MODEL`, which defaults to `models/Qwen3-0.6B-Q8_0.gguf`.
- Browser tests use the Playwright Chromium; `GNE_CHROME` sets its path.
- Tests whose model or browser is missing are skipped.

## Documentation

The documentation is in Japanese:
- [docs/verification.md](docs/verification.md): what was checked against llama.cpp, and the findings.
- [docs/measurements.md](docs/measurements.md): timings and throughput.
- [docs/agent_playbook.md](docs/agent_playbook.md): how to read the results.

## License

MIT — see [LICENSE](LICENSE).

Bundled third-party files:
- three.js (MIT), in `gne/ui/static/vendor/three/`.
- The default statistics corpus (`configs/corpus/default.txt`), made of public-domain texts from Project Gutenberg and Aozora Bunko. Details are in [configs/corpus/README.md](configs/corpus/README.md).
