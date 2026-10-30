"""MCP サーバ `gne mcp`(stdio)。設計書 §6.13 (d)。ツールは gne.api の写像。

    claude mcp add gne -- gne mcp --model <path>

- 戻り値は --json と同じ形の dict(トークン列は文字列)。失敗は {"error": {"code", "message", "hint"}}。
- GPU を使う呼び出しは worker スレッドで GpuArbiter を通して実行する。長い処理は job_start で投げ、
  job_status で進捗を見る。ジョブはユニットの切れ目で対話的な呼び出しに GPU を譲る。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import anyio

from .jobs import GpuArbiter, JobRunner, standard_kinds
from .steer import GneError
from .store import CONFIGS

INSTRUCTIONS = """\
gguf-neuron-explorer: llama.cpp 上の GGUF モデルの MLP ニューロンを調べ、押し(steer)、レシピを control vector に書き出す。
進め方(詳しくは docs/agent_playbook.md):
1. model_overview → layer_summary(L) で、どの層にどのデータ(ρ・lens・sweep・逆引き)があるかを見る。無いものは job_start で作る。
2. 候補は neuron(L, n) の調書で判断する。null_pct(sweep が対照群の何割より上か)と kl_null_p95(steer lens の対照の 95% 点)を必ず見る。対照群を超えないものは報告しない。
3. 「名乗り・作者名が変わる」はランダム方向でも起こる安い効果。複数のプロンプトで同じテーマが出るものだけを本物とみなす。
4. rel dose: 0.5 で微かな挿入、0.8 で話題を上書き、1.0 以上は崩壊域。層 0〜2 には rel dose がそのまま通用しない。
5. steer(units, compare=True) で無介入との差を確かめ、recipe_save → eval(recipe) で副作用(能力・ppl・言語の混入)を測ってから export_cvec。
   抑える用途は add の − より gain の x0 を先に試す。位置(when)は gen を第一候補にして all と比べる。
   ニューロンで断片しか見つからない性格・振る舞いは direction(contrast, select="auto", mode="add"|"ablate") で方向を作る。
   消したい振る舞いは ablate(steer の ablate="NAME")と上位ニューロンの x0 を eval の比較(gne eval --compare-ablation)で比べてから選ぶ。
   方向も対照を超えているかを見る(direction の null_pct、eval(random_controls=N) の novelty_null_pct)。null_pct が 0.95 未満の方向は報告しない。
   見積もりが 30 分を超えるジョブは NEEDS_CONFIRM と見積もりを返して止まる。params に budget("30m")で絞るか yes: true。
   大型モデル(GPU に載らない、全数の lens が 1 層 1 時間を超える)は逆引きから: reverse → 上位 N の lens_steer → quick sweep → eval。
ユニットの書き方: "L:N:dose"(例 "14:8949:+1.0")、gain は "L:N:x2"(出力を k 倍、x0 で消去)、ランダム方向は "L:random:SEED:dose"、
direction で作った方向は "L:dir:NAME:dose"、attention ヘッドの gain は "L:Hh:xk"(例 "21:H7:x0")。
7. 反応性(トリガーの有無で差が出るか)はヘッドで作る。まず heads_contrast → 上位を x0 / x2 で eval(reactivity=対照集合) →
   対照(同じ k のランダムなヘッド)を超えたものだけ残す。
6. 人が UI(gne serve)を見ていれば、ui_open で調べているニューロンを開き、ui_highlight で候補を強調し、ui_note で判断の要点を
   ニューロンに貼る(--ui-url で繋いだときだけ。繋いでいなければ UI_NOT_RUNNING が返るので気にせず続ける)。
8. MoE(専門家を選ぶモデル)では、専門家のニューロンは "L:Ee:Nn:dose"(例 "12:E37:N511:+0.8")。MoE の add は「書き込み方向の押し」
   (専門家が選ばれたかに関係なく全トークンに足す)で、ニューロンを押すなら gain("12:E37:N511:x2"、選ばれたときだけ効く)。
   専門家全体は "L:Ee:x0"(消す)/ "L:Ee:x2"、ルーターのバイアスは "L:Ee:route:+2"(バイアスを持つ arch だけ)。
   まず experts_contrast で専門家を絞り(usage_shift = ペルソナのときに呼ばれる、contrib_cos = 方向を書く)、その中のニューロンを gain で試す。
"""


def _err(e: Exception) -> dict:
    if isinstance(e, GneError):
        out = {"error": {"code": e.code, "message": e.message, "hint": e.hint}}
        if getattr(e, "estimate", None) is not None:          # NEEDS_CONFIRM: 見積もりを返して止まる(CHANGE-03 C)
            out["estimate"] = e.estimate
        return out
    return {"error": {"code": "INTERNAL", "message": f"{type(e).__name__}: {e}", "hint": None}}


def _strip_ids(d: Any) -> Any:
    """token_ids を落とす(トークン列は文字列で返す、§6.13 (d))。"""
    if isinstance(d, dict):
        return {k: _strip_ids(v) for k, v in d.items() if k != "token_ids"}
    if isinstance(d, list):
        return [_strip_ids(x) for x in d]
    return d


def build_server(model: str | Path, *, root: str | Path | None = None, ui_url: str | None = None,
                 n_gpu_layers: int = -1, n_ctx: int = 4096, seed: int = 0):
    from mcp.server.mcpserver import MCPServer

    from .api import Session
    session = Session(model, root=root, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    arbiter = GpuArbiter()
    jobs = JobRunner(session, arbiter, standard_kinds())
    srv = MCPServer("gne", instructions=INSTRUCTIONS, version="0.1.0", log_level="WARNING")

    async def call(fn, *a, gpu: bool = True, **kw) -> dict:
        def run():
            session.warnings = []
            try:
                if gpu:
                    with arbiter.interactive():
                        out = fn(*a, **kw)
                else:
                    out = fn(*a, **kw)
                out = _strip_ids(out)
                if session.warnings:
                    out = {**out, "warnings": list(session.warnings)}
                return out
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await anyio.to_thread.run_sync(run)

    @srv.tool()
    async def model_overview() -> dict:
        """ModelMeta、arch の警告、bake の可否、層ごとの coverage(ρ_L・活性統計・logit lens・steer lens・sweep・逆引き)。"""
        return await call(session.model_overview, gpu=False)

    @srv.tool()
    async def layer_summary(layer: int, top: int = 20) -> dict:
        """層の要約: ρ_L、steer lens の KL(中央値・p90・対照の 95% 点を超える割合)、sweep の対照群超えの割合と上位ユニット、logit lens の上位、逆引きの上位。"""
        return await call(session.layer_summary, layer, top=top, gpu=False)

    @srv.tool()
    async def neuron(layer: int, neuron: int, rollouts: int = 4, compute: bool = False) -> dict:
        """ニューロン 1 個の調書(sweep の順位・語・rollout、steer lens、logit lens、逆引きの cos、活性統計、coverage)。
        compute=True なら、無い logit lens と steer lens をこのニューロンだけ計算する(数秒)。"""
        return await call(session.neuron, layer, neuron, rollouts=rollouts, compute=compute, gpu=compute)

    @srv.tool()
    async def search(q: str, mode: str = "lens", layer: int | None = None, limit: int = 50) -> dict:
        """一致するニューロンの一覧。mode: lens(steer/logit lens の上位語)、words(sweep の語)、name(付けた名前)。"""
        return await call(session.search, q, mode=mode, layer=layer, limit=limit, gpu=False)

    @srv.tool()
    async def steer(units: list[str], prompt: str | None = None, messages: list[dict[str, str]] | None = None,
                    max_tokens: int = 128, compare: bool = True, dose_unit: str = "rel",
                    system: str | None = None, when: str = "all", ablate: str | None = None) -> dict:
        """ユニット(例 ["14:8949:+1.0", "13:7956:-1.0"])で押して生成する(greedy)。compare=True なら無介入も並べる。
        "14:8949:x2" のように x を付けると gain(そのニューロンの出力を k 倍、x0 で消去。§6.16)。
        when は足す位置: all(全位置) / gen(生成中だけ、プロンプトの理解を壊さない) / assistant(assistant の発話だけ) /
        after:<文字列>(出力にその文字列が出てから) / ramp:<a>-<b>(倍率を a から b へ)。
        ablate は direction で作った方向の射影消去("NAME"、"NAME@層"、"NAME:0.5")。units は空でもよい。"""
        def fn():
            r = session.recipe(units=units or None, dose_unit=dose_unit, when=when, ablate=ablate)
            return session.steer(r, prompt=prompt, messages=messages, system=system, max_tokens=max_tokens,
                                 compare=compare)
        return await call(fn)

    @srv.tool()
    async def steer_lens(layer: int, neurons: str, dose: float = 0.8, random_controls: int = 16) -> dict:
        """指定ニューロン(例 "2742" や "1,5,9-12")の steer lens(± で押したときの次トークンの変化、KL と対照)。少数向け。"""
        return await call(session.lens_steer, layer, neurons=neurons, dose=dose, random_controls=random_controls)

    @srv.tool()
    async def fires(text: str, top: int = 50) -> dict:
        """文章を流し、トークンごとに最も光る(活性 ÷ 基準 が大きい)ニューロンを top 件ずつ返す。"""
        return await call(session.fires, text, top=top)

    @srv.tool()
    async def reverse(contrast: str | None = None, system_a: str | None = None, system_b: str | None = None,
                      user_prompts: list[str] | None = None, name: str | None = None, top: int = 50,
                      method: str = "mean_diff", positions: str = "all") -> dict:
        """逆引き: 対照プロンプト集(configs/contrast/ の名前かパス、または system_a / system_b / user_prompts を直接)から
        方向を作り、全層・全ニューロンを順位づけする。検証(rollout)は job_start("reverse", {..., "verify": true})。"""
        def fn():
            path = _contrast_path(session, contrast, system_a, system_b, user_prompts, name)
            return session.reverse(contrast=path, top=top, method=method, positions=positions, name=name)
        return await call(fn)

    @srv.tool()
    async def direction(contrast: str, name: str | None = None, select: str | None = None, mode: str = "ablate",
                        layers: str | None = None, dose: float = 1.0, positions: str = "last",
                        random_controls: int = 16) -> dict:
        """対照集合(configs/contrast/ の名前かパス。system_a / system_b / user_prompts、層選びには validate.match)から
        全層の方向 r_L = mean_A − mean_B(生成プロンプトの最終トークン)を作って保存する(§6.17)。
        select="auto" なら候補層ごとに効果を測ってジョブで層を選ぶ(job_status で結果)。mode="add" は層 L に平均差の dose 倍を
        足して判定の一致率が上がるか、mode="ablate" は全層から r_L を消して下がるか(B 側の KL が 0.1 以下の層から)。
        作った方向は steer の units に "L:dir:NAME:+0.5"(rel dose)で足せる。
        positions="response" は A / B の system で返事を生成し、返事のトークンで測る(性格を足す用途ではこちらが効きやすい)。
        random_controls 本の同じ大きさのランダムな向きを層ごとに対照として流し、効果が対照の p95 を超えた層だけを選ぶ
        (行の ctrl_effect_p95・null_pct、選んだ層の selected_null_pct。0 で対照なし)。"""
        try:
            params = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                      "select": select, "mode": mode, "layers": layers, "dose": dose, "positions": positions,
                      "random_controls": random_controls}
        except Exception as e:  # noqa: BLE001
            return _err(e)
        if select == "auto":
            try:
                return jobs.start("direction", params).view()
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await call(session.direction, **params)

    @srv.tool()
    async def recipe_save(name: str, units: list[str], dose_unit: str = "rel", ablate: str | None = None,
                          when: str | None = None) -> dict:
        """レシピを <root>/recipes/<name>.yaml に保存する。ablate は方向の射影消去("NAME"、"NAME@層"、"NAME:倍率")、
        when は足す位置(all / gen / assistant / skip-replies / after:<文字列> / ramp:<a>-<b>)。"""
        return await call(session.recipe_save, name, units, dose_unit=dose_unit, ablate=ablate, when=when, gpu=False)

    @srv.tool()
    async def export_cvec(name: str, format: str = "cvec", verify: bool = False) -> dict:
        """保存したレシピを control vector GGUF(または npz / json / safetensors)に書き出す。verify=True で llama-completion と比べる。"""
        def fn():
            from .cli import _EXT
            if format not in _EXT:
                raise GneError("BAD_FORMAT", f"format は {', '.join(_EXT)} のどれかです")
            r = session.recipe(path=session.recipe_path(name))
            out = session.store.export_dir() / f"{name}{_EXT[format]}"
            return session.export_cvec(r, out, fmt=format, verify=verify)
        return await call(fn, gpu=True)

    @srv.tool()
    async def gain_export(name: str, bake: bool = False) -> dict:
        """保存したレシピの gain ユニット("14:8949:x2" のように出力を k 倍、x0 で消去)を書き出す(§6.16)。
        既定は LoRA アダプタ GGUF(llama.cpp の --lora で使える、Qwen 系でも可)。bake=True なら W_up を書き換えた GGUF。"""
        def fn():
            r = session.recipe(path=session.recipe_path(name))
            out = session.store.export_dir() / (f"{name}-gain.gguf" if bake else f"{name}.lora.gguf")
            return session.gain(r, out, bake=bake)
        return await call(fn, gpu=False)

    @srv.tool(name="eval")
    async def eval_recipe(recipe: str | None = None, units: list[str] | None = None, doses: list[float] | None = None,
                   when: str | None = None, random_controls: int = 0, reactivity: str | None = None,
                   head_controls: int = 0) -> dict:
        """副作用チェック(§6.14)をジョブとして投げる。recipe は保存したレシピの名前、units は単発指定(どちらか)。
        doses はレシピ全体の倍率(既定 [0, 0.25, 0.5, 0.75, 1]、0 が無介入)。倍率ごとに exact / format の正答率、ppl、
        言語の混入(簡体字・ハングルなど)、性格の変化量(無介入にない語の割合など)を測り、配布可(能力 −5pt 以内、
        ppl +10% 以内、混入が増えない)の倍率を返す。4B で数分。job_status で結果を見る(保存先 <root>/evals/)。
        random_controls=N なら各倍率で同じ大きさのランダムな対照レシピ N 本も測る(時間は約 N 倍)。行の ctrl_novelty_p95・
        novelty_null_pct(新語が対照の何割より多いか)、ctrl_capability_drop_p50・ctrl_ppl_worse_p50(大きさだけで起きる悪化)。
        reactivity(対照集合の名前かパス。reactivity の対と validate.match が要る)を渡すと、行の reactivity に
        トリガーあり/なしの一致率とその差を入れる。head_controls 本のランダムなヘッド(同じ k)と比べた null_pct も。"""
        try:
            p = {"recipe": recipe, "units": units, "when": when, "random_controls": random_controls}
            if reactivity:
                p["reactivity"] = str(_contrast_path(session, reactivity, None, None, None, None))
                p["head_controls"] = head_controls
            if doses is not None:
                p["doses"] = doses
            return jobs.start("eval", p).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def ablate_export(name: str, layers: str = "all", scale: float = 1.0, bake: bool = False,
                            from_layer: int | None = None) -> dict:
        """方向 name(gne direction / direction ツールで作ったもの)の射影消去を書き出す(§6.17、いわゆる abliteration)。
        既定は LoRA アダプタ GGUF(llama.cpp の --lora、--lora-scaled で部分消去)。bake=True なら重みを Q8_0 で書き換えた GGUF を
        ジョブで作る(大きい。job_status で結果)。layers は "all" / "10-40"、scale は消す割合(1 で全部)。
        使う r は from_layer の層で測ったもの(既定は direction の select で選んだ層)。"""
        from .cli import parse_layers
        try:
            Ls = parse_layers(layers)
            suffix = "-ablate.gguf" if bake else "-ablate.lora.gguf"
            out = str(session.store.export_dir() / f"{name}{suffix}")
            params = {"direction": name, "out": out, "layers": Ls, "scale": scale, "bake": bake, "from_layer": from_layer,
                      "verify": True}
        except Exception as e:  # noqa: BLE001
            return _err(e)
        if bake:
            try:
                return jobs.start("ablate", params).view()
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await call(lambda: session.ablate(**{**params, "out": out}))

    @srv.tool()
    async def bake(name: str, dst: str, verify: bool = True) -> dict:
        """保存したレシピを FFN バイアスとして焼き付けた GGUF を dst に作る(対応 arch のみ、Qwen 系は不可)。"""
        def fn():
            r = session.recipe(path=session.recipe_path(name))
            return session.bake(r, dst, verify=verify)
        return await call(fn)

    @srv.tool()
    async def head(layer: int, head: int, lens: bool = True) -> dict:
        """attention ヘッド 1 個の調書(CHANGE-04): kind(attn / linear)、幅、K/V グループ、保存済みの heads_contrast での
        指標と順位、名前、当てたことのある gain、W_O の列ブロックの logit lens(lens=False で省く)。"""
        return await call(session.head, layer, head, lens=lens, gpu=False)

    @srv.tool()
    async def heads_contrast(contrast: str, name: str | None = None, direction: str | None = None,
                             layers: str | None = None, max_tokens: int = 48) -> dict:
        """ヘッドの対照分析をジョブとして投げる(job_status で結果)。対照集合(configs/contrast/ の名前かパス)の会話を
        ペルソナあり(system_a)と中立(system_b)で流し、ヘッドごとに trigger_mass(トリガー語への注目の割合の差、
        対照集合の triggers が要る)・pattern_shift(注目の分布の JS)・out_shift(出力の差)・dir_cos(出力差が方向 r_L に
        書く cos、direction か同じ名前の方向があるとき)を測り、中立どうしの差を床にして z で順位づける。"""
        from .cli import parse_layers
        try:
            params = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                      "direction": direction, "layers": parse_layers(layers), "max_tokens": max_tokens}
            return jobs.start("heads_contrast", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def expert(layer: int, expert: int) -> dict:
        """MoE の専門家 1 個の調書(CHANGE-05): 選択率・選ばれたときの重みの平均・よく選ぶトークン(stats act があるとき)、
        保存済みの experts_contrast での指標と順位、書き込みの大きいニューロン(q999 × ‖W_down[:, n]‖)。"""
        return await call(session.expert, layer, expert)

    @srv.tool()
    async def experts_contrast(contrast: str, name: str | None = None, direction: str | None = None,
                               layers: str | None = None, max_tokens: int = 48) -> dict:
        """専門家の対照分析をジョブとして投げる(job_status で結果)。対照集合の同じ返事をペルソナあり(system_a)と中立(system_b)で
        読ませ、MoE の層の専門家ごとに usage_shift(選ばれた割合の差)・weight_shift(実効の重みの差)・contrib_cos(寄与の差と
        その層の状態の向きの cos)・contrib_norm を測り、中立どうしを床にして z で順位づける。数秒〜数十秒。"""
        from .cli import parse_layers
        try:
            params = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                      "direction": direction, "layers": parse_layers(layers), "max_tokens": max_tokens}
            return jobs.start("experts_contrast", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def experts_sweep(layer: int, experts: list[int] | None = None, ks: list[float] | None = None,
                            max_tokens: int = 48) -> dict:
        """専門家単位のスイープをジョブとして投げる。層の各専門家を k 倍(既定 0 と 2)するスケールパッチで生成し、語の出方で
        順位づける(帰無は同じ層の全専門家)。1 専門家・1 倍率ごとにモデルを読み直す(小さいモデルで 0.3〜1 秒)。"""
        try:
            p = {"layer": layer, "experts": experts, "ks": tuple(ks) if ks else (0.0, 2.0), "max_tokens": max_tokens}
            return jobs.start("experts_sweep", p).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def moe_route(text: str, layers: str | None = None, chat: bool = False) -> dict:
        """文を流し、MoE の層ごとに各トークンで選ばれた専門家と実効のルーティング重み(大きい順)を返す。"""
        from .cli import parse_layers
        return await call(session.moe_route, text, layers=parse_layers(layers), chat=chat)

    @srv.tool()
    async def expert_patch(layer: int, expert: int, k: float, prompt: str, system: str | None = None,
                           max_tokens: int = 64) -> dict:
        """専門家 e の出力を k 倍(0 で消す)するスケールパッチを作業用コピーに当てて生成し、無介入と並べる。終わったら元に戻す
        (元の GGUF は触らない)。レシピに入れるなら "L:Ee:xk" を recipe_save に渡す。"""
        return await call(session.steer, [f"{layer}:E{expert}:x{k:g}"], prompt=prompt, system=system, max_tokens=max_tokens)

    @srv.tool()
    async def writers(contrast: str, name: str | None = None, ref_layer: int | None = None,
                      replies_a: str | None = None, max_tokens: int = 96) -> dict:
        """書き手の強さをジョブとして投げる(job_status で結果)。対照集合の A(見たい状態、例 ヤンデレの system)と B(中立)の
        返事のトークンで、各ニューロンが残差に書き足した量(活性の差 × 書く向き)の、参照の層の方向への直接の寄与を測る。
        方向を埋め込み・mixer(attention など)・MLP が何割書いたかも返す。replies_a は A の返事を読ませるファイル
        (別のモデルの返事を借りるとき)。活性の差(読み手)と違い、方向に実際に書いている「書き手」が上に来る。"""
        try:
            p = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                 "ref_layer": ref_layer, "replies_a": replies_a, "max_tokens": max_tokens}
            return jobs.start("writers", p).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def attention(text: str, layer: int | None = None, head: int | None = None, top: int = 5, pos: int = -1,
                        chat: bool = False, system: str | None = None) -> dict:
        """文を流し、位置 pos(既定は最後)のトークンが各ヘッドでどこを見ているか(上位 top)と per-head 出力のノルム。
        layer と head を両方指定すると T × T のパターンも返す。chat=True なら text をユーザーの発話としてテンプレートに入れる。"""
        return await call(session.attention, text, layer=layer, head=head, top=top, pos=pos, chat=chat, system=system)

    @srv.tool()
    async def heads_listen(user: str, heads: str, system: str | None = None, reply: str | None = None, max_tokens: int = 64,
                           direction: str | None = None, top: int = 20) -> dict:
        """返事のトークンが、選んだヘッド(例 "14:H0-3")で文脈のどのトークンに注目しているか(注目の平均)。
        direction(保存済みの方向)を渡すと、文脈の各トークンがヘッドの出力を通してその層の方向に寄与した量(注目 × ゲート × OV、
        attention の後の norm 込み)と、層の全ヘッドの寄与のうち選んだヘッドの分も返す。reply を省くと greedy に作る。"""
        return await call(session.heads_listen, user, heads=heads, system=system, reply=reply, max_tokens=max_tokens,
                          direction=direction, top=top)

    @srv.tool()
    async def set_name(layer: int, neuron: int, name: str) -> dict:
        """ニューロンに名前を付ける(空文字で消す)。調書と search(mode="name") に出る。"""
        return await call(session.set_name, layer, neuron, name, gpu=False)

    @srv.tool()
    async def job_start(kind: str, params: dict[str, Any] | None = None, workers: str | None = None) -> dict:
        """長時間の処理を投げる。kind: stats_resid / stats_act / lens_logit / lens_steer / sweep / rank / reverse / eval / direction / ablate / heads_contrast。
        params は gne.api の同名メソッドの引数(例 sweep: {"layer": 22, "neurons": "random:64", "design": "quick"})。
        見積もりが 30 分を超えるジョブは走らせずに NEEDS_CONFIRM と見積もりを返す。params に budget("30m" など)を入れると
        予算に収まるように絞る(lens_steer は方向に近いニューロンの上位、sweep は quick 設計で steer lens の上位、reverse は
        確かめる候補の数、connections は window)。それでも走らせるなら params に yes: true。
        workers("auto" か名前のカンマ区切り)を渡すと、gne worker のファイルのキューに入れる(sweep / lens_steer は
        ニューロンで分ける、CHANGE-03 B)。返り値の jobs の id で job_status を見る。"""
        try:
            if workers:
                from .estimate import NeedsConfirm, gate
                from .workers import submit
                try:
                    p, _ = gate(session, kind, dict(params or {}))
                except NeedsConfirm as e:
                    e.error.estimate = e.estimate
                    raise e.error from e
                return submit(session, kind, p, workers=workers)
            return jobs.start(kind, params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def job_status(job_id: str | None = None) -> dict:
        """ジョブの進捗(done / total / eta_s)と結果。job_id を省くと全ジョブの一覧。"""
        try:
            from .workers import job_files
            if job_id is None:
                return {"jobs": [j.view() for j in jobs.jobs.values()],
                        "worker_jobs": [{k: v for k, v in j.items() if k != "result"} for j in job_files(session.store.root)]}
            if job_id not in jobs.jobs:                            # ワーカーのジョブ(id か group)
                fj = [j for j in job_files(session.store.root) if job_id in (j["id"], j.get("group"))]
                if fj:
                    return _strip_ids({"jobs": fj})
            return _strip_ids(jobs.get(job_id).view())
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def job_cancel(job_id: str) -> dict:
        """ジョブを止める(次のユニットの切れ目で止まる。途中までの結果は残り、同じ条件で続きから再開できる)。"""
        try:
            return jobs.cancel(job_id).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    for action in ("highlight", "note", "open"):
        _register_ui_tool(srv, action, ui_url)
    return srv


def _contrast_path(session, contrast, system_a, system_b, user_prompts, name) -> Path:
    if contrast:
        p = Path(contrast)
        if not p.exists():
            p = CONFIGS / "contrast" / f"{contrast}.yaml"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"対照プロンプト集 {contrast} がありません",
                           "configs/contrast/ の名前かパス、または system_a / system_b / user_prompts を渡してください")
        return p
    if not (system_a and system_b and user_prompts):
        raise GneError("BAD_ARGS", "contrast か、system_a・system_b・user_prompts の 3 つを渡してください")
    import yaml
    nm = name or "contrast"
    if not re.fullmatch(r"[\w.\-]+", nm):
        raise GneError("BAD_ARGS", "name は英数字・_・-・. だけにしてください")
    p = session.store.root / "contrast" / f"{nm}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"name": nm, "system_a": system_a, "system_b": system_b,
                                 "user_prompts": list(user_prompts)}, allow_unicode=True), encoding="utf-8")
    return p


def _register_ui_tool(srv, action: str, ui_url: str | None) -> None:
    """ui_* ツール(別冊 §11)。gne serve の POST /api/ui/{action} を叩く。UI に繋いでいなければ UI_NOT_RUNNING。"""

    async def post(body: dict) -> dict:
        if not ui_url:
            return {"error": {"code": "UI_NOT_RUNNING", "message": "UI に接続していません",
                              "hint": "gne serve を起動し、gne mcp --ui-url http://127.0.0.1:8791 で繋いでください"}}
        req = urllib.request.Request(f"{ui_url.rstrip('/')}/api/ui/{action}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})

        def send():
            try:
                with urllib.request.urlopen(req, timeout=5) as r:     # noqa: S310  127.0.0.1 の UI だけ
                    return json.loads(r.read() or b"{}")
            except urllib.error.HTTPError as e:                      # 400 は UI 側のエラー({"error": ...})をそのまま返す
                try:
                    return json.loads(e.read())
                except ValueError:
                    return {"error": {"code": "UI_ERROR", "message": f"UI が {e.code} を返しました", "hint": None}}
        try:
            return await anyio.to_thread.run_sync(send)
        except (urllib.error.URLError, OSError) as e:
            return {"error": {"code": "UI_NOT_RUNNING", "message": f"UI に届きません: {e}", "hint": None}}

    if action == "highlight":
        async def ui_highlight(units: list[str], label: str | None = None) -> dict:
            """UI のマップでユニットを強調する(水色の破線の輪と札)。units は ["22:2742", "14:8949:+1"] の形。
            前の強調は置き換わる。空のリストで消す。label はまとまりの見出し(例 "宇宙人の候補")。
            強調は「人に見てほしいもの」だけにする。見送った候補は強調せず、ui_note で理由だけ貼る。"""
            return await post({"units": units, "label": label})
        srv.tool(name="ui_highlight")(ui_highlight)
    elif action == "note":
        async def ui_note(layer: int, neuron: int, text: str) -> dict:
            """UI のマップのニューロンに短い注釈(400 字まで)を貼る。同じニューロンには上書き、空の text で消す。
            地図には最初の 28 字だけが出るので、結論から書く(例 "本命: 宇宙人の話題。sweep 3 位・全対照超え、3 プロンプトで
            extraterrestrial" / "見送り: 量の語(many/most)")。事実と解釈を分ける。"""
            return await post({"layer": layer, "neuron": neuron, "text": text})
        srv.tool(name="ui_note")(ui_note)
    else:
        async def ui_open(layer: int, neuron: int, tab: str | None = None) -> dict:
            """UI の NEURON パネルでニューロンを開き、マップをその層の正面ビューにする。tab: examples / chat / details / web。"""
            return await post({"layer": layer, "neuron": neuron, "tab": tab})
        srv.tool(name="ui_open")(ui_open)


def main(model: str, *, root: str | None = None, ui_url: str | None = None, n_gpu_layers: int = -1,
         n_ctx: int = 4096, seed: int = 0) -> None:
    srv = build_server(model, root=root, ui_url=ui_url, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    srv.run("stdio")
