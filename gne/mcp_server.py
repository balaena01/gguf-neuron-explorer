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
direction で作った方向は "L:dir:NAME:dose"、attention ヘッドの gain は "L:Hh:xk"(例 "21:H7:x0")、ヘッドの方向成分の射影は "L:Hh:sS@NAME"。
7. 反応性(トリガーの有無で差が出るか)はヘッドで作る。まず heads_contrast → 上位を x0 / x2 で eval(reactivity=対照集合) →
   対照(同じ k のランダムなヘッド)を超えたものだけ残す。
6. 人が UI(gne serve)を見ていれば、ui_open で調べているニューロンを開き、ui_highlight で候補を強調し、ui_note で判断の要点を
   ニューロンに貼る(--ui-url で繋いだときだけ。繋いでいなければ UI_NOT_RUNNING が返るので気にせず続ける)。
8. MoE(専門家を選ぶモデル)では、専門家のニューロンは "L:Ee:Nn:dose"(例 "12:E37:N511:+0.8")。MoE の add は「書き込み方向の押し」
   (専門家が選ばれたかに関係なく全トークンに足す)で、ニューロンを押すなら gain("12:E37:N511:x2"、選ばれたときだけ効く)。
   専門家全体は "L:Ee:x0"(消す)/ "L:Ee:x2"、ルーターのバイアスは "L:Ee:route:+2"(バイアスを持つ arch だけ)。
   まず experts_contrast で専門家を絞り(usage_shift = ペルソナのときに呼ばれる、contrib_cos = 方向を書く)、その中のニューロンを gain で試す。
9. ヘッドの方向成分だけを変える(CHANGE-06、"L:Hh:sS@NAME"、例 "33:H16:s6@evil"。s = 1 でそのヘッドの方向成分が消え、2 で反転、
   直交成分は変えない)。手順: direction_score(基準層と分離度を確認)→ heads_causal(mode="ablate" で候補出し)→
   heads_allocate(効率 η で強さを配り、ヘッド数を減らした組を比べる)→ eval(random_controls=N, compare_methods=True)→ headproj_export。
   一通り回すだけなら heads_pipeline(spec)(方向づくりから配分まで。質問は何百個でもよく、止めて測る段階は 64 個に間引く)。
   強さを決める最後は、heads_candidates(組 × 倍率で生成し、名前を伏せた採点のファイルを作る)→ 採点 → heads_candidates_tally。
   組で消し切れない質問(配分の表の remain)が残るなら、heads_refine で残りの質問だけから追加のヘッドを探して組に足す。
   配る前の確かめは fieldtest(焼き込んだファイルと元のファイルを、本番と同じ動かし方で比べる。eval では見えない長い作業の崩れとツール呼び出しを見る)。
   - 方向を使う介入は、まず direction_score で基準層と分離度(cohen_d)を確認してから進める。
   - heads_causal の ablate と marginal は順位が違う。最初の候補出しは ablate、配り直しは marginal(heads_allocate が使う)。
   - 候補と強さの案は生成なしの指標(direction_score・heads_causal・heads_allocate)で作る。方向への射影の値は押した量で、振る舞いそのものではないので、止める強さは少数の問題を生成して確かめる。
   - 報告には外から見える指標(eval の rate = 対照集合の validate.match の一致率、能力・ppl)を主にする。方向への射影の値だけでは
     「方向で押したものを方向で測った」ことにしかならない。
   - ランダム対照(eval の random_controls、同数・同じ s のランダムなヘッド)と方式の比較(compare_methods)を通していないレシピは報告しない。
   - 目的の指標が数 % まで来たら強さを上げるのをやめ、残る例を担う別のヘッドを探す。
10. 鍵つきの書き込み(keyed、CHANGE-07、"L:ffn:keyed:aα@DIR#KEY"、例 "19:ffn:keyed:a1@zundamon_endings#zundamon_endings")。
   proj が「どの向きを」変えるかを選ぶのに対し、keyed は「どの状態で」書くかを選ぶ(残差に書き込む重み行列に α d κᵀ を足し、
   活性に鍵 κ が反応する位置でだけ方向 d を書く。α は raw で、1 = 観測した A − B の差そのまま)。両者は重ねられる(同じ LoRA に入る)。
   手順: direction(書く向き)→ direction_key(鍵。介入なしで作る。対照集合の positions を "before:<正規表現>" にすると、推論時と
   同じ文脈のモデル自身の返事で、変えたい文字を書く直前の位置が書き込みたい位置になる)→ direction_key_check(作成に使って
   いない例で。既定 subset="odd"。held_out が false の値は甘い。key-check の選択性が比較用の鍵(ラベルをシャッフルして同じ
   手順で作った鍵)の p95 を超えない鍵は使わない)→ steer / eval → keyed_export(配るなら LoRA。焼き込みの既定は、元が
   Q8_0 のテンソルは Q8_0 のまま、それ以外は f16)。when は keyed に効かない。eval は keyed のレシピに eval の選択性の
   ブロック(行の keyed.selectivity、結果の keyed の方式の比較と比較用の鍵)を足す。鍵と方向を別の対照集合で作ったなど、鍵の対照集合を推定できないときは
   eval(keyed_contrast=鍵を作った対照集合)で渡す(渡さないと keyed は skipped)。
   例: ずんだもんの語尾 configs/contrast/zundamon_endings.yaml(direction(positions="response") → direction_key(layers=[…]) → direction_key_check)。
   一通り回すだけなら direction_key_pipeline(contrast, layers)(方向 → 鍵 → 作成と反対側の subset での key-check を 1 回のジョブで。
   同じ名前の方向があれば使い回し、rebuild_direction=True で作り直す)。用途別のひな形は configs/keyed/(文末の言い方・特定の場面の返事・
   system で作った状態。about は説明だけで読むときに捨てる)。
   CHiPS のレシピで消し切れなかった場面だけを keyed で抑えるなら、behave(ジョブ。CHiPS のレシピを当てて A 側の質問に答えさせる)の結果から
   direction_key_from_behave(behave, recipe, direction) で JSON を作り(<root>/keyed/<name>.json。keyed.iv にレシピ、keyed.direction に
   CHiPS の方向、alpha −1)、direction_key_pipeline(contrast=その名前, layers) で鍵を作る。
   どの層・書き込み先に書けば効くかは key-check では分からない(鍵が場面を分けるかしか見ない)。direction_key_causal(contrast) が
   全層 × ffn,attn の候補で、生成なしで振る舞いの部分の対数確率の下がり方を比較用の鍵と並べて測る(第 1 段、ふるい分け。
   印の数を「偶然でも印が付く数」と比べる)。生成なしの値は保証ではないので、使う人が候補を数個選び、direction_key_causal_verify
   (units, result) の生成で振る舞いの数を iv だけ・比較用の鍵と比べる(第 2 段)。自動で 1 つに決めない。
   方向の作り方: direction(positions="response", same_reply=True)は、同じ返事を A 側・B 側の条件の両方で読ませた差
   (条件の違いだけの成分)。既定の作り方(それぞれ自分の返事を読んだ差)の方向には「読んでいる語の違い」の成分も入り、それを
   書き続けると繰り返しに落ちやすかった(Qwen3.5-4B の性格の 1 例。docs/keyed-write.md §19)。
   MoE の層は direction_key(target="shared")で shared expert に、target="expert:e" で専門家 e に書く(ユニットは L:S / L:Ee:keyed:…)。
   専門家は direction_key_experts で書き込みたい位置でよく選ばれる(rate_a が大きく rate_b が小さい)ものを選ぶ。焼き込みは
   既定(auto)と requant="keep" なら、元が Q8_0 の専門家のテンソルは元の型のまま(f16 だとテンソル全体が F16 になって大きくなる)。
11. FiSH(Formula-instantiated Scene-gated Hidden units。計算で作るニューロン、mode nw、docs/fish.md)。MLP のニューロンを学習なしで計算した重みで上書きして、鍵が反応する
   位置でだけ方向を書く。ユニット: "L:Nn:nw:aα@方向#鍵"(鍵 1 つ)、"L:Nn:nw:aα@方向#場面の鍵&位置の鍵"(鍵 2 つ = 場面 かつ 位置)、
   "L:Nn+Nm:nw:aα@方向#鍵&1を返す鍵"(2 個を組にして足す量に上限を付ける。逆向きの書き込みが小さい既定 s 16・しきい値 0.4-0.8)。
   手順: direction(positions="response", same_reply=True, segments="0,1-3,4-")で方向(区切り i は "NAME.seg<i>")→ nw_neurons(layers)で
   働きの小さいニューロンの表を見て番号を選ぶ → direction_key(target="mlpin", positions="pos:1-3")で位置の鍵、
   direction_key(role="scene")で場面の鍵、direction_key(role="one")でほぼ 1 を返す鍵 → nw_verify(式どおりか)→ nw_check(位置ごとの
   足した量。狙う場面で 1 前後、書き込みたくない場面で 0 前後か)→ steer / behave で返事を読む → nw_export(LoRA。焼き込みは無い)。
   対応するのはゲートつきの MLP(SiLU のゲート × 入力、バイアスなし)の dense の層だけ(MoE の層・ほかの形は ARCH_UNSUPPORTED)。
   口調を変えるには返事の間ずっと書き続ける必要があった(最初の数トークンだけでは足りない)。強さの幅は狭く、返事を読んで決める。
   まとめて作るなら fish_pipeline(contrast, layers, alpha)。強さは区切りごとに変えられる(segments "0,1-3,4-" に alpha [1, 1, 0.8])。
   作る前に、nw_try(direction, contrast, layers, alpha)で、同じ方向を層の出力に直接足した返事を見て、層と強さを比べられる。
   Qwen3.5-4B の 1 例では、離れた 3 層より続いた 13 層(11〜23)・強さ 1 / 1 / 0.8 のほうが良かった(口調らしい返事 27〜29% → 53%)。
   層の番号はモデルで違う。同じ語句から抜けられなくなる返事(繰り返し)が質問によって 1〜8% 出る(docs/fish.md §8)。
   鍵を作る質問は多いほうがよい(狙う場面の質問は 100 問以上を勧める)。返事の書き出し方が何通りかあるとき(かぎかっこで始まる、
   など)、少ない質問で作った鍵はその一部を拾えない。nw_check の位置ごとの値は平均で、一部の質問でまったく書けていないことが
   隠れるので、質問ごとの値(by_question の low、question_summary.worst の質問と返事の出だし)も見る。fish_pipeline は、direction で
   すでにある方向を使い回せ(作り直さない)、cap で上限つきの形のしきい値を区切りごとに変えられる。
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

    from . import __version__
    from .api import Session
    session = Session(model, root=root, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    arbiter = GpuArbiter()
    jobs = JobRunner(session, arbiter, standard_kinds())
    srv = MCPServer("gne", instructions=INSTRUCTIONS, version=__version__, log_level="WARNING")

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
                        random_controls: int = 16, same_reply: bool = False, segments: str | None = None,
                        stack_layers: str | list[int] | None = None, response_tokens: int = 96) -> dict:
        """対照集合(configs/contrast/ の名前かパス。system_a / system_b / user_prompts、層選びには validate.match)から
        全層の方向 r_L = mean_A − mean_B(生成プロンプトの最終トークン)を作って保存する(§6.17)。
        select="auto" なら候補層ごとに効果を測ってジョブで層を選ぶ(job_status で結果)。mode="add" は層 L に平均差の dose 倍を
        足して判定の一致率が上がるか、mode="ablate" は全層から r_L を消して下がるか(B 側の KL が 0.1 以下の層から)。
        作った方向は steer の units に "L:dir:NAME:+0.5"(rel dose)で足せる。
        positions="response" は A / B の system で返事を生成し、返事のトークンで測る(性格を足す用途ではこちらが効きやすい)。
        random_controls 本の同じ大きさのランダムな向きを層ごとに対照として流し、効果が対照の p95 を超えた層だけを選ぶ
        (行の ctrl_effect_p95・null_pct、選んだ層の selected_null_pct。0 で対照なし)。
        方向の作り方(positions="response" のとき。結果の method に記録): same_reply=False(既定)は今までどおり、A 側・B 側が
        それぞれ自分の返事を読んだ差。same_reply=True は、同じ返事(A 側の返事)を A 側の条件と B 側の条件の両方で読ませた差
        (条件の違いだけの成分。読んでいる語の違いから来る成分が入らない)。segments="0,1-3,4-" で返事の位置で分けた方向も作る
        (位置 0 = 返事を書き始める直前。区切り i は "NAME.seg<i>" という名前でユニットに書ける)。stack_layers="11-23"(範囲)・"11,18,23"・[11, 18, 23] は、
        区切りつきの方向をその層に重ねて書く前提の値にする(上の層は、下の層に足した後の残り。時間がかかる)。"""
        try:
            params = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                      "select": select, "mode": mode, "layers": layers, "dose": dose, "positions": positions,
                      "random_controls": random_controls, "same_reply": same_reply, "segments": segments,
                      "stack_layers": stack_layers, "response_tokens": response_tokens}
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
            out = session.default_export_path(r, bake=bake, kind="gain", name=_recipe_stem(name))
            return session.gain(r, out, bake=bake)
        return await call(fn, gpu=False)

    @srv.tool(name="eval")
    async def eval_recipe(recipe: str | None = None, units: list[str] | None = None, doses: list[float] | None = None,
                   when: str | None = None, random_controls: int = 0, reactivity: str | None = None,
                   head_controls: int = 0, compare_methods: bool = False, match_effect: float | None = None,
                   score_layer: int | None = None, alloc: str | None = None, n_heads: list[int] | None = None,
                   control_pool: str = "before", direction_metrics: bool = True, dir_max_examples: int = 64,
                   keyed_contrast: str | None = None, keyed_metrics: bool = True, keyed_examples: int = 8,
                   keyed_match: bool = True, keyed_pseudo: int = 8, keyed_subset: str | None = None) -> dict:
        """副作用チェック(§6.14)をジョブとして投げる。recipe は保存したレシピの名前、units は単発指定(どちらか)。
        doses はレシピ全体の倍率(既定 [0, 0.25, 0.5, 0.75, 1]、0 が無介入)。倍率ごとに exact / format の正答率、ppl、
        言語の混入(簡体字・ハングルなど)、性格の変化量(無介入にない語の割合など)を測り、配布可(能力 −5pt 以内、
        ppl +10% 以内、混入が増えない)の倍率を返す。4B で数分。job_status で結果を見る(保存先 <root>/evals/)。
        random_controls=N なら各倍率で同じ大きさのランダムな対照レシピ N 本も測る(時間は約 N 倍)。行の ctrl_novelty_p95・
        novelty_null_pct(新語が対照の何割より多いか)、ctrl_capability_drop_p50・ctrl_ppl_worse_p50(大きさだけで起きる悪化)。
        reactivity(対照集合の名前かパス。reactivity の対と validate.match が要る)を渡すと、行の reactivity に
        トリガーあり/なしの一致率とその差を入れる。head_controls 本のランダムなヘッド(同じ k)と比べた null_pct も。
        方向を使うレシピ(proj・方向のユニット・ablate)の行には、射影の値の下がり方 dir_drop_a・残差の差の直交成分 drift_orth_a /
        drift_orth_b・判定の一致率 rate が入る(CHANGE-06 F.1)。proj の random_controls は同数・同じ s のランダムなヘッド
        (control_pool: before / same-layers)で、dir_null_pct・rate_null_pct・drift_orth_b_null_pct を出す。
        compare_methods=True は、proj のレシピと同じ方向で全層射影・ヘッドの方向成分・ヘッドの丸ごと倍率・ニューロンの消去を
        A 側の射影の値の水準(match_effect、省けばこのレシピ)をそろえて比べる(F.3)。alloc(heads_allocate の名前)と n_heads で、
        ヘッド数ごとの組を評価する(F.4、recipe・units は要らない)。方向の指標は direction_metrics=False で測らない。
        dir_max_examples はその測定で読む質問の数(既定 64、A・B それぞれ等間隔に間引く。0 で全部)。
        keyed のユニットを持つレシピ(CHANGE-07)では、行の keyed(目的の層の書き込みたい位置・書き込みたくない位置・dont_touch の残差の
        r̂ 方向の変化と eval の選択性 selectivity、random_controls があれば比較用の鍵の対照の null_pct)と、結果の keyed(方式の比較
        methods・compare、比較用の鍵 pseudo、text)が入る(生成なしの prefill だけ)。keyed_contrast は鍵を作った対照集合(名前かパス。方向を作った
        対照集合と同じなら省ける。分からなければ keyed は skipped と hint)、keyed_examples は会話の組ごとに読む数(0 で全部)、
        keyed_pseudo は比較用の鍵の本数、keyed_subset は読む例(even / odd / all、既定は鍵の作成と反対側)、keyed_match=False で
        大きさ合わせを省く、keyed_metrics=False で測らない。"""
        try:
            if alloc:
                return jobs.start("eval", {"alloc": alloc, "n_heads": n_heads, "when": when,
                                           "random_controls": random_controls}).view()
            p = {"recipe": recipe, "units": units, "when": when, "random_controls": random_controls}
            if compare_methods:
                p.update({"compare_methods": True, "match_effect": match_effect, "score_layer": score_layer})
                p.pop("random_controls")
            elif control_pool != "before":
                p["control_pool"] = control_pool
            if reactivity:
                p["reactivity"] = str(_contrast_path(session, reactivity, None, None, None, None))
                p["head_controls"] = head_controls
            if doses is not None:
                p["doses"] = doses
            if not compare_methods:
                p["direction_metrics"] = bool(direction_metrics)
                p["dir_max_examples"] = int(dir_max_examples) or None
                p.update({"keyed_metrics": bool(keyed_metrics), "keyed_examples": int(keyed_examples) or None,
                          "keyed_match": bool(keyed_match), "keyed_pseudo": int(keyed_pseudo)})
                if keyed_contrast:
                    p["keyed_contrast"] = str(_contrast_path(session, keyed_contrast, None, None, None, None))
                if keyed_subset:
                    p["keyed_subset"] = keyed_subset
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
        書く cos、direction か同じ名前の方向があるとき)を測り、中立どうしの差の p95 を基準値にして z で順位づける。"""
        from .cli import parse_layers
        try:
            params = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                      "direction": direction, "layers": parse_layers(layers), "max_tokens": max_tokens}
            return jobs.start("heads_contrast", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_score(name: str, contrast: str | None = None, layers: list[int] | None = None,
                              units: list[str] | None = None, positions: str = "last", side: str = "both") -> dict:
        """速い指標(CHANGE-06 B、生成なしなので数秒)。対照集合(省けば方向を作ったもの)の A 側・B 側をプロンプトの最後まで
        読ませ、層ごとに 残差 · r̂_L の mean / sd / cohen_d と、基準層(cohen_d 最大。介入なし・layers 省略のとき記録する)、
        基準層の方向とのコサイン(方向が層でどれだけ回るか)を返す。units("33:H16:s6@NAME" など)を渡すと、その介入での
        射影の値の下がり方 drop と、残差の差の方向成分・直交成分 drift(par / orth、A・B それぞれ)。
        方向を使う介入は、まずこれで基準層と分離度を確認してから進める。"""
        def fn():
            d = session.direction_score(name, contrast=contrast, layers=layers, units=units, positions=positions, side=side)
            d.pop("values", None)
            d.pop("values_after", None)
            return d
        return await call(fn)

    @srv.tool()
    async def heads_causal(name: str, contrast: str | None = None, mode: str = "ablate", layers: str = "auto",
                           candidates: str | None = None, score_layers: list[int] | None = None,
                           base_units: list[str] | None = None, delta_s: float = 3.0, hard: int = 16,
                           b_side_max: float | None = None, top_layers: int = 4) -> dict:
        """因果スクリーニングをジョブで投げる(CHANGE-06 C、job_status で結果。生成なし)。方向 name への射影の値(score_layers、
        既定は基準層と後段 2 層)が、ヘッドを止めたとき(mode="ablate")どれだけ下がるか。layers="auto" は層単位で全ヘッドを
        止めて下がる top_layers 層を選び、その層のヘッドを 1 個ずつ測る。candidates は all / contrib:N / "L:Hh,…"。
        mode="marginal" は base_units(いまの組 "L:Hh:sS")から s を delta_s 上げたときの、効きにくい例 hard 件での
        下がり方 ÷ delta_s(効率 eta)。ablate と marginal は順位が違う(候補出しは ablate、配り直しは marginal)。
        行には直接の寄与 contrib と ratio = causal / contrib(1 より大きい = 間接の経路、小さい = 補い合い)、
        B 側への影響(b_side_delta・drift_orth_b、b_side_max で除外の印)。保存先 <root>/heads/<model_id>/<name>.causal.jsonl。"""
        try:
            params = {"direction": name, "contrast": None if contrast is None else str(_contrast_path(session, contrast, None, None, None, None)),
                      "mode": mode, "layers": layers, "candidates": candidates, "score_layers": score_layers,
                      "base_units": base_units, "delta_s": delta_s, "hard": hard, "b_side_max": b_side_max,
                      "top_layers": top_layers}
            return jobs.start("heads_causal", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_allocate(name: str, target: float, contrast: str | None = None, candidates: str | None = None,
                             n_heads: list[int] | None = None, score_layer: int | None = None, hard: int = 16,
                             b_side_max: float | None = None, method: str = "proportional", s_max: float | None = None) -> dict:
        """効率ベースの強さ配分をジョブで投げる(CHANGE-06 D、job_status で結果。生成なし)。heads_causal の結果(candidates、
        既定は方向の名前)から、効きにくい例(A 側で射影の値が高く残る hard 件)を選び、効率 eta を測り、n_heads(既定 [6, 5, 4])の
        各ヘッド数で s ∝ eta に配って、効きにくい例の射影の値の平均が target になるよう二分探索する。比較表(ヘッド数・各ヘッドの s・
        最大の s・残る件数・A 側の射影の値の最大・B 側の直交成分のずれ)と recommended(残る件数と B 側のずれが同程度なら最少の
        ヘッド数)、各組のレシピのパスを返す。method="greedy" は比を固定せず、いちばん効くヘッドを 1 本ずつ上げる(s_max の既定はなし、
        止まるのは目標か B 側のずれ)。行の stop が止まった理由、capped が上限に当たったヘッド。目標は射影の値の目標なので、組は振る舞いで確かめる。"""
        try:
            params = {"direction": name, "target": target, "candidates": candidates, "score_layer": score_layer,
                      "hard": hard, "b_side_max": b_side_max, "method": method, "s_max": s_max,
                      "contrast": None if contrast is None else str(_contrast_path(session, contrast, None, None, None, None))}
            if n_heads:
                params["n_heads"] = list(n_heads)
            return jobs.start("heads_allocate", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_pipeline(spec: dict[str, Any] | str, target: float | str | None = None, n_heads: list[int] | None = None,
                             max_examples: int | None = None, resume: bool = True) -> dict:
        """対照集合(name・user_prompts か側ごとの質問 prompts_a / prompts_b・system_a / system_b か recipe_a / recipe_b、
        任意で pipeline の設定。表か JSON / YAML の
        パス)から、方向づくり → 基準層 → 因果スクリーニング → 強さの配分(効き目に比例した最初の組 → 効率で配り直す →
        ヘッド数の比較)をジョブで 1 回で回す(job_status で結果)。質問は何百個でもよく、止めて測る段階は max_examples 個
        (既定 64)に間引く。target は数か "B"(B 側の平均まで消す)か "mid"(省いたときの既定、A と B の真ん中)。
        真ん中では振る舞いが半分ほどしか消えないことが多いので、消し切りたいときは "B"。結果の recommended が推す組のレシピ。resume(既定 True)なら、
        同じ対照集合・同じ設定で作り済みの方向と因果の結果を使い回す(止めて流し直したとき、配分からやり直せる)。"""
        try:
            p = {"spec": spec, "target": target, "n_heads": n_heads, "max_examples": max_examples,
                 "resume": None if resume else False}
            return jobs.start("head_pipeline", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def fieldtest(model: str | None = None, against: str | None = None, lora: str | None = None, name: str | None = None,
                        prompts: dict[str, Any] | None = None, match: str | None = None, long_tokens: int = 3072, tool_tokens: int = 1536,
                        prompt_tokens: int = 1536, n_parallel: int = 4, llama_server: str | None = None, ppl: bool = True,
                        bench: list[str] | None = None, bench_n: int | None = None, bench_tokens: int | None = None,
                        presence_penalty: float | None = None) -> dict:
        """実地テストをジョブで投げる(job_status で結果)。本番と同じ動かし方(llama-server、思考あり、サンプリングあり)で、長い作業
        (上限で切れた割合・ループ・本文が空)、ツール呼び出し(呼ぶか・名前・引数・不要なときに呼ばない・結果の後に続ける)、質問集
        (prompts は {名前: ファイルか質問のリスト}、match で一致率)、ベンチマーク(bench: gsm8k・mmlu_pro・math500・ifeval の正答率と、片方だけ正解の数)、perplexity を測り、against のファイルと比べる(compare の worse、
        same_level)。焼き込んだファイルもそのまま測れる。model を省くと読み込んでいるモデル。走っている間はそのモデルを GPU から下ろし、
        ほかの GPU の処理は待たされる。27B で質問集 200 問を足して 1 ファイル 30〜45 分。生成した文章は結果に入らず、replies_path に書く。"""
        try:
            p = {"model": model, "against": against, "lora": lora, "name": name, "prompts": prompts, "match": match, "long_tokens": long_tokens,
                 "tool_tokens": tool_tokens, "prompt_tokens": prompt_tokens, "n_parallel": n_parallel, "llama_server": llama_server, "ppl": ppl,
                 "bench": bench, "bench_n": bench_n, "bench_tokens": bench_tokens, "presence_penalty": presence_penalty}
            return jobs.start("fieldtest", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_refine(name: str, n_heads: int | None = None, new_heads: list[int] | None = None, resume: bool = True) -> dict:
        """組を当てても目標まで下がらない A 側の質問(残り)だけで、追加のヘッドを探すジョブを投げる(job_status で結果)。
        name は一括(heads_pipeline)か heads_allocate の名前、n_heads は残りを測る組(既定は推す組)。残りの質問と B 側の質問で、
        組を当てたまま方向を作り直し(方向 <name>_rK)、基準層 → 因果 → 配分で足すヘッドと強さ(new_heads、既定 [3, 2, 1])を決め、
        元の組と合わせたレシピ(combined.recipe)を作って、元の方向で残りの数(remain_before → remain_after)と B 側のずれを測り直す。"""
        try:
            p = {"name": name, "n_heads": n_heads, "new_heads": new_heads, "resume": None if resume else False}
            return jobs.start("heads_refine", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_candidates(alloc: str, prompts: list[str] | dict[str, list[str]] | str | None = None,
                               scales: list[float] | None = None, n_heads: list[int] | None = None, max_prompts: int = 24,
                               max_tokens: int = 120, sampling: str | dict[str, Any] = "llama_server", seed: int = 0,
                               judge_b: bool = False, with_recipe: str | None = None) -> dict:
        """強さの候補づくりをジョブで投げる(job_status で結果)。配分の結果(alloc、heads_allocate / heads_pipeline の名前)の
        比較表の組 × 全体の倍率 scales(既定 [0.5, 1, 1.5, 2, 3])と無介入で、A 側と B 側の質問に返事を作る(同じ種のサンプリング)。
        正の倍率は A 側の振る舞いを消す向き(A 側を採点)、負の倍率は符号を返して B 側で強める向き(B 側を採点)。
        候補ごとの判定の一致率(rate_a / rate_b)・繰り返しの少なさ(d4)・無介入と同じ返事の割合と、採点する側の返事を
        どの候補か伏せて並べた採点のファイル(judge.round、基準は judge.rubric)を返す。prompts は調整に使っていない質問
        (省くと対照集合の validate.prompts、無ければ調整に使った質問で held_out が False)。with_recipe は全部の候補
        (無介入の base も)に重ねるレシピ(ニューロンの種火など)。採点は heads_candidates_tally。"""
        try:
            p = {"alloc": alloc, "prompts": prompts, "scales": scales, "n_heads": n_heads, "max_prompts": max_prompts,
                 "max_tokens": max_tokens, "sampling": sampling, "seed": seed, "judge_b": judge_b,
                 "with_recipe": with_recipe}
            return jobs.start("heads_candidates", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_candidates_tally(name: str, scores: str | dict[str, list[int]] | None = None) -> dict:
        """候補の採点を集計する(GPU を使わない)。scores に「番号 P C」の行(または {番号: [P, C]})を渡すと採点のファイルに
        書き足してから集計する。P = 目標の振る舞いの強さ、C = 自然さ(どちらも 1〜5、基準は RUBRIC.md)。返り値の scores が
        候補 → 側 → P・C・min_PC・P_ge3・remove(min(6 − P, C)、消す向きの目的)・add(min(P, C)、強める向きの目的)、
        best が向きごとのいちばん良い候補。返事は名前を伏せたまま採点し、集計の前に候補を見ない。"""
        try:
            d = session.heads_candidates_tally(name, scores=scores)
            d.pop("replies", None)
            return d
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def headproj_export(recipe: str, bake: bool = False, requant: str | None = None, verify: bool = True,
                              strip_meta: bool = False, into: str | None = None, adapter_name: str | None = None) -> dict:
        """保存したレシピの proj のユニット(ヘッドの方向成分の射影、と gain・keyed)を書き出す(CHANGE-06 E)。既定は LoRA アダプタ
        GGUF(同じ層・同じ方向は階数 1 に畳む。数百 KB)。bake=True なら残差に書き込む重み行列だけ量子化し直した GGUF をジョブで作り
        (requant の既定は q8_0。keyed を含むなら auto: 元が Q8_0 のテンソルは Q8_0 のまま、それ以外は f16。Q8_0 のまま
        では編集の 1 割以上が丸めに消えるテンソルだけ f16。"q8_0" / "f16" / "keep" も選べる)、
        verify なら焼き込み版と LoRA 版を方向への射影の値・最初のトークンの KL・最尤トークンの一致率・ppl で比べる(V26)。射影の値は
        層・側ごとに例ごとの差の最大(max_abs_diff)を合格の線(threshold)と比べる。線は許容誤差(射影の値の標準偏差の 1 割、resolution)で、
        最後の層(last_layer)だけ許容誤差 + 計算方法の違いだけで出る誤差(noise.max_abs_diff = 書き込みがほぼ 0 の LoRA と
        LoRA なしの射影の値の差の最大。Q8_0 では LoRA を当てるだけで最後の層の射影の値が動くため)。
        strip_meta=True なら gne の記録(gne.* のメタデータ・レシピ・名前の印・横の .headproj.json)を残さない(人に渡す版)。LoRA では
        元のモデルを特定できる値(識別子、名前の欄の元のモデルの名前・レシピの名前)も書かず、名前の欄は "adapter"。
        adapter_name は LoRA の名前の欄(general.name。省くと「<元のモデルの名前> gain <レシピの名前>」。bake では使えない)。
        結果の local_path_keys は、メタデータに手元のパスらしい文字列が残っているキー。into は焼き込む先の別の GGUF
        (同じ構造の量子化していない F16 / BF16 など。方向はこのモデルのものを使う。requant="keep" で元の型のまま、verify はできない)。"""
        return await _export(recipe, None, None, None, None, bake=bake, requant=requant, verify=verify,
                             strip_meta=strip_meta, into=into, adapter_name=adapter_name)

    async def _export(recipe, units, name, key, direction, *, bake, requant, verify, strip_meta, into,
                      lora_f16: bool = False, adapter_name: str | None = None) -> dict:
        """headproj_export・keyed_export の中身(jobs._headproj_export。ファイル名の既定は Session.default_export_path)。
        焼き込みはジョブ、LoRA はその場で書く。"""
        from .jobs import _headproj_export
        p = {"recipe": recipe, "units": units, "name": name if name else (_recipe_stem(recipe) if recipe else None),
             "key": key, "direction": direction}
        p = {k: v for k, v in p.items() if v is not None}
        if adapter_name is not None:
            if bake:
                return _err(GneError("BAD_ARGS", "adapter_name(名前の欄)は LoRA の書き出しだけです"))
            p["adapter_name"] = adapter_name
        if bake:
            try:
                q = {**p, "bake": True, "requant": requant, "verify": verify and not into, "strip_meta": strip_meta}
                if into:
                    q["into"] = into
                return jobs.start("headproj_export", q).view()
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await call(lambda: _headproj_export(session, **p, strip_meta=strip_meta, lora_f16=lora_f16))

    @srv.tool()
    async def keyed_export(recipe: str | None = None, units: list[str] | None = None, name: str | None = None,
                           key: str | None = None, direction: str | None = None, bake: bool = False,
                           requant: str | None = None, verify: bool = False, strip_meta: bool = False,
                           into: str | None = None, lora_f16: bool = False, adapter_name: str | None = None) -> dict:
        """鍵つきの書き込み(keyed、CHANGE-07)を書き出す(headproj_export の keyed 向けの別名。proj・gain が混ざっていても同じ
        ファイルに入る)。recipe は保存したレシピの名前、または units(例 ["19:ffn:keyed:a1@zundamon_endings#zundamon_endings"])と
        name。key・direction は #KEY・@NAME を省いたユニットの鍵と方向。既定は LoRA アダプタ GGUF(層・鍵ごとに階数 1、llama.cpp の
        --lora でそのまま当たる。配るならこれが第一候補。keyed だけなら <名前>-keyed.lora.gguf)。bake=True なら残差に書き込む
        重み行列(ffn_down など)だけ書き直した GGUF をジョブで作る(requant の既定は auto: テンソルごとに、元が Q8_0 なら Q8_0 のまま
        (f16 にすると計算方法が変わり、LoRA 版とかえってずれた)、それ以外(K-quant・IQ・F16 など)は f16。Q8_0 のままでは
        rank 1 の小さな編集の 1 割以上が丸めに消えるテンソルだけ f16(記録の fallback)。結果の edit_kept_min(残った編集の割合)が
        0.9 を下回ったら警告。requant_error_max は編集と無関係な丸めのゆらぎも数えるので、小さな編集では大きく出る)。MoE の専門家のテンソル(ffn_down_exps、expert:e)も同じ規則で、
        Q8_0 のままなら編集しない専門家はバイトのまま、f16 だとテンソル全体が F16 になる(結果の expert_bytes_added が増えた大きさ)。
        verify=True で焼き込み版と LoRA 版を比べる(V33。判定は headproj_export の verify と同じで、最後の層の射影の値の合格の線
        threshold は許容誤差 + 計算方法の違いだけで出る誤差 noise.max_abs_diff)。
        strip_meta=True(人に渡す版)の LoRA は、gne の記録と元のモデルを特定できる値(識別子、名前の欄の元のモデルの名前・レシピの名前)を
        書かず、名前の欄は "adapter"。adapter_name は LoRA の名前の欄(general.name。name はレシピの名前)。"""
        if bool(recipe) == bool(units):
            return _err(GneError("BAD_ARGS", "recipe(保存したレシピの名前)か units のどちらかを渡してください"))
        return await _export(recipe, units, name, key, direction, bake=bake, requant=requant, verify=verify,
                             strip_meta=strip_meta, into=into, lora_f16=lora_f16, adapter_name=adapter_name)

    @srv.tool()
    async def direction_key(contrast: str, layers: list[int] | str, name: str | None = None, positions: str | None = None,
                            positions_b: str | None = None, skip: int | None = None, max_tokens: int = 96,
                            q_mode: str = "examples", q_energy: float = 0.99, q_max_rank: int | None = None,
                            q_max_samples: int = 4096, subset: str = "even", pseudo: int = 16,
                            replies_a: str | list[str] | None = None, replies_b: str | list[str] | None = None,
                            iv: str | list[str] | None = None, target: str = "ffn", seed: int = 0,
                            dont_touch_start: bool | None = None, role: str | None = None) -> dict:
        """鍵つきの書き込みの鍵 κ を作るジョブを投げる(CHANGE-07、job_status で結果)。対照集合(configs/contrast/ の名前かパス)の
        書き込みたい位置の活性の平均 μ を、書き込みたくない位置と dont_touch の活性が張る空間に直交させ、κᵀμ = 1 にして
        <root>/keys/<model_id>/<name>.npz に保存する。layers は必須(例 [8, 14, 20])。positions(省くと対照集合の値)が
        "before:<正規表現>" なら推論時と同じ文脈のモデル自身の返事で、一致する文字を書く直前を書き込みたい位置、残りの位置を書き込みたくない位置にする。
        replies_a / replies_b は返事を読ませるファイルかリスト(生成しない)。q_mode: examples(既定)/ means / samples。
        subset は既定 even(direction_key_check は odd で確かめる)。iv はこの介入(レシピかユニット)を当てた状態で作る。
        target は書き込み先: ffn(既定、dense の ffn_down の入力)/ attn(attn_output・ssm_out の入力 = ヘッド出力の連結)/
        shared(MoE の層の shared expert の ffn_down_shexp の入力)/ expert:e(ffn_down_exps の専門家 e の入力。専門家 e が
        選ばれた位置だけで作るので、書き込みたい位置でよく選ばれる専門家を選ぶ)。ユニットは L:ffn / L:attn / L:S / L:Ee:keyed:…。
        dont_touch の各群の質問の、返事を書き始める直前の位置も Q に入る(既定。dont_touch_start=False か
        対照集合の dont_touch_start: false で切る。入れた行の数は結果の dont_touch_start。書き込みたくない側の、返事を書き始める直前の位置は入れない)。
        結果の層ごとの kn(1 に近いほど書き込みたくない位置と重ならない、0.05 未満は弱い)と build_auc(作成に使った例での値で甘い)。
        warnings を必ず見る: 書き込みたい位置を含む例が 2 未満か、書き込みたい位置が 8 未満(専門家の鍵は層ごとの行)なら少ない。
        "before:<正規表現>" で書き込みたい位置の 25% 以上(μ に占める重み、結果の reply_start.share)が返事の冒頭(最初の
        3 トークンを決める位置)にあると、鍵がどの返事の冒頭にも反応しやすい。dont_touch の群があれば、その質問の、返事を
        書き始める直前の位置が Q に入っているので注(notes)になり、無ければ
        警告になる(ふつうの質問を dont_touch に足して作り直す)。
        FiSH(計算で作るニューロン、mode nw)の鍵は target="mlpin"(dense の MLP の入力の空間)。返事の位置の区切りで反応する鍵は
        positions="pos:1-3"(位置 0 = 返事を書き始める直前、"pos:4-" は 4 以降)。role="scene" は狙う場面の質問(user_prompts)への
        返事の間ずっと反応し、dont_touch の質問では反応しない鍵、role="one" はどの位置でもほぼ 1 を返す鍵(上限つきの形用)。
        role を渡すと target は mlpin で、positions・q_mode などは使わない。"""
        from .jobs import _layers
        try:
            if not _layers(layers):
                raise GneError("BAD_ARGS", "layers に鍵を作る層を指定してください(例 [8, 14, 20])")
            p = {"contrast": str(_contrast_path(session, contrast, None, None, None, None)), "layers": layers, "name": name,
                 "positions": positions, "positions_b": positions_b, "skip": skip, "max_tokens": max_tokens,
                 "q_mode": q_mode, "q_energy": q_energy, "q_max_rank": q_max_rank, "q_max_samples": q_max_samples,
                 "subset": subset, "pseudo": pseudo, "replies_a": replies_a, "replies_b": replies_b, "iv": iv,
                 "target": target, "seed": seed, "dont_touch_start": dont_touch_start}
            if role in ("scene", "one"):
                p = {"contrast": p["contrast"], "layers": layers, "name": name, "max_tokens": max_tokens, "subset": subset,
                     "pseudo": pseudo, "iv": iv, "seed": seed, "role": role}
            elif role not in (None, "position"):
                raise GneError("BAD_ARGS", f"role は position / scene / one です({role!r})")
            return jobs.start("direction_key", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def nw_neurons(layers: list[int] | str, top: int = 20, sort: str = "sum", max_tokens: int = 96,
                         corpus_tokens: int = 2048, iv: str | list[str] | None = None) -> dict:
        """FiSH(計算で作るニューロン、mode nw)で上書きするニューロンを選ぶ材料のジョブを投げる(job_status で結果)。層ごとに、各ニューロンの
        ふだんの働き(|ゲート後の活性| × 出力の列の長さ = 1 位置で残差に書く長さ)を、ふつうの質問への無介入の返事と既定のコーパスで
        測り、働きの小さい順の表(番号・平均・最大・最大 ÷ MLP の出力の長さ・順位)を返す。sort: sum(最大の順位 + 平均の順位、既定)/
        max / mean。どれを使うかは表を見て選ぶ(自動では決めない)。上書きすると、そのニューロンの元の働きは無くなる。
        iv(保存したレシピの名前かパス、またはユニットの書き方のリスト)を渡すと、その介入を当てた状態で測る(例: CHiPS のレシピを
        当てた状態でニューロンを選ぶ)。nw_verify・nw_check・direction_key の iv も同じ。"""
        try:
            return jobs.start("nw_neurons", {k: v for k, v in {"layers": layers, "top": top, "sort": sort, "max_tokens": max_tokens,
                                                               "corpus_tokens": corpus_tokens, "iv": iv}.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def nw_verify(units: list[str] | None = None, recipe: str | None = None, texts: list[str] | None = None,
                        iv: str | list[str] | None = None) -> dict:
        """nw のユニット("L:Nn:nw:aα@方向#鍵" = 鍵 1 つ、"…#場面の鍵&位置の鍵" = 鍵 2 つ、"L:Nn+Nm:nw:aα@方向#鍵&1を返す鍵" = 上限つき)の
        LoRA が式どおりに動くかを確かめる。上書きしたニューロンのゲート・入力・活性が式どおりか、ほかのニューロンが変わっていないか、
        MLP の出力の変化が合っているか(層ごとの ok)。recipe は保存したレシピの名前、または units。"""
        try:
            if bool(recipe) == bool(units):
                raise GneError("BAD_ARGS", "recipe(保存したレシピの名前)か units のどちらかを渡してください")
            r = session.recipe(path=session.recipe_path(recipe)) if recipe else session.recipe(units=units)
        except Exception as e:  # noqa: BLE001
            return _err(e)
        return await call(session.nw_verify, r, texts=texts, iv=iv)

    @srv.tool()
    async def nw_check(contrast: str, units: list[str] | None = None, recipe: str | None = None, subset: str = "odd",
                       replies: str = "own", max_tokens: int = 96, n_pos: int = 8, iv: str | list[str] | None = None,
                       low: float = 0.4, over: float = 1.2, top: int = 5, save: str | None = None) -> dict:
        """nw のレシピについて、位置の種類ごとの鍵の反応と実際に足した量(α·d を 1 として)を測るジョブを投げる(job_status で結果)。
        対照集合の質問(狙う場面 = groups.scene)と dont_touch の群(書き込みたくない場面)で、返事の位置 0(書き始める直前)〜
        n_pos−1 とそれ以降について、ユニットごとに amount・rho_gate・rho_up の平均・最小・最大。replies="own" はレシピを当てて
        作った返事を当てたまま読む(実際に起きること。groups.<名前>.loops に繰り返しの数)、"base" は無介入の返事を読む。
        見るところ: 狙う場面で足す量が 1 から大きく外れる位置、書き込みたくない場面で 0 から離れる位置、負(逆向き)の位置。
        平均だけでは、一部の質問でまったく書けていないことが隠れるので、質問ごとの値も返す。区間 = ユニットが担当する返事の位置の
        範囲(位置の鍵の pos:<範囲>)。groups.<名前>.questions は質問ごとの行(prompt・reply_start = 返事の出だし・segments.<区間> の
        amount = その区間で書いた量(狙った量 = 1)、total = 全部のユニットの合計、rho = 鍵の反応)。groups.<名前>.by_question は区間ごとの
        数(狙う場面は low = amount が low 未満の質問、over = total が over を超えた質問。書き込みたくない場面は leak = 書いてしまった質問)。
        question_summary.worst は、ほとんど書けていない質問の上位 top 件。書けていない質問が、返事の書き出し方(かぎかっこで始まる、
        など)でまとまっていたら、鍵を作る質問を増やす。save(名前)で <root>/behave/<model_id>/<名前>_check.json にも書く。"""
        try:
            if bool(recipe) == bool(units):
                raise GneError("BAD_ARGS", "recipe(保存したレシピの名前)か units のどちらかを渡してください")
            r = str(session.recipe_path(recipe)) if recipe else session.recipe(units=units)
            return jobs.start("nw_check", {"recipe": r, "contrast": str(_contrast_path(session, contrast, None, None, None, None)),
                                           "subset": subset, "replies": replies, "max_tokens": max_tokens, "n_pos": n_pos,
                                           "low": low, "over": over, "top": top, **({"save": save} if save else {}),
                                           **({"iv": iv} if iv is not None else {})}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    def _fish_contrast(contrast) -> dict:
        """FiSH のジョブの対照集合: 表(spec)、configs/contrast/ の名前・パス、configs/fish/ のひな形の名前。"""
        if isinstance(contrast, dict):
            return {"spec": {k: v for k, v in contrast.items() if k != "about"}}
        try:
            return {"contrast": str(_contrast_path(session, contrast, None, None, None, None))}
        except GneError:
            tpl = CONFIGS / "fish" / f"{str(contrast).removesuffix('.json')}.json"
            if not tpl.exists():
                raise
            d = json.loads(tpl.read_text(encoding="utf-8"))
            d.pop("about", None)
            return {"spec": d}

    @srv.tool()
    async def fish_pipeline(contrast: dict[str, Any] | str, layers: list[int] | str, alpha: float | list[float] | str | None = None,
                            segments: str | None = None, form: str | None = None, name: str | None = None,
                            same_reply: bool | None = None, stack: bool | None = None, rekey: bool | None = None,
                            pseudo: int | None = None, max_tokens: int | None = None, response_tokens: int | None = None,
                            iv: str | list[str] | None = None, neurons: dict[str, list[int]] | None = None, top: int = 20,
                            save: str | None = None, summary: bool = False, direction: str | None = None,
                            cap: str | list[str] | None = None) -> dict:
        """FiSH を最初から最後まで 1 回のジョブで作る(job_status で結果。UI の FiSH のタブの「まとめて作る」と同じ): 方向 → ニューロンの表 →
        鍵 → ユニット → そのユニットを当てた状態で鍵を作り直す → 式どおりか → 足した量。contrast は対照集合の表(name・system_a・
        user_prompts・dont_touch と、設定の fish)か、configs/contrast/ の名前・パス、configs/fish/ のひな形の名前。
        layers は必須で、使う人が決める(リストか "11-23"・"11,18,23"。モデルで違う)。alpha は強さで、数 1 つなら全部の区切りに同じ値、
        区切りごとに変えるなら区切りの順に [1, 1, 0.8] か "1,1,0.8"(segments "0,1-3,4-" に対して)。form は cap(上限つきの 2 個組、既定)/
        two / one。渡していない引数は、表の fish の設定、その次に既定(segments "0,1-3,4-"、form cap、alpha 0.9、same_reply・stack・rekey あり)。
        層を増やすと、要るニューロンの数(層 × 区切り × 形の個数)と作成時間が増える(Qwen3.5-4B の 13 層で 78 個、方向と鍵に 16 分ほど)。
        結果: units(レシピのユニットの書き方)、direction・usage・neurons・keys・rekeyed・verify(ok)・check(groups.scene と
        dont_touch:<名前> の、位置ごとの足した量)、stage_seconds、summary。作る前に層と強さを比べるなら nw_try。
        save(名前か .yaml で終わるパス)を渡すと、できたユニットをレシピのファイルに書く(名前なら <root>/recipes/<名前>.yaml。結果の
        recipe_path。nw_verify・nw_check・nw_export の recipe にその名前を渡せる)。省くと書かない。
        結果は数十万字になる(ニューロンの表・鍵の表・足した量の細目)。summary=True なら、代わりに短い要約だけを返す: ok(式どおりか)・
        n_units・n_neurons・recipe_path・check(場面の群 × 位置ごとの足した量の平均・最小・最大と、by_question = 区間ごとの、ほとんど
        書けていない質問・狙った量を超えた質問の数)・questions(書けていない質問の上位)・units(ユニットの書き方)。
        direction(すでにある方向の名前)を渡すと、方向を作り直さずにそれを使う(区切り・層・重ねる層が合わなければ BAD_DIRECTION)。
        cap は上限つきの形のしきい値 "a-b"(既定 0.4-0.8。鍵の反応が a より小さいとほぼ書かず、b より大きいと狙った量で頭打ち)。1 つなら
        全部の区切りに同じ値、区切りごとに変えるなら区切りの順に ["0.4-0.8", "0.3-0.7", "0.3-0.7"] か "0.4-0.8,0.3-0.7,0.3-0.7"。
        鍵は、返事を 1 回だけ作って読み、全部の区切りの分を作る。鍵を作る質問は多いほうがよい(狙う場面の質問は 100 問以上を勧める)。"""
        from .jobs import _layers
        try:
            if not _layers(layers):
                raise GneError("BAD_ARGS", "layers に書く層を指定してください(例 [11, 18, 23] か \"11-23\")")
            p = {**_fish_contrast(contrast), "layers": layers, "alpha": alpha, "segments": segments, "form": form, "name": name,
                 "same_reply": same_reply, "stack": stack, "rekey": rekey, "pseudo": pseudo, "max_tokens": max_tokens,
                 "response_tokens": response_tokens, "iv": iv, "neurons": neurons, "top": top, "save": save,
                 "direction": direction, "cap": cap, "summary": True if summary else None}
            return jobs.start("fish_pipeline", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def nw_try(direction: str, contrast: dict[str, Any] | str, layers: list[int] | str | None = None,
                     alpha: float | list[float] | str | None = None, prompts: list[str] | None = None, n: int = 12,
                     max_tokens: int = 120, match: str | None = None, temperature: float = 0.0, seed: int = 0,
                     others: bool = True, baseline: bool = True, name: str | None = None, iv: str | list[str] | None = None) -> dict:
        """FiSH を重みに入れる前に試すジョブを投げる(job_status で結果。gne nw try)。区切りつきの方向 direction
        (direction(positions="response", same_reply=True, segments="0,1-3,4-", stack_layers=…) で作ったもの)を、ニューロンにも鍵にも
        入れずに、層 layers の出力へ直接足して返事を作る(返事の位置の区切りが変わるところで control vector を付け替える)。
        足すのは狙う場面の質問(prompts か、対照集合の質問の先頭 n 問)の返事だけ(場面は質問の群で決める。重みに入れたときの場面の鍵の
        代わり)。alpha は区切りごとの強さ([1, 1, 0.8] か "1,1,0.8"。数 1 つなら全部の区切りに同じ値)。layers を省くと、方向を作った
        ときの重ねる層。結果: scene(足した返事)・baseline(同じ質問に何も足さない返事)・others(dont_touch の質問に何も足さない返事)の
        それぞれに counts(n・loop = 繰り返しに落ちた返事・empty・at_limit・tokens・match = 正規表現の一致)と replies。層の組・区切り
        ごとの強さを変えて数回投げ、返事を読んで比べてから fish_pipeline に進む。temperature > 0 でサンプリング。
        結果は <root>/behave/<model_id>/<name>_try.json にも残る。name を省くと「<方向の名前>_<設定から作った 6 文字>」で、設定(層・強さ・
        質問・長さ・温度・種など)が違えば別のファイルになる。name を渡すとその名前で、同じ名前の前の結果は上書きする(notes で知らせる)。
        制約: 足せるのは層 1 〜 最終層の 1 つ手前。狙う場面は 1 問ずつ生成する(付け替えは 1 本の列ごと)。数は機械的に数えたもので、
        口調が出たか・相手の発言に合っているかは返事を読んで確かめる。"""
        try:
            p = {**_fish_contrast(contrast), "direction": direction, "layers": layers, "alpha": alpha, "prompts": prompts, "n": n,
                 "max_tokens": max_tokens, "match": match, "temperature": temperature, "seed": seed, "others": others,
                 "baseline": baseline, "name": name, "iv": iv}
            return jobs.start("fish_try", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def nw_export(recipe: str | None = None, units: list[str] | None = None, name: str | None = None,
                        strip_meta: bool = False, adapter_name: str | None = None) -> dict:
        """nw のユニットを LoRA アダプタ GGUF(<root>/export/<名前>-nw.lora.gguf)に書き出す。llama.cpp の --lora で当てる(倍率 1)。
        ffn_gate・ffn_up・ffn_down への差し替えで、ニューロン 1 個あたり 3 つのテンソルに階数 1。焼き込みは無い(Q8_0 の ffn_down は
        32 個の重みが 1 つの倍率を共有するので、1 列だけを方向に差し替えると誤差が 9 割を超える)。レシピのほかの介入(keyed・gain など)は
        このファイルに入らない。proj(CHiPS)・keyed・gain と一緒に 1 つの LoRA にするなら headproj_export / keyed_export を使う
        (nw のユニットも同じファイルに入る。bake=True はエラー)。
        strip_meta=True(人に渡す版)は、gne の記録(gne.nw)と、元のモデルを特定できる値(元のモデルの識別子、名前の欄に入る元のモデルの
        名前・レシピの名前)を書かない。adapter_name はファイルの名前の欄(general.name。省くと、ふつうは「<元のモデルの名前> nw
        <レシピの名前>」、strip_meta では "adapter")。name は units を渡したときのレシピの名前(出力のファイル名に使う)。"""
        try:
            if bool(recipe) == bool(units):
                raise GneError("BAD_ARGS", "recipe(保存したレシピの名前)か units のどちらかを渡してください")
            r = session.recipe(path=session.recipe_path(recipe)) if recipe else session.recipe(units=units, name=name or "units")
            out = session.store.root / "export" / f"{r.name or 'recipe'}-nw.lora.gguf"
        except Exception as e:  # noqa: BLE001
            return _err(e)
        return await call(session.nw_export, r, out, strip_meta=strip_meta, name=adapter_name)

    @srv.tool()
    async def direction_key_check(key: str, contrast: str | None = None, subset: str = "odd", layers: list[int] | str | None = None,
                                  positions: str | None = None, positions_b: str | None = None, skip: int | None = None,
                                  max_tokens: int | None = None, replies_a: str | list[str] | None = None,
                                  replies_b: str | list[str] | None = None, iv: str | list[str] | None = None,
                                  strict: bool = False) -> dict:
        """鍵の分離の確認のジョブを投げる(CHANGE-07、job_status で結果)。対照集合(省くと鍵に保存した作成時の例で、甘い値)の
        各位置で鍵の反応の強さ ρ = κᵀx を測り、層ごとに書き込みたい位置 対 書き込みたくない位置の AUC(auc は例単位、
        auc_token はトークン単位)、key-check の選択性 selectivity(例ごとの ρ の平均の、書き込みたい位置 − 書き込みたくない
        位置)と比較用の鍵(ラベルをシャッフルして同じ手順で作った鍵)の p95、beats_pseudo(選択性が比較用の鍵の p95 を超えた)、
        dont_touch の ρ の中央値を返す。位置・skip・max_tokens の既定は鍵を作ったときの値。作成に使った例と重なると
        held_out が false(strict なら BAD_ARGS)。判断は held_out が true の結果で、beats_pseudo の層だけを使う。
        層ごとの start_rho は書き込みたくない位置と dont_touch の、返事の冒頭(位置 0 = 返事を書き始める直前の位置、1・2 = 返事の
        最初の 2 トークンを読んだ位置)での ρ の中央値。dont_touch の ρ の中央値は返事の全位置の中央値なので、返事の冒頭の
        1 位置で反応しても見えない。中央値が 0.4 以上なら warnings に出る(dont_touch なら書き込みたくない場面への漏れ。
        ふつうの質問を dont_touch に足して鍵を作り直す。dont_touch の質問の、返事を書き始める直前の位置は鍵の作成で自動で Q に入る)。
        この位置を Q に入れて作った鍵では、key-check も dont_touch の各群のこの位置を測り(返事の冒頭が positions の正規表現に
        一致しても)、start_rho の位置 0 と層ごとの dont_touch_start_median、counts の dts:<群>、dont_touch_start に入れる
        (AUC・選択性には入れない)。保留集合の書き込みたい位置を含む例が
        2 未満か、書き込みたい位置が 8 未満でも warnings に出る(選択性が揺れる)。"""
        try:
            p = {"key": key, "subset": subset, "layers": layers, "positions": positions, "positions_b": positions_b,
                 "skip": skip, "max_tokens": max_tokens, "replies_a": replies_a, "replies_b": replies_b, "iv": iv,
                 "strict": strict, "compact": True,
                 "contrast": None if contrast is None else str(_contrast_path(session, contrast, None, None, None, None))}
            return jobs.start("direction_key_check", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_key_verify(layer: int, key: str, direction: str, alphas: list[float] | None = None,
                                   target: str = "ffn", direction_layer: str = "own") -> dict:
        """V30(CHANGE-07、数秒): keyed の LoRA が式どおり ΔW x = α (κᵀx) d を書くか。無介入で重み行列の入力 x と出力を
        捕捉し、LoRA を当てた出力の差と比べる(rows の rel_err_median < 3%、鍵の作成に使った書き込みたくない側の例で leak_b < 0.1、v30)。
        target は鍵の書き込み先と同じ ffn / attn / shared / expert:e(例 "expert:31")、alphas の既定は [1, 2]。shared は
        ゲートの前、expert:e はルーターの重みの前の出力を、専門家 e が選ばれた枠で測る。expert:e の rows には other_max
        (ほかの専門家の枠の差、0 のはず)と selection_changed(選ばれる専門家が変わったトークンの数、0 のはず)も入る。
        MoE では結果に gate(鍵の gate_a)・res_scale(granite 系の residual_scale)も出る。"""
        def fn():
            from .keyed import parse_target
            kind, expert = parse_target(target)
            return session.check_keyed(layer, key=key, direction=direction, alphas=tuple(alphas or (1.0, 2.0)),
                                       kind=kind, expert=expert, direction_layer=direction_layer)
        return await call(fn)

    @srv.tool()
    async def direction_key_experts(contrast: str, layers: list[int] | str, positions: str | None = None,
                                    positions_b: str | None = None, skip: int | None = None, max_tokens: int = 96,
                                    subset: str = "even", replies_a: str | list[str] | None = None,
                                    replies_b: str | list[str] | None = None, iv: str | list[str] | None = None,
                                    top: int = 8) -> dict:
        """expert:e の鍵の専門家を選ぶ材料のジョブを投げる(CHANGE-07 MoE、job_status で結果)。鍵と同じ対照集合(configs/contrast/
        の名前かパス)・位置・返事で、書き込みたい位置・書き込みたくない位置で各専門家が選ばれた割合を層ごとに返す: layers[].top[] の expert・
        rate_a・rate_b(例ごとの選ばれた割合の平均)・diff・lift(rate_a / rate_b)・weight_a(選ばれたときのルーターの重みの平均
        = 鍵の gate_a の見込み)・rows_a(書き込みたい位置で選ばれる行の数の見込み)・rate_dt(dont_touch の群)。rate_a の大きい順に top 個。
        専門家の鍵は書き込みたい位置で専門家 e が選ばれた行だけでできるので、rate_a が大きく rate_b が小さい専門家を direction_key の
        target="expert:e" に選ぶ。layers は必須(MoE の層、例 [6, 12, 18])。ほかの引数は direction_key と同じ。"""
        from .jobs import _layers
        try:
            if not _layers(layers):
                raise GneError("BAD_ARGS", "layers に調べる MoE の層を指定してください(例 [6, 12, 18])")
            p = {"contrast": str(_contrast_path(session, contrast, None, None, None, None)), "layers": layers,
                 "positions": positions, "positions_b": positions_b, "skip": skip, "max_tokens": max_tokens,
                 "subset": subset, "replies_a": replies_a, "replies_b": replies_b, "iv": iv, "top": top}
            return jobs.start("direction_key_experts", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_key_pipeline(contrast: dict[str, Any] | str, layers: list[int] | str | None = None,
                                     name: str | None = None, direction: str | None = None, rebuild_direction: bool = False,
                                     response_tokens: int = 96, target: str | None = None, positions: str | None = None,
                                     positions_b: str | None = None, skip: int | None = None, max_tokens: int | None = None,
                                     q_mode: str | None = None, subset: str | None = None, pseudo: int | None = None,
                                     replies_a: str | list[str] | None = None, replies_b: str | list[str] | None = None,
                                     iv: str | list[str] | None = None, seed: int | None = None,
                                     dont_touch_start: bool | None = None, check: bool = True,
                                     same_reply: bool | None = None) -> dict:
        """鍵つきの書き込みを、対照集合 1 つから「方向 → 鍵 → key-check」まで 1 回のジョブで回す(CHANGE-07、job_status で結果。
        UI の鍵のタブの「方向から key-check まで」と同じ)。contrast は対照集合の表(name・user_prompts・positions・dont_touch・
        system_a など。<root>/contrasts/<name>.yaml に書いてから使う)か、configs/contrast/ の名前・configs/keyed/ のひな形の名前・パス。
        1. 方向 direction(省くと対照集合の name): gne direction --positions response と同じ(A 側 = system_a、B 側 = system_b か system なしの
        返事の差)。同じ名前の方向がもうあれば使い回す(rebuild_direction=True で作り直す。中身の違う対照集合から作った方向なら warnings)。
        2. 鍵 name(省くと対照集合の name): direction_key と同じ引数(target・positions・q_mode・subset・pseudo・replies_a など)。
        3. key-check: 作成と反対側の subset(even ↔ odd)で、同じ対照集合・同じ返事で測る(結果の check は direction_key_check を
        compact にしたもの)。check=False で回さない。layers は必須(引数か表の keyed.layers)。表の keyed(layers・target・q_mode・
        subset・pseudo・max_tokens・alpha)は渡していない引数の既定になる(alpha は結果の alpha_hint に返すだけ。レシピの α の目安)。
        結果: direction(name・reused・rebuilt)、key(direction_key の結果)、check(key-check の結果)、contrast_path、warnings。
        判断は direction_key_check と同じ: check の held_out が true で、beats_pseudo の層だけを使う。
        same_reply=True で、方向を「同じ返事(A 側の返事)を A 側・B 側の条件の両方で読ませた差」で作る(既定は今までどおり。
        表の keyed.same_reply でも指定できる。結果の direction.same_reply に記録)。"""
        from .jobs import _layers
        try:
            spec, path = None, None
            if isinstance(contrast, dict):
                spec = contrast
            else:
                try:
                    path = str(_contrast_path(session, contrast, None, None, None, None))
                except GneError:
                    # ひな形の名前: configs/keyed/ の次に、gne direction key-from-behave などで作った <root>/keyed/
                    tpl = CONFIGS / "keyed" / f"{contrast}.json"
                    if not tpl.exists() and re.fullmatch(r"[\w.\-]+", str(contrast)) and getattr(session, "store", None) is not None:
                        tpl = session.store.root / "keyed" / f"{contrast}.json"
                    if not tpl.exists():
                        raise
                    spec = json.loads(tpl.read_text(encoding="utf-8"))
            if layers is not None and not _layers(layers):
                raise GneError("BAD_ARGS", "layers に鍵を作る層を指定してください(例 [8, 14, 20])")
            p = {"spec": spec, "contrast": path, "layers": layers, "name": name, "direction": direction,
                 "rebuild_direction": bool(rebuild_direction) or None, "response_tokens": response_tokens, "target": target,
                 "positions": positions, "positions_b": positions_b, "skip": skip, "max_tokens": max_tokens, "q_mode": q_mode,
                 "subset": subset, "pseudo": pseudo, "replies_a": replies_a, "replies_b": replies_b, "iv": iv, "seed": seed,
                 "dont_touch_start": dont_touch_start, "check": None if check else False, "compact": True,
                 "same_reply": same_reply}
            return jobs.start("direction_key_pipeline", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_key_causal(contrast: dict[str, Any] | str, layers: list[int] | str | None = None,
                                   targets: list[str] | str | None = None, alphas: list[float] | str | None = None,
                                   pseudo: int = 3, iv: str | list[str] | None = None, no_iv: bool = False,
                                   direction: str | None = None, measure_match: str | None = None,
                                   probe: str | list[dict[str, Any]] | None = None, max_tokens: int | None = None,
                                   side_tokens: int = 32, name: str | None = None) -> dict:
        """keyed の効く場所(層・書き込み先・α)を探す第 1 段: 生成なしのふるい分け(ジョブ、job_status で結果)。
        key-check は鍵が場面を分けるか(鍵の反応の強さ)しか見ないので、どこに書けば振る舞いが変わるかはこれで測る。
        候補(layers の層 × targets の書き込み先。既定は全層 × ffn,attn)ごとに、対照集合の偶数番目で鍵を作り(<name>-ffn などで保存)、
        作成に使っていない奇数番目の返事で、介入 iv(既定は対照集合の keyed.iv)を当てた状態に keyed のユニット(alphas の段ごと、
        既定は 1,2,4 に keyed.alpha の符号)を足したときの変化を測る: rows[].reply = 振る舞いの部分(既定は positions の正規表現に
        一致した文字、measure_match で別に)の対数確率の変化(負 = 出にくくなる)、rows[].probe = probe(JSONL のパスか行のリスト。
        1 行 {prompt, prefix(例 "Answer:"), behavior(振る舞いの答え), desired(望む答え), group?(書き込みたくない場面の行)})の
        log p(振る舞いの答え) − log p(望む答え) の変化、rows[].side = dont_touch の群と書き込みたくない位置の 1 トークンあたりの
        対数確率の変化、rows[].pseudo = 候補ごとの比較用の鍵(pseudo 本)の値。mark = 全候補の比較用の鍵の 5% 点より下がった
        (marks[α] に候補の数・印の数・偶然でも印が付く数 expected = 候補の数 × 5%)。key_check は参考、linear.ratio は比例の目安。
        candidates は 1 か所(書き込み先 × 層)につき 1 つ: by_alpha[α] に reply(下がり方)・mark・side_b(副作用)・ratio(下がり方 ÷ 副作用、
        副作用の下限 ratio_floor = 0.01)・unit、all_marked(全部の強さで有意)。下がり方・副作用・all_marked の 3 つを並べて候補を選ぶ
        (根拠は Qwen3.5-4B の語尾の 1 例。下がり方だけで選ぶと、副作用も大きい=返事を繰り返しに落としている候補を拾う)。
        notes[0] のとおり、これは生成なしの目安で、比較用の鍵と見分けがつかない候補を外すためのもの。印の数が expected と
        同じくらいなら、印だけでは比較用の鍵と見分けがつかない(それでも、いくつかの α で続けて印が付き、効き目に比べて
        side.b(書き込みたくない位置の変化)が小さい候補は試す価値がある。効き目が最大でも side.b が比較用の鍵並みに大きい候補は、
        返事を繰り返しに落としているだけのことがある)。効くかどうかは direction_key_causal_verify の生成で数える。
        contrast は対照集合の表かパス・configs/keyed/ のひな形の名前・<root>/keyed/ の名前。"""
        from .jobs import _layers
        try:
            spec, path = None, None
            if isinstance(contrast, dict):
                spec = contrast
            else:
                try:
                    path = str(_contrast_path(session, contrast, None, None, None, None))
                except GneError:
                    tpl = CONFIGS / "keyed" / f"{contrast}.json"
                    if not tpl.exists() and re.fullmatch(r"[\w.\-]+", str(contrast)) and getattr(session, "store", None) is not None:
                        tpl = session.store.root / "keyed" / f"{contrast}.json"
                    if not tpl.exists():
                        raise
                    spec = json.loads(tpl.read_text(encoding="utf-8"))
            if layers is not None and layers != "all" and not _layers(layers):
                raise GneError("BAD_ARGS", "layers は層のリストか \"8-20\"・\"all\" です")
            p = {"spec": spec, "contrast": path, "layers": layers, "targets": targets, "alphas": alphas, "pseudo": pseudo,
                 "iv": iv, "no_iv": no_iv or None, "direction": direction, "measure_match": measure_match, "probe": probe,
                 "max_tokens": max_tokens, "side_tokens": side_tokens, "name": name, "compact": True}
            return jobs.start("direction_key_causal", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_key_causal_verify(units: list[str] | str | None = None, result: str | None = None, contrast: str | None = None,
                                          iv: str | list[str] | None = None, no_iv: bool = False, pseudo_index: int = 0,
                                          match: str | None = None, judge: str | None = None, labels: str = "R,P,C",
                                          label: list[str] | str | None = None, max_tokens: int = 256,
                                          temperature: float = 0.0, seeds: list[int] | str | None = None,
                                          dont_touch: bool = True, name: str | None = None,
                                          combos: list[list[str]] | list[str] | None = None) -> dict:
        """keyed の効く場所の第 2 段: 選んだ候補を生成で確かめる(ジョブ、job_status で結果)。units は direction_key_causal の
        rows[].unit(例 "14:ffn:keyed:a-2@方向#鍵")を数個。result はその結果の名前(対照集合・iv・方向・鍵・測った正規表現の既定)。
        候補ごとに比較用の鍵 1 本(#鍵~pseudo_index)も足し、作成に使っていない奇数番目の質問(と dont_touch の奇数番目)に
        behave と同じ生成をさせる(返事の長さ max_tokens、温度と種も behave と同じ)。結果: base(iv だけ)と rows[] の a(書き込みたい場面)・
        dt(群ごと)の keyed・pseudo それぞれの n・behavior(振る舞いが出た数。match の正規表現か、judge の label)・hits(一致の総数)・
        changed(iv だけのときから返事が変わった数、greedy のとき)・loop・empty・at_limit。比較用の鍵でも同じくらい変わるなら、
        その変化は鍵が場面を分けたからではない。返事は path のファイルに。combos は複数の層の組を 1 つの候補として確かめる
        (組ごとにユニットのリストか "11:attn:keyed:a2,18:attn:keyed:a2" の文字列。強さはユニットの書き方どおり。比較用の鍵は、
        組の各ユニットを同じ番号の比較用の鍵に替えた組)。units と一緒に渡せば、1 か所の候補と組が結果の rows に並ぶ(組は combo: true)。"""
        try:
            p = {"units": units, "combos": combos, "result": result, "contrast": None if contrast is None else str(_contrast_path(session, contrast, None, None, None, None)),
                 "iv": iv, "no_iv": no_iv or None, "pseudo_index": pseudo_index or None, "match": match, "judge": judge,
                 "labels": labels, "label": label, "max_tokens": max_tokens, "temperature": temperature or None,
                 "seeds": seeds, "dont_touch": None if dont_touch else False, "name": name}
            return jobs.start("direction_key_causal_verify", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_key_from_behave(behave: str, recipe: str, direction: str | None = None, contrast: str | None = None,
                                        labels: str | list[str] | None = None, match: str | None = None,
                                        min_rate: float | None = None, positions: str | None = None, name: str | None = None,
                                        out: str | None = None, replies: bool = True, prompts: str | list | None = None,
                                        strict: bool = False) -> dict:
        """CHiPS のレシピを当てても消したい振る舞いが残った A 側の質問を集めて、keyed の対照集合の JSON を作る(モデルは動かさない)。
        behave は gne behave の結果の名前かパス(CHiPS のレシピ recipe を当てて A 側の質問に答えさせたもの)。direction は CHiPS の方向の名前、
        contrast はその対照集合(どちらか。B 側の質問・validate.match・A 側の質問を取る)。判定は labels(読み手のラベル、例 "P")か
        match(正規表現)。省くと behave の結果の match、無ければ validate.match。min_rate(既定 1.0 = 全部の種で出た質問だけ)。
        positions の既定は before:<validate.match>(大文字と小文字のある文字を含めば (?i) を付ける)。出力は <root>/keyed/<name>.json
        (out でパス): user_prompts = 抜き出した質問、replies = その behave の返事(replies=False なら入れない)、dont_touch の ordinary =
        B 側の質問(user_prompts と同じものは外す)、validate、keyed = {direction, alpha: -1, iv: recipe}。direction_key_pipeline の
        contrast にこの名前を渡せば鍵まで作れる。warnings: レシピ違い・A 側に無い質問(strict ならエラー)・少なすぎ・広すぎる正規表現
        (position_share.share が 0.25 以上 = 返事のトークンの大部分で一致の始まりになる)・一致の無い質問(no_match)・B 側の重なり。"""
        def fn():
            d = session.direction_key_from_behave(behave, recipe=recipe, direction=direction, contrast=contrast, labels=labels,
                                                  match=match, min_rate=min_rate, positions=positions, name=name, out=out,
                                                  replies=replies, prompts=prompts, strict=strict)
            for r in d.get("extracted") or []:                    # 返事の全文は JSON(replies)にある。ここでは頭だけ
                if len(r.get("reply") or "") > 300:
                    r["reply"] = r["reply"][:300] + "…"
            return d
        return await call(fn, gpu=False)

    @srv.tool()
    async def behave(prompts: str | list, recipes: list[str] | str | None = None, match: str | None = None,
                     judge: str | None = None, labels: str = "R,P,C", max_tokens: int = 400, thinking: bool = False,
                     temperature: float = 0.0, seeds: list[int] | str | None = None, name: str = "behave") -> dict:
        """振る舞いを数えるジョブを投げる(gne behave と同じ、job_status で結果)。レシピ recipes(名前かパス、複数なら重ねる)を当てて
        質問集 prompts(パス・configs/contrast/ の名前・質問のリスト。対照集合なら A 側の質問)に答えさせ、正規表現 match の一致と、
        読み手(介入なしの同じモデル)の判定 judge(文面。{q} に質問、{a} に返事が入り、返事の最後に出てきた labels の文字を判定にする)を
        数える。temperature > 0 なら seeds の種ごとに 1 回。返事は <root>/behave/<model_id>/<name>.json に残り(質問と当てたレシピの名前も)、
        結果は数だけ(runs[].counts)。CHiPS で残った場面を keyed で抑えるときは、この結果を direction_key_from_behave に渡す。"""
        try:
            p = {"prompts": prompts, "recipes": recipes, "match": match, "judge": judge, "labels": labels,
                 "max_tokens": max_tokens, "thinking": thinking or None, "temperature": temperature or None, "seeds": seeds,
                 "name": name}
            return jobs.start("behave", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def keys() -> dict:
        """保存した鍵の一覧(CHANGE-07、<root>/keys/<model_id>/): 名前・書き込み先・層・層ごとの kn・作成の条件(位置・subset・q_mode)。"""
        return await call(lambda: {"keys": session.keys()}, gpu=False)

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
        その層の状態の向きの cos)・contrib_norm を測り、中立どうしの差の p95 を基準値にして z で順位づける。数秒〜数十秒。"""
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
        """長時間の処理を投げる。kind: stats_resid / stats_act / lens_logit / lens_steer / sweep / rank / reverse / eval / direction / ablate / heads_contrast / heads_causal / heads_allocate / head_pipeline / heads_candidates / heads_refine / headproj_export / fieldtest / direction_key / direction_key_check / direction_key_experts / direction_key_pipeline / direction_key_verify / direction_key_causal / direction_key_causal_verify / behave。
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


def _recipe_stem(recipe: str) -> str:
    """書き出しのファイル名に使うレシピの名前(YAML のパスならファイル名の stem)。"""
    p = Path(str(recipe))
    return p.stem if p.suffix in (".yaml", ".yml") else str(recipe)


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
    from .yamlio import write_yaml
    nm = name or "contrast"
    if not re.fullmatch(r"[\w.\-]+", nm):
        raise GneError("BAD_ARGS", "name は英数字・_・-・. だけにしてください")
    p = session.store.root / "contrast" / f"{nm}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(p, {"name": nm, "system_a": system_a, "system_b": system_b, "user_prompts": list(user_prompts)})
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
