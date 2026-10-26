"""Phase 3 の単体テスト(モデル不要): rank の指標(語の数え方・Poisson z・mass・null_pct・補助指標)と選択の混在指定。"""

import math

import numpy as np
import pytest

from gne.rank import distinct4, rank_records, rollout_words
from gne.selection import parse_neurons

WORDS = [None, "alien", "planet", "the", "space", "cat", "dog", None, "evil", "good"]   # token id → 語


def rec(kind, neuron, prompt, dose, ids, *, rel_push=None, finish="length", text=""):
    return {"kind": kind, "neuron": neuron, "layer": 22, "prompt": prompt, "dose": dose,
            "rel_push": abs(dose) if rel_push is None else rel_push, "token_ids": ids, "finish": finish,
            "logprob": [-1.0] * len(ids), "text": text, "batch": [prompt]}


def test_rollout_words_tokens_and_regex():
    r = rec("neuron", 1, "who", 0.8, [1, 2, 3, 7, 9], finish="eog", text="Alien planets, 42 cats and évil")
    assert rollout_words(r, "tokens", WORDS) == {"alien", "planet", "the"}       # EOG(最後のトークン)は除く
    assert rollout_words(r, "regex", None) == {"alien", "planets", "cats", "and", "évil"}


def test_distinct4():
    assert distinct4([1, 2, 3]) == 1.0
    assert distinct4([1, 2, 3, 4, 5]) == 1.0
    assert distinct4([1, 1, 1, 1, 1, 1]) == 1 / 3


def _records():
    rs = [rec("baseline", None, "who", 0.0, [3, 5]), rec("baseline", None, "orange", 0.0, [3])]
    # ニューロン 7: 2 つのプロンプトで alien / planet(本物のテーマ)
    for p in ("who", "orange"):
        rs.append(rec("neuron", 7, p, 0.8, [1, 2, 3]))
        rs.append(rec("neuron", 7, p, 0.5, [1, 4]))
    rs.append(rec("neuron", 7, "free", 3.0, [8, 8, 8, 8, 8, 8]))                  # rel_push 3.0 > 1.6 は数えない
    # ニューロン 9: cat が 1 回だけ(count < 2 は z を付けない)
    rs.append(rec("neuron", 9, "who", 0.8, [5, 6]))
    rs.append(rec("neuron", 9, "orange", 0.8, [6]))
    # 対照 0..2
    for j in range(3):
        rs.append(rec("random", j, "who", 0.8, [9, 9] if j == 0 else [4]))
        rs.append(rec("random", j, "orange", 0.8, [9] if j == 0 else [2]))
    return rs


def test_rank_records_scores():
    units = rank_records(_records(), token_words=WORDS)
    by = {(u["kind"], u["neuron"]): u for u in units}
    n7 = by[("neuron", 7)]
    words = {w: (c, z) for w, c, z in n7["words"]}
    # alien は n7 の 4 本に出る。全 5 ユニットの平均 4/5
    assert words["alien"][0] == 4 and math.isclose(words["alien"][1], (4 - 0.8) / math.sqrt(0.8 + 0.5), rel_tol=1e-3)
    assert "the" not in words and "evil" not in words                       # ベースラインの語・max_rel 超えは除く
    assert set(n7["per_prompt"]["alien"]) == {"who", "orange"}
    assert units[0]["kind"] == "neuron" and units[0]["neuron"] == 7 and units[0]["rank"] == 1
    assert n7["null_pct"] == 1.0
    n9 = by[("neuron", 9)]
    assert n9["mass"] == 0.0 or all(c >= 2 for _, c, _ in n9["words"])     # count ≥ 2 のみ
    assert by[("random", 0)]["null_pct"] == 1.0 and by[("random", 1)]["null_pct"] == 0.0   # 対照は自分を除いて比べる
    d = n7["dose"]
    assert set(d) == {"+0.5", "+0.8", "+3"} and d["+3"]["distinct4"] == pytest.approx(1 / 3)
    assert d["+0.8"]["capped"] == 1.0 and d["+0.8"]["rel_push"] == pytest.approx(0.8)


def test_rank_records_max_rel_changes_counts():
    units = rank_records(_records(), token_words=WORDS, max_rel=5.0)
    n7 = next(u for u in units if u["kind"] == "neuron" and u["neuron"] == 7)
    assert "evil" not in {w for w, _, _ in n7["words"]}                     # free の 1 本では count 1(< 2)
    assert any(w == "alien" for w, _, _ in n7["words"])


def test_rank_records_empty_and_no_controls():
    assert rank_records([rec("baseline", None, "who", 0.0, [3])], token_words=WORDS) == []
    units = rank_records([rec("neuron", 1, "who", 0.8, [1]), rec("neuron", 1, "o", 0.8, [1])], token_words=WORDS)
    assert units[0]["null_pct"] is None and units[0]["mass"] >= 0


def test_parse_neurons_mixed():
    a = parse_neurons("random:5,2742", 9216, seed=0)
    assert 2742 in a and a.size in (5, 6) and np.all(np.diff(a) > 0)
    assert np.array_equal(np.setdiff1d(a, [2742]), np.setdiff1d(parse_neurons("random:5", 9216, seed=0), [2742]))
