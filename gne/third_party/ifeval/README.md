# IFEval checkers

`instructions.py`, `instructions_registry.py` and `instructions_util.py` are the official checkers of
[IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval), copied from
google-research commit `e6890f85757dd84e27ca6df2dd30651dafad28e0`, so that scores stay comparable with other reports. The
code is under the Apache License 2.0 (`LICENSE`), copyright The Google Research Authors. No package of it is published.

Changes, each noted at the top of the file:

- imports are relative;
- `instructions.py` uses the standard `logging` instead of `absl.logging`;
- `instructions_util.py` loads the sentence tokenizer with `PunktTokenizer("english")` (the `punkt_tab` data), because
  nltk 3.9 and later no longer load the pickled `punkt` data.

`gne fieldtest --bench ifeval` uses them. They need the `bench` extra (`langdetect`, `nltk`, `immutabledict`).
