# contract/

The shared contract for the soccer-coach pipeline. Put this folder at `soccer-coach/contract/`.

```
CONTRACT.md          paste into every parallel chat
schemas/             JSON Schema (draft 2020-12) for each pipeline file
examples/            consistent example files for every stage (synthetic two-shot video)
tools/
  generate_synthetic.py   regenerates examples/ (edit rep params to make new test cases)
  validate.py             schema + cross-file checks; exit code 1 on any failure
  preview.py              draws skeleton + ball at each rep's key frames
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python tools/validate.py examples/          # should print OK for every file
python tools/preview.py examples/pose.example.json examples/ball.example.json examples/reps.example.json preview.png
```

## Making more test cases

Edit `reps_cfg` in `tools/generate_synthetic.py` (plant gap, lean, hip position, ankle angle, ball speed,
which foot) and rerun it. Each change produces a new video's worth of data with known correct answers,
which is useful for unit tests in C, D and E before real footage exists.

## Limits of the synthetic data

It is a 2D side-on stick figure, not real motion capture. It is good for testing formats, rep detection,
foot detection and metric formulas. It is not good for tuning thresholds or judging accuracy; use your
real labeled footage for that.
