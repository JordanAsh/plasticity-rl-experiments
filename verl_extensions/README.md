# verl extensions

Custom reward functions and data preprocessing added on top of verl 0.4.1.

## Reward functions

Drop these into `verl/utils/reward_score/` and register in `__init__.py`:

```python
elif data_source == "countdown":
    from . import countdown
    res = countdown.compute_score(solution_str, ground_truth)
elif data_source == "kk_logic":
    from . import kk
    res = kk.compute_score(solution_str, ground_truth)
```

- **`countdown.py`**: ported from [TinyZero](https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py).
  Scores `0.0` (no answer) / `0.1` (format-only) / `1.0` (correct equation).
- **`kk.py`**: ported from [Logic-RL](https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py).
  Logic-RL faithful scoring: format ±1 + answer (±2 / -1.5). Total in [-3, +3].

## Data preprocessing

See `prep_countdown.py` and `prep_kk.py` in the repo root.
