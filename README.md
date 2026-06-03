# Plasticity Loss in RL Post-Training

Experiments studying plasticity loss when fine-tuning a base model with RL (GRPO) and
then doing SFT on its positive generations. Compares RL vs SFT across four tasks
(GSM8K, MATH, Countdown, Knights & Knaves) and two model sizes (Qwen2.5-1.5B, 3B).

## Setup

- **Hardware**: 4× A100 80GB
- **Models**: Qwen/Qwen2.5-1.5B (base), Qwen/Qwen2.5-3B (base, KK only)
- **Data**: GSM8K + MATH + Countdown + Knights & Knaves (verl parquet format under `~/data/`)
- **RL framework**: [verl](https://github.com/volcengine/verl) v0.4.1
- **Conda env**: `plasticity-rl` (Python 3.12, torch 2.6.0+cu124, vllm 0.8.5, flash-attn 2.8.4)

### Environment

```bash
conda create -n plasticity-rl python=3.12 -y
conda activate plasticity-rl
# Install verl 0.4.1 in editable mode from the verl repo
# Then install its deps (torch 2.6, vllm 0.8.5, flash-attn 2.8.4, ray, etc.)
# Note: VLLM_USE_V1=0 is required for current driver compatibility
```

### Data preprocessing

verl's standard preprocessing scripts produce the parquet files used here. The prompts
contain a single user message; Qwen2.5's chat template injects the default
`You are a helpful assistant.` system prompt automatically.

## Repository layout

```
.
├── run_sft.py                   # DDP SFT script (1 epoch over RL positives)
├── eval_model.py                # Greedy vLLM eval on GSM8K + MATH
├── eval_pass_at_k.py            # pass@k vLLM eval (samples n, scores k)
├── scripts/                     # Pipeline runners (training + eval)
└── results/
    ├── greedy/                  # summary.json per model
    └── pass16/                  # summary.json per model
```

## Pipeline

### 1. RL training (GRPO)

```bash
bash scripts/run_grpo_qwen2.5_1.5b.sh
# Or run 3 seed replicates sequentially:
bash scripts/run_replicates.sh
```

Default config: GSM8K + MATH training data, 435 steps total, batch 512,
mini-batch 128, rollout n=5, LR 1e-6, KL coef 0.001. Saves checkpoints at
steps 150/300/435 and dumps generation logs each step.

### 2. SFT on RL positives

Trains the base model on positive (correct) generations harvested from the RL run.
One epoch, configurable LR / schedule / batch / order.

```bash
torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_<date> \
    --output_dir sft_outputs/seed42_shuffled_lr5e5 \
    --lr 5e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03
```

Key flags:
- `--ordered`: train in the order generations were produced (curriculum from RL trajectory). Default is shuffled.
- `--effective_batch_size`: total batch across all GPUs; `grad_accum_steps` is computed automatically from `world_size × batch_size`.
- `--schedule {cosine,linear,constant}`: LR schedule.
- `--warmup_ratio`: fraction of total steps for warmup.

Pre-built scripts for the LR sweeps:
- `scripts/run_sft_5e5.sh` — lr=5e-5 cosine, both ordered + shuffled
- `scripts/run_sft_1e4.sh` — lr=1e-4 cosine
- `scripts/run_sft_const.sh` — lr=5e-5 constant, no warmup

**Data templating is aligned with vLLM eval.** Prompts are reconstructed via
`tokenizer.apply_chat_template`, then concatenated with response tokens and
`<|im_end|>`. Prompt tokens are masked with `-100` so loss is only on the response.

### 3. Merge RL FSDP shards (for evaluation)

verl saves model weights as DTensor-sharded FSDP per-rank `.pt` files. To convert
to HuggingFace format for vLLM:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ckpt_base = "checkpoints/.../actor"
shards = [torch.load(f"{ckpt_base}/model_world_size_4_rank_{r}.pt", map_location="cpu",
                     weights_only=False) for r in range(4)]
merged = {}
for key in shards[0]:
    shard_dim = shards[0][key].placements[0].dim
    locals_ = [shards[r][key]._local_tensor for r in range(4)]
    merged[key] = torch.cat(locals_, dim=shard_dim)

config = AutoConfig.from_pretrained(f"{ckpt_base}/huggingface")
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
model.load_state_dict(merged)
model.save_pretrained("checkpoints/rl_merged")
AutoTokenizer.from_pretrained(f"{ckpt_base}/huggingface").save_pretrained("checkpoints/rl_merged")
```

### 4. Evaluation

Two eval modes, both using vLLM with TP=4 and the same chat template / reward
functions as verl (so there's no distribution shift from training to eval).

**Greedy** (temperature=0, single sample per prompt):
```bash
python eval_model.py \
    --model_path sft_outputs/seed42_shuffled_lr5e5 \
    --output_dir results/greedy/sft_seed42_shuffled_lr5e5 \
    --tensor_parallel_size 4
```

**pass@k** (n samples per prompt with temperature > 0, unbiased estimator):
```bash
python eval_pass_at_k.py \
    --model_path sft_outputs/seed42_shuffled_lr5e5 \
    --output_dir results/pass16/sft_seed42_shuffled_lr5e5 \
    --k 16 --n 16 --temperature 0.8 --top_p 0.95 \
    --tensor_parallel_size 4
```

Scoring uses verl's reward functions:
- GSM8K: `verl.utils.reward_score.gsm8k.compute_score` (strict `#### N` extraction)
- MATH: `verl.utils.reward_score.math.compute_score` (last `\boxed{...}` + equivalence)

## Results

All from seed 42 RL runs. SFT uses positive generations (rule-verified correct
responses) from the corresponding RL run, one epoch.

### GSM8K + MATH (Qwen2.5-1.5B)

| Model                       | GSM8K greedy | GSM8K pass@16 | MATH greedy | MATH pass@16 |
|-----------------------------|--------------|---------------|-------------|--------------|
| **RL (GRPO, step 435)**     | **80.1%**    | 92.7%         | **57.0%**   | 78.3%        |
| SFT ordered  lr=1e-5 cosine | 72.0%        | 93.1%         | 56.6%       | 78.9%        |
| SFT shuffled lr=1e-5 cosine | 57.5%        | 94.8%         | 53.9%       | **79.8%**    |
| SFT ordered  lr=5e-5 cosine | 78.4%        | 93.9%         | 56.6%       | 79.0%        |
| SFT shuffled lr=5e-5 cosine | 75.7%        | 94.7%         | 56.6%       | 79.5%        |
| SFT ordered  lr=1e-4 cosine | 75.8%        | 93.0%         | 51.5%       | 77.4%        |
| SFT shuffled lr=1e-4 cosine | 78.3%        | 94.5%         | 51.0%       | 77.0%        |
| SFT ordered  lr=5e-5 const  | 77.6%        | 95.0%         | 55.0%       | (partial)    |
| SFT shuffled lr=5e-5 const  | 77.4%        | **95.1%**     | 55.0%       | 78.3%        |

### Countdown (Qwen2.5-1.5B)

| Model                       | Greedy       | pass@16      |
|-----------------------------|--------------|--------------|
| RL (GRPO, step 1280)        | 41.7%        | 42.0%        |
| SFT shuffled lr=1e-5 cosine | **44.1%**    | **46.5%**    |
| SFT ordered  lr=1e-5 cosine | 43.0%        | 44.3%        |

### Knights & Knaves (Qwen2.5-1.5B)

| Model                       | Greedy       | pass@16      |
|-----------------------------|--------------|--------------|
| RL (GRPO, 20 epochs)        | 25.0%        | 26.6%        |
| SFT shuffled lr=1e-5 cosine | 25.9%        | **30.0%**    |
| SFT ordered  lr=1e-5 cosine | **26.1%**    | 29.3%        |

### Knights & Knaves (Qwen2.5-3B)

| Model                       | Greedy       | pass@16      |
|-----------------------------|--------------|--------------|
| **RL (GRPO, 10 epochs)**    | **37.1%**    | 48.7%        |
| SFT shuffled lr=1e-5 cosine | 33.6%        | **57.3%**    |
| SFT ordered  lr=1e-5 cosine | 35.6%        | 52.1%        |

## Training hyperparameters

All RL runs use **GRPO** (`algorithm.adv_estimator=grpo`, `kl_loss_type=low_var_kl`,
`kl_loss_coef=0.001`), no critic. Greedy/pass@16 evals always use vLLM TP=4 with
`stop_token_ids=[<|im_end|>]` to match the chat template's stop signal.

### RL training

| Setting | GSM8K+MATH | Countdown | KK (1.5B) | KK (3B) |
|---|---|---|---|---|
| Model | Qwen2.5-1.5B | Qwen2.5-1.5B | Qwen2.5-1.5B | Qwen2.5-3B |
| Train batch size | 512 | 256 | 256 | 256 |
| PPO mini-batch | 128 | 64 | 64 | 64 |
| Rollouts per prompt (n) | 5 | 8 | 16 | 16 |
| Actor LR | 1e-6 | 1e-6 | 3e-7 | 3e-7 |
| Max prompt len | 1024 | 256 | 400 | 400 |
| Max response len | 2048 | 512 | 2048 | 2048 |
| GPU mem util (rollout) | 0.6 | 0.85 | 0.85 | 0.7 |
| Epochs | 15 | 1 | 20 | 10 |
| Total optimizer steps | 435 | 1280 | 380 | 190 |
| Wall clock | ~9h | ~6.5h | ~8.7h | ~10.7h |
| Reward function | gsm8k + math (binary 0/1) | countdown (0 / 0.1 / 1) | kk (Logic-RL: ±1 format ±2 answer) | kk (same) |

### SFT training (on RL positives)

All SFT runs are **1 epoch**, DDP on 4 GPUs with effective batch size 128
(per-GPU batch=4, grad accum auto-derived), weight decay 0.01, max seq length
1024-3072 depending on task.

| Setting | GSM8K+MATH (sweep) | Countdown | KK (1.5B) | KK (3B) |
|---|---|---|---|---|
| Base model | Qwen2.5-1.5B | Qwen2.5-1.5B | Qwen2.5-1.5B | Qwen2.5-3B |
| Positive filter | `score > 0` | `score > 0.5` (correct only) | `score > 2.5` (correct + format) | same |
| # positives | 841,512 | 961,135 | 284,483 | 176,989 |
| Learning rate | {1e-5, 5e-5, 1e-4} | 1e-5 | 1e-5 | 1e-5 |
| LR schedule | {cosine, constant} | cosine | cosine | cosine |
| Warmup ratio | 0.03 (0 for constant) | 0.03 | 0.03 | 0.03 |
| Max seq length | 3072 | 1024 | 2048 | 2048 |
| Data order | {shuffled, ordered} | {shuffled, ordered} | {shuffled, ordered} | {shuffled, ordered} |

"Ordered" = train on positives in the step order they were generated (curriculum
from RL trajectory). "Shuffled" = uniform random.

### Evaluation

- **Greedy**: `temperature=0, top_p=1.0`, single sample per prompt.
- **pass@16**: `temperature=0.8, top_p=0.95, n=16, k=16`, Codex-style unbiased
  estimator (Chen et al. 2021).
- Both use `stop_token_ids=[<|im_end|>]` and rule-based scoring via the
  corresponding `verl.utils.reward_score.<task>.compute_score` function.

### Reproducing the table

Each `summary.json` in `results/{greedy,pass16}/<model>/` contains the accuracy
and total problem count.

```bash
for f in results/greedy/*/summary.json; do
    echo "--- $(dirname $f | xargs basename) ---"
    cat "$f"
done
```

## Notes

- All SFT runs use seed=42 RL generation logs.
- Generation logs (~1-2GB per RL run) and model checkpoints (~3-6GB per SFT
  model, ~36GB per FSDP RL checkpoint) are not tracked in git. Re-run the pipeline
  to regenerate them.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` is required to avoid CUDA fork errors
  when vLLM is launched from a process that already imported torch.
- The Countdown and KK eval results were computed before the `stop_token_ids`
  fix was applied. Countdown scoring is robust to runaway generations (it takes
  the last `<answer>` tag); KK was re-run with the fix.
