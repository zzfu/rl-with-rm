# CLAUDE.md

## Project Description

A self-educational project for learning to train Large Language Models (LLMs) using Reward Models (RM). This covers reinforcement learning from human feedback (RLHF) and related techniques.

## Development Instructions

- **Run Python scripts with venv**: `.venv/Scripts/python.exe script.py` (Windows)
- **After every edit**: Check if changes to `CLAUDE.md` and `README.md` are needed
- `CLAUDE.md` serves as context for Claude Code sessions - update with key decisions, architecture choices, and important implementation details
- `README.md` is user-facing documentation - update when features, setup steps, or usage instructions change

## Project Structure

```
models.py         # Model definitions (RewardModel, PolicyModel, LoRA loading)
data.py           # HH-RLHF dataset loader (preference pairs + prompts)
train_rm.py       # Reward model training script (full-param and LoRA)
train_grpo.py     # GRPO policy training script
quantize_rm.py    # RM quantization (INT8/INT4) and evaluation script
quantize_policy.py # Policy quantization with KL divergence evaluation
chat.py           # Gradio chatbot UI for testing models
utils.py          # CLI utilities (auto argparse from dataclass)
manage_runs.py    # Interactive tool to manage RM+GRPO runs and snapshots
requirements.txt  # Python dependencies
```

## Key Decisions

### Base Model
- **Qwen3-0.6B**: Small enough for full-parameter tuning (~10-12GB VRAM)
- Architecture: 28 layers, 16 Q-heads, 8 KV-heads (GQA), 32K context

### Model Architecture
- **RewardModel**: `Qwen3Model` (no LM head) + `nn.Linear(hidden_size, 1)` value head
  - Outputs scalar reward from last token's hidden state
  - Value head initialized with small random weights (std=0.02)
- **PolicyModel**: Standard `Qwen3ForCausalLM` for RL training

### Training Approach
- Full parameter tuning (default) or LoRA with `--lora` flag
- bf16 precision
- **RM Loss**: Bradley-Terry pairwise ranking: `-log(sigmoid(r_chosen - r_rejected))`
- `TrainConfig` dataclass holds all hyperparameters (single source of truth for defaults)
- CLI auto-generated from dataclass via `utils.add_dataclass_args()`
- `--config train_config.json` loads saved training config; CLI args override file values
- `--resume_from` auto-loads train_config.json from checkpoint (no need for separate `--config`)
- Test evaluation samples with replacement (fast, configurable num_batches)
- Checkpoints save optimizer state + train_config.json for resuming; auto-save on crash/interrupt

### LoRA Training
- Uses peft library with `task_type="SEQ_CLS"` (RM) or `task_type="CAUSAL_LM"` (policy)
- Default targets: `q_proj,k_proj,v_proj,o_proj` (attention projections)
- Scaling formula: `W' = W + (α/r) × BA` where α=lora_alpha, r=lora_r
- Checkpoints save only adapter weights (~few MB vs full model)

### GRPO (Group Relative Policy Optimization)
- Trains policy using trained reward model
- **Group sampling**: N completions per prompt, advantages computed within-group
- **Loss**: PPO with clipped objective + KL penalty from frozen reference policy
- **Key hyperparams**: `group_size=8`, `kl_coef=0.05`, `cliprange=0.2`
- Three models in memory: policy (trainable), ref_policy (frozen), reward_model (frozen)
- `PromptDataset` extracts prompts from HH-RLHF with left-padding for generation
  - Uses mixed train/test splits (default: 1.5k train + 8.5k test = 10k) to reduce RM overfitting
- **n_minibatches**: Multiple gradient steps per batch for PPO-style sample efficiency
  - Batch data stored on CPU during grad accumulation, enabling both features together
  - `old_logprobs` computed once, reused across minibatch updates (ratio diverges from 1)
- **Rollout logging**: `--save_rollouts` saves all generations to SQLite for inspection
  - Training rollouts: step, epoch, prompt_index, rollout_index, prompt, completion, reward, advantage
  - Eval rollouts: separate `eval_rollouts` table (no advantage column)
  - Async writes via ThreadPoolExecutor (non-blocking)
- **eval_steps=-1** disables evaluation entirely (useful for quick testing)

### Quantization
- Uses bitsandbytes for INT8/INT4 quantization
- INT4 uses NF4 (Normal Float 4) with double quantization for best quality
- `quantize_rm.py` compares original vs quantized RM on test set (accuracy)
- `quantize_policy.py` compares original vs quantized policy (KL divergence on generated tokens)
- Outputs saved to `checkpoint/quantized_{int8|int4}/` subfolder (or `quantized_models/` for base models)

### Dataset
- **Anthropic/hh-rlhf**: Human preference data (chosen/rejected pairs)
- Train: 161k, Test: 8.5k examples
- Token stats: Mean ~219, P99 ~843, Max ~1780 (at 2048: ~0% truncated)
- Runtime filtering: examples exceeding max_length are skipped during sampling (fast init)
- Thinking tags (`<think>...</think>`) stripped from formatted output for RM training
