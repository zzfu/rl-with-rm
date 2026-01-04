# rl-with-rm

A self-educational project for learning to train LLMs with Reward Models (RM) and RLHF.

## Setup

```bash
# Create virtual environment
uv venv

# Activate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -r requirements.txt
```

## Project Structure

```
models.py         # RewardModel and PolicyModel definitions
data.py           # HH-RLHF dataset loader (preference pairs + prompts)
train_rm.py       # Reward model training script
train_grpo.py     # GRPO policy training script
quantize_rm.py    # RM quantization (INT8/INT4) and evaluation
quantize_policy.py # Policy quantization with KL divergence evaluation
chat.py           # Gradio chatbot for testing models
manage_runs.py    # Interactive tool to manage RM+GRPO runs and snapshots
utils.py          # CLI utilities
```

## Training the Reward Model

Uses Bradley-Terry pairwise loss on the [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset.

```bash
python train_rm.py
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Qwen/Qwen3-0.6B | Base model |
| `--epochs` | 1 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-5 | Learning rate |
| `--max_length` | 2048 | Max sequence length |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--max_grad_norm` | 10.0 | Gradient clipping (0 to disable) |
| `--gradient_checkpointing` | True | Trade compute for memory |
| `--eval_steps` | 50 | Evaluate every N steps |
| `--eval_num_batches` | 32 | Batches per evaluation |
| `--save_steps` | 500 | Save checkpoint every N steps |
| `--output_dir` | ./checkpoints/rm | Output directory |
| `--resume_from` | | Path to checkpoint to resume from |
| `--log_dir` | ./logs/rm | TensorBoard log directory |
| `--run_name` | (datetime) | Run name for organizing outputs |

### LoRA Training

For parameter-efficient fine-tuning with much lower VRAM usage:

```bash
python train_rm.py --lora
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora` | False | Enable LoRA training |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha (scaling) |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--lora_target_modules` | q_proj,k_proj,v_proj,o_proj | Target modules (comma-separated) |

LoRA checkpoints only save adapter weights (~few MB vs full model).

### Resuming Training

Training automatically saves checkpoints on crash or interrupt (Ctrl+C). To resume:

```bash
python train_rm.py --resume_from ./checkpoints/rm/20240115_143022/step-1000
```

Config is auto-loaded from the checkpoint. To extend training with new hyperparameters:

```bash
# Resume and train for one more epoch with different batch size
python train_rm.py --resume_from ./checkpoints/rm/run_name/step-1000 --epochs 2 --batch_size 16
```

### Loading Config from File

Load a saved `train_config.json` as base config, with CLI args overriding specific values:

```bash
# Load from saved config (useful when not resuming)
python train_rm.py --config ./checkpoints/rm/run_name/step-500/train_config.json

# Load and override specific values
python train_rm.py --config ./checkpoints/rm/step-500/train_config.json --lr 1e-6 --epochs 2
```

### TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir ./logs/rm
```

Metrics logged:
- `train/loss`, `train/accuracy` - training metrics by step
- `train/grad_norm`, `train/lr`, `train/step_time` - optimization metrics
- `eval/loss`, `eval/accuracy` - evaluation metrics
- `*_by_examples` variants - same metrics with examples seen as x-axis

## GRPO Policy Training

Group Relative Policy Optimization - trains a policy model using a trained reward model.

```bash
python train_grpo.py --reward_model_path ./checkpoints/rm/run_name/step-500
```

### How GRPO Works

1. Generate N completions per prompt (group)
2. Score each completion with the reward model
3. Compute within-group advantages (relative ranking)
4. Update policy with PPO loss + KL penalty (prevents drift from reference)

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Qwen/Qwen3-0.6B | Policy base model |
| `--reward_model_path` | (required) | Path to trained RM checkpoint |
| `--rm_quant_type` | (none) | RM quantization: `int8` or `int4` |
| `--ref_quant_type` | (none) | Ref policy quantization: `int8` or `int4` |
| `--epochs` | 1 | Number of epochs |
| `--batch_size` | 2 | Prompts per batch |
| `--group_size` | 8 | Completions per prompt |
| `--lr` | 1e-6 | Learning rate |
| `--grad_accum_steps` | 4 | Gradient accumulation steps |
| `--max_grad_norm` | 10.0 | Gradient clipping (0 to disable) |
| `--n_minibatches` | 1 | Gradient steps per batch (PPO-style reuse) |
| `--kl_coef` | 0.01 | KL penalty coefficient |
| `--cliprange` | 0.2 | PPO clipping range |
| `--eos_penalty` | 1.0 | Reward penalty for incomplete responses |
| `--max_new_tokens` | 512 | Max tokens to generate |
| `--temperature` | 1.0 | Sampling temperature |
| `--eval_steps` | 50 | Evaluate every N steps (-1 to disable) |
| `--eval_num_prompts` | 128 | Prompts per evaluation |
| `--output_dir` | ./checkpoints/grpo | Output directory |
| `--log_dir` | ./logs/grpo | TensorBoard log directory |
| `--save_rollouts` | True | Save all generations to SQLite |

### LoRA Training

```bash
python train_grpo.py --reward_model_path ./checkpoints/rm/step-500 --lora
```

Same LoRA arguments as reward model training (`--lora_r`, `--lora_alpha`, etc.).

### TensorBoard

```bash
tensorboard --logdir ./logs/grpo
```

Metrics logged:
- `train/loss`, `train/policy_loss` - training losses
- `train/kl_div` - KL divergence from reference policy
- `train/reward_mean` - average reward from RM
- `train/completion_length_avg`, `train/completion_length_max` - token counts
- `train/proper_ending_rate` - fraction of completions ending with EOS
- `eval/reward`, `eval/kl_div` - evaluation metrics

## Quantization

Quantize a trained RM to reduce memory usage and compare accuracy impact:

```bash
# INT8 quantization (default)
python quantize_rm.py --checkpoint ./checkpoints/rm/run_name/step-1000

# INT4 quantization (more compression, potentially more accuracy loss)
python quantize_rm.py --checkpoint ./checkpoints/rm/run_name/step-1000 --quant_type int4
```

The script:
1. Loads and evaluates the original bf16 model
2. Loads and evaluates the quantized model
3. Saves results to `checkpoint/quantized_{int8|int4}/evaluation_results.json`
4. Attempts to save the quantized model (works for INT4, may fail for INT8)

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to trained RM checkpoint |
| `--quant_type` | int8 | Quantization type: `int8` or `int4` |
| `--eval_batch_size` | 4 | Evaluation batch size |
| `--eval_num_batches` | 500 | Number of batches (~2000 examples) |
| `--max_length` | 2048 | Max sequence length |
| `--skip_save` | False | Skip saving quantized model |

## Policy Quantization

Quantize a policy model and evaluate KL divergence vs original:

```bash
# From a GRPO checkpoint
python quantize_policy.py --model ./checkpoints/grpo/run_name/step-1000

# From a base model (e.g., to test before training)
python quantize_policy.py --model Qwen/Qwen3-0.6B --quant_type int4
```

The script:
1. Generates completions with the original bf16 model
2. Computes log probabilities from both models on generated tokens
3. Reports KL divergence: `KL(original || quantized)`

For checkpoints, outputs go to `checkpoint/quantized_{int8|int4}/`. For base models, outputs go to `./quantized_models/`.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Model name or checkpoint path |
| `--quant_type` | int8 | Quantization type: `int8` or `int4` |
| `--num_prompts` | 100 | Number of prompts to evaluate |
| `--max_new_tokens` | 512 | Max tokens to generate per prompt |
| `--temperature` | 1.0 | Sampling temperature |
| `--prompt_max_length` | 1536 | Max prompt length |
| `--skip_save` | False | Skip saving quantized model |

## Chat Interface

Test models interactively:

```bash
python chat.py --autoload  # Load default model on startup
python chat.py --model ./checkpoints/rm/step-1000  # Load checkpoint
```

## Base Model

- **Qwen3-0.6B**: 0.6B parameters, ~10-12GB VRAM for full-param tuning
- bf16 precision
