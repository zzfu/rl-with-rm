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
data.py           # HH-RLHF dataset loader
train_rm.py       # Reward model training script
chat.py           # Gradio chatbot for testing models
manage_runs.py    # Interactive tool to list/delete training runs
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

## Chat Interface

Test models interactively:

```bash
python chat.py --autoload  # Load default model on startup
python chat.py --model ./checkpoints/rm/step-1000  # Load checkpoint
```

## Base Model

- **Qwen3-0.6B**: 0.6B parameters, ~10-12GB VRAM for full-param tuning
- bf16 precision
