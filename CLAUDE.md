# CLAUDE.md

## Project Description

A self-educational project for learning to train Large Language Models (LLMs) using Reward Models (RM). This covers reinforcement learning from human feedback (RLHF) and related techniques.

## Development Instructions

- **After every edit**: Check if changes to `CLAUDE.md` and `README.md` are needed
- `CLAUDE.md` serves as context for Claude Code sessions - update with key decisions, architecture choices, and important implementation details
- `README.md` is user-facing documentation - update when features, setup steps, or usage instructions change

## Project Structure

```
models.py         # Model definitions (RewardModel, PolicyModel)
chat.py           # Gradio chatbot UI for testing models
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
- Full parameter tuning (not LoRA/QLoRA)
- bf16 precision
