"""
GRPO (Group Relative Policy Optimization) training.

Trains a policy model using group-based advantage estimation:
1. Generate N completions per prompt
2. Score with reward model
3. Compute within-group advantages
4. Update policy with PPO loss + KL penalty
"""

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import (
    load_policy_model,
    load_policy_model_lora,
    load_tokenizer,
    RewardModel,
)
from data import PromptDataset
from utils import add_dataclass_args, build_config_from_args


def strip_think_tags(text: str) -> str:
    """Strip thinking tags for RM scoring (matches data.py behavior)."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


@dataclass
class GRPOConfig:
    """GRPO Training configuration."""

    # Models
    policy_model_name: str = "Qwen/Qwen3-0.6B"
    reward_model_path: str = ""  # Required: path to trained RM checkpoint
    max_length: int = 2048
    max_new_tokens: int = 256
    prompt_max_length: int = 512

    # Training
    epochs: int = 1
    batch_size: int = 4  # prompts per batch
    group_size: int = 4  # completions per prompt
    lr: float = 1e-6
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    # GRPO hyperparameters
    kl_coef: float = 0.05  # KL penalty coefficient
    cliprange: float = 0.2  # PPO clipping range
    normalize_advantages: bool = True  # Normalize within group

    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Evaluation & Checkpointing
    eval_steps: int = 50
    eval_num_batches: int = 16
    save_steps: int = 500
    output_dir: str = "./checkpoints/grpo"
    resume_from: str = ""

    # Logging
    log_dir: str = "./logs/grpo"
    run_name: str = ""

    # LoRA
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"


def compute_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute sum of log probabilities for completion tokens only.

    Args:
        model: Policy model
        input_ids: Full sequences (prompt + completion) [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        prompt_lengths: Length of each prompt [batch_size]

    Returns:
        Sum of log probs for completion tokens [batch_size]
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
    labels = input_ids[:, 1:]  # [batch, seq_len-1]

    # Compute per-token log probs
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Create mask for completion tokens only
    batch_size, seq_len = token_log_probs.shape
    positions = torch.arange(seq_len, device=token_log_probs.device).unsqueeze(0)
    # Mask: 1 for completion tokens, 0 for prompt tokens
    # prompt_lengths is the index of first completion token
    completion_mask = (positions >= (prompt_lengths.unsqueeze(1) - 1)).float()
    # Also mask padding
    completion_mask = completion_mask * attention_mask[:, 1:]

    # Sum log probs over completion tokens
    return (token_log_probs * completion_mask).sum(dim=-1)


def compute_group_advantages(
    rewards: torch.Tensor, group_size: int, normalize: bool = True
) -> torch.Tensor:
    """
    Compute within-group advantages.

    Args:
        rewards: Flat rewards [batch_size * group_size]
        group_size: Number of completions per prompt
        normalize: Whether to normalize by std

    Returns:
        Advantages [batch_size * group_size]
    """
    # Reshape to groups
    grouped = rewards.view(-1, group_size)

    # Subtract group mean
    advantages = grouped - grouped.mean(dim=1, keepdim=True)

    if normalize:
        std = grouped.std(dim=1, keepdim=True)
        advantages = advantages / (std + 1e-8)

    return advantages.view(-1)


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """
    Generate multiple completions per prompt.

    Returns:
        full_ids: [batch * group_size, seq_len] - prompt + completion
        full_mask: [batch * group_size, seq_len]
        prompt_lengths: [batch * group_size] - length of prompt for each
    """
    batch_size = prompt_ids.shape[0]
    device = prompt_ids.device

    # Repeat prompts for group generation
    expanded_ids = prompt_ids.repeat_interleave(group_size, dim=0)
    expanded_mask = prompt_mask.repeat_interleave(group_size, dim=0)

    # Store original prompt lengths (accounting for left padding)
    prompt_lengths = prompt_mask.sum(dim=1).repeat_interleave(group_size)

    # Generate
    outputs = model.generate(
        input_ids=expanded_ids,
        attention_mask=expanded_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Create attention mask for generated sequences
    full_mask = torch.ones_like(outputs)
    full_mask[outputs == tokenizer.pad_token_id] = 0

    return outputs, full_mask, prompt_lengths


def save_checkpoint(
    policy, tokenizer, optimizer, config, output_dir: str, step: int, epoch: int = 0
):
    """Save checkpoint with training state."""
    save_path = os.path.join(output_dir, f"step-{step}")
    print(f"Saving checkpoint to {save_path}")

    policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(save_path, "training_state.pt"),
    )

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)


@torch.no_grad()
def evaluate(
    policy,
    ref_policy,
    reward_model,
    tokenizer,
    dataset,
    config: GRPOConfig,
    num_batches: int,
):
    """Evaluate policy on held-out prompts."""
    policy.eval()
    device = next(policy.parameters()).device

    total_reward = 0.0
    total_kl = 0.0
    count = 0

    for _ in tqdm(range(num_batches), desc="Evaluating"):
        batch = dataset.sample_batch(config.batch_size)
        prompt_ids = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)

        # Generate single completion per prompt for eval
        full_ids, full_mask, prompt_lengths = generate_completions(
            policy,
            tokenizer,
            prompt_ids,
            prompt_mask,
            group_size=1,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )

        # Score with reward model (strip think tags to match RM training)
        completion_texts = tokenizer.batch_decode(full_ids, skip_special_tokens=False)
        stripped_texts = [strip_think_tags(t) for t in completion_texts]
        stripped_tokens = tokenizer(
            stripped_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        ).to(device)
        rewards = reward_model(
            input_ids=stripped_tokens["input_ids"],
            attention_mask=stripped_tokens["attention_mask"],
        )

        # Compute KL
        policy_logprobs = compute_logprobs(policy, full_ids, full_mask, prompt_lengths)
        ref_logprobs = compute_logprobs(ref_policy, full_ids, full_mask, prompt_lengths)
        kl = (policy_logprobs - ref_logprobs).mean()

        total_reward += rewards.mean().item()
        total_kl += kl.item()
        count += 1

    return total_reward / count, total_kl / count


def train(
    policy,
    ref_policy,
    reward_model,
    tokenizer,
    train_dataset,
    test_dataset,
    config: GRPOConfig,
):
    """Train policy with GRPO."""
    device = next(policy.parameters()).device

    # Setup run name and directories
    run_name = config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, run_name)
    log_dir = os.path.join(config.log_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
    writer = SummaryWriter(log_dir=log_dir)
    effective_batch_size = config.batch_size * config.group_size * config.grad_accum_steps

    # Resume from checkpoint
    global_step = 0
    start_epoch = 0
    if config.resume_from:
        state_path = os.path.join(config.resume_from, "training_state.pt")
        if os.path.exists(state_path):
            print(f"Resuming from checkpoint: {config.resume_from}")
            state = torch.load(state_path, weights_only=True)
            global_step = state["step"]
            start_epoch = state["epoch"]
            optimizer.load_state_dict(state["optimizer_state_dict"])
            print(f"  Resumed at step {global_step}, epoch {start_epoch}")
        else:
            print(f"Warning: No training_state.pt found, starting fresh")

    print(f"\nGRPO Training config:")
    print(f"  Run name: {run_name}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size} prompts")
    print(f"  Group size: {config.group_size} completions/prompt")
    print(f"  Gradient accumulation: {config.grad_accum_steps}")
    print(f"  Effective batch size: {effective_batch_size} completions")
    print(f"  Learning rate: {config.lr}")
    print(f"  KL coefficient: {config.kl_coef}")
    print(f"  Clip range: {config.cliprange}")
    print(f"  Output dir: {output_dir}")
    print(f"  Log dir: {log_dir}")
    print()
    print(f"To monitor training, run: tensorboard --logdir {log_dir}")
    print()

    try:
        _train_loop(
            policy,
            ref_policy,
            reward_model,
            tokenizer,
            optimizer,
            writer,
            train_dataset,
            test_dataset,
            config,
            output_dir,
            effective_batch_size,
            global_step,
            start_epoch,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        save_checkpoint(policy, tokenizer, optimizer, config, output_dir, global_step, start_epoch)
        print("Checkpoint saved. You can resume with --resume_from flag.")
    except Exception as e:
        print(f"\n\nTraining crashed: {e}")
        save_checkpoint(policy, tokenizer, optimizer, config, output_dir, global_step, start_epoch)
        print("Emergency checkpoint saved. You can resume with --resume_from flag.")
        raise
    finally:
        writer.close()


def _train_loop(
    policy,
    ref_policy,
    reward_model,
    tokenizer,
    optimizer,
    writer,
    train_dataset,
    test_dataset,
    config,
    output_dir,
    effective_batch_size,
    global_step,
    start_epoch,
):
    """Inner training loop."""
    device = next(policy.parameters()).device

    # Estimate steps per epoch
    dataset_size = len(train_dataset)
    steps_per_epoch = dataset_size // (config.batch_size * config.grad_accum_steps)

    for epoch in range(start_epoch, config.epochs):
        print(f"=== Epoch {epoch + 1}/{config.epochs} ===")

        policy.train()
        optimizer.zero_grad()
        step_start_time = time.time()

        accum_loss = 0.0
        accum_policy_loss = 0.0
        accum_kl = 0.0
        accum_reward = 0.0

        pbar = tqdm(range(steps_per_epoch * config.grad_accum_steps), desc="Training")
        for step in pbar:
            # Sample prompts
            batch = train_dataset.sample_batch(config.batch_size)
            prompt_ids = batch["input_ids"].to(device)
            prompt_mask = batch["attention_mask"].to(device)

            # Generate completions
            with torch.no_grad():
                full_ids, full_mask, prompt_lengths = generate_completions(
                    policy,
                    tokenizer,
                    prompt_ids,
                    prompt_mask,
                    group_size=config.group_size,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                )

                # Score with reward model (strip think tags to match RM training)
                completion_texts = tokenizer.batch_decode(full_ids, skip_special_tokens=False)
                stripped_texts = [strip_think_tags(t) for t in completion_texts]
                stripped_tokens = tokenizer(
                    stripped_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                ).to(device)
                rewards = reward_model(
                    input_ids=stripped_tokens["input_ids"],
                    attention_mask=stripped_tokens["attention_mask"],
                )

                # Compute advantages
                advantages = compute_group_advantages(
                    rewards, config.group_size, normalize=config.normalize_advantages
                )

                # Get reference log probs
                ref_logprobs = compute_logprobs(
                    ref_policy, full_ids, full_mask, prompt_lengths
                )

                # Get old log probs (for PPO ratio)
                old_logprobs = compute_logprobs(
                    policy, full_ids, full_mask, prompt_lengths
                )

            # Forward pass with gradients
            policy_logprobs = compute_logprobs(
                policy, full_ids, full_mask, prompt_lengths
            )

            # PPO loss
            ratio = (policy_logprobs - old_logprobs).exp()
            clipped_ratio = ratio.clamp(1 - config.cliprange, 1 + config.cliprange)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # KL penalty
            kl_div = (ref_logprobs - policy_logprobs).mean()

            # Total loss
            loss = policy_loss + config.kl_coef * kl_div

            # Backward
            (loss / config.grad_accum_steps).backward()

            # Accumulate for logging
            accum_loss += loss.item()
            accum_policy_loss += policy_loss.item()
            accum_kl += kl_div.item()
            accum_reward += rewards.mean().item()

            # Optimizer step
            if (step + 1) % config.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    config.max_grad_norm if config.max_grad_norm > 0 else float("inf"),
                ).item()

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                examples_seen = global_step * effective_batch_size

                step_time = time.time() - step_start_time
                step_start_time = time.time()

                # Average over accumulation steps
                avg_loss = accum_loss / config.grad_accum_steps
                avg_policy_loss = accum_policy_loss / config.grad_accum_steps
                avg_kl = accum_kl / config.grad_accum_steps
                avg_reward = accum_reward / config.grad_accum_steps

                # Log metrics
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/policy_loss", avg_policy_loss, global_step)
                writer.add_scalar("train/kl_div", avg_kl, global_step)
                writer.add_scalar("train/reward_mean", avg_reward, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/step_time", step_time, global_step)

                writer.add_scalar("train/loss_by_examples", avg_loss, examples_seen)
                writer.add_scalar("train/reward_by_examples", avg_reward, examples_seen)

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "reward": f"{avg_reward:.4f}",
                    "kl": f"{avg_kl:.4f}",
                })

                # Reset accumulators
                accum_loss = 0.0
                accum_policy_loss = 0.0
                accum_kl = 0.0
                accum_reward = 0.0

                # Evaluate
                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    eval_reward, eval_kl = evaluate(
                        policy,
                        ref_policy,
                        reward_model,
                        tokenizer,
                        test_dataset,
                        config,
                        config.eval_num_batches,
                    )
                    print(f"\n[Step {global_step}] Eval - Reward: {eval_reward:.4f}, KL: {eval_kl:.4f}")
                    writer.add_scalar("eval/reward", eval_reward, global_step)
                    writer.add_scalar("eval/kl_div", eval_kl, global_step)
                    writer.add_scalar("eval/reward_by_examples", eval_reward, examples_seen)
                    policy.train()
                    step_start_time = time.time()

                # Save checkpoint
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    save_checkpoint(
                        policy, tokenizer, optimizer, config, output_dir, global_step, epoch
                    )

        # End of epoch evaluation
        eval_reward, eval_kl = evaluate(
            policy,
            ref_policy,
            reward_model,
            tokenizer,
            test_dataset,
            config,
            config.eval_num_batches,
        )
        print(f"Epoch {epoch + 1} - Eval Reward: {eval_reward:.4f}, Eval KL: {eval_kl:.4f}")
        examples_seen = global_step * effective_batch_size
        writer.add_scalar("eval/reward", eval_reward, global_step)
        writer.add_scalar("eval/kl_div", eval_kl, global_step)
        writer.add_scalar("eval/reward_by_examples", eval_reward, examples_seen)

    # Save final model
    save_checkpoint(policy, tokenizer, optimizer, config, output_dir, global_step, config.epochs)
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train policy with GRPO (Group Relative Policy Optimization)",
    )
    reverse_renames = add_dataclass_args(
        parser,
        GRPOConfig,
        renames={
            "policy_model_name": "model",
            "grad_accum_steps": "grad_accum",
            "use_lora": "lora",
        },
    )
    args = parser.parse_args()
    config = build_config_from_args(GRPOConfig, args, reverse_renames)

    # Validate required args
    if not config.reward_model_path:
        parser.error("--reward_model_path is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(config.policy_model_name)

    # Load policy model
    policy_path = config.resume_from if config.resume_from else config.policy_model_name
    print(f"Loading policy model from {policy_path}...")
    if config.use_lora:
        policy = load_policy_model_lora(
            policy_path,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules.split(","),
            device_map="auto",
        )
        policy.print_trainable_parameters()
    else:
        policy = load_policy_model(policy_path, device_map="auto")
        print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    if config.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Load reference policy (frozen copy)
    print(f"Loading reference policy from {config.policy_model_name}...")
    ref_policy = load_policy_model(config.policy_model_name, device_map="auto")
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False
    print("Reference policy frozen")

    # Load reward model
    print(f"Loading reward model from {config.reward_model_path}...")
    reward_model = RewardModel.from_pretrained(
        config.reward_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False
    print("Reward model frozen")

    # Load datasets
    print("Loading datasets...")
    train_dataset = PromptDataset(
        tokenizer, split="train", max_length=config.prompt_max_length
    )
    test_dataset = PromptDataset(
        tokenizer, split="test", max_length=config.prompt_max_length
    )

    train(
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
    )


if __name__ == "__main__":
    main()
