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
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from models import (
    load_policy_model,
    load_policy_model_lora,
    load_tokenizer,
    RewardModel,
)
from data import PromptDataset
from utils import add_dataclass_args, build_config_from_args, print_config


_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Strip thinking tags for RM scoring (matches data.py behavior)."""
    return _THINK_TAG_RE.sub("", text)


def init_rollouts_db(db_path: str) -> None:
    """Initialize SQLite database for rollout logging."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rollouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step INTEGER,
            epoch INTEGER,
            prompt_index INTEGER,
            rollout_index INTEGER,
            prompt TEXT,
            completion TEXT,
            reward REAL,
            advantage REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_step ON rollouts(step)")
    # Eval rollouts table (no rollout_index or advantage - single completion per prompt)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_rollouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step INTEGER,
            epoch INTEGER,
            prompt_index INTEGER,
            prompt TEXT,
            completion TEXT,
            reward REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_step ON eval_rollouts(step)")
    conn.commit()
    conn.close()


class RolloutWriter:
    """Manages persistent SQLite connection for async rollout writes."""

    _local = threading.local()

    @classmethod
    def _get_connection(cls, db_path: str) -> sqlite3.Connection:
        """Get or create a thread-local connection."""
        if not hasattr(cls._local, "conn") or cls._local.db_path != db_path:
            if hasattr(cls._local, "conn"):
                cls._local.conn.close()
            cls._local.conn = sqlite3.connect(db_path)
            cls._local.db_path = db_path
        return cls._local.conn

    @classmethod
    def save_rollouts_batch(cls, db_path: str, rows: list) -> None:
        """Save a batch of rollouts to the database (called async from thread pool)."""
        conn = cls._get_connection(db_path)
        conn.executemany(
            """INSERT INTO rollouts
               (step, epoch, prompt_index, rollout_index, prompt, completion, reward, advantage)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()

    @classmethod
    def save_eval_rollouts_batch(cls, db_path: str, rows: list) -> None:
        """Save eval rollouts to database (called async from thread pool)."""
        conn = cls._get_connection(db_path)
        conn.executemany(
            """INSERT INTO eval_rollouts
               (step, epoch, prompt_index, prompt, completion, reward)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()


@dataclass
class GRPOConfig:
    """GRPO Training configuration."""

    # Models
    policy_model_name: str = "Qwen/Qwen3-0.6B"
    reward_model_path: str = ""  # Required: path to trained RM checkpoint
    rm_quant_type: str = field(default="", metadata={"help": "RM quantization: '' (none), 'int8', or 'int4'"})
    ref_quant_type: str = field(default="", metadata={"help": "Ref policy quantization: '' (none), 'int8', or 'int4'"})

    # Sequence lengths
    prompt_max_length: int = field(default=1536, metadata={"help": "Max prompt tokens (longer prompts skipped)"})
    max_new_tokens: int = field(default=2048, metadata={"help": "Max tokens to generate per completion"})
    max_length: int = field(default=2048, metadata={"help": "Truncation limit when tokenizing for RM scoring"})

    # Training
    epochs: int = 1
    batch_size: int = 4  # prompts per batch
    group_size: int = 8  # completions per prompt
    lr: float = 1e-6
    grad_accum_steps: int = 2
    max_grad_norm: float = 10.0
    gradient_checkpointing: bool = True

    # GRPO hyperparameters
    kl_coef: float = 0.01  # KL penalty coefficient
    cliprange: float = 0.2  # PPO clipping range
    normalize_advantages: bool = True  # Normalize within group
    n_minibatches: int = field(default=1, metadata={"help": "Gradient steps per batch (reusing old_logprobs)"})
    eos_penalty: float = field(default=1.0, metadata={"help": "Reward penalty for completions not ending with EOS"})

    # Generation
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True

    # Evaluation & Checkpointing
    eval_steps: int = field(default=15, metadata={"help": "Evaluate every N steps (-1 to disable)"})
    eval_batch_size: int = field(default=0, metadata={"help": "Eval batch size (0 = batch_size * group_size)"})
    eval_num_prompts: int = 64
    save_steps: int = 15
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

    # Rollout logging
    save_rollouts: bool = True
    rollouts_dir: str = "./rollouts"

    # Debug
    verbose: bool = False  # Print detailed progress logs


def get_vram_usage() -> str:
    """Get current VRAM usage as string (allocated/reserved)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"[VRAM: {alloc:.1f}/{reserved:.1f}GB]"
    return ""


def vlog(verbose: bool, step: int, msg: str, accum_step: int = None, accum_total: int = None):
    """Print timestamped message if verbose mode enabled."""
    if verbose:
        timestamp = datetime.now().strftime("%H:%M:%S")
        accum_str = f" [{accum_step}/{accum_total}]" if accum_step is not None else ""
        vram = get_vram_usage()
        print(f"[{timestamp}] [Step {step}]{accum_str} {msg} {vram}")


def compute_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log probabilities for completion tokens.

    Args:
        model: Policy model
        input_ids: Full sequences (prompt + completion) [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        prompt_lengths: Padded prompt length (index of first completion token) [batch_size]

    Returns:
        token_log_probs: Per-token log probs [batch_size, seq_len-1]
        completion_mask: Mask for completion tokens [batch_size, seq_len-1]
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

    return token_log_probs, completion_mask


def compute_kl_divergence(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence using GRPO estimator (always >= 0, lower variance).

    Args:
        policy_logprobs: Policy log probs [batch_size, seq_len]
        ref_logprobs: Reference policy log probs [batch_size, seq_len]
        completion_mask: Mask for completion tokens [batch_size, seq_len]

    Returns:
        Scalar KL divergence (masked mean over completion tokens)
    """
    log_ratio = ref_logprobs - policy_logprobs  # log(π_ref/π_policy)
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1
    return (per_token_kl * completion_mask).sum() / completion_mask.sum()


def compute_rewards(
    reward_model,
    full_ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
    config,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[int], list[bool]]:
    """
    Compute rewards with EOS penalty.

    Args:
        reward_model: Reward model
        full_ids: Full sequences (prompt + completion) [batch_size, seq_len]
        prompt_len: Padded prompt length (same for all sequences in batch)
        tokenizer: Tokenizer
        config: Training config (for max_length and eos_penalty)
        device: Device

    Returns:
        rewards: Reward tensor [batch_size]
        completion_texts: Decoded completion-only texts (no prompt, no padding)
        completion_lengths: List of completion lengths
        proper_endings: List of bools indicating if completion ended with EOS
    """
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Process each sequence: extract non-pad tokens, compute stats
    full_texts = []
    completion_texts = []
    completion_lengths = []
    proper_endings = []

    for i in range(full_ids.shape[0]):
        seq = full_ids[i]

        # Find non-pad tokens (handles both left and right padding)
        non_pad_mask = seq != pad_token_id
        non_pad_tokens = seq[non_pad_mask]

        # Decode full transcript (prompt + completion, no padding)
        full_text = tokenizer.decode(non_pad_tokens, skip_special_tokens=False)
        full_texts.append(full_text)

        # Completion tokens start after the padded prompt
        comp_tokens = seq[prompt_len:]
        non_pad_comp = (comp_tokens != pad_token_id).sum().item()
        completion_lengths.append(non_pad_comp)

        # Check if last non-pad token is EOS
        if non_pad_comp > 0:
            last_token = comp_tokens[non_pad_comp - 1].item()
            proper_endings.append(last_token == eos_token_id)
            # Decode only the non-pad completion tokens
            comp_text = tokenizer.decode(comp_tokens[:non_pad_comp], skip_special_tokens=False)
        else:
            proper_endings.append(False)
            comp_text = ""
        completion_texts.append(comp_text)

    # Score with reward model (strip think tags to match RM training)
    stripped_texts = [strip_think_tags(t) for t in full_texts]
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

    # Apply penalty for non-proper endings
    if config.eos_penalty > 0:
        penalties = torch.tensor(
            [0.0 if ended else -config.eos_penalty for ended in proper_endings],
            device=rewards.device
        )
        rewards = rewards + penalties

    return rewards, completion_texts, completion_lengths, proper_endings


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
        prompt_lengths: [batch * group_size] - padded prompt length (index of first completion token)
    """
    batch_size = prompt_ids.shape[0]
    device = prompt_ids.device

    # Repeat prompts for group generation
    expanded_ids = prompt_ids.repeat_interleave(group_size, dim=0)
    expanded_mask = prompt_mask.repeat_interleave(group_size, dim=0)

    # Padded prompt length = index of first completion token (same for all in batch)
    padded_prompt_len = prompt_ids.shape[1]
    prompt_lengths = torch.full(
        (batch_size * group_size,), padded_prompt_len, device=device, dtype=torch.long
    )

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

    # Save training config (use different name to not overwrite model's config.json)
    with open(os.path.join(save_path, "train_config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)


@torch.inference_mode()
def evaluate(
    policy,
    ref_policy,
    reward_model,
    tokenizer,
    dataset,
    config: GRPOConfig,
    num_prompts: int,
    global_step: int,
    epoch: int,
    rollout_executor=None,
    rollout_db_path: str = None,
):
    """Evaluate policy on held-out prompts."""
    policy.eval()
    device = next(policy.parameters()).device

    total_reward = 0.0
    total_kl = 0.0
    count = 0
    prompt_index = 0

    num_batches = (num_prompts + config.eval_batch_size - 1) // config.eval_batch_size
    for _ in tqdm(range(num_batches), desc="Evaluating"):
        batch = dataset.sample_batch(config.eval_batch_size)
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

        # Score with reward model (includes EOS penalty)
        rewards, completion_texts, _, _ = compute_rewards(
            reward_model, full_ids, prompt_lengths[0].item(), tokenizer, config, device
        )

        # Compute KL
        policy_logprobs, completion_mask = compute_logprobs(policy, full_ids, full_mask, prompt_lengths)
        ref_logprobs, _ = compute_logprobs(ref_policy, full_ids, full_mask, prompt_lengths)
        kl = compute_kl_divergence(policy_logprobs, ref_logprobs, completion_mask)

        total_reward += rewards.mean().item()
        total_kl += kl.item()
        count += 1

        # Save eval rollouts asynchronously
        if rollout_executor is not None:
            # Decode prompts without left-padding
            prompt_texts = []
            for j in range(prompt_ids.shape[0]):
                non_pad = prompt_ids[j][prompt_ids[j] != tokenizer.pad_token_id]
                prompt_texts.append(tokenizer.decode(non_pad, skip_special_tokens=False))
            rewards_list = rewards.tolist()
            rows = []
            for i, (prompt_text, comp_text, reward) in enumerate(
                zip(prompt_texts, completion_texts, rewards_list)
            ):
                rows.append((
                    global_step,
                    epoch,
                    prompt_index + i,
                    prompt_text,
                    comp_text,
                    reward,
                ))
            rollout_executor.submit(RolloutWriter.save_eval_rollouts_batch, rollout_db_path, rows)
            prompt_index += len(prompt_texts)

        # Free GPU memory between batches
        torch.cuda.empty_cache()

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

    # Initialize rollout logging
    rollout_executor = None
    rollout_db_path = None
    if config.save_rollouts:
        rollout_db_path = os.path.join(config.rollouts_dir, run_name, "rollouts.db")
        init_rollouts_db(rollout_db_path)
        rollout_executor = ThreadPoolExecutor(max_workers=1)
        print(f"Rollout logging enabled: {rollout_db_path}")

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

    print_config(config, "GRPO Training config")
    print(f"\n  Effective batch size: {effective_batch_size} completions")
    print(f"  Output dir: {output_dir}")
    print(f"  Log dir: {log_dir}")
    # Use forward slashes for display (works on all platforms)
    print(f"\nTo monitor training, run: tensorboard --logdir {log_dir.replace(os.sep, '/')}")
    if config.save_rollouts:
        print(f"To view rollouts, run: python view_rollouts.py {rollout_db_path.replace(os.sep, '/')}")
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
            rollout_executor,
            rollout_db_path,
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
        if rollout_executor:
            rollout_executor.shutdown(wait=True)


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
    rollout_executor,
    rollout_db_path,
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
        accum_completion_lengths = []
        accum_proper_endings = []

        # Storage for batch data during gradient accumulation
        stored_batches = []

        # Track prompt offset across accumulation steps for rollout logging
        accum_prompt_offset = 0

        pbar = tqdm(range(steps_per_epoch * config.grad_accum_steps), desc="Training", disable=config.verbose)
        for step in pbar:
            # Sample prompts
            batch = train_dataset.sample_batch(config.batch_size)
            prompt_ids = batch["input_ids"].to(device)
            prompt_mask = batch["attention_mask"].to(device)

            # Generate completions (eval mode to enable KV cache with gradient checkpointing)
            accum_step = step % config.grad_accum_steps + 1
            policy.eval()
            with torch.inference_mode():
                vlog(config.verbose, global_step, "Generating completions...", accum_step, config.grad_accum_steps)
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

                # Score with reward model (includes EOS penalty)
                vlog(config.verbose, global_step, "Scoring with reward model...", accum_step, config.grad_accum_steps)
                rewards, completion_texts, completion_lengths, proper_endings = compute_rewards(
                    reward_model, full_ids, prompt_lengths[0].item(), tokenizer, config, device
                )

                # Accumulate completion stats for logging
                accum_completion_lengths.extend(completion_lengths)
                accum_proper_endings.extend(proper_endings)

                # Compute advantages
                advantages = compute_group_advantages(
                    rewards, config.group_size, normalize=config.normalize_advantages
                )

                # Get reference log probs (per-token)
                vlog(config.verbose, global_step, "Computing ref policy logprobs...", accum_step, config.grad_accum_steps)
                ref_logprobs, completion_mask = compute_logprobs(
                    ref_policy, full_ids, full_mask, prompt_lengths
                )

                # Get old log probs (per-token, for PPO ratio)
                vlog(config.verbose, global_step, "Computing old policy logprobs...", accum_step, config.grad_accum_steps)
                old_logprobs, _ = compute_logprobs(
                    policy, full_ids, full_mask, prompt_lengths
                )

            # Accumulate reward for logging
            accum_reward += rewards.mean().item()

            # Save rollouts asynchronously (once per batch)
            if rollout_executor is not None:
                # Decode prompts without left-padding
                prompt_texts = []
                for j in range(prompt_ids.shape[0]):
                    non_pad = prompt_ids[j][prompt_ids[j] != tokenizer.pad_token_id]
                    prompt_texts.append(tokenizer.decode(non_pad, skip_special_tokens=False))
                rows = []
                rewards_list = rewards.tolist()
                advantages_list = advantages.tolist()
                for i, (comp_text, reward, advantage) in enumerate(
                    zip(completion_texts, rewards_list, advantages_list)
                ):
                    batch_prompt_idx = i // config.group_size
                    rollout_idx = i % config.group_size
                    rows.append((
                        global_step,
                        epoch,
                        accum_prompt_offset + batch_prompt_idx,
                        rollout_idx,
                        prompt_texts[batch_prompt_idx],
                        comp_text,
                        reward,
                        advantage,
                    ))
                rollout_executor.submit(RolloutWriter.save_rollouts_batch, rollout_db_path, rows)
                accum_prompt_offset += config.batch_size

            # Store batch data on CPU for later gradient computation
            stored_batches.append({
                "full_ids": full_ids.cpu(),
                "full_mask": full_mask.cpu(),
                "prompt_lengths": prompt_lengths.cpu(),
                "old_logprobs": old_logprobs.cpu(),
                "ref_logprobs": ref_logprobs.cpu(),
                "completion_mask": completion_mask.cpu(),
                "advantages": advantages.cpu(),
            })
            # Free GPU cache to prevent VRAM accumulation during grad accum
            torch.cuda.empty_cache()

            # Back to train mode for gradient computation
            policy.train()

            # When accumulation complete, run n_minibatches updates over all stored data
            if (step + 1) % config.grad_accum_steps == 0:
                for inner_step in range(config.n_minibatches):
                    vlog(config.verbose, global_step, f"Minibatch {inner_step + 1}/{config.n_minibatches}: forward & backward...")
                    # Accumulate gradients over all stored batches
                    for batch_data in stored_batches:
                        # Move batch data to GPU
                        b_full_ids = batch_data["full_ids"].to(device)
                        b_full_mask = batch_data["full_mask"].to(device)
                        b_prompt_lengths = batch_data["prompt_lengths"].to(device)
                        b_old_logprobs = batch_data["old_logprobs"].to(device)
                        b_ref_logprobs = batch_data["ref_logprobs"].to(device)
                        b_completion_mask = batch_data["completion_mask"].to(device)
                        b_advantages = batch_data["advantages"].to(device)

                        # Forward pass with gradients (per-token log probs)
                        policy_logprobs, _ = compute_logprobs(
                            policy, b_full_ids, b_full_mask, b_prompt_lengths
                        )

                        # Per-token PPO ratio (stays close to 1.0)
                        ratio = (policy_logprobs - b_old_logprobs).exp()
                        clipped_ratio = ratio.clamp(1 - config.cliprange, 1 + config.cliprange)

                        # Per-token PPO loss, broadcast advantages to token level
                        # b_advantages: [batch], need to expand to [batch, seq_len]
                        adv_expanded = b_advantages.unsqueeze(1)  # [batch, 1]
                        per_token_loss = -torch.min(ratio * adv_expanded, clipped_ratio * adv_expanded)
                        # Masked average over completion tokens
                        policy_loss = (per_token_loss * b_completion_mask).sum() / b_completion_mask.sum()

                        # Per-token KL penalty (GRPO estimator: always ≥ 0, lower variance)
                        kl_div = compute_kl_divergence(policy_logprobs, b_ref_logprobs, b_completion_mask)

                        # Total loss (divided by grad_accum_steps for proper averaging)
                        loss = policy_loss + config.kl_coef * kl_div
                        (loss / config.grad_accum_steps).backward()

                        # Log metrics from last inner step, last batch
                        if inner_step == config.n_minibatches - 1:
                            accum_loss += loss.item()
                            accum_policy_loss += policy_loss.item()
                            accum_kl += kl_div.item()

                    vlog(config.verbose, global_step, f"Minibatch {inner_step + 1}/{config.n_minibatches}: optimizer step...")
                    # Optimizer step after processing all batches
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        config.max_grad_norm if config.max_grad_norm > 0 else float("inf"),
                    ).item()
                    optimizer.step()
                    optimizer.zero_grad()

                # Clear stored batches and reset prompt offset for next step
                stored_batches = []
                accum_prompt_offset = 0

                # Update step counter and log
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

                # Log completion stats
                if accum_completion_lengths:
                    avg_comp_len = sum(accum_completion_lengths) / len(accum_completion_lengths)
                    max_comp_len = max(accum_completion_lengths)
                    proper_end_rate = sum(accum_proper_endings) / len(accum_proper_endings)
                    writer.add_scalar("train/completion_length_avg", avg_comp_len, global_step)
                    writer.add_scalar("train/completion_length_max", max_comp_len, global_step)
                    writer.add_scalar("train/proper_ending_rate", proper_end_rate, global_step)

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
                accum_completion_lengths = []
                accum_proper_endings = []

                # Evaluate
                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    eval_reward, eval_kl = evaluate(
                        policy,
                        ref_policy,
                        reward_model,
                        tokenizer,
                        test_dataset,
                        config,
                        config.eval_num_prompts,
                        global_step,
                        epoch,
                        rollout_executor,
                        rollout_db_path,
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
        if config.eval_steps >= 0:
            eval_reward, eval_kl = evaluate(
                policy,
                ref_policy,
                reward_model,
                tokenizer,
                test_dataset,
                config,
                config.eval_num_prompts,
                global_step,
                epoch,
                rollout_executor,
                rollout_db_path,
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

    # Resolve dynamic defaults
    if config.eval_batch_size == 0:
        config.eval_batch_size = config.batch_size * config.group_size

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
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")

    # Load reference policy (optionally quantized, frozen copy)
    if config.ref_quant_type:
        print(f"Loading reference policy from {config.policy_model_name} ({config.ref_quant_type})...")
        if config.ref_quant_type == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif config.ref_quant_type == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            raise ValueError(f"Unknown ref_quant_type: {config.ref_quant_type}. Use '', 'int8', or 'int4'.")
        ref_policy = AutoModelForCausalLM.from_pretrained(
            config.policy_model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        print(f"Loading reference policy from {config.policy_model_name}...")
        ref_policy = load_policy_model(config.policy_model_name, device_map="auto")
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False
    print("Reference policy frozen")

    # Load reward model (optionally quantized)
    if config.rm_quant_type:
        print(f"Loading reward model from {config.reward_model_path} ({config.rm_quant_type})...")
        if config.rm_quant_type == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif config.rm_quant_type == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            raise ValueError(f"Unknown rm_quant_type: {config.rm_quant_type}. Use '', 'int8', or 'int4'.")
        reward_model = RewardModel.from_pretrained(
            config.reward_model_path,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
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

    # Load datasets (mixed train/test for GRPO to reduce RM overfitting)
    print("Loading datasets...")
    train_dataset = PromptDataset(
        tokenizer,
        train_samples=1500,
        test_samples=8500,
        max_length=config.prompt_max_length,
    )
    # Use same dataset for eval (we're using all test data in training anyway)
    test_dataset = train_dataset

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
