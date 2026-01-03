"""
Reward Model training with Bradley-Terry pairwise loss.

Loss = -log(sigmoid(r_chosen - r_rejected))
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import load_reward_model, load_reward_model_lora, load_tokenizer
from data import HHRLHFDataset
from utils import add_dataclass_args, build_config_from_args, print_config


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    max_length: int = 2048

    # Training
    epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-5
    grad_accum_steps: int = 4
    max_grad_norm: float = 10.0  # Gradient clipping (0 to disable)
    gradient_checkpointing: bool = True  # Trade compute for memory

    # Evaluation
    eval_steps: int = 50  # 0 to disable step-based eval
    eval_batch_size: int = 4
    eval_num_batches: int = 32

    # Checkpointing
    save_steps: int = 500  # 0 to disable step-based saving (~5 saves per epoch)
    output_dir: str = "./checkpoints/rm"
    resume_from: str = ""  # Path to checkpoint to resume from

    # Logging
    log_dir: str = "./logs/rm"
    run_name: str = ""  # Run name (default: datetime)

    # LoRA (ignored if use_lora=False)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"  # comma-separated


def bradley_terry_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry pairwise ranking loss.

    Args:
        chosen_rewards: Scalar rewards for chosen responses (batch_size,)
        rejected_rewards: Scalar rewards for rejected responses (batch_size,)

    Returns:
        Scalar loss value
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def compute_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """Compute accuracy: how often chosen reward > rejected reward."""
    return (chosen_rewards > rejected_rewards).float().mean().item()


@torch.no_grad()
def evaluate(model, dataset, device, batch_size: int, num_batches: int):
    """Evaluate by sampling with replacement."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for _ in tqdm(range(num_batches), desc="Evaluating"):
        batch = dataset.sample_batch(batch_size)

        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)

        loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
        acc = compute_accuracy(chosen_rewards, rejected_rewards)

        total_loss += loss.item()
        total_acc += acc

    return total_loss / num_batches, total_acc / num_batches


def save_checkpoint(model, tokenizer, optimizer, config, output_dir: str, step: int, epoch: int = 0):
    """Save model checkpoint with training state and config."""
    save_path = os.path.join(output_dir, f"step-{step}")
    print(f"Saving checkpoint to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # Save training state for resuming
    torch.save({
        "step": step,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(save_path, "training_state.pt"))
    # Save config for reproducibility
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)


def train(
    model,
    tokenizer,
    train_loader: DataLoader,
    test_dataset,
    config: TrainConfig,
):
    """Train the reward model."""
    device = next(model.parameters()).device

    # Setup run name and directories
    run_name = config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, run_name)
    log_dir = os.path.join(config.log_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    writer = SummaryWriter(log_dir=log_dir)
    effective_batch_size = config.batch_size * config.grad_accum_steps

    # Resume from checkpoint if specified
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
            print(f"Warning: No training_state.pt found in {config.resume_from}, starting fresh")

    print_config(config, "Training config")
    print(f"\n  Effective batch size: {config.batch_size * config.grad_accum_steps}")
    print(f"  Output dir: {output_dir}")
    print(f"  Log dir: {log_dir}")
    print(f"\nTo monitor training, run: tensorboard --logdir {log_dir}")
    print()

    try:
        _train_loop(
            model, tokenizer, optimizer, writer, train_loader, test_dataset,
            config, output_dir, effective_batch_size, global_step, start_epoch
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        save_checkpoint(model, tokenizer, optimizer, config, output_dir, global_step, start_epoch)
        print("Checkpoint saved. You can resume with --resume_from flag.")
    except Exception as e:
        print(f"\n\nTraining crashed: {e}")
        save_checkpoint(model, tokenizer, optimizer, config, output_dir, global_step, start_epoch)
        print("Emergency checkpoint saved. You can resume with --resume_from flag.")
        raise
    finally:
        writer.close()


def _train_loop(
    model, tokenizer, optimizer, writer, train_loader, test_dataset,
    config, output_dir, effective_batch_size, global_step, start_epoch
):
    """Inner training loop (separated for crash handling)."""
    device = next(model.parameters()).device

    for epoch in range(start_epoch, config.epochs):
        print(f"=== Epoch {epoch + 1}/{config.epochs} ===")

        model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        optimizer.zero_grad()
        step_start_time = time.time()

        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)

            loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
            loss_value = loss.item()  # Per-example mean loss (before scaling)
            (loss / config.grad_accum_steps).backward()

            if (step + 1) % config.grad_accum_steps == 0:
                # Compute grad norm (before clipping)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm if config.max_grad_norm > 0 else float("inf")
                ).item()
                grad_clipped = 1.0 if (config.max_grad_norm > 0 and grad_norm > config.max_grad_norm) else 0.0

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                examples_seen = global_step * effective_batch_size

                # Compute step time
                step_time = time.time() - step_start_time
                step_start_time = time.time()

                # Log training metrics (loss_value is per-example mean from last micro-batch)
                step_acc = compute_accuracy(chosen_rewards, rejected_rewards)
                current_lr = optimizer.param_groups[0]["lr"]

                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/accuracy", step_acc, global_step)
                writer.add_scalar("train/loss_by_examples", loss_value, examples_seen)
                writer.add_scalar("train/accuracy_by_examples", step_acc, examples_seen)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/grad_clipped", grad_clipped, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/step_time", step_time, global_step)

                # Evaluate every N steps
                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    test_loss, test_acc = evaluate(
                        model, test_dataset, device,
                        config.eval_batch_size, config.eval_num_batches
                    )
                    print(f"\n[Step {global_step}] Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2%}")
                    writer.add_scalar("eval/loss", test_loss, global_step)
                    writer.add_scalar("eval/accuracy", test_acc, global_step)
                    writer.add_scalar("eval/loss_by_examples", test_loss, examples_seen)
                    writer.add_scalar("eval/accuracy_by_examples", test_acc, examples_seen)
                    model.train()
                    step_start_time = time.time()  # Reset after eval

                # Save every N steps
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    save_checkpoint(model, tokenizer, optimizer, config, output_dir, global_step, epoch)

            with torch.no_grad():
                acc = compute_accuracy(chosen_rewards, rejected_rewards)

            total_loss += loss_value
            total_acc += acc
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss_value:.4f}", "acc": f"{acc:.2%}"})

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2%}")

        # End of epoch evaluation
        test_loss, test_acc = evaluate(
            model, test_dataset, device,
            config.eval_batch_size, config.eval_num_batches
        )
        print(f"Epoch {epoch + 1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")
        examples_seen = global_step * effective_batch_size
        writer.add_scalar("eval/loss", test_loss, global_step)
        writer.add_scalar("eval/accuracy", test_acc, global_step)
        writer.add_scalar("eval/loss_by_examples", test_loss, examples_seen)
        writer.add_scalar("eval/accuracy_by_examples", test_acc, examples_seen)

    # Save final model
    save_checkpoint(model, tokenizer, optimizer, config, output_dir, global_step, config.epochs)
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train Reward Model with Bradley-Terry pairwise loss",
    )
    reverse_renames = add_dataclass_args(parser, TrainConfig, renames={
        "model_name": "model",
        "grad_accum_steps": "grad_accum",
        "use_lora": "lora",
    })
    args = parser.parse_args()

    config = build_config_from_args(TrainConfig, args, reverse_renames)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load from checkpoint if resuming, otherwise from base model
    model_path = config.resume_from if config.resume_from else config.model_name

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)

    print(f"Loading reward model from {model_path}...")
    if config.use_lora:
        model = load_reward_model_lora(
            model_path,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules.split(","),
            device_map="auto",
        )
        model.print_trainable_parameters()
    else:
        model = load_reward_model(model_path, device_map="auto")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    print("Loading datasets...")
    train_dataset = HHRLHFDataset(tokenizer, split="train", max_length=config.max_length)
    test_dataset = HHRLHFDataset(tokenizer, split="test", max_length=config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0,
    )

    train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_dataset=test_dataset,
        config=config,
    )


if __name__ == "__main__":
    main()
