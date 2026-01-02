"""
Reward Model training with Bradley-Terry pairwise loss.

Loss = -log(sigmoid(r_chosen - r_rejected))
"""

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import load_reward_model, load_tokenizer
from data import HHRLHFDataset


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    max_length: int = 2048

    # Training
    epochs: int = 1
    batch_size: int = 4
    lr: float = 1e-5
    grad_accum_steps: int = 4

    # Evaluation
    eval_steps: int = 500  # 0 to disable step-based eval
    eval_batch_size: int = 4
    eval_num_batches: int = 100

    # Checkpointing
    save_steps: int = 2000  # 0 to disable step-based saving (~5 saves per epoch)
    output_dir: str = "./checkpoints/rm"

    # Logging
    log_dir: str = "./logs/rm"


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


def save_checkpoint(model, tokenizer, output_dir: str, step: int):
    """Save model checkpoint."""
    save_path = os.path.join(output_dir, f"step-{step}")
    print(f"Saving checkpoint to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def train(
    model,
    tokenizer,
    train_loader: DataLoader,
    test_dataset,
    config: TrainConfig,
):
    """Train the reward model."""
    device = next(model.parameters()).device
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    writer = SummaryWriter(log_dir=config.log_dir)
    effective_batch_size = config.batch_size * config.grad_accum_steps

    print(f"\nTraining config:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.grad_accum_steps}")
    print(f"  Effective batch size: {config.batch_size * config.grad_accum_steps}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Eval every: {config.eval_steps} steps" if config.eval_steps > 0 else "  Eval: end of epoch only")
    print(f"  Save every: {config.save_steps} steps" if config.save_steps > 0 else "  Save: end of training only")
    print()

    global_step = 0

    for epoch in range(config.epochs):
        print(f"=== Epoch {epoch + 1}/{config.epochs} ===")

        model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)

            loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
            loss = loss / config.grad_accum_steps
            loss.backward()

            if (step + 1) % config.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                examples_seen = global_step * effective_batch_size

                # Log training metrics (log every step for smooth curves)
                step_loss = loss.item() * config.grad_accum_steps
                step_acc = compute_accuracy(chosen_rewards, rejected_rewards)
                writer.add_scalar("train/loss", step_loss, global_step)
                writer.add_scalar("train/accuracy", step_acc, global_step)
                writer.add_scalar("train/loss_by_examples", step_loss, examples_seen)
                writer.add_scalar("train/accuracy_by_examples", step_acc, examples_seen)

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

                # Save every N steps
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    save_checkpoint(model, tokenizer, config.output_dir, global_step)

            with torch.no_grad():
                acc = compute_accuracy(chosen_rewards, rejected_rewards)

            total_loss += loss.item() * config.grad_accum_steps
            total_acc += acc
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item() * config.grad_accum_steps:.4f}", "acc": f"{acc:.2%}"})

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
    save_checkpoint(model, tokenizer, config.output_dir, global_step)
    writer.close()
    print("\nTraining complete!")


def main():
    # Use TrainConfig defaults as source of truth
    defaults = TrainConfig()

    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument("--model", type=str, default=defaults.model_name)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--max_length", type=int, default=defaults.max_length)
    parser.add_argument("--grad_accum", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--eval_steps", type=int, default=defaults.eval_steps)
    parser.add_argument("--eval_num_batches", type=int, default=defaults.eval_num_batches)
    parser.add_argument("--save_steps", type=int, default=defaults.save_steps)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--log_dir", type=str, default=defaults.log_dir)
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_steps=args.grad_accum,
        eval_steps=args.eval_steps,
        eval_batch_size=args.batch_size,
        eval_num_batches=args.eval_num_batches,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name)

    print("Loading reward model...")
    model = load_reward_model(config.model_name, device_map="auto")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
