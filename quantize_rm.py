"""
Quantize a trained Reward Model and evaluate both original and quantized versions.

Uses bitsandbytes for INT8/INT4 quantization. The quantized model is saved along
with evaluation results comparing original vs quantized performance.

Usage:
    python quantize_rm.py --checkpoint checkpoints/rm/run_name/step-1000
    python quantize_rm.py --checkpoint checkpoints/rm/run_name/step-1000 --quant_type int4
"""

import argparse
import json
import os
import time

import torch
from transformers import BitsAndBytesConfig
from tqdm import tqdm

from models import RewardModel, load_tokenizer
from data import HHRLHFDataset
from train_rm import bradley_terry_loss, compute_accuracy


def load_original_model(checkpoint_path: str):
    """Load the original (non-quantized) model from checkpoint."""
    model = RewardModel.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


def load_quantized_model(checkpoint_path: str, quant_type: str):
    """Load a quantized version of the model."""
    if quant_type == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quant_type == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}. Use 'int8' or 'int4'.")

    model = RewardModel.from_pretrained(
        checkpoint_path,
        quantization_config=quantization_config,
        device_map="auto",
    )
    return model


def get_model_memory_footprint(model) -> dict:
    """Get memory usage of model parameters."""
    total_bytes = 0
    param_count = 0

    for param in model.parameters():
        param_count += param.numel()
        # Handle quantized parameters which may have different storage
        if hasattr(param, "quant_state"):
            # bitsandbytes 4-bit
            total_bytes += param.numel() // 2  # 4-bit = 0.5 bytes per param
        elif param.dtype == torch.int8:
            total_bytes += param.numel()
        else:
            total_bytes += param.numel() * param.element_size()

    return {
        "param_count": param_count,
        "size_mb": total_bytes / (1024 * 1024),
    }


@torch.no_grad()
def evaluate(model, dataset, batch_size: int, num_batches: int, desc: str = "Evaluating"):
    """Evaluate model on test dataset."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_acc = 0.0
    all_chosen_rewards = []
    all_rejected_rewards = []

    for _ in tqdm(range(num_batches), desc=desc):
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
        all_chosen_rewards.extend(chosen_rewards.cpu().tolist())
        all_rejected_rewards.extend(rejected_rewards.cpu().tolist())

    n = len(all_chosen_rewards)
    return {
        "loss": total_loss / num_batches,
        "accuracy": total_acc / num_batches,
        "mean_chosen_reward": sum(all_chosen_rewards) / n,
        "mean_rejected_reward": sum(all_rejected_rewards) / n,
        "reward_gap": (sum(all_chosen_rewards) - sum(all_rejected_rewards)) / n,
        "num_examples": n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a trained RM and evaluate both versions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained RM checkpoint",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="int8",
        choices=["int8", "int4"],
        help="Quantization type (default: int8)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size (default: 4)",
    )
    parser.add_argument(
        "--eval_num_batches",
        type=int,
        default=500,
        help="Number of batches for evaluation (default: 500, ~2000 examples)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max sequence length (default: 2048)",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="Skip saving the quantized model (only save results)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.join(args.checkpoint, f"quantized_{args.quant_type}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Quantization: {args.quant_type}")
    print(f"Output dir: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(args.checkpoint)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = HHRLHFDataset(tokenizer, split="test", max_length=args.max_length)

    results = {
        "checkpoint": args.checkpoint,
        "quant_type": args.quant_type,
        "eval_batch_size": args.eval_batch_size,
        "eval_num_batches": args.eval_num_batches,
        "max_length": args.max_length,
    }

    # === Evaluate Original Model ===
    print("\n" + "=" * 50)
    print("Evaluating ORIGINAL model (bf16)")
    print("=" * 50)

    start_time = time.time()
    original_model = load_original_model(args.checkpoint)
    load_time_original = time.time() - start_time
    print(f"Load time: {load_time_original:.2f}s")

    mem_info = get_model_memory_footprint(original_model)
    print(f"Parameters: {mem_info['param_count']:,}")
    print(f"Size (estimated): {mem_info['size_mb']:.1f} MB")

    eval_start = time.time()
    original_results = evaluate(
        original_model, test_dataset,
        args.eval_batch_size, args.eval_num_batches,
        desc="Eval original"
    )
    eval_time_original = time.time() - eval_start

    original_results["load_time_s"] = load_time_original
    original_results["eval_time_s"] = eval_time_original
    original_results["size_mb"] = mem_info["size_mb"]
    original_results["param_count"] = mem_info["param_count"]
    results["original"] = original_results

    print(f"\nOriginal Results:")
    print(f"  Loss:     {original_results['loss']:.4f}")
    print(f"  Accuracy: {original_results['accuracy']:.2%}")
    print(f"  Reward gap (chosen - rejected): {original_results['reward_gap']:.4f}")
    print(f"  Eval time: {eval_time_original:.1f}s")

    # Clear from memory
    del original_model
    torch.cuda.empty_cache()

    # === Evaluate Quantized Model ===
    print("\n" + "=" * 50)
    print(f"Evaluating QUANTIZED model ({args.quant_type})")
    print("=" * 50)

    start_time = time.time()
    quant_model = load_quantized_model(args.checkpoint, args.quant_type)
    load_time_quant = time.time() - start_time
    print(f"Load time: {load_time_quant:.2f}s")

    # Memory footprint for quantized model
    quant_mem = get_model_memory_footprint(quant_model)
    print(f"Size (estimated): {quant_mem['size_mb']:.1f} MB")

    eval_start = time.time()
    quant_results = evaluate(
        quant_model, test_dataset,
        args.eval_batch_size, args.eval_num_batches,
        desc=f"Eval {args.quant_type}"
    )
    eval_time_quant = time.time() - eval_start

    quant_results["load_time_s"] = load_time_quant
    quant_results["eval_time_s"] = eval_time_quant
    quant_results["size_mb"] = quant_mem["size_mb"]
    results[f"quantized_{args.quant_type}"] = quant_results

    print(f"\nQuantized Results:")
    print(f"  Loss:     {quant_results['loss']:.4f}")
    print(f"  Accuracy: {quant_results['accuracy']:.2%}")
    print(f"  Reward gap (chosen - rejected): {quant_results['reward_gap']:.4f}")
    print(f"  Eval time: {eval_time_quant:.1f}s")

    # === Comparison ===
    acc_drop = original_results["accuracy"] - quant_results["accuracy"]
    acc_drop_pct = acc_drop / original_results["accuracy"] * 100 if original_results["accuracy"] > 0 else 0
    loss_increase = quant_results["loss"] - original_results["loss"]
    size_reduction = (1 - quant_mem["size_mb"] / mem_info["size_mb"]) * 100

    results["comparison"] = {
        "accuracy_drop": acc_drop,
        "accuracy_drop_relative_pct": acc_drop_pct,
        "loss_increase": loss_increase,
        "size_reduction_pct": size_reduction,
        "speedup_eval": eval_time_original / eval_time_quant if eval_time_quant > 0 else 0,
    }

    # === Save quantized model ===
    if not args.skip_save:
        print(f"\nSaving quantized model to {output_dir}...")
        try:
            quant_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            results["model_saved"] = True
            print("Model saved successfully.")
        except Exception as e:
            print(f"Warning: Could not save quantized model: {e}")
            print("This is expected for INT8 models. INT4 models with recent bitsandbytes can be saved.")
            results["model_saved"] = False
            results["save_error"] = str(e)

            # Save a config file so users know how to load the quantized model
            quant_config = {
                "base_checkpoint": args.checkpoint,
                "quant_type": args.quant_type,
                "load_command": f"RewardModel.from_pretrained('{args.checkpoint}', quantization_config=BitsAndBytesConfig(load_in_{'8' if args.quant_type == 'int8' else '4'}bit=True), device_map='auto')",
            }
            with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
                json.dump(quant_config, f, indent=2)
    else:
        results["model_saved"] = False
        print("\nSkipping model save (--skip_save)")

    # === Save results ===
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Original':>12} {'Quantized':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'Accuracy':<30} {original_results['accuracy']:>11.2%} {quant_results['accuracy']:>11.2%} {-acc_drop:>+11.2%}")
    print(f"{'Loss':<30} {original_results['loss']:>12.4f} {quant_results['loss']:>12.4f} {loss_increase:>+12.4f}")
    print(f"{'Reward Gap':<30} {original_results['reward_gap']:>12.4f} {quant_results['reward_gap']:>12.4f} {quant_results['reward_gap'] - original_results['reward_gap']:>+12.4f}")
    print(f"{'Size (MB)':<30} {mem_info['size_mb']:>12.1f} {quant_mem['size_mb']:>12.1f} {-size_reduction:>+11.1f}%")
    print(f"{'Eval Time (s)':<30} {eval_time_original:>12.1f} {eval_time_quant:>12.1f} {(eval_time_quant/eval_time_original - 1)*100:>+11.1f}%")
    print("-" * 60)

    if acc_drop <= 0.01:
        print("\n✓ Quantization has minimal impact on accuracy (<1% drop)")
    elif acc_drop <= 0.03:
        print("\n⚠ Quantization has moderate impact on accuracy (1-3% drop)")
    else:
        print("\n✗ Quantization significantly impacts accuracy (>3% drop)")


if __name__ == "__main__":
    main()
