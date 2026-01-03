"""
Quantize a Policy Model and evaluate KL divergence vs original.

Generates completions with the original model, then computes KL divergence
between original and quantized models on those generated tokens.

Usage:
    # From a GRPO checkpoint
    python quantize_policy.py --model ./checkpoints/grpo/run_name/step-1000

    # From a base model (saves to ./quantized_models/)
    python quantize_policy.py --model Qwen/Qwen3-0.6B --quant_type int4
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from models import load_tokenizer
from data import PromptDataset


def is_checkpoint_path(path: str) -> bool:
    """Check if path looks like a checkpoint (vs a model name like Qwen/...)."""
    return os.path.exists(path) or path.startswith("./") or path.startswith("/") or "\\" in path


def load_original_model(model_path: str):
    """Load the original (non-quantized) policy model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


def load_quantized_model(model_path: str, quant_type: str):
    """Load a quantized version of the policy model."""
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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
        if hasattr(param, "quant_state"):
            total_bytes += param.numel() // 2
        elif param.dtype == torch.int8:
            total_bytes += param.numel()
        else:
            total_bytes += param.numel() * param.element_size()

    return {
        "param_count": param_count,
        "size_mb": total_bytes / (1024 * 1024),
    }


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: list[dict],
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
    """
    Generate completions for a batch of prompts.

    Returns list of dicts with input_ids, attention_mask, prompt_length.
    """
    device = next(model.parameters()).device
    results = []

    for prompt_data in tqdm(prompts, desc="Generating"):
        input_ids = prompt_data["input_ids"].unsqueeze(0).to(device)
        attention_mask = prompt_data["attention_mask"].unsqueeze(0).to(device)
        prompt_length = attention_mask.sum().item()

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # outputs is [1, seq_len]
        full_ids = outputs[0]
        full_mask = torch.ones_like(full_ids)

        results.append({
            "input_ids": full_ids.cpu(),
            "attention_mask": full_mask.cpu(),
            "prompt_length": prompt_length,
        })

    return results


@torch.no_grad()
def compute_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for completion tokens.

    Returns tensor of shape [num_completion_tokens].
    """
    device = next(model.parameters()).device
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0, :-1, :]  # [seq_len-1, vocab]
    labels = input_ids[0, 1:]  # [seq_len-1]

    # Compute per-token log probs
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Return only completion tokens (after prompt)
    # prompt_length is the number of prompt tokens, so completion starts at index prompt_length-1
    # in the shifted sequence (since we're predicting token i+1 from position i)
    completion_start = prompt_length - 1
    return token_log_probs[completion_start:].cpu()


@torch.no_grad()
def evaluate_kl_divergence(
    original_model,
    quant_model,
    completions: list[dict],
    desc: str = "Computing KL",
) -> dict:
    """
    Compute KL divergence between original and quantized models on completions.

    KL(P||Q) = E_P[log P - log Q] where P is original, Q is quantized.
    """
    all_kl = []
    all_original_logprobs = []
    all_quant_logprobs = []
    total_tokens = 0

    for comp in tqdm(completions, desc=desc):
        input_ids = comp["input_ids"]
        attention_mask = comp["attention_mask"]
        prompt_length = comp["prompt_length"]

        # Get log probs from both models
        orig_logprobs = compute_token_logprobs(
            original_model, input_ids, attention_mask, prompt_length
        )
        quant_logprobs = compute_token_logprobs(
            quant_model, input_ids, attention_mask, prompt_length
        )

        # KL per token
        kl_per_token = orig_logprobs - quant_logprobs

        all_kl.extend(kl_per_token.tolist())
        all_original_logprobs.extend(orig_logprobs.tolist())
        all_quant_logprobs.extend(quant_logprobs.tolist())
        total_tokens += len(kl_per_token)

    mean_kl = sum(all_kl) / len(all_kl) if all_kl else 0
    mean_orig_logprob = sum(all_original_logprobs) / len(all_original_logprobs) if all_original_logprobs else 0
    mean_quant_logprob = sum(all_quant_logprobs) / len(all_quant_logprobs) if all_quant_logprobs else 0

    return {
        "kl_divergence": mean_kl,
        "mean_original_logprob": mean_orig_logprob,
        "mean_quantized_logprob": mean_quant_logprob,
        "total_tokens": total_tokens,
        "num_completions": len(completions),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a policy model and evaluate KL divergence"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., Qwen/Qwen3-0.6B) or checkpoint path",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="int8",
        choices=["int8", "int4"],
        help="Quantization type (default: int8)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=100,
        help="Number of prompts to evaluate (default: 100)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per prompt (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=1536,
        help="Max prompt length (default: 1536)",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="Skip saving the quantized model",
    )

    args = parser.parse_args()

    # Determine output directory
    if is_checkpoint_path(args.model):
        output_dir = os.path.join(args.model, f"quantized_{args.quant_type}")
        tokenizer_path = args.model
    else:
        # Base model - save to quantized_models folder
        model_name_safe = args.model.replace("/", "_")
        output_dir = os.path.join("./quantized_models", f"{model_name_safe}_{args.quant_type}")
        tokenizer_path = args.model

    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Quantization: {args.quant_type}")
    print(f"Output dir: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path)

    # Load prompt dataset and sample prompts
    print("Loading prompt dataset...")
    prompt_dataset = PromptDataset(
        tokenizer,
        train_samples=500,
        test_samples=500,
        max_length=args.prompt_max_length,
    )

    print(f"Sampling {args.num_prompts} prompts...")
    sampled_prompts = []
    for _ in range(args.num_prompts):
        idx = torch.randint(0, len(prompt_dataset), (1,)).item()
        item = prompt_dataset._try_get_item(idx)
        if item:
            sampled_prompts.append(item)
        if len(sampled_prompts) >= args.num_prompts:
            break

    results = {
        "model": args.model,
        "quant_type": args.quant_type,
        "num_prompts": len(sampled_prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    # === Load and evaluate original model ===
    print("\n" + "=" * 50)
    print("Loading ORIGINAL model (bf16)")
    print("=" * 50)

    start_time = time.time()
    original_model = load_original_model(args.model)
    load_time_original = time.time() - start_time
    print(f"Load time: {load_time_original:.2f}s")

    mem_info = get_model_memory_footprint(original_model)
    print(f"Parameters: {mem_info['param_count']:,}")
    print(f"Size (estimated): {mem_info['size_mb']:.1f} MB")

    # Generate completions with original model
    print(f"\nGenerating {len(sampled_prompts)} completions...")
    gen_start = time.time()
    completions = generate_completions(
        original_model,
        tokenizer,
        sampled_prompts,
        args.max_new_tokens,
        args.temperature,
    )
    gen_time = time.time() - gen_start
    print(f"Generation time: {gen_time:.1f}s")

    results["original"] = {
        "load_time_s": load_time_original,
        "generation_time_s": gen_time,
        "size_mb": mem_info["size_mb"],
        "param_count": mem_info["param_count"],
    }

    # === Load quantized model ===
    print("\n" + "=" * 50)
    print(f"Loading QUANTIZED model ({args.quant_type})")
    print("=" * 50)

    start_time = time.time()
    quant_model = load_quantized_model(args.model, args.quant_type)
    load_time_quant = time.time() - start_time
    print(f"Load time: {load_time_quant:.2f}s")

    quant_mem = get_model_memory_footprint(quant_model)
    print(f"Size (estimated): {quant_mem['size_mb']:.1f} MB")

    # === Compute KL divergence ===
    print("\n" + "=" * 50)
    print("Computing KL divergence")
    print("=" * 50)

    eval_start = time.time()
    kl_results = evaluate_kl_divergence(original_model, quant_model, completions)
    eval_time = time.time() - eval_start
    print(f"Evaluation time: {eval_time:.1f}s")

    results[f"quantized_{args.quant_type}"] = {
        "load_time_s": load_time_quant,
        "size_mb": quant_mem["size_mb"],
    }
    results["kl_evaluation"] = {
        **kl_results,
        "eval_time_s": eval_time,
    }

    print(f"\nKL Divergence: {kl_results['kl_divergence']:.6f}")
    print(f"Mean original log prob: {kl_results['mean_original_logprob']:.4f}")
    print(f"Mean quantized log prob: {kl_results['mean_quantized_logprob']:.4f}")
    print(f"Total tokens evaluated: {kl_results['total_tokens']:,}")

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
            results["model_saved"] = False
            results["save_error"] = str(e)

            quant_config = {
                "base_model": args.model,
                "quant_type": args.quant_type,
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
    size_reduction = (1 - quant_mem["size_mb"] / mem_info["size_mb"]) * 100

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'Value':>20}")
    print("-" * 60)
    print(f"{'KL Divergence (orig || quant)':<35} {kl_results['kl_divergence']:>20.6f}")
    print(f"{'Mean Original Log Prob':<35} {kl_results['mean_original_logprob']:>20.4f}")
    print(f"{'Mean Quantized Log Prob':<35} {kl_results['mean_quantized_logprob']:>20.4f}")
    print(f"{'Size Reduction':<35} {size_reduction:>19.1f}%")
    print(f"{'Tokens Evaluated':<35} {kl_results['total_tokens']:>20,}")
    print("-" * 60)

    # Interpret KL divergence
    kl = kl_results['kl_divergence']
    if kl < 0.01:
        print("\n✓ Very low KL divergence - quantization has minimal impact")
    elif kl < 0.05:
        print("\n⚠ Low KL divergence - quantization has small but measurable impact")
    elif kl < 0.1:
        print("\n⚠ Moderate KL divergence - quantization noticeably affects outputs")
    else:
        print("\n✗ High KL divergence - quantization significantly changes model behavior")


if __name__ == "__main__":
    main()
