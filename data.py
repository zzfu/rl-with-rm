"""
Data loader for Anthropic/hh-rlhf dataset.

Used for training reward models with chosen/rejected preference pairs.
"""

import random
import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


def parse_hh_dialogue(text: str) -> list[dict]:
    """
    Convert hh-rlhf dialogue format to chat messages.

    Input format: "Human: ... Assistant: ... Human: ... Assistant: ..."
    Output format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    messages = []

    # Split by "Human:" or "Assistant:" while keeping the delimiter
    pattern = r"(Human:|Assistant:)"
    parts = re.split(pattern, text)

    # Process parts in pairs (delimiter, content)
    current_role = None
    for part in parts:
        part = part.strip()
        if part == "Human:":
            current_role = "user"
        elif part == "Assistant:":
            current_role = "assistant"
        elif part and current_role:
            messages.append({"role": current_role, "content": part})
            current_role = None

    return messages


class HHRLHFDataset(Dataset):
    """
    Dataset for Anthropic/hh-rlhf preference data.

    Each item contains a chosen/rejected pair for reward model training.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 2048,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer (used for chat template and tokenization)
            split: "train" or "test"
            max_length: Maximum sequence length (examples exceeding this are skipped in sampling)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading hh-rlhf dataset (split={split})...")
        self.dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        print(f"Loaded {len(self.dataset)} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def _format_dialogue(self, text: str) -> str:
        """Parse dialogue and apply chat template."""
        messages = parse_hh_dialogue(text)
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Strip thinking tags (RM training uses responses without thinking)
        formatted = re.sub(r"<think>.*?</think>\s*", "", formatted, flags=re.DOTALL)
        return formatted

    def _get_token_length(self, text: str) -> int:
        """Get token count for a dialogue without full tokenization overhead."""
        formatted = self._format_dialogue(text)
        # Use fast tokenization without creating tensors
        tokens = self.tokenizer(formatted, add_special_tokens=False)
        return len(tokens["input_ids"])

    def _tokenize_dialogue(self, text: str) -> dict:
        """Parse and tokenize a dialogue string."""
        formatted = self._format_dialogue(text)

        tokens = self.tokenizer(
            formatted,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single chosen/rejected pair.

        Returns:
            dict with keys:
                - chosen_input_ids, chosen_attention_mask
                - rejected_input_ids, rejected_attention_mask
        """
        item = self.dataset[idx]

        chosen = self._tokenize_dialogue(item["chosen"])
        rejected = self._tokenize_dialogue(item["rejected"])

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        """
        Collate function for DataLoader.

        Pads chosen and rejected sequences separately.
        """

        def pad_sequences(sequences, attention_masks):
            max_len = max(seq.size(0) for seq in sequences)
            padded_seqs = []
            padded_masks = []

            for seq, mask in zip(sequences, attention_masks):
                pad_len = max_len - seq.size(0)
                if pad_len > 0:
                    # Right padding
                    seq = torch.cat([seq, torch.full((pad_len,), self.tokenizer.pad_token_id)])
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                padded_seqs.append(seq)
                padded_masks.append(mask)

            return torch.stack(padded_seqs), torch.stack(padded_masks)

        chosen_ids, chosen_masks = pad_sequences(
            [item["chosen_input_ids"] for item in batch],
            [item["chosen_attention_mask"] for item in batch],
        )

        rejected_ids, rejected_masks = pad_sequences(
            [item["rejected_input_ids"] for item in batch],
            [item["rejected_attention_mask"] for item in batch],
        )

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_masks,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_masks,
        }

    def _try_get_item(self, idx: int) -> dict | None:
        """
        Try to get an item, returning None if it exceeds max_length.
        """
        item = self.dataset[idx]

        # Check lengths before full tokenization
        chosen_len = self._get_token_length(item["chosen"])
        if chosen_len > self.max_length:
            return None

        rejected_len = self._get_token_length(item["rejected"])
        if rejected_len > self.max_length:
            return None

        # Both fit, do full tokenization
        chosen = self._tokenize_dialogue(item["chosen"])
        rejected = self._tokenize_dialogue(item["rejected"])

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    def sample_batch(self, batch_size: int) -> dict:
        """
        Sample a random batch from the dataset, skipping examples that exceed max_length.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Collated batch dict ready for model input
        """
        batch = []
        max_attempts = batch_size * 10  # safety limit
        attempts = 0

        while len(batch) < batch_size and attempts < max_attempts:
            idx = random.randint(0, len(self) - 1)
            item = self._try_get_item(idx)
            if item is not None:
                batch.append(item)
            attempts += 1

        if len(batch) < batch_size:
            raise RuntimeError(
                f"Could not sample {batch_size} valid examples in {max_attempts} attempts. "
                f"Consider increasing max_length (current: {self.max_length})."
            )

        return self.collate_fn(batch)


class PromptDataset(Dataset):
    """
    Prompt-only dataset for GRPO training.

    Extracts user prompts from HH-RLHF for policy model generation.
    Supports mixing train/test splits to reduce overfitting to RM training data.
    """

    def __init__(
        self,
        tokenizer,
        train_samples: int = 1500,
        test_samples: int = 8500,
        max_length: int = 512,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            train_samples: Number of samples from train split (0 to skip)
            test_samples: Number of samples from test split (0 to skip)
            max_length: Maximum prompt length (longer prompts skipped in sampling)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading hh-rlhf prompts (train={train_samples}, test={test_samples})...")
        datasets_to_combine = []

        if train_samples > 0:
            train = load_dataset("Anthropic/hh-rlhf", split="train")
            train = train.shuffle(seed=42).select(range(min(train_samples, len(train))))
            datasets_to_combine.append(train)
            print(f"  Train: {len(train)} samples")

        if test_samples > 0:
            test = load_dataset("Anthropic/hh-rlhf", split="test")
            test = test.shuffle(seed=42).select(range(min(test_samples, len(test))))
            datasets_to_combine.append(test)
            print(f"  Test: {len(test)} samples")

        if not datasets_to_combine:
            raise ValueError("Must have at least one of train_samples or test_samples > 0")

        # Combine and shuffle to mix train/test
        self.dataset = concatenate_datasets(datasets_to_combine).shuffle(seed=42)
        print(f"  Total: {len(self.dataset)} samples (shuffled)")

    def __len__(self) -> int:
        return len(self.dataset)

    def _extract_prompt(self, text: str) -> str:
        """
        Extract the prompt (user turns) from a dialogue.

        Returns the dialogue up to (and including) the last user turn,
        formatted with chat template and generation prompt.
        """
        messages = parse_hh_dialogue(text)

        # Find last user message index
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                last_user_idx = i

        if last_user_idx == -1:
            # Fallback: use all messages if no user found
            prompt_messages = messages
        else:
            # Include everything up to and including last user turn
            prompt_messages = messages[: last_user_idx + 1]

        # Apply chat template with generation prompt
        formatted = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted

    def _get_prompt_length(self, text: str) -> int:
        """Get token count for prompt."""
        formatted = self._extract_prompt(text)
        tokens = self.tokenizer(formatted, add_special_tokens=False)
        return len(tokens["input_ids"])

    def _tokenize_prompt(self, text: str) -> dict:
        """Extract and tokenize prompt from dialogue."""
        formatted = self._extract_prompt(text)

        tokens = self.tokenizer(
            formatted,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> dict:
        """Get a single prompt."""
        item = self.dataset[idx]
        # Use chosen dialogue (could also use rejected, both have same prompt)
        return self._tokenize_prompt(item["chosen"])

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate function for DataLoader. Pads prompts."""
        max_len = max(item["input_ids"].size(0) for item in batch)
        padded_ids = []
        padded_masks = []

        for item in batch:
            seq = item["input_ids"]
            mask = item["attention_mask"]
            pad_len = max_len - seq.size(0)

            if pad_len > 0:
                # Left padding for generation (model generates from right)
                seq = torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id), seq])
                mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])

            padded_ids.append(seq)
            padded_masks.append(mask)

        return {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_masks),
        }

    def _try_get_item(self, idx: int) -> dict | None:
        """Try to get an item, returning None if it exceeds max_length."""
        item = self.dataset[idx]

        prompt_len = self._get_prompt_length(item["chosen"])
        if prompt_len > self.max_length:
            return None

        return self._tokenize_prompt(item["chosen"])

    def sample_batch(self, batch_size: int) -> dict:
        """
        Sample a random batch of prompts.

        Args:
            batch_size: Number of prompts to sample

        Returns:
            Collated batch dict ready for model input
        """
        batch = []
        max_attempts = batch_size * 10
        attempts = 0

        while len(batch) < batch_size and attempts < max_attempts:
            idx = random.randint(0, len(self) - 1)
            item = self._try_get_item(idx)
            if item is not None:
                batch.append(item)
            attempts += 1

        if len(batch) < batch_size:
            raise RuntimeError(
                f"Could not sample {batch_size} valid prompts in {max_attempts} attempts. "
                f"Consider increasing max_length (current: {self.max_length})."
            )

        return self.collate_fn(batch)


if __name__ == "__main__":
    # Quick test
    import argparse
    from models import load_tokenizer

    parser = argparse.ArgumentParser(description="Test HH-RLHF dataset loading")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path for tokenizer (default: Qwen/Qwen3-0.6B)",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = load_tokenizer(args.model)

    print("\nCreating dataset...")
    dataset = HHRLHFDataset(tokenizer, split="train", max_length=2048)

    print(f"\nDataset size: {len(dataset)}")

    print("\nSample item:")
    item = dataset[0]
    for key, value in item.items():
        print(f"  {key}: shape={value.shape}")

    print("\nSample batch (size=4):")
    batch = dataset.sample_batch(4)
    for key, value in batch.items():
        print(f"  {key}: shape={value.shape}")

    print("\nDecoded chosen example (padding stripped):")
    # Use attention mask to find actual content length
    content_len = batch["chosen_attention_mask"][0].sum().item()
    print(tokenizer.decode(batch["chosen_input_ids"][0][:content_len]))

    print("\n" + "=" * 50)
    print("Testing PromptDataset...")
    prompt_dataset = PromptDataset(tokenizer, train_samples=100, test_samples=100, max_length=512)
    print(f"Prompt dataset size: {len(prompt_dataset)}")

    print("\nSample prompt batch (size=4):")
    prompt_batch = prompt_dataset.sample_batch(4)
    for key, value in prompt_batch.items():
        print(f"  {key}: shape={value.shape}")

    print("\nDecoded prompt (first in batch):")
    mask = prompt_batch["attention_mask"][0]
    content_start = (mask == 0).sum().item()  # Left padding
    print(tokenizer.decode(prompt_batch["input_ids"][0][content_start:]))
