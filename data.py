"""
Data loader for Anthropic/hh-rlhf dataset.

Used for training reward models with chosen/rejected preference pairs.
"""

import random
import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


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


if __name__ == "__main__":
    # Quick test
    from models import load_tokenizer

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

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
