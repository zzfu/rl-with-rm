"""
Model definitions for RL with Reward Model training.

- RewardModel: Qwen3 base + value head for reward prediction
- PolicyModel: Standard Qwen3 causal LM for RL policy
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel, Qwen3Model


class RewardModel(Qwen3PreTrainedModel):
    """
    Reward Model for RLHF training.

    Uses Qwen3 base model (without LM head) + a scalar value head
    that predicts reward for a given sequence.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list = None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        """
        Forward pass that returns scalar rewards.

        Returns:
            rewards: Shape (batch_size,) - scalar reward per sequence
            For training, we typically compare rewards between chosen/rejected pairs.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)

        # Get reward from last token position (or last non-padding token)
        if attention_mask is not None:
            # Find last non-padding position for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            last_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]
        else:
            last_hidden = hidden_states[:, -1, :]

        rewards = self.value_head(last_hidden).squeeze(-1)  # (batch_size,)

        return rewards

    @classmethod
    def from_pretrained_base(cls, model_name_or_path: str, **kwargs):
        """Load from a pretrained Qwen3 model, initializing value head randomly."""
        model = cls.from_pretrained(model_name_or_path, **kwargs)
        # Re-initialize value head with small weights
        nn.init.normal_(model.value_head.weight, mean=0.0, std=0.02)
        return model


def load_policy_model(model_name: str = "Qwen/Qwen3-0.6B", **kwargs):
    """
    Load the policy model for RL training.

    This is the standard causal LM that will be trained with PPO/GRPO
    using rewards from the RewardModel.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=kwargs.pop("dtype", torch.bfloat16),
        **kwargs
    )
    return model


def load_reward_model(model_name: str = "Qwen/Qwen3-0.6B", **kwargs):
    """
    Load the reward model for RM training.

    Initializes from pretrained Qwen3 weights with a randomly initialized value head.
    """
    model = RewardModel.from_pretrained_base(
        model_name,
        dtype=kwargs.pop("dtype", torch.bfloat16),
        **kwargs
    )
    return model


def load_tokenizer(model_name: str = "Qwen/Qwen3-0.6B"):
    """Load tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


if __name__ == "__main__":
    # Quick test
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading reward model...")
    rm = load_reward_model(device_map="auto")
    print(f"Reward model parameters: {sum(p.numel() for p in rm.parameters()):,}")

    print("Loading policy model...")
    policy = load_policy_model(device_map="auto")
    print(f"Policy model parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Test forward pass
    test_input = tokenizer("Hello, how are you?", return_tensors="pt")
    test_input = {k: v.to(rm.device) for k, v in test_input.items()}

    with torch.no_grad():
        reward = rm(**test_input)
        print(f"Test reward: {reward.item():.4f}")
