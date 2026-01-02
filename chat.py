"""
Simple Gradio chatbot for testing models and custom checkpoints.

Usage:
    python chat.py                          # Load default Qwen3-0.6B
    python chat.py --model path/to/checkpoint  # Load custom checkpoint
"""

import argparse
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


class ChatBot:
    def __init__(self, model_path: str = "Qwen/Qwen3-0.6B"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str = None):
        """Load or reload model from path."""
        if model_path:
            self.model_path = model_path

        print(f"Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        print(f"Model loaded on {self.device}")
        return f"Loaded: {self.model_path}"

    def chat(self, message: str, history: list, max_new_tokens: int = 512, temperature: float = 0.7):
        """Generate response with streaming."""
        if self.model is None:
            yield "Please load a model first using the 'Load Model' button."
            return

        # Build conversation from history
        # Gradio 4.x uses list of dicts: [{"role": "user", "content": "..."}, ...]
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Streaming generation
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        response = ""
        for token in streamer:
            response += token
            yield response

        thread.join()


def create_ui(chatbot: ChatBot, initial_model: str):
    """Create Gradio interface."""

    with gr.Blocks(title="RL-RM Chat") as demo:
        gr.Markdown("# Chat with Model")
        gr.Markdown("Load a base model or custom checkpoint and chat with it.")

        with gr.Row():
            model_path_input = gr.Textbox(
                label="Model Path",
                value=initial_model,
                placeholder="Qwen/Qwen3-0.6B or path/to/checkpoint",
                scale=4,
            )
            load_btn = gr.Button("Load Model", scale=1)

        status = gr.Textbox(label="Status", interactive=False)

        chat_interface = gr.ChatInterface(
            fn=chatbot.chat,
            additional_inputs=[
                gr.Slider(64, 2048, value=512, step=64, label="Max New Tokens"),
                gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature"),
            ],
        )

        load_btn.click(
            fn=chatbot.load_model,
            inputs=[model_path_input],
            outputs=[status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Chat with a model")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link",
    )
    parser.add_argument(
        "--autoload",
        action="store_true",
        help="Automatically load the model on startup",
    )
    args = parser.parse_args()

    chatbot = ChatBot(args.model)

    if args.autoload:
        chatbot.load_model()

    demo = create_ui(chatbot, args.model)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
