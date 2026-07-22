import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from config import *
import os
from model import LLaDA, LLaDAConfig


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def validate_config():
    """Validate hyperparameter constraints before training or inference."""
    if PROMPT_LEN >= MAX_SEQ_LEN:
        raise ValueError(
            f"PROMPT_LEN ({PROMPT_LEN}) must be < MAX_SEQ_LEN ({MAX_SEQ_LEN}), "
            f"otherwise there is no room for the model to generate."
        )
    if MAX_SEQ_LEN <= 0:
        raise ValueError(f"MAX_SEQ_LEN must be positive, got {MAX_SEQ_LEN}")
    if PROMPT_LEN < 0:
        raise ValueError(f"PROMPT_LEN must be non-negative, got {PROMPT_LEN}")


def process_tokenize_data():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    def tokenize_and_truncate(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length"
        )

    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(TOKENIZE_SUBSET))

    tokenized_dataset = dataset.map(
        tokenize_and_truncate,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )

    tokenized_dataset.set_format(type='torch')

    os.makedirs(DATA_DIR, exist_ok=True)
    tokenized_dataset.save_to_disk(f"{DATA_DIR}/tokenized_tinystories_dataset")

    print(f"dataset len: {len(tokenized_dataset)}")
    print(f"column names: {tokenized_dataset.column_names}")
    print(f"shape: {tokenized_dataset[0]['input_ids'].shape}")


def load_model(checkpoint_path):
    device = get_device()
    model = LLaDA()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params, "
          f"epoch={checkpoint.get('epoch', '?')}, device={device}")
    return model, device

