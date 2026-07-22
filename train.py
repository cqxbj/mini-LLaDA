# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
from model import LLaDA
from datasets import load_from_disk
from transformers import AutoTokenizer
from config import *
from helper import get_device, validate_config,load_model
import random
import config as cfg


class LLaDADataCollator:
    def __init__(self, mask_token_id, diffusion_steps=DIFFUSION_STEPS):
        """
        Args:
            mask_token_id: [MASK] token ID
            diffusion_steps: number of discrete diffusion steps;
                             t = randint(1, S) / S  →  t ∈ [1/S, 1]
                             1/t max = S, naturally bounded, no eps needed
        """
        self.mask_token_id = mask_token_id
        self.diffusion_steps = diffusion_steps

    def __call__(self, features):
        input_ids = torch.stack([f['input_ids'] for f in features])
        batch_size, seq_len = input_ids.shape

        masked_input_ids = input_ids.clone()
        labels = input_ids.clone()
        t_values = torch.zeros(batch_size)

        for i in range(batch_size):
            prompt_len = cfg.PROMPT_LEN

            t_int = random.randint(1, self.diffusion_steps)
            t = t_int / self.diffusion_steps
            t_values[i] = t

            for j in range(prompt_len, seq_len):
                if random.random() < t:
                    masked_input_ids[i, j] = self.mask_token_id
                else:
                    labels[i, j] = -100

            labels[i, :prompt_len] = -100

        return {
            "input_ids": masked_input_ids,
            "labels": labels,
            "t_values": t_values,
        }


# training loss function
def training_loss_LLaDA(logits, labels, t_values):
    """
    LLaDA loss from Algorithm 2 of the paper.

    L(θ) = -E_{t, x0, xt} [ 1/(t·L) · Σ 1[x^i_t=M] · log p_θ(x^i_0 | xt) ]

    Args:
        logits:   (B, L, V)  model output logits
        labels:   (B, L)     ground-truth token IDs; -100 = ignore (unmasked / prompt)
        t_values: (B,)       per-sample masking probability t ∈ (0, 1]

    Returns:
        scalar loss
    """
    B, L, V = logits.shape

    token_loss = F.cross_entropy(
        logits.view(-1, V),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(B, L)                                                         # (B, L)

    t = t_values.to(logits.device).clamp(min=1e-8).view(B, 1)          # (B, 1)

    masked = (labels != -100)                                            # (B, L)

    weighted = token_loss / t                                            # (B, L)  — 1/t scaling
    weighted = weighted * masked.float()                                 # zero out non-masked
    loss = weighted.sum() / (B * L)                                      # scalar

    return loss


@torch.no_grad()
def evaluate(model, val_loader, device, val_passes=3):
    """
    Evaluate the model on a validation set.

    Runs val_passes over the data with different random masks each time and
    averages the results to reduce variance from the stochastic masking.

    Returns:
        avg_loss: LLaDA loss averaged over all batches and passes
    """
    model.eval()
    all_losses = []

    for _ in range(val_passes):
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            t_values = batch["t_values"]

            logits = model(input_ids)
            loss = training_loss_LLaDA(logits, labels, t_values)
            all_losses.append(loss.item())

    model.train()
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else float('inf')
    return avg_loss


def train_model():
    device = get_device()
    print(f"Using device: {device}")
    # Configuration
    batch_size = cfg.BATCH_SIZE
    learning_rate = cfg.LEARNING_RATE
    epochs = cfg.EPOCHS
    save_steps = cfg.SAVE_EVERY
    output_dir = cfg.CHECKPOINT_DIR

    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    model = LLaDA()
    # checkpoint_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    # model, _ = load_model(checkpoint_path=checkpoint_path)    
    model.to(device)

    # Load dataset & split into train / val
    loaded_dataset = load_from_disk(f"{cfg.DATA_DIR}/tokenized_tinystories_dataset")
    full_dataset = loaded_dataset.select(range(cfg.TRAIN_SUBSET))
    split = full_dataset.train_test_split(test_size=cfg.VAL_SPLIT, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    collator = LLaDADataCollator(mask_token_id=tokenizer.mask_token_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"Data: train={len(train_dataset)}, val={len(val_dataset)} "
          f"({cfg.VAL_SPLIT*100:.0f}% split)")


    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_dataloader) * epochs
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )

    # Training loop
    model.train()
    global_step = 0

    print("Starting training...")
    # Count and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model parameter count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Total steps: {total_steps} needs to complete")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            t_values = batch["t_values"]  # keep on CPU for now; moved inside loss

            # Forward pass
            logits = model(input_ids)

            # Calculate LLaDA training loss
            loss = training_loss_LLaDA(logits, labels, t_values)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })
            epoch_loss += loss.item()
            # Save checkpoint periodically
            if global_step % save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                }, os.path.join(checkpoint_path, "model_checkpoint.pth"))

                print(f"Saved checkpoint at step {global_step}")

        avg_train_loss = epoch_loss / len(train_dataloader)

        # ---- Validation ----
        if (epoch + 1) % cfg.VAL_EVERY == 0:
            val_loss = evaluate(model, val_dataloader, device, cfg.VAL_PASSES)
            print(f"  Epoch {epoch+1}: "
                  f"train_loss={avg_train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}")

        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         best_epoch = epoch + 1
        #         torch.save(
        #             {"model_state_dict": model.state_dict()},
        #             os.path.join(CHECKPOINT_DIR, "best_model.pth"),
        #         )
        #         print(f"  >>> Best model saved (val_loss={val_loss:.4f})")
        # else:
        #     print(f"  Epoch {epoch+1} avg loss: {avg_train_loss:.4f}")

    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "final_model.pth"),
    )

    print(f"Training completed! Final model saved.")
    if best_val_loss < float('inf'):
        print(f"Best model: epoch {best_epoch}, val_loss={best_val_loss:.4f}  "
              f"→ saved as best_model.pth")



if __name__ == "__main__":
    validate_config()
    train_model()