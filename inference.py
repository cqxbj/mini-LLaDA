"""
LLaDA Inference Algorithms

This module implements two inference strategies for the LLaDA diffusion language model.
Unlike autoregressive models that generate tokens left-to-right, LLaDA starts with
all [MASK] tokens and iteratively denoises (de-masks) them over multiple steps,
similar to diffusion models in image generation.

Both algorithms follow the same high-level loop:
  1. Start with a fully masked response region (after an unmasked prompt prefix).
  2. At each step, run the model to predict clean tokens for all positions.
  3. Decide which predictions to "commit" (keep permanently) and which to re-mask.
  4. Repeat until all positions are decoded or we run out of steps.

The two strategies differ only in step 3 — how they choose which tokens to keep:
  - Random remasking: randomly re-mask positions with decreasing probability.
  - Low-confidence remasking: keep high-confidence predictions first, re-mask
    uncertain ones so the model can revise them with more context later.
"""

import torch
import random
from transformers import AutoTokenizer
from config import *
from helper import get_device
import time


# =============================================================================
# Algorithm 1: Random Remasking (paper §3.3, simpler baseline)
# =============================================================================
#
# At each step t → s:
#   - The model predicts tokens for all positions.
#   - Already-decoded tokens are kept (frozen).
#   - Each still-masked position is independently re-masked with probability s/t.
#     As s decreases toward 0, fewer tokens get re-masked, so more tokens survive.
#
# This is simple but wasteful: high-confidence predictions may be discarded
# while low-confidence ones survive by chance.
# =============================================================================
@torch.no_grad()
def random_remasking_inference(
    model, prompt_ids=None,
    sampling_steps=30,
    max_length=MAX_SEQ_LEN,
    show_each_step=True,
    show_mode="decoding",  # "decoding" or "correction"
):

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    device = get_device()

    if prompt_ids is None:
        prompt_ids = []
    prompt_len = len(prompt_ids)
    answer_length = max_length - prompt_len

    # Initialize the response region as all [MASK] tokens.
    # The model will progressively de-mask these over the sampling loop.
    response = [tokenizer.mask_token_id] * answer_length

    # Track the previous step's raw predictions so we can count how many
    # still-masked positions changed their prediction ("corrections").
    # This is purely for display purposes in "correction" mode.
    prev_correction = None

    for step in range(sampling_steps):
        # Noise schedule: linearly decrease from 1.0 (fully masked) to 0.0 (fully decoded).
        # t = current noise level, s = next noise level (one step lower).
        # The ratio s/t controls the remasking probability at this step.
        t = 1.0 - step * (1.0 / sampling_steps)      # current noise level
        s = t - (1.0 / sampling_steps)                # next noise level

        # Concatenate prompt (frozen) + response (being decoded) and run the model.
        full_input = prompt_ids + response.copy()
        logits = model(torch.tensor([full_input]).to(device))

        # Take argmax predictions for the response region only (skip prompt prefix).
        pred_ids = torch.argmax(logits, dim=-1)[0][prompt_len:].tolist()

        # ---- Correction view (display only) ----
        # Build a "what the model thinks right now" view: confirmed tokens plus
        # raw predictions at still-masked positions. This lets us observe the model
        # revising its guesses over time as more context becomes available.
        if show_each_step and show_mode == "correction":
            correction_view = []
            for i in range(answer_length):
                if response[i] != tokenizer.mask_token_id:
                    correction_view.append(response[i])          # already confirmed
                else:
                    correction_view.append(pred_ids[i])          # model's current guess

            # Count how many still-masked positions changed prediction since last step.
            num_changed = 0
            if prev_correction is not None:
                for i in range(answer_length):
                    if response[i] == tokenizer.mask_token_id:
                        if correction_view[i] != prev_correction[i]:
                            num_changed += 1
            prev_correction = correction_view.copy()

            # Count newly decoded tokens this step (for display).
            num_newly_decoded = 0
            for i in range(answer_length):
                if response[i] == tokenizer.mask_token_id and random.random() >= s / t:
                    num_newly_decoded += 1

        # Decide which tokens to keep vs. re-mask.
        # Already-decoded tokens are always kept (once unmasked, they stay).
        # Still-masked tokens are re-masked with probability s/t.
        for i in range(answer_length):
            if response[i] != tokenizer.mask_token_id:
                # Already decoded in a previous step — keep the committed token.
                pred_ids[i] = response[i]
            else:
                # Still masked — re-mask with prob s/t so the model can refine it later.
                if random.random() < s / t:
                    pred_ids[i] = tokenizer.mask_token_id  # re-mask

        response = pred_ids

        # ---- Visualize the current step ----
        if show_each_step:
            time.sleep(0.05)
            # Clear the terminal screen (ANSI escape) for animated output.
            print("\033[H\033[J", end="", flush=True)

            # Display header with the prompt.
            mode_label = "CORRECTION (raw predictions)" if show_mode == "correction" else "DECODING (after masking)"
            print(f"  Prompt : {tokenizer.decode(prompt_ids)}")

            if show_mode == "correction":
                # Show raw model predictions at every position, including masked ones.
                pred_text = tokenizer.decode(correction_view, skip_special_tokens=False)
                pred_text = pred_text.replace(tokenizer.pad_token, " ")
                print(f"  Predict: {pred_text}")
                # Count currently masked tokens.
                num_masked = sum(1 for r in response if r == tokenizer.mask_token_id)
                print(f"  newly decoded ~{num_newly_decoded}  |  remask ~{num_masked}  |  changed {num_changed}")
            else:
                # Show the response as it gradually fills in.
                decoded_text = tokenizer.decode(response, skip_special_tokens=False)
                decoded_text = decoded_text.replace(tokenizer.mask_token, "######").replace(tokenizer.pad_token, " ")
                print()
                print(f"  : \n {decoded_text}")

    return tokenizer.decode(response)


# =============================================================================
# Algorithm 2: Low-Confidence Remasking (paper §3.3, higher quality)
# =============================================================================
#
# At each step t → s:
#   - The model predicts tokens AND their confidence (softmax max probability).
#   - Among still-masked positions, sort by confidence (highest first).
#   - Decode the top-k most confident predictions (k proportional to step progress).
#   - Re-mask the rest so the model can revise them with more context next step.
#
# This produces better output because uncertain predictions are deferred until
# the model has more surrounding context to condition on.
#
# Two display modes are supported:
#   - "decoding": shows the response as it gradually fills in (tokens appear over time).
#   - "correction": shows raw model predictions at every position, including masked ones,
#     so you can watch the model "change its mind" as more context is decoded.
# =============================================================================
@torch.no_grad()
def low_confidence_remasking_inference(
    model, prompt_ids=None,
    sampling_steps=128,
    max_length=MAX_SEQ_LEN,
    show_each_step=True,
    show_mode="decoding",  # "decoding" or "correction"
):

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    device = get_device()

    if prompt_ids is None:
        prompt_ids = []
    prompt_len = len(prompt_ids)
    answer_length = max_length - prompt_len

    # Initialize the response region as all [MASK] tokens.
    response = [tokenizer.mask_token_id] * answer_length

    # Track the previous step's raw predictions so we can count how many
    # still-masked positions changed their prediction ("corrections").
    # This is purely for display purposes in "correction" mode.
    prev_correction = None

    for step in range(sampling_steps):
        # Linear noise schedule: t goes from 1.0 → 0.0, s = t - 1/T.
        t = 1.0 - step * (1.0 / sampling_steps)
        s = t - (1.0 / sampling_steps)

        # Run the model on prompt + current response state.
        full_input = prompt_ids + response.copy()
        logits = model(torch.tensor([full_input]).to(device))

        # Convert logits to probabilities and extract both the predicted token
        # and its confidence (max softmax probability) for every position.
        probs = torch.softmax(logits[0], dim=-1)             # (seq_len, vocab_size)
        confidences, predictions = probs.max(dim=-1)          # (seq_len,)

        # Identify which positions in the response are still masked.
        # These are the candidates for decoding or re-masking this step.
        masked_indices = [
            i for i in range(answer_length)
            if response[i] == tokenizer.mask_token_id
        ]
        num_masked = len(masked_indices)

        # Early exit: all positions already decoded.
        if num_masked == 0:
            break

        # How many tokens to commit this step.
        # We decode a fraction proportional to progress: (1 - s/t) of remaining masked.
        # At step 1 (t≈1, s≈1-1/T) we decode ~1/T of tokens; at the last step we decode all.
        # max(1, ...) ensures we always make at least some progress.
        num_to_decode = max(1, round(num_masked * (1.0 - s / t)))

        # Sort masked positions by prediction confidence (highest first).
        # We'll keep the top-k most confident and re-mask the uncertain ones.
        masked_conf = [
            (i, confidences[prompt_len + i].item()) for i in masked_indices
        ]
        masked_conf.sort(key=lambda x: x[1], reverse=True)

        # Set of positions to decode (keep prediction) this step.
        decode_indices = {i for i, _ in masked_conf[:num_to_decode]}

        # Extract argmax predictions for the response region.
        pred_ids = predictions[prompt_len:].tolist()

        # ---- Correction view (display only) ----
        # Build a "what the model thinks right now" view: confirmed tokens plus
        # raw predictions at still-masked positions. This lets us observe the model
        # revising its guesses over time as more context becomes available.
        if show_each_step and show_mode == "correction":
            correction_view = []
            for i in range(answer_length):
                if response[i] != tokenizer.mask_token_id:
                    correction_view.append(response[i])          # already confirmed
                else:
                    correction_view.append(pred_ids[i])          # model's current guess

            # Count how many still-masked positions changed prediction since last step.
            # A high change count means the model is still actively revising its output.
            num_changed = 0
            if prev_correction is not None:
                for i in range(answer_length):
                    if response[i] == tokenizer.mask_token_id:
                        if correction_view[i] != prev_correction[i]:
                            num_changed += 1
            prev_correction = correction_view.copy()

        # ---- Apply the masking decision ----
        # For each position in the response, decide what the new state will be:
        #   - Already decoded → keep the committed token (frozen).
        #   - Masked + in decode_indices → keep the model's prediction (newly decoded).
        #   - Masked + not in decode_indices → re-mask (model was uncertain, try again later).
        for i in range(answer_length):
            if response[i] != tokenizer.mask_token_id:
                pred_ids[i] = response[i]           # already decoded → keep
            elif i in decode_indices:
                pass                                  # newly decoded → keep prediction
            else:
                pred_ids[i] = tokenizer.mask_token_id  # uncertain → re-mask for next step

        response = pred_ids

        # ---- Visualize the current step ----
        if show_each_step:
            time.sleep(0.05)
            # Clear the terminal screen (ANSI escape) for animated output.
            print("\033[H\033[J", end="", flush=True)

            # Display header with the prompt.
            mode_label = "CORRECTION (raw predictions)" if show_mode == "correction" else "DECODING (after masking)"
            # print(f"{'='*60}")
            # # print(f"  Step {step+1}/{sampling_steps}  [{mode_label}]")
            # print(f"{'='*60}")
            print(f"  Prompt : {tokenizer.decode(prompt_ids)}")

            if show_mode == "correction":
                # Show raw model predictions at every position, including masked ones.
                # This reveals how the model "changes its mind" over successive steps
                # as more surrounding tokens get decoded and provide additional context.
                pred_text = tokenizer.decode(correction_view, skip_special_tokens=False)
                pred_text = pred_text.replace(tokenizer.pad_token, " ")
                print(f"  Predict: {pred_text}")
                print(f"  Decode {num_to_decode} new  |  remask {num_masked - num_to_decode}  |  changed {num_changed}")
            else:
                # Show the response as it gradually fills in.
                # Masked positions are shown as "######" so you can see tokens appear over time.
                decoded_text = tokenizer.decode(response, skip_special_tokens=False)
                decoded_text = decoded_text.replace(tokenizer.mask_token, "######").replace(tokenizer.pad_token, " ")
                print()
                print(f"  : \n {decoded_text}")

            # print(f"  t={t:.3f}  s={s:.3f}  masked={num_masked}")
            # print("-" * 50)


    return tokenizer.decode(response)
