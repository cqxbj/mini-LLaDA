from datasets import load_from_disk
from helper import load_model, validate_config
from inference import random_remasking_inference, low_confidence_remasking_inference
from config import MAX_SEQ_LEN, PROMPT_LEN, CHECKPOINT_DIR, DATA_DIR
import os


def main():
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "final_model-1.29.pth")
    model, _ = load_model(checkpoint_path=checkpoint_path)

    ds = load_from_disk(os.path.join(DATA_DIR, "tokenized_tinystories_dataset"))

    test_index = 99
    prompts = [
        ds[test_index]["input_ids"][:PROMPT_LEN].tolist(),
    ]

    for i, prompt_ids in enumerate(prompts):
        # Generate with low-confidence remasking (better quality)
        # show_mode: "decoding" = gradual reveal (###### → text)
        #            "correction" = raw predictions before masking (model "changing its mind")
        generated = low_confidence_remasking_inference(
            model, 
            prompt_ids=prompt_ids,
            sampling_steps = 100, 
            max_length=MAX_SEQ_LEN,
            show_each_step=True,
            show_mode="decoding",  
        )


if __name__ == "__main__":
    validate_config()
    main()
