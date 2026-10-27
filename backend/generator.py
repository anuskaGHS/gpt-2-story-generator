from typing import Any
import torch


def generate_story(
    prompt: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    seed: int = 42,
) -> str:
    """Generate a story continuation from a prompt using GPT-2 with temperature sampling."""
    # Step 1: Set random seed for reproducibility so identical inputs produce the same story
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Step 2: Convert the prompt text into PyTorch input tensors and send them to the model's device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Step 3: Run autoregressive generation without gradient calculation to save memory and compute
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Step 4: Slice off the prompt token IDs so only newly generated continuation tokens remain
    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][prompt_length:]

    # Step 5: Decode the generated token IDs back to a readable string, omitting special tokens
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generated_text

