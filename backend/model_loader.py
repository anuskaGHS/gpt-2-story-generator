from typing import Tuple
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model_and_tokenizer(model_name: str = "gpt2") -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load and return the pre-trained GPT-2 model and tokenizer from Hugging Face."""
    # Step 1: Load the tokenizer for converting text into tokens (numbers)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Step 2: Set the pad token to the eos token because GPT-2 does not have a default pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Load the pre-trained GPT-2 model with weights
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Step 4: Move the model to GPU if available for faster execution, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Step 5: Put the model in evaluation mode to turn off dropout during inference
    model.eval()

    # Step 6: Return both the loaded model and tokenizer
    return model, tokenizer

