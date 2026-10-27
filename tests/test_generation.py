import pytest
import torch
from backend.model_loader import load_model_and_tokenizer
from backend.generator import generate_story
from backend.preprocessing import build_prompt
from backend.output import count_tokens, get_token_view, unique_word_ratio


# Fixture to load the model and tokenizer only once for all tests
@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer once for the entire test module to save time and memory."""
    return load_model_and_tokenizer()


# ==========================================
# A) Tokenization
# ==========================================

# 1. Proves that token encoding and subsequent decoding is lossless for text
def test_encode_decode_roundtrip(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    original_sentence = "The lighthouse keeper heard a knock at the door."
    token_ids = tokenizer.encode(original_sentence)
    reconstructed_sentence = tokenizer.decode(token_ids)
    assert reconstructed_sentence == original_sentence


# 2. Proves that get_token_view and count_tokens return consistent token lengths
def test_token_view_length_matches_count_tokens(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    text = "A mystery story. The door opened slowly."
    token_strings = get_token_view(text, tokenizer)
    total_token_count = count_tokens(text, tokenizer)
    assert len(token_strings) == total_token_count


# 3. Proves GPT-2 tokenizes words with spaces using the special 'Ġ' character marker
def test_leading_space_token_marker(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    tokens = tokenizer.tokenize("Hello world")
    assert tokens[1].startswith("Ġ")


# ==========================================
# B) Autoregressive generation
# ==========================================

# 4. Proves greedy generation preserves the prompt token prefix and produces bounded new tokens
def test_greedy_generation_prefix_and_length(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    inputs = tokenizer("The cat sat on the", return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    total_len = output_ids.shape[-1]
    # First part of output ids equals prompt ids
    assert torch.equal(output_ids[0, :prompt_len], inputs["input_ids"][0])
    # Total output length is longer than prompt but at most prompt + 10
    assert total_len > prompt_len
    assert total_len <= prompt_len + 10


# 5. Proves model.generate matches a manual token-by-token argmax autoregressive loop
def test_manual_loop_matches_greedy_generate(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    inputs = tokenizer("Once upon a", return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    # Run standard greedy generate for 5 new tokens
    with torch.no_grad():
        expected_ids = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
    expected_new_tokens = expected_ids[0, prompt_len:]

    # Run manual step-by-step autoregressive loop for 5 steps
    curr_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(5):
            outputs = model(input_ids=curr_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token_id], dim=-1)
    manual_new_tokens = curr_ids[0, prompt_len:]

    assert torch.equal(manual_new_tokens, expected_new_tokens)


# 6. Proves generate_story slices off prompt tokens and returns only the newly generated continuation
def test_generate_story_returns_continuation_only(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "A mystery story. The door opened."
    story = generate_story(prompt, model, tokenizer, max_new_tokens=20, seed=42)
    assert isinstance(story, str) and len(story.strip()) > 0
    assert not story.strip().startswith(prompt)


# ==========================================
# C) Model parameters
# ==========================================

# 7. Proves GPT-2 (small) contains approximately 124M total parameters (between 120M and 130M)
def test_model_total_parameter_count(model_and_tokenizer):
    model, _ = model_and_tokenizer
    total_params = sum(p.numel() for p in model.parameters())
    assert 120_000_000 <= total_params <= 130_000_000


# 8. Proves model architecture dimensions match GPT-2 small specifications (12 layers, 12 heads, 768 emb, 50257 vocab)
def test_model_architecture_config(model_and_tokenizer):
    model, _ = model_and_tokenizer
    assert model.config.n_layer == 12
    assert model.config.n_head == 12
    assert model.config.n_embd == 768
    assert model.config.vocab_size == 50257


# 9. Proves the loaded model is in evaluation mode with training disabled
def test_model_is_in_eval_mode(model_and_tokenizer):
    model, _ = model_and_tokenizer
    assert model.training is False


# ==========================================
# D) How sampling affects text
# ==========================================

# 10. Proves setting identical seeds and temperature guarantees deterministic output
def test_reproducibility_with_same_seed(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "The secret chamber was hidden"
    story1 = generate_story(prompt, model, tokenizer, max_new_tokens=20, temperature=0.7, seed=42)
    story2 = generate_story(prompt, model, tokenizer, max_new_tokens=20, temperature=0.7, seed=42)
    assert story1 == story2


# 11. Proves changing random seeds produces different text under temperature sampling
def test_different_seeds_produce_different_text(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "The secret chamber was hidden"
    story1 = generate_story(prompt, model, tokenizer, max_new_tokens=20, temperature=1.0, seed=1)
    story2 = generate_story(prompt, model, tokenizer, max_new_tokens=20, temperature=1.0, seed=2)
    assert story1 != story2


# 12. Proves lower temperature sharpens probability distribution by increasing top token probability
def test_temperature_sharpens_probability_distribution(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    inputs = tokenizer("The sky is", return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    probs_low = torch.softmax(logits / 0.3, dim=-1)
    probs_high = torch.softmax(logits / 1.5, dim=-1)
    assert torch.max(probs_low).item() > torch.max(probs_high).item()


# ==========================================
# E) Helpers
# ==========================================

# 13. Proves build_prompt attaches the correct prefix and rejects empty input sentences
def test_build_prompt_prefix_and_validation():
    prompt = build_prompt("The detective looked around.", "Mystery")
    assert prompt == "A mystery story. The detective looked around."

    with pytest.raises(ValueError):
        build_prompt("   ", "Mystery")


# 14. Proves unique_word_ratio accurately computes lexical diversity and handles empty text
def test_unique_word_ratio():
    assert unique_word_ratio("a b a") == 0.67
    assert unique_word_ratio("") == 0.0
