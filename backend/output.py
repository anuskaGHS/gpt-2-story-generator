from typing import Any, Dict, List
from backend.generator import generate_story


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count and return the number of tokens in the given text using tokenizer.encode."""
    # Step 1: Encode the text into token IDs and return the total count
    token_ids = tokenizer.encode(text)
    return len(token_ids)


def get_token_view(text: str, tokenizer: Any) -> List[str]:
    """Break text into individual token strings for the token-view expander."""
    # Step 1: Use tokenizer.tokenize to obtain the list of individual token strings
    return tokenizer.tokenize(text)


def unique_word_ratio(text: str) -> float:
    """Calculate the ratio of distinct lowercase words to total words, rounded to 2 decimals."""
    # Step 1: Split the text into individual words
    words = text.split()

    # Step 2: Return 0.0 if the text has no words
    if not words:
        return 0.0

    # Step 3: Convert all words to lowercase and collect unique words
    unique_words = set(word.lower() for word in words)

    # Step 4: Divide distinct words by total words and round to 2 decimal places
    return round(len(unique_words) / len(words), 2)


def compare_temperatures(
    prompt: str,
    model: Any,
    tokenizer: Any,
    temp1: float,
    temp2: float,
    max_new_tokens: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate and compare stories produced at two different temperature values using the same seed."""
    # Step 1: Generate the first story continuation using temperature 1
    story1 = generate_story(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temp1,
        seed=seed,
    )

    # Step 2: Generate the second story continuation using temperature 2 with the exact same seed
    story2 = generate_story(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temp2,
        seed=seed,
    )

    # Step 3: Build and return dictionary containing stories, token counts, and diversity metrics
    return {
        "temp1": {
            "temperature": temp1,
            "story": story1,
            "generated_tokens": count_tokens(story1, tokenizer),
            "unique_word_ratio": unique_word_ratio(story1),
        },
        "temp2": {
            "temperature": temp2,
            "story": story2,
            "generated_tokens": count_tokens(story2, tokenizer),
            "unique_word_ratio": unique_word_ratio(story2),
        },
    }

