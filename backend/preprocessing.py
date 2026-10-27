from typing import List

# Dictionary mapping friendly style names to short prompt prefixes
STYLE_PREFIXES = {
    "Mystery": "A mystery story.",
    "Fantasy": "A fantasy story.",
    "Horror": "A horror story.",
    "Sci-Fi": "A science fiction story.",
    "Romance": "A romance story.",
    "Fairy tale": "A fairy tale.",
}


def get_style_names() -> List[str]:
    """Return the list of available writing style names."""
    # Step 1: Return the dictionary keys as a list for the dropdown
    return list(STYLE_PREFIXES.keys())


def build_prompt(opening_sentence: str, style: str) -> str:
    """Combine a style prefix with the user's opening sentence into a single prompt."""
    # Step 1: Strip extra whitespace from the user opening sentence
    cleaned_sentence = opening_sentence.strip()

    # Step 2: Validate that the sentence is not empty
    if not cleaned_sentence:
        raise ValueError("Opening sentence cannot be empty.")

    # Step 3: Look up the style prefix for the chosen style
    prefix = STYLE_PREFIXES.get(style, "")

    # Step 4: Return the prefix and sentence joined by a single space
    return f"{prefix} {cleaned_sentence}".strip()

