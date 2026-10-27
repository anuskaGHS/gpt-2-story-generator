import streamlit as st

# Import backend modules for model loading, text preprocessing, and output evaluation
from backend.model_loader import load_model_and_tokenizer
from backend.preprocessing import get_style_names, build_prompt
from backend.output import count_tokens, get_token_view, compare_temperatures

# Step 1: Configure page title and use wide layout for clean side-by-side comparison
st.set_page_config(page_title="Once Upon a Token", layout="wide")

# Step 2: Display main application title and introductory caption
st.title("Once Upon a Token")
st.caption(
    "A GPT-2 story generator. Write an opening sentence, pick a style, and compare how "
    "temperature changes what GPT-2 writes. The model runs on this computer through "
    "Hugging Face Transformers. No API calls."
)


# Step 3: Cache the model and tokenizer so they load once into memory across user interactions
@st.cache_resource
def get_model_and_tokenizer():
    """Load and cache the GPT-2 model and tokenizer."""
    return load_model_and_tokenizer()


# Load the model and tokenizer
model, tokenizer = get_model_and_tokenizer()

# Step 4: Create the sidebar controls for model and generation settings
st.sidebar.header("Story settings")
st.sidebar.caption("Model: gpt2 (small), loaded locally")

style = st.sidebar.selectbox("Writing style", options=get_style_names())
temp_a = st.sidebar.slider("Temperature A", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
temp_b = st.sidebar.slider("Temperature B", min_value=0.1, max_value=1.5, value=1.3, step=0.1)
max_new_tokens = st.sidebar.slider("Max new tokens", min_value=30, max_value=200, value=100, step=10)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
st.sidebar.caption("Both temperature runs use this seed, so the only difference between them is temperature.")

# Step 5: Main area input for user opening sentence and generate button
opening_sentence = st.text_area(
    "Opening sentence",
    value="The lighthouse keeper heard a knock at the door at midnight.",
)

generate_btn = st.button("Generate story", type="primary")

# Step 6: When generate button is clicked, validate input and run story generation
if generate_btn:
    # Build prompt with selected style prefix and catch validation errors
    try:
        prompt = build_prompt(opening_sentence, style)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    # Generate stories at both temperatures inside a spinner
    with st.spinner("Generating stories..."):
        result = compare_temperatures(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            temp1=temp_a,
            temp2=temp_b,
            max_new_tokens=max_new_tokens,
            seed=int(seed),
        )

    # Save outputs to session state so they persist across widget interactions
    st.session_state["saved_result"] = result
    st.session_state["saved_sentence"] = opening_sentence
    st.session_state["saved_prompt"] = prompt

# Step 7: Display results if a previously generated story exists in session state
if "saved_result" in st.session_state:
    result = st.session_state["saved_result"]
    saved_sentence = st.session_state["saved_sentence"]
    saved_prompt = st.session_state["saved_prompt"]

    # Combine opening sentence with story continuation from Temperature A
    full_story_a = f"{saved_sentence.strip()} {result['temp1']['story'].strip()}"
    full_story_b = f"{saved_sentence.strip()} {result['temp2']['story'].strip()}"

    # 1) Your story section
    st.header("Your story")
    with st.container(border=True):
        st.write(full_story_a)

    # Display key token metrics in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    prompt_tokens = count_tokens(saved_prompt, tokenizer)
    gen_tokens = result["temp1"]["generated_tokens"]
    total_tokens = prompt_tokens + gen_tokens

    col1.metric("Prompt tokens", prompt_tokens)
    col2.metric("Generated tokens", gen_tokens)
    col3.metric("Total tokens", total_tokens)
    col4.metric("Temperature", result["temp1"]["temperature"])

    # 2) Temperature comparison section
    st.header("Temperature comparison")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.subheader(f"Temperature A: {result['temp1']['temperature']}")
        with st.container(border=True):
            st.write(full_story_a)
        pct1 = int(round(result["temp1"]["unique_word_ratio"] * 100))
        st.caption(f"Generated tokens: {result['temp1']['generated_tokens']} | Unique words: {pct1}%")

    with comp_col2:
        st.subheader(f"Temperature B: {result['temp2']['temperature']}")
        with st.container(border=True):
            st.write(full_story_b)
        pct2 = int(round(result["temp2"]["unique_word_ratio"] * 100))
        st.caption(f"Generated tokens: {result['temp2']['generated_tokens']} | Unique words: {pct2}%")

    # 3) How your prompt was built section
    with st.expander("How your prompt was built", expanded=True):
        st.write("GPT-2 receives the style prefix followed by your opening sentence. It is then split into tokens.")
        st.code(saved_prompt)
        prompt_tokens_list = get_token_view(saved_prompt, tokenizer)
        st.code(" | ".join(prompt_tokens_list))

