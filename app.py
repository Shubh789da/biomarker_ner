"""Streamlit app for biomarker NER prediction using trained model."""
import re
import sys
from pathlib import Path

import streamlit as st
import torch

#sys.path.append(str(Path(__file__).parent.parent / "model_training"))

from data_utils import ID2LABEL, LABEL2ID, LABELS
from model_advanced import BioBertNerAdvanced, POS2ID, chars_to_ids

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import AutoTokenizer
except ImportError:
    st.error("Please install transformers: pip install transformers torch")
    st.stop()

st.set_page_config(
    page_title="Biomarker NER",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model_from_hf(repo_name: str, token: str = None):
    """Load model from Hugging Face Hub."""
    from huggingface_hub import snapshot_download
    
    with st.spinner(f"Downloading model from {repo_name}..."):
        model_path = snapshot_download(
            repo_name,
            token=token,
            cache_dir=".cache/huggingface"
        )
    return load_model_local(model_path)


@st.cache_resource
def load_model_local(model_path: str):
    """Load model from local path."""
    import json
    
    model_path = Path(model_path)
    config_path = model_path / "ner_config.json"
    
    if not config_path.exists():
        st.error(f"Config file not found: {config_path}")
        st.stop()
    
    with open(config_path) as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Load model
    model = BioBertNerAdvanced(
        bert_model_name=config["model_name"],
        num_labels=len(LABELS),
        use_crf=config.get("use_crf", True),
        use_char_cnn=config.get("use_char_cnn", True),
        use_pos=config.get("use_pos", True),
    )
    
    state_dict = torch.load(model_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load POS tagger if needed
    pos_tagger = None
    if config.get("use_pos", True) and SPACY_AVAILABLE:
        try:
            pos_tagger = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            st.warning("spaCy model not found. POS features disabled.")
    
    return model, tokenizer, config, pos_tagger, device


def tokenize_text(text: str):
    """Simple tokenization with punctuation handling."""
    tokens = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text)
    return tokens


def predict_biomarkers(model, tokenizer, tokens, pos_tagger, device, config):
    """Run inference on tokens."""
    max_length = config.get("max_length", 128)
    max_char_len = 20
    
    # Tokenize with BERT
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    word_ids = encoding.word_ids()
    
    # Character IDs
    char_ids = [chars_to_ids(t, max_char_len) for t in tokens]
    max_words = min(len(tokens), max_length)
    while len(char_ids) < max_words:
        char_ids.append([0] * max_char_len)
    char_ids = char_ids[:max_words]
    
    # POS IDs
    if pos_tagger is not None:
        doc = pos_tagger(" ".join(tokens))
        pos_ids = [POS2ID.get(token.pos_, POS2ID["OTHER"]) for token in doc]
    else:
        pos_ids = [POS2ID["OTHER"]] * len(tokens)
    while len(pos_ids) < max_words:
        pos_ids.append(0)
    pos_ids = pos_ids[:max_words]
    
    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    char_ids_tensor = torch.tensor([char_ids], dtype=torch.long, device=device)
    pos_ids_tensor = torch.tensor([pos_ids], dtype=torch.long, device=device)
    
    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            char_ids=char_ids_tensor,
            pos_ids=pos_ids_tensor,
            word_ids_list=[word_ids],
        )
    
    predictions = outputs["predictions"][0]
    results = [
        (tokens[i], ID2LABEL[predictions[i]])
        for i in range(min(len(tokens), len(predictions)))
    ]
    
    return results


def extract_biomarkers(predictions):
    """Extract biomarker entities from BIO predictions."""
    entities = []
    current_entity = []
    
    for token, label in predictions:
        if label == "B-BIO":
            if current_entity:
                entities.append(" ".join(current_entity))
            current_entity = [token]
        elif label == "I-BIO" and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    
    if current_entity:
        entities.append(" ".join(current_entity))
    
    return entities


def highlight_biomarkers(text: str, biomarkers: list[str]) -> str:
    """Highlight biomarkers in text with HTML."""
    highlighted = text
    for biomarker in sorted(biomarkers, key=len, reverse=True):
        pattern = re.compile(re.escape(biomarker), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{biomarker}</span>',
            highlighted
        )
    return highlighted


def main():
    st.title("🧬 Biomarker Named Entity Recognition")
    st.markdown("Extract biomarkers from clinical text using BioBERT + Char-CNN + POS + CRF")
    
    # Auto-load model from Hugging Face
    if "model_data" not in st.session_state:
        with st.spinner("Loading model from Hugging Face..."):
            try:
                st.session_state.model_data = load_model_from_hf("Postlyt/biomarker-ner-advanced", None)
                st.success("✓ Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()
    
    model, tokenizer, config, pos_tagger, device = st.session_state.model_data
    
    # Display model info
    with st.sidebar.expander("Model Info"):
        st.write(f"**Base Model:** {config['model_name']}")
        st.write(f"**CRF Layer:** {config.get('use_crf', False)}")
        st.write(f"**Char-CNN:** {config.get('use_char_cnn', False)}")
        st.write(f"**POS Features:** {config.get('use_pos', False)}")
        st.write(f"**Device:** {device}")
        if "best_f1" in config:
            st.write(f"**Best F1 Score:** {config['best_f1']:.4f}")
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Input Text")
        # Pre-fill with selected example if available
        default_value = st.session_state.get("selected_example", "")
        if "selected_example" in st.session_state:
            del st.session_state.selected_example
        
        text_input = st.text_area(
            "Enter clinical text:",
            value=default_value,
            height=200,
            placeholder="Example: Her HER2 and IL-6 levels are elevated, indicating potential therapeutic targets.",
            help="Enter any clinical text containing biomarker mentions"
        )
        
        analyze_button = st.button("🔍 Analyze Text", type="primary", width="stretch")
    
    with col2:
        st.subheader("Quick Examples")
        examples = [
            "The IL-6, HER2, and EGFR biomarkers showed significant changes",
            "Increased insulin sensitivity index was observed",
            "Patient has high glucose and elevated hemoglobin A1c",
            "L-dopa onset time improved after treatment",
            "Her HER2 levels are elevated",
            "PD-L1 expression was high in tumor cells"
        ]
        
        for i, example in enumerate(examples):
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.selected_example = example
                st.rerun()
    
    if analyze_button:
        if not text_input.strip():
            st.warning("Please enter some text to analyze")
            st.stop()
        
        with st.spinner("Analyzing text..."):
            # Tokenize
            tokens = tokenize_text(text_input)
            
            # Predict
            predictions = predict_biomarkers(
                model, tokenizer, tokens, pos_tagger, device, config
            )
            
            # Extract biomarkers
            biomarkers = extract_biomarkers(predictions)
        
        # Display results
        st.markdown("---")
        st.subheader("Results")
        
        if biomarkers:
            st.success(f"Found {len(biomarkers)} biomarker(s)")
            
            # Highlighted text
            st.markdown("### Annotated Text")
            highlighted_html = highlight_biomarkers(text_input, biomarkers)
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            # Biomarker list
            st.markdown("### Extracted Biomarkers")
            cols = st.columns(min(len(biomarkers), 4))
            for i, biomarker in enumerate(biomarkers):
                with cols[i % len(cols)]:
                    st.markdown(f"**{i+1}.** `{biomarker}`")
            
            # Token-level predictions
            with st.expander("View Token-level Predictions"):
                import pandas as pd
                
                df = pd.DataFrame(predictions, columns=["Token", "Label"])
                df_colored = df.style.apply(
                    lambda x: ['background-color: #ffeb3b' if v in ['B-BIO', 'I-BIO'] else '' 
                              for v in x], 
                    subset=['Label']
                )
                st.dataframe(df_colored, width="stretch")
        else:
            st.info("No biomarkers detected in the text")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with BioBERT + Char-CNN + POS + CRF | "
        "[Model Training Code](https://github.com/Shubh789da/biomarker_ner) | "
        "[Hugging Face Model](https://huggingface.co/Postlyt/biomarker-ner-advanced)"
    )


if __name__ == "__main__":
    main()
