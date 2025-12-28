# Biomarker NER Streamlit App

Interactive web application for extracting biomarkers from clinical text using the trained BioBERT + Char-CNN + POS + CRF model.

## Features

- 🚀 Load model from local path or Hugging Face Hub
- 🧬 Real-time biomarker extraction
- 🎨 Highlighted text visualization
- 📊 Token-level predictions view
- 💡 Quick example texts

## Installation

```bash
cd streamlit_app
pip install -r requirements.txt

# Download spaCy model (required for POS features)
python -m spacy download en_core_web_sm
```

## Usage

### Option 1: Local Model

```bash
streamlit run app.py
```

Then in the sidebar:
1. Select "Local Path"
2. Enter path: `../model_training/output_advanced/final_model`
3. Click "Load Model"

### Option 2: Hugging Face Model

First, upload your model to HF Hub:
```bash
cd ..
python hugging_face/upload_to_hf.py \
    --model_path model_training/output_advanced/final_model \
    --repo_name "your-username/biomarker-ner-advanced" \
    --token "hf_..."
```

Then in the app:
1. Select "Hugging Face Hub"
2. Enter repo name: `your-username/biomarker-ner-advanced`
3. Enter HF token (if private)
4. Click "Load Model"

## Screenshots

The app provides:
- Text input area for clinical text
- Quick example buttons
- Highlighted biomarker visualization
- List of extracted biomarkers
- Token-level prediction table

## Example Queries

Try these examples:
- "Her HER2 levels are elevated"
- "The IL-6, HER2, and EGFR biomarkers showed significant changes"
- "Increased insulin sensitivity index was observed"
- "Patient has high glucose and elevated hemoglobin A1c"

## Model Architecture

The app uses the advanced model with:
- **BioBERT** contextual embeddings
- **Character-CNN** for gene name patterns
- **POS embeddings** for grammatical awareness
- **CRF layer** for valid BIO sequences

## Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set `streamlit_app/app.py` as the main file
5. Add secrets for HF token (if using private model)

### Deploy to Hugging Face Spaces

1. Create new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select Streamlit as SDK
3. Upload `app.py` and `requirements.txt`
4. Add model files or link to model repo

## Troubleshooting

**Issue**: spaCy model not found  
**Fix**: Run `python -m spacy download en_core_web_sm`

**Issue**: Model not loading from HF  
**Fix**: Check repo name and token. Make sure model was uploaded correctly.

**Issue**: CUDA out of memory  
**Fix**: The model will automatically use CPU if CUDA is not available.

## API Usage

For programmatic access, use the inference script instead:
```bash
cd ../model_training
python inference.py \
    --model_path ./output_advanced/final_model \
    --model_type advanced \
    --text "Your clinical text here"
```
