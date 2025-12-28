"""Advanced BioBERT + Char-CNN + POS + CRF trainer for robust biomarker NER."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from data_utils import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    read_token_bio_csv,
    train_val_split,
)
from model_advanced import (
    BioBertNerAdvanced,
    POS2ID,
    chars_to_ids,
)

DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.2"


class NerDataset(Dataset):
    """Dataset for advanced NER model with char and POS features."""

    def __init__(
        self,
        examples,
        tokenizer,
        pos_tagger=None,
        max_length: int = 128,
        max_char_len: int = 20,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.pos_tagger = pos_tagger
        self.max_length = max_length
        self.max_char_len = max_char_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex.tokens
        labels = ex.labels

        # Tokenize with BERT
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()

        # Character IDs for each word
        char_ids = [chars_to_ids(t, self.max_char_len) for t in tokens]
        # Pad to max words
        max_words = min(len(tokens), self.max_length)
        while len(char_ids) < max_words:
            char_ids.append([0] * self.max_char_len)
        char_ids = char_ids[:max_words]

        # POS tags
        if self.pos_tagger is not None:
            doc = self.pos_tagger(" ".join(tokens))
            pos_ids = [POS2ID.get(token.pos_, POS2ID["OTHER"]) for token in doc]
        else:
            pos_ids = [POS2ID["OTHER"]] * len(tokens)
        while len(pos_ids) < max_words:
            pos_ids.append(0)
        pos_ids = pos_ids[:max_words]

        # Labels (word-level)
        label_ids = [LABEL2ID[l] for l in labels]
        while len(label_ids) < max_words:
            label_ids.append(0)
        label_ids = label_ids[:max_words]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "word_ids": word_ids,
        }


def collate_fn(batch):
    """Custom collate function to handle word_ids and pad variable-length tensors."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    word_ids_list = [b["word_ids"] for b in batch]
    
    # Find max word length in batch for padding
    max_words = max(b["char_ids"].size(0) for b in batch)
    max_char_len = batch[0]["char_ids"].size(1)
    
    # Pad char_ids, pos_ids, and labels to same length
    char_ids_list = []
    pos_ids_list = []
    labels_list = []
    
    for b in batch:
        char_len = b["char_ids"].size(0)
        
        # Pad char_ids
        if char_len < max_words:
            pad = torch.zeros(max_words - char_len, max_char_len, dtype=torch.long)
            char_ids_padded = torch.cat([b["char_ids"], pad], dim=0)
        else:
            char_ids_padded = b["char_ids"]
        char_ids_list.append(char_ids_padded)
        
        # Pad pos_ids
        if b["pos_ids"].size(0) < max_words:
            pad = torch.zeros(max_words - b["pos_ids"].size(0), dtype=torch.long)
            pos_ids_padded = torch.cat([b["pos_ids"], pad], dim=0)
        else:
            pos_ids_padded = b["pos_ids"]
        pos_ids_list.append(pos_ids_padded)
        
        # Pad labels
        if b["labels"].size(0) < max_words:
            pad = torch.zeros(max_words - b["labels"].size(0), dtype=torch.long)
            labels_padded = torch.cat([b["labels"], pad], dim=0)
        else:
            labels_padded = b["labels"]
        labels_list.append(labels_padded)
    
    char_ids = torch.stack(char_ids_list)
    pos_ids = torch.stack(pos_ids_list)
    labels = torch.stack(labels_list)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "char_ids": char_ids,
        "pos_ids": pos_ids,
        "labels": labels,
        "word_ids_list": word_ids_list,
    }


def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            char_ids = batch["char_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            labels = batch["labels"]
            word_ids_list = batch["word_ids_list"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                char_ids=char_ids,
                pos_ids=pos_ids,
                word_ids_list=word_ids_list,
            )

            predictions = outputs["predictions"]

            # Calculate actual sequence lengths from word_ids
            for i, (pred_seq, label_seq) in enumerate(zip(predictions, labels.tolist())):
                # Find actual sequence length from word_ids (count unique non-None word IDs)
                word_ids = word_ids_list[i]
                actual_len = max((wid for wid in word_ids if wid is not None), default=-1) + 1
                
                # Ensure we don't exceed available predictions/labels
                actual_len = min(actual_len, len(pred_seq), len(label_seq))
                
                if actual_len > 0:
                    pred_labels = [ID2LABEL[p] for p in pred_seq[:actual_len]]
                    true_labels = [ID2LABEL[l] for l in label_seq[:actual_len]]
                    all_preds.append(pred_labels)
                    all_labels.append(true_labels)

    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    return {"f1": f1, "report": report}


def main():
    parser = argparse.ArgumentParser(description="Train advanced BioBERT+CRF NER model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/biomarkers-training/bio_tags_output_27DEC.csv",
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, default="./output_advanced")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_crf", action="store_true", default=True)
    parser.add_argument("--use_char_cnn", action="store_true", default=True)
    parser.add_argument("--use_pos", action="store_true", default=True)
    parser.add_argument("--no_crf", action="store_true")
    parser.add_argument("--no_char_cnn", action="store_true")
    parser.add_argument("--no_pos", action="store_true")
    args = parser.parse_args()

    # Override flags
    use_crf = not args.no_crf
    use_char_cnn = not args.no_char_cnn
    use_pos = not args.no_pos and SPACY_AVAILABLE

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path
    print(f"Loading data from {data_path}")
    examples = read_token_bio_csv(data_path)
    print(f"Loaded {len(examples)} sentences")

    train_examples, val_examples = train_val_split(
        examples, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load POS tagger
    pos_tagger = None
    if use_pos:
        try:
            pos_tagger = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            print("Loaded spaCy POS tagger")
        except OSError:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            print("Continuing without POS features...")
            use_pos = False

    # Create datasets
    train_ds = NerDataset(train_examples, tokenizer, pos_tagger, args.max_length)
    val_ds = NerDataset(val_examples, tokenizer, pos_tagger, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model
    print(f"Creating model with CRF={use_crf}, CharCNN={use_char_cnn}, POS={use_pos}")
    model = BioBertNerAdvanced(
        bert_model_name=args.model_name,
        num_labels=len(LABELS),
        use_crf=use_crf,
        use_char_cnn=use_char_cnn,
        use_pos=use_pos,
    )
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            char_ids = batch["char_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            labels = batch["labels"].to(device)
            word_ids_list = batch["word_ids_list"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                char_ids=char_ids,
                pos_ids=pos_ids,
                labels=labels,
                word_ids_list=word_ids_list,
            )

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Evaluate
        metrics = evaluate(model, val_loader, device)
        print(f"Validation F1: {metrics['f1']:.4f}")

        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"Saved best model with F1: {best_f1:.4f}")

    # Save final model and config
    final_path = output_dir / "final_model"
    final_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_path / "model.pt")
    tokenizer.save_pretrained(str(final_path))

    config = {
        "model_name": args.model_name,
        "labels": list(LABELS),
        "max_length": args.max_length,
        "use_crf": use_crf,
        "use_char_cnn": use_char_cnn,
        "use_pos": use_pos,
        "best_f1": best_f1,
    }
    with open(final_path / "ner_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. Best F1: {best_f1:.4f}")
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
