"""Data utilities for loading and preprocessing BIO-tagged biomarker data."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

LABELS: Tuple[str, ...] = ("O", "B-BIO", "I-BIO")
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}


@dataclass
class SentenceExample:
    sentence_id: str
    tokens: List[str]
    labels: List[str]


def fix_bio_sequence(labels: List[str]) -> List[str]:
    """Ensure valid BIO sequence: I-BIO must follow B-BIO or I-BIO."""
    fixed: List[str] = []
    prev = "O"
    for lab in labels:
        lab = (lab or "O").strip()
        if lab not in LABELS:
            lab = "O"
        if lab == "I-BIO" and prev not in {"B-BIO", "I-BIO"}:
            lab = "B-BIO"
        fixed.append(lab)
        prev = lab
    return fixed


def read_token_bio_csv(path: Path) -> List[SentenceExample]:
    """Read CSV with columns: sentence_id, token, BIO_tag."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    order: List[str] = []
    tok_map: Dict[str, List[str]] = {}
    lab_map: Dict[str, List[str]] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return []

        for row in reader:
            if not row or len(row) < 2:
                continue
            sid = row[0].strip()
            token = row[1] if len(row) > 1 else ""
            label = row[2].strip() if len(row) > 2 else "O"
            if not sid:
                continue
            if sid not in tok_map:
                order.append(sid)
                tok_map[sid] = []
                lab_map[sid] = []
            tok_map[sid].append(token)
            lab_map[sid].append(label)

    examples = []
    for sid in order:
        tokens = tok_map[sid]
        labels = fix_bio_sequence(lab_map[sid])
        if tokens:
            examples.append(SentenceExample(sid, tokens, labels))
    return examples


def train_val_split(
    examples: List[SentenceExample], val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[SentenceExample], List[SentenceExample]]:
    """Split examples into train and validation sets."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def get_shape_feature(token: str) -> str:
    """Extract shape feature for a token (useful for genes/proteins)."""
    shape = []
    for c in token:
        if c.isupper():
            shape.append("X")
        elif c.islower():
            shape.append("x")
        elif c.isdigit():
            shape.append("d")
        else:
            shape.append(c)
    # Collapse consecutive same chars
    collapsed = []
    for s in shape:
        if not collapsed or collapsed[-1] != s:
            collapsed.append(s)
    return "".join(collapsed)


def get_prefix_suffix(token: str, n: int = 3) -> Tuple[str, str]:
    """Get prefix and suffix of length n."""
    return token[:n].lower(), token[-n:].lower()
