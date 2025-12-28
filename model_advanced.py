"""Advanced BioBERT + Char-CNN + POS + CRF model for robust biomarker NER."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

from data_utils import LABELS


class CharCNN(nn.Module):
    """Character-level CNN for capturing morphological features (gene names, casing)."""

    def __init__(
        self,
        char_vocab_size: int = 256,
        char_embed_dim: int = 30,
        num_filters: int = 50,
        kernel_sizes: tuple = (2, 3, 4),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (batch, seq_len, max_word_len)
        Returns:
            char_features: (batch, seq_len, output_dim)
        """
        batch, seq_len, max_word_len = char_ids.shape
        # Flatten to (batch * seq_len, max_word_len)
        char_ids = char_ids.view(-1, max_word_len)
        # Embed: (batch * seq_len, max_word_len, char_embed_dim)
        char_embeds = self.char_embedding(char_ids)
        # Conv expects (batch, channels, length)
        char_embeds = char_embeds.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(char_embeds))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        # Concat and reshape
        char_features = torch.cat(conv_outputs, dim=1)
        char_features = self.dropout(char_features)
        char_features = char_features.view(batch, seq_len, -1)
        return char_features


class BioBertNerAdvanced(nn.Module):
    """
    Advanced NER model combining:
    - BioBERT contextual embeddings
    - Character-level CNN (for gene names, casing, special chars)
    - POS tag embeddings (grammatical awareness)
    - CRF layer (valid BIO sequence constraints)
    """

    def __init__(
        self,
        bert_model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        num_labels: int = len(LABELS),
        num_pos_tags: int = 20,
        pos_embed_dim: int = 25,
        char_cnn_output_dim: int = 150,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_crf: bool = True,
        use_char_cnn: bool = True,
        use_pos: bool = True,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.use_char_cnn = use_char_cnn
        self.use_pos = use_pos

        # BioBERT encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size

        # Character CNN
        if use_char_cnn:
            self.char_cnn = CharCNN(num_filters=char_cnn_output_dim // 3)
            char_dim = self.char_cnn.output_dim
        else:
            char_dim = 0

        # POS embeddings
        if use_pos:
            self.pos_embedding = nn.Embedding(num_pos_tags, pos_embed_dim, padding_idx=0)
            pos_dim = pos_embed_dim
        else:
            pos_dim = 0

        # Combined feature dimension
        combined_dim = bert_hidden + char_dim + pos_dim

        # Projection layers
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(combined_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # CRF layer
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_ids: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        word_ids_list: list | None = None,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) - BERT token IDs
            attention_mask: (batch, seq_len)
            char_ids: (batch, word_seq_len, max_char_len) - character IDs per word
            pos_ids: (batch, word_seq_len) - POS tag IDs per word
            labels: (batch, word_seq_len) - BIO labels per word
            word_ids_list: list of word_ids for each example (for subword alignment)
        """
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden = bert_output.last_hidden_state  # (batch, seq_len, hidden)

        # If we have word-level features, we need to align BERT subwords to words
        # For simplicity, we'll use first-subword pooling
        if word_ids_list is not None:
            batch_size = bert_hidden.size(0)
            max_word_len = max(
                max(w for w in wids if w is not None) + 1 if any(w is not None for w in wids) else 1
                for wids in word_ids_list
            )
            
            # Pool BERT to word level (first subword)
            word_bert = torch.zeros(
                batch_size, max_word_len, bert_hidden.size(2),
                device=bert_hidden.device, dtype=bert_hidden.dtype
            )
            word_mask = torch.zeros(batch_size, max_word_len, device=bert_hidden.device)
            
            for b, wids in enumerate(word_ids_list):
                seen = set()
                for t, wid in enumerate(wids):
                    if wid is not None and wid not in seen:
                        word_bert[b, wid] = bert_hidden[b, t]
                        word_mask[b, wid] = 1
                        seen.add(wid)
        else:
            word_bert = bert_hidden
            word_mask = attention_mask.float()

        # Combine features
        features = [word_bert]

        if self.use_char_cnn and char_ids is not None:
            char_features = self.char_cnn(char_ids)
            # Pad/truncate to match word_bert
            if char_features.size(1) < word_bert.size(1):
                pad = torch.zeros(
                    char_features.size(0),
                    word_bert.size(1) - char_features.size(1),
                    char_features.size(2),
                    device=char_features.device,
                )
                char_features = torch.cat([char_features, pad], dim=1)
            elif char_features.size(1) > word_bert.size(1):
                char_features = char_features[:, :word_bert.size(1), :]
            features.append(char_features)

        if self.use_pos and pos_ids is not None:
            pos_embeds = self.pos_embedding(pos_ids)
            if pos_embeds.size(1) < word_bert.size(1):
                pad = torch.zeros(
                    pos_embeds.size(0),
                    word_bert.size(1) - pos_embeds.size(1),
                    pos_embeds.size(2),
                    device=pos_embeds.device,
                )
                pos_embeds = torch.cat([pos_embeds, pad], dim=1)
            elif pos_embeds.size(1) > word_bert.size(1):
                pos_embeds = pos_embeds[:, :word_bert.size(1), :]
            features.append(pos_embeds)

        # Concatenate all features
        combined = torch.cat(features, dim=-1)
        combined = self.dropout(combined)
        hidden = torch.relu(self.hidden_layer(combined))
        hidden = self.dropout(hidden)
        emissions = self.classifier(hidden)

        # Compute loss and predictions
        output = {"emissions": emissions}
        mask = word_mask.bool() if word_mask.dim() == 2 else attention_mask.bool()

        if self.use_crf:
            if labels is not None:
                # Truncate labels if needed
                if labels.size(1) > emissions.size(1):
                    labels = labels[:, :emissions.size(1)]
                elif labels.size(1) < emissions.size(1):
                    pad = torch.full(
                        (labels.size(0), emissions.size(1) - labels.size(1)),
                        0, device=labels.device, dtype=labels.dtype
                    )
                    labels = torch.cat([labels, pad], dim=1)
                
                # CRF loss (negative log-likelihood)
                loss = -self.crf(emissions, labels, mask=mask[:, :emissions.size(1)], reduction="mean")
                output["loss"] = loss
            
            # Decode best path
            predictions = self.crf.decode(emissions, mask=mask[:, :emissions.size(1)])
            output["predictions"] = predictions
        else:
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(emissions.view(-1, self.num_labels), labels.view(-1))
                output["loss"] = loss
            output["predictions"] = torch.argmax(emissions, dim=-1).tolist()

        return output


# POS tag mapping (simplified universal tags)
POS_TAGS = [
    "PAD", "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM",
    "CONJ", "PUNCT", "X", "PROPN", "PART", "INTJ", "SYM", "AUX", "SCONJ", "CCONJ", "OTHER"
]
POS2ID = {tag: i for i, tag in enumerate(POS_TAGS)}


def chars_to_ids(token: str, max_len: int = 20) -> list[int]:
    """Convert token characters to IDs (ASCII-based)."""
    ids = [min(ord(c), 255) for c in token[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids
