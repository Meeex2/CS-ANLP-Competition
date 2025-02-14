# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Imports

# %% [code] {"execution":{"iopub.status.busy":"2025-02-09T18:21:05.741382Z","iopub.execute_input":"2025-02-09T18:21:05.741752Z","iopub.status.idle":"2025-02-09T18:21:05.747286Z","shell.execute_reply.started":"2025-02-09T18:21:05.741723Z","shell.execute_reply":"2025-02-09T18:21:05.746144Z"},"jupyter":{"outputs_hidden":false}}
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Positional Encoding


# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-02-09T18:12:23.432769Z","iopub.execute_input":"2025-02-09T18:12:23.433674Z","iopub.status.idle":"2025-02-09T18:12:23.443314Z","shell.execute_reply.started":"2025-02-09T18:12:23.433634Z","shell.execute_reply":"2025-02-09T18:12:23.442066Z"}}
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): The dimension of the embeddings.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on
        # position and dimension.
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # In case of odd d_model, handle the last dimension separately.
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Tensor with positional encoding added (and dropout applied).
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Transformer Model


# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-02-09T18:12:25.648096Z","iopub.execute_input":"2025-02-09T18:12:25.648502Z","iopub.status.idle":"2025-02-09T18:12:25.658812Z","shell.execute_reply.started":"2025-02-09T18:12:25.648467Z","shell.execute_reply":"2025-02-09T18:12:25.657519Z"}}
class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        output_size,
        dropout=0.1,
        max_seq_length=5000,
        pretrained_embedding=None,
    ):
        """
        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            output_size (int): Number of classes for classification.
            dropout (float): Dropout rate.
            max_seq_length (int): Maximum sequence length (for positional encoding).
            pretrained_embedding (nn.Module): Pre-trained embedding layer.
        """
        super(TransformerModel, self).__init__()
        if pretrained_embedding is not None:
            for param in pretrained_embedding.parameters():
                param.requires_grad = False
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_length)

        # Use batch_first=True so that the transformer expects input as (batch_size, seq_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Enable batch-first mode
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, output_size)

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, seq_length) containing token IDs.
        Returns:
            Output logits of shape (batch_size, output_size).
        """
        # Create key padding mask: True for padding tokens (assuming padding id = 0)
        src_key_padding_mask = src == 0  # (batch_size, seq_length)

        x = self.embedding(src)  # (batch_size, seq_length, embedding_dim)
        x = self.pos_encoder(x)  # (batch_size, seq_length, embedding_dim)
        # With batch_first=True, no need to transpose the input.
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Use the first token (e.g., a [CLS] token) for classification.
        x = x[:, 0, :]  # (batch_size, embedding_dim)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Text Dataset


# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-02-09T18:12:27.634029Z","iopub.execute_input":"2025-02-09T18:12:27.634354Z","iopub.status.idle":"2025-02-09T18:12:27.641997Z","shell.execute_reply.started":"2025-02-09T18:12:27.634331Z","shell.execute_reply":"2025-02-09T18:12:27.641016Z"}}
class TextDataset(Dataset):
    def __init__(
        self, dataframe, label2idx, idx2label, tokenizer, max_length, augment=False
    ):
        """
        Args:
            csv_path (str): Path to the CSV file. The CSV should have columns 'Text' and 'Label'.
            tokenizer: A Huggingface tokenizer (e.g., BertTokenizer).
            max_length (int): Maximum tokenized sequence length.
            augment (bool): Whether to apply data augmentation.
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

        # Remove rows with missing labels and create a deterministic label mapping.
        self.data = self.data[self.data["Label"].notnull()]
        self.label2idx = label2idx
        self.idx2label = idx2label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["Text"]
        label = row["Label"]

        # Tokenize text without automatic truncation.
        token_ids = self.tokenizer.encode(
            text, add_special_tokens=True, truncation=False
        )

        # Data augmentation: take a random contiguous slice if augmentation is enabled
        if self.augment and len(token_ids) > self.max_length:
            start_idx = random.randint(0, len(token_ids) - self.max_length)
            token_ids = token_ids[start_idx : start_idx + self.max_length]
        else:
            token_ids = token_ids[: self.max_length]

        length = len(token_ids)
        label_idx = self.label2idx[label]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            length,
            torch.tensor(label_idx, dtype=torch.long),
        )


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Collate function


# %% [code] {"jupyter":{"source_hidden":true,"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-09T18:12:29.517007Z","iopub.execute_input":"2025-02-09T18:12:29.517306Z","iopub.status.idle":"2025-02-09T18:12:29.521903Z","shell.execute_reply.started":"2025-02-09T18:12:29.517285Z","shell.execute_reply":"2025-02-09T18:12:29.521027Z"}}
def collate_fn(batch):
    """
    Pads a batch of variable-length sequences.
    Args:
        batch: List of tuples (token_ids, length, label)
    Returns:
        padded_token_ids: Tensor of shape (batch_size, max_seq_length)
        lengths: Tensor of original sequence lengths
        labels: Tensor of labels
    """
    token_ids, lengths, labels = zip(*batch)
    padded_token_ids = torch.nn.utils.rnn.pad_sequence(
        token_ids, batch_first=True, padding_value=0
    )
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(labels)
    return padded_token_ids, lengths, labels


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Utils


# %% [code] {"jupyter":{"source_hidden":true,"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-09T18:12:31.049921Z","iopub.execute_input":"2025-02-09T18:12:31.050199Z","iopub.status.idle":"2025-02-09T18:12:31.057118Z","shell.execute_reply.started":"2025-02-09T18:12:31.050180Z","shell.execute_reply":"2025-02-09T18:12:31.056244Z"}}
def decode_token_ids(token_ids, tokenizer):
    """
    Decodes a list or tensor of token ids back to text.
    Args:
        token_ids: List or tensor of ints.
        tokenizer: A Huggingface tokenizer instance.
    Returns:
        Decoded string.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def format_time(seconds):
    """Convert time in seconds to a human-readable format (HH:MM:SS)."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def print_model_summary(model):
    # Calculate total and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print header information.
    separator = "=" * 80
    print(separator)
    print("MODEL ARCHITECTURE".center(80))
    print(separator)
    print(model)
    print(separator)
    print(f"Total Parameters    : {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(separator)
    print("Detailed Layer Information:".center(80))
    print(separator)

    # Build table data for each parameter.
    table_data = []
    for name, param in model.named_parameters():
        table_data.append([name, list(param.shape), param.numel(), param.requires_grad])

    headers = ["Layer (Parameter Name)", "Shape", "Param #", "Trainable"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print(separator)


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training Loop


# %% [code] {"jupyter":{"source_hidden":true,"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-09T18:12:33.783726Z","iopub.execute_input":"2025-02-09T18:12:33.784079Z","iopub.status.idle":"2025-02-09T18:12:33.792691Z","shell.execute_reply.started":"2025-02-09T18:12:33.784050Z","shell.execute_reply":"2025-02-09T18:12:33.791813Z"}}
def train_model(
    model,
    dataloader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    scheduler=None,
):  # UPDATED SIGNATURE
    for epoch in range(epochs):
        start_time = time.time()  # Record the start time of the epoch

        model.train()
        epoch_loss = 0.0
        n = 0

        # Training phase
        for inputs, lengths, targets in dataloader:
            n += 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            avg_loss = epoch_loss / n

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, lengths, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute validation metrics
        val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # Calculate time per epoch and estimated time to finish
        epoch_time = time.time() - start_time
        time_per_epoch = epoch_time
        time_left = epoch_time * (epochs - (epoch + 1))

        # Format times for better readability
        time_per_epoch_readable = format_time(time_per_epoch)
        time_left_readable = format_time(time_left)

        # Print metrics along with time information
        print(
            f"=== Epoch [{epoch + 1:0{len(str(epochs))}d}/{epochs}] ===\n"
            f"\tTrain Loss: {avg_loss:.4f}\n"
            f"\tVal Loss: {val_loss:.4f}\n"
            f"\tVal Acc: {accuracy:.4f}\n"
            f"\tVal Precision: {precision:.4f}\n"
            f"\tVal Recall: {recall:.4f}\n"
            f"\tVal F1-Score: {f1:.4f}\n"
            f"\tTime per epoch: {time_per_epoch_readable}\n"
            f"\tEstimated time left: {time_left_readable}\n"
            f"================================="
        )

        # Use ReduceLROnPlateau scheduler
        if scheduler is not None:
            scheduler.step(val_loss)


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Custom Tokenizer


# %% [code] {"jupyter":{"source_hidden":true,"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-09T18:12:37.464326Z","iopub.execute_input":"2025-02-09T18:12:37.464689Z","iopub.status.idle":"2025-02-09T18:12:37.470226Z","shell.execute_reply.started":"2025-02-09T18:12:37.464662Z","shell.execute_reply":"2025-02-09T18:12:37.469331Z"}}
# Modified Custom Tokenizer function: now accepts a list of raw texts from the training set
def train_custom_tokenizer(
    train_texts,
    vocab_size=30522,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    output_path="custom_tokenizer.json",
):
    # train_texts: list of raw text strings from the training set
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Create a trainer for the tokenizer
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Train the tokenizer on your texts
    tokenizer.train_from_iterator(train_texts, trainer=trainer)

    # Set a decoder (optional)
    tokenizer.decoder = decoders.BPEDecoder()

    # Set post-processor to add special tokens automatically
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # Save the trained tokenizer to a file
    tokenizer.save(output_path)
    print(f"Custom tokenizer saved to {output_path}")
    return tokenizer


# %% [markdown] {"execution":{"iopub.status.busy":"2025-02-08T09:09:56.324461Z","iopub.execute_input":"2025-02-08T09:09:56.324877Z","iopub.status.idle":"2025-02-08T09:09:56.425856Z","shell.execute_reply.started":"2025-02-08T09:09:56.324852Z","shell.execute_reply":"2025-02-08T09:09:56.424914Z"},"jupyter":{"outputs_hidden":false}}
# ```
# from transformers import PreTrainedTokenizerFast
# #
# # Load the tokenizer file you just saved.
# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file="custom_tokenizer.json",
#     bos_token="[CLS]",
#     eos_token="[SEP]",
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     mask_token="[MASK]",
# )
# #
# # You can now use this tokenizer just like any Hugging Face tokenizer.
# print("Vocabulary size:", tokenizer.vocab_size)
# print("Example encoding:", tokenizer.encode("Hello, how are you?"))
# ```

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Hyperparameters

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2025-02-09T18:41:42.611199Z","iopub.execute_input":"2025-02-09T18:41:42.611586Z","iopub.status.idle":"2025-02-09T18:41:42.616414Z","shell.execute_reply.started":"2025-02-09T18:41:42.611556Z","shell.execute_reply":"2025-02-09T18:41:42.615238Z"}}
# Hyperparameters
embedding_dim = 384  # Embedding dimension
nhead = 4  # Number of attention heads
num_encoder_layers = 2  # Number of transformer encoder layers
dim_feedforward = 128  # Dimension of the feedforward network in the transformer
output_size = 389  # Number of classes
batch_size = 256
epochs = 64
learning_rate = 0.002
max_length = 128  # Maximum sequence length for tokenization
dropout = 0.2

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training Preperations

# %% [code] {"execution":{"iopub.status.busy":"2025-02-09T18:41:45.840131Z","iopub.execute_input":"2025-02-09T18:41:45.840488Z","iopub.status.idle":"2025-02-09T18:41:52.438160Z","shell.execute_reply.started":"2025-02-09T18:41:45.840457Z","shell.execute_reply":"2025-02-09T18:41:52.437038Z"}}
# Device configuration: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

data = pd.read_csv("/kaggle/input/cs-anlp-competition/train.csv")
label2idx = {label: idx for idx, label in enumerate(data["Label"].unique())}
idx2label = {idx: label for label, idx in label2idx.items()}

# Split the data into training and validation sets
train_data, valid_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
)

# Train tokenizer on the training set only
train_texts = train_data["Text"].tolist()
custom_tokenizer = train_custom_tokenizer(train_texts)

# Create dataset from CSV
train_dataset = TextDataset(
    data=train_data,
    label2idx=label2idx,
    idx2label=idx2label,
    tokenizer=custom_tokenizer,
    max_length=max_length,
    augment=False,
)

valid_dataset = TextDataset(
    data=valid_data,
    label2idx=label2idx,
    idx2label=idx2label,
    tokenizer=custom_tokenizer,
    max_length=max_length,
    augment=False,
)


# Get class labels from the dataset (you may want to ensure this column exists)
labels = data["Label"]

# Compute class weights using sklearn's compute_class_weight function
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
# class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Compute sample weights for oversampling minority classes in the training set.
sample_weights = []
for idx in train_dataset.indices:
    label = train_dataset.data.iloc[idx]["Label"]
    label_idx = train_dataset.label2idx[label]
    sample_weights.append(class_weights[label_idx])
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(train_dataset), replacement=True
)

# Create DataLoaders for batching
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

# Display a sample batch for verification.
sample_batch = next(iter(train_loader))
sample_inputs, sample_lengths, sample_labels = sample_batch

separator = "=" * 60
print(separator)
print(" SAMPLE BATCH VERIFICATION ".center(60, "="))
print(separator)
print(f"Inputs shape           : {sample_inputs.shape}")
print(f"Sequence lengths shape : {sample_lengths.shape}")
print(f"Labels shape           : {sample_labels.shape}")
print(separator)
print(" First Sample Details ".center(60, "="))
print(separator)
print(f"Token IDs              : {sample_inputs[0]}")
print(f"Sequence Length        : {sample_lengths[0]}")
print(f"Label Index            : {sample_labels[0]}")
decoded_text = decode_token_ids(sample_inputs[0], tokenizer)
print(f"Decoded Text           : {decoded_text}")
print(f"Label                  : {train_dataset.idx2label[sample_labels[0].item()]}")
print(separator)


# Load the pre-trained model
pretrained_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Extract the embedding layer
pretrained_embedding = pretrained_model[0].auto_model.embeddings

# Initialize the Transformer model, loss function, and optimizer.
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=embedding_dim,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    output_size=output_size,
    dropout=dropout,
    max_seq_length=max_length,
    pretrained_embedding=pretrained_embedding,
)
model.to(device)

# Define the loss criterion with class weights
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Initialize the ReduceLROnPlateau scheduler with patience=5 and factor 0.8.
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.8, patience=5, verbose=True
)

print_model_summary(model)


# %% [code] {"execution":{"iopub.status.busy":"2025-02-09T11:39:03.287613Z","iopub.execute_input":"2025-02-09T11:39:03.287941Z","iopub.status.idle":"2025-02-09T11:48:08.522824Z","shell.execute_reply.started":"2025-02-09T11:39:03.287898Z","shell.execute_reply":"2025-02-09T11:48:08.521884Z"},"jupyter":{"outputs_hidden":false}}
# Train the model (pass scheduler to the training function)
train_model(
    model, train_loader, valid_loader, criterion, optimizer, device, epochs, scheduler
)

# Save the trained model
torch.save(model.state_dict(), "transformer.pth")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Load model .pth

# %% [code] {"execution":{"iopub.status.busy":"2025-02-09T18:23:30.844357Z","iopub.execute_input":"2025-02-09T18:23:30.844752Z","iopub.status.idle":"2025-02-09T18:23:31.242261Z","shell.execute_reply.started":"2025-02-09T18:23:30.844723Z","shell.execute_reply":"2025-02-09T18:23:31.241237Z"},"jupyter":{"outputs_hidden":false}}
# Load the model's state dictionary from the .pth file
model.load_state_dict(
    torch.load(
        "/kaggle/input/transformer-for-language-detection/transformer.pth",
        weights_only=False,
    )
)

# Set the model to evaluation mode
model.eval()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Evaluate


# %% [code] {"execution":{"iopub.status.busy":"2025-02-09T18:24:37.238267Z","iopub.execute_input":"2025-02-09T18:24:37.238602Z","iopub.status.idle":"2025-02-09T18:25:12.192008Z","shell.execute_reply.started":"2025-02-09T18:24:37.238577Z","shell.execute_reply":"2025-02-09T18:25:12.190972Z"},"jupyter":{"outputs_hidden":false}}
def plot_confusion_matrix(
    true_labels, pred_labels, class_names, output_path="confusion_matrix.png"
):
    """
    Plots and saves a confusion matrix heatmap.
    Args:
        true_labels: List of true labels.
        pred_labels: List of predicted labels.
        class_names: List of class names.
        output_path: Path to save the confusion matrix figure.
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
    )
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.xlabel("")  # Remove x-axis label
    plt.ylabel("")  # Remove y-axis label
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")


def evaluate_model(model, dataloader, criterion, device, dataset, train=False):
    """
    Evaluate model performance with detailed per-class metrics.

    Args:
        model: ...
        dataloader: DataLoader containing the evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        dataset: TextDataset instance for label mapping
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, lengths, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Get predictions
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert numeric labels back to original language codes
    pred_labels = [dataset.idx2label[idx] for idx in all_preds]
    true_labels = [dataset.idx2label[idx] for idx in all_labels]

    # Calculate average loss
    val_loss = val_loss / len(dataloader)

    # Generate classification report
    report = classification_report(
        true_labels, pred_labels, zero_division=0, output_dict=True
    )

    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

    # Prepare per-class metrics
    class_metrics = []
    for label in report:
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            class_metrics.append(
                {
                    "label": label,
                    "f1": report[label]["f1-score"],
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "support": report[label]["support"],
                }
            )

    # Sort by F1 score
    class_metrics.sort(key=lambda x: x["f1"], reverse=True)

    # Print full table for all classes
    headers = ["Language", "F1", "Precision", "Recall", "Support"]
    rows = [
        [
            metric["label"],
            metric["f1"],
            metric["precision"],
            metric["recall"],
            metric["support"],
        ]
        for metric in class_metrics
    ]
    print("\nFull Class Metrics Table:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Print top 5 and bottom 5 performing classes
    print("\nTop 5 Performing Languages:")
    for metric in class_metrics[:5]:
        print(
            f"Language: {metric['label']:<5} "
            f"F1: {metric['f1']:.4f} "
            f"Precision: {metric['precision']:.4f} "
            f"Recall: {metric['recall']:.4f} "
            f"Support: {metric['support']}"
        )

    print("\nBottom 5 Performing Languages:")
    for metric in class_metrics[-5:]:
        print(
            f"Language: {metric['label']:<5} "
            f"F1: {metric['f1']:.4f} "
            f"Precision: {metric['precision']:.4f} "
            f"Recall: {metric['recall']:.4f} "
            f"Support: {metric['support']}"
        )

    # Plot confusion matrix heatmap
    class_names = list(dataset.label2idx.keys())

    output_path = "train_confusion_matrix.png" if train else "val_confusion_matrix.png"
    plot_confusion_matrix(
        true_labels, pred_labels, class_names, output_path=output_path
    )

    return val_loss, report


# Usage
print("\nEvaluating on training set:")
train_metrics = evaluate_model(model, train_loader, criterion, device, train_dataset)

print("\nEvaluating on validation set:")
val_metrics = evaluate_model(model, valid_loader, criterion, device, valid_dataset)
