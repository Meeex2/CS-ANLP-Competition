import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import BertTokenizer


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0.0,
    ):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # Set bidirectional=True and use dropout if more than 1 layer.
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Additional dropout after LSTM.
        self.dropout2 = nn.Dropout(dropout)
        # Fully connected layer input dimension is doubled for bidirectional LSTM.
        self.fc1 = nn.Linear(hidden_size * 2, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = self.dropout(x)
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)
        out = self.dropout2(out[:, -1, :])  # Apply dropout before FC.
        out = torch.nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# Custom Dataset that reads a CSV file and tokenizes text using a Huggingface tokenizer
class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        """
        Args:
            csv_path (str): Path to the CSV file. The CSV should have columns 'Text' and 'Label'.
            tokenizer: A Huggingface tokenizer (e.g., BertTokenizer).
            max_length (int): Maximum tokenized sequence length.
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create a mapping from string language codes to integer indices.
        # Sorting ensures the mapping is deterministic.
        # Remove rows where "Label" is None or NaN
        self.data = self.data[self.data["Label"].notnull()]
        unique_labels = sorted(self.data["Label"].unique())
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row from the dataframe
        row = self.data.iloc[idx]
        text = row["Text"]
        label = row["Label"]  # e.g., "eng", "ara", etc.

        # Tokenize the text; add special tokens, truncate if necessary.
        token_ids = self.tokenizer.encode(
            text, add_special_tokens=True, max_length=self.max_length, truncation=True
        )
        length = len(token_ids)

        # Map the string label to its integer index.
        label_idx = self.label2idx[label]

        # Return token IDs as a tensor, the length, and the label index as a tensor.
        return (
            torch.tensor(token_ids, dtype=torch.long),
            length,
            torch.tensor(label_idx, dtype=torch.long),
        )


# Collate function to pad variable-length sequences in a batch
def collate_fn(batch):
    """
    Args:
        batch: A list of tuples (token_ids, length, label)
    Returns:
        padded_token_ids: (batch_size, max_seq_length) tensor of padded token ids.
        lengths: (batch_size,) tensor of original sequence lengths.
        labels: (batch_size,) tensor of labels.
    """
    token_ids, lengths, labels = zip(*batch)
    # Pad the sequences so that all have the same length within the batch.
    padded_token_ids = torch.nn.utils.rnn.pad_sequence(
        token_ids, batch_first=True, padding_value=0
    )
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(labels)
    return padded_token_ids, lengths, labels


# New function to decode token ids back to text.
def decode_token_ids(token_ids, tokenizer):
    """
    Decodes a list or tensor of token ids back to text using the specified tokenizer.
    Args:
        token_ids: list or tensor of ints.
        tokenizer: A Huggingface tokenizer instance.
    Returns:
        A decoded string.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# Modify train_model to accept a validation loader and compute accuracy after each epoch.
def train_model(model, dataloader, val_loader, criterion, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        )
        for inputs, lengths, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (progress_bar.n + 1)
        # Validation phase
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, lengths, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}"
        )


def main():
    # Hyperparameters
    embedding_dim = 256  # Dimension of word embeddings.
    hidden_size = 128  # Hidden state dimension for LSTM.
    output_size = 389  # Number of classes.
    num_layers = 4  # Number of LSTM layers.
    batch_size = 64
    epochs = 50
    learning_rate = 0.005
    max_length = 128  # Maximum sequence length for tokenization.

    # Device configuration: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained tokenizer from Huggingface
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create dataset from CSV
    dataset = TextDataset(
        csv_path="data/train_submission.csv", tokenizer=tokenizer, max_length=max_length
    )

    # Create train and validation splits (80/20 split)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create DataLoaders for batching
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Print a sample batch to show data examples and shapes.
    sample_batch = next(iter(train_loader))
    sample_inputs, sample_lengths, sample_labels = sample_batch
    print("Sample inputs shape:", sample_inputs.shape)  # (batch_size, seq_length)
    print("Sample sequence lengths shape:", sample_lengths.shape)  # (batch_size,)
    print("Sample labels shape:", sample_labels.shape)  # (batch_size,)
    print("First sample input token IDs:", sample_inputs[0])
    print("First sample sequence length:", sample_lengths[0])
    print("First sample label:", sample_labels[0])

    print(decode_token_ids(sample_inputs[0], tokenizer))
    print("Label:", dataset.idx2label[sample_labels[0].item()])

    # Initialize the model, loss function, and optimizer
    model = LSTMModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=0.5,  # Example dropout rate
    )
    model.to(device)

    # For classification, we use CrossEntropyLoss.
    criterion = nn.CrossEntropyLoss()
    # Switch to AdamW with weight decay for L2 regularization.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Train the model
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs)

    # Save model
    torch.save(model.state_dict(), "lstm_model.pth")


if __name__ == "__main__":
    main()
