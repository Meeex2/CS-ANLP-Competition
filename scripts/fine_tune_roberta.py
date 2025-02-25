# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Imports

# %% [code] {"execution":{"iopub.status.busy":"2025-02-17T21:00:23.886752Z","iopub.execute_input":"2025-02-17T21:00:23.887047Z","iopub.status.idle":"2025-02-17T21:00:44.354386Z","shell.execute_reply.started":"2025-02-17T21:00:23.887023Z","shell.execute_reply":"2025-02-17T21:00:44.353507Z"},"jupyter":{"outputs_hidden":false}}
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import the PEFT (Parameter-Efficient Fine-Tuning) components for LoRA.
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# %% [code]
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Dataset


# %% [code] {"execution":{"iopub.status.busy":"2025-02-17T21:08:38.722940Z","iopub.execute_input":"2025-02-17T21:08:38.723226Z","iopub.status.idle":"2025-02-17T21:08:38.730366Z","shell.execute_reply.started":"2025-02-17T21:08:38.723205Z","shell.execute_reply":"2025-02-17T21:08:38.729520Z"},"jupyter":{"outputs_hidden":false}}
class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length, augment=False):
        """
        Args:
            csv_path (str): Path to the CSV file. The CSV should have columns 'Text' and 'Label'.
            tokenizer: A Huggingface tokenizer (e.g., RobertaTokenizerFast).
            max_length (int): Maximum tokenized sequence length.
            augment (bool): Whether to apply data augmentation.
        """
        print(f"Loading data from {csv_path} ...")
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

        # Remove rows with missing labels and create a deterministic label mapping.
        self.data = self.data[self.data["Label"].notnull()]
        unique_labels = sorted(self.data["Label"].unique())
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["Text"]
        label = row["Label"]

        # Tokenize text without automatic truncation.
        token_ids = self.tokenizer.encode(
            text, add_special_tokens=True, truncation=True
        )

        # Data augmentation: take a random contiguous slice if augmentation is enabled.
        if self.augment and len(token_ids) > self.max_length:
            start_idx = random.randint(0, len(token_ids) - self.max_length)
            token_ids = token_ids[start_idx : start_idx + self.max_length]
        else:
            token_ids = token_ids[: self.max_length]

        # Create an attention mask (1 for each token).
        attention_mask = [1] * len(token_ids)

        label_idx = self.label2idx[label]
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": label_idx,
        }


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training preaeration

# %% [code] {"execution":{"iopub.status.busy":"2025-02-17T21:13:05.948320Z","iopub.execute_input":"2025-02-17T21:13:05.948683Z","iopub.status.idle":"2025-02-17T21:13:10.966367Z","shell.execute_reply.started":"2025-02-17T21:13:05.948653Z","shell.execute_reply":"2025-02-17T21:13:10.965715Z"},"jupyter":{"outputs_hidden":false}}
# ----- Settings and Hyperparameters -----
model_name = "papluca/xlm-roberta-base-language-detection"
csv_path = "/kaggle/input/nlp-cs-2025/train_submission.csv"
max_length = 64
augment = True  # set to True to enable data augmentation for training data

# Training hyperparameters
num_train_epochs = 32
learning_rate = 4e-4
weight_decay = 0.01
per_device_train_batch_size = 64
per_device_eval_batch_size = 64
logging_steps = 100
early_stopping_patience = 5  # Number of epochs to wait for improvement before stopping

# Define warmup, stable, and decay durations
warmup_epochs = 5
stable_epochs = 10
decay_epochs = num_train_epochs - warmup_epochs - stable_epochs


# Learning rate scheduling function
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # Linear warmup
    elif epoch < warmup_epochs + stable_epochs:
        return 1.0  # Keep it stable
    else:
        return max(
            0.1, 1.0 - (epoch - warmup_epochs - stable_epochs) / decay_epochs
        )  # Linear decay


# ----- Load Tokenizer and Prepare Dataset -----
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Instantiating the full dataset...")
full_dataset = TextDataset(csv_path, tokenizer, max_length, augment=augment)
num_labels = len(full_dataset.label2idx)
print(f"Number of labels: {num_labels}")

# Split into training (90%) and validation (10%) sets.
dataset_size = len(full_dataset)
train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size
print(
    f"Total dataset size: {dataset_size}, Training size: {train_size}, Validation size: {val_size}"
)
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

# Create DataLoaders.
train_dataloader = DataLoader(
    train_dataset,
    batch_size=per_device_train_batch_size,
    shuffle=True,
    collate_fn=data_collator,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=per_device_eval_batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

# ----- Load the Model and Wrap with LoRA (PEFT) -----
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, ignore_mismatched_sizes=True
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
    inference_mode=False,
    r=8,  # Rank of the LoRA update matrices (hyperparameter)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability applied to LoRA layers
    target_modules=["query", "value"],  # Target modules to apply LoRA on
)
model = get_peft_model(model, lora_config)
print("LoRA-modified model loaded.")

# ----- Prepare for Training -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Compute label frequencies from the training split.
# train_labels = [full_dataset[idx]["labels"] for idx in train_dataset.indices]
# label_counts = Counter(train_labels)
# print("Training label distribution:", label_counts)

# Compute weights as the inverse frequency.
# weights = [1.0 / label_counts[i] for i in range(num_labels)]
# weights = torch.tensor(weights, dtype=torch.float, device=device)
# print("Loss function weights:", weights)

# Create a weighted CrossEntropyLoss.
loss_fn = torch.nn.CrossEntropyLoss()

# Create optimizer. Here we use AdamW with weight decay.
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

# Define scheduler
scheduler = LambdaLR(optimizer, lr_lambda)

best_val_accuracy = 0.0
global_step = 0
epochs_without_improvement = 0

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training Loop

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-17T21:14:03.786217Z","iopub.execute_input":"2025-02-17T21:14:03.786514Z","iopub.status.idle":"2025-02-17T21:14:33.338990Z","shell.execute_reply.started":"2025-02-17T21:14:03.786493Z","shell.execute_reply":"2025-02-17T21:14:33.337878Z"}}
for epoch in range(num_train_epochs):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        # Move batch tensors to the device.
        batch = {k: v.to(device) for k, v in batch.items()}
        # Remove labels from the batch to avoid the model computing its own loss.
        labels = batch.pop("labels")

        # Forward pass (without labels).
        outputs = model(**batch)
        logits = outputs.logits
        # Compute the weighted loss manually.
        loss = loss_fn(logits, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        global_step += 1

        if global_step % logging_steps == 0:
            avg_loss = running_loss / logging_steps
            print(
                f"Epoch [{epoch + 1}/{num_train_epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}"
            )
            running_loss = 0.0

    # ----- Validation at the End of Each Epoch -----
    model.eval()
    all_preds = []
    all_labels = []
    eval_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Pop labels to compute loss manually.
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            eval_loss += loss_fn(logits, labels).item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1

    avg_eval_loss = eval_loss / num_batches
    val_accuracy = accuracy_score(all_labels, all_preds)
    print(
        f"Epoch [{epoch + 1}/{num_train_epochs}] Validation Loss: {avg_eval_loss:.4f} | Accuracy: {val_accuracy:.4f}"
    )

    # Save best model (if desired)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict()
        print("Best model updated.")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Save model

# %% [code] {"execution":{"iopub.status.busy":"2025-02-17T21:01:51.252335Z","iopub.status.idle":"2025-02-17T21:01:51.252609Z","shell.execute_reply":"2025-02-17T21:01:51.252486Z"},"jupyter":{"outputs_hidden":false}}
output_dir = "."

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-17T21:01:51.253206Z","iopub.status.idle":"2025-02-17T21:01:51.253502Z","shell.execute_reply":"2025-02-17T21:01:51.253360Z"}}
# (Optionally, reload the best model state)
model.load_state_dict(best_model_state)

# ----- Save the Model and Tokenizer -----
os.makedirs(output_dir, exist_ok=True)
print("Saving model and tokenizer...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Final evaluation

# %% [code] {"execution":{"iopub.status.busy":"2025-02-17T21:01:51.254056Z","iopub.status.idle":"2025-02-17T21:01:51.254376Z","shell.execute_reply":"2025-02-17T21:01:51.254258Z"},"jupyter":{"outputs_hidden":false}}

# ----- Final Evaluation on the Validation Set -----
model.eval()
all_preds = []
all_labels = []
probs = []

with torch.no_grad():
    for batch in val_dataloader:
        # Move batch tensors to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        outputs = model(**batch)
        logits = outputs.logits

        # Compute probabilities using softmax
        probabilities = F.softmax(logits, dim=-1)

        # Get predicted labels
        preds = torch.argmax(logits, dim=-1)

        # Store results
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        probs.extend(probabilities.cpu().numpy())

# %% [code] {"execution":{"iopub.status.busy":"2025-02-17T21:01:51.255295Z","iopub.status.idle":"2025-02-17T21:01:51.255568Z","shell.execute_reply":"2025-02-17T21:01:51.255463Z"},"jupyter":{"outputs_hidden":false}}

# Extract unique labels present in the validation set
unique_labels = np.unique(all_labels)


# Map these labels to class names using idx2label
target_names = [full_dataset.idx2label[label] for label in unique_labels]

# Generate the classification report with explicit labels
report = classification_report(
    all_labels,
    all_preds,
    labels=unique_labels,
    target_names=target_names,
    zero_division=0,
)

# Print the classification report
print("Classification Report:\n", report)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Generate Submission Output

# %% [code] {"execution":{"iopub.status.busy":"2025-02-17T21:01:51.256056Z","iopub.status.idle":"2025-02-17T21:01:51.256376Z","shell.execute_reply":"2025-02-17T21:01:51.256258Z"},"jupyter":{"outputs_hidden":false}}
test_csv_path = "/kaggle/input/nlp-cs-2025/test_submission.csv"

# Load test dataset
test_dataset = TextDataset(test_csv_path, tokenizer, max_length, augment=False)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=per_device_eval_batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

# Predict labels for the test dataset
model.eval()
test_preds = []

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        test_preds.extend(preds.cpu().numpy())

# Create submission DataFrame
submission_df = pd.DataFrame(
    {
        "ID": range(1, len(test_preds) + 1),
        "Label": [test_dataset.idx2label[pred] for pred in test_preds],
    }
)

# Save submission file
submission_file_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(submission_file_path, index=False)
print(f"Submission file saved to {submission_file_path}")
