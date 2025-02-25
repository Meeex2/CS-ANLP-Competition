# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Imports

# %% [code] {"execution":{"iopub.status.busy":"2025-02-25T13:29:17.758212Z","iopub.execute_input":"2025-02-25T13:29:17.758530Z","iopub.status.idle":"2025-02-25T13:29:38.210304Z","shell.execute_reply.started":"2025-02-25T13:29:17.758503Z","shell.execute_reply":"2025-02-25T13:29:38.209378Z"},"jupyter":{"outputs_hidden":false}}
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import the PEFT (Parameter-Efficient Fine-Tuning) components for LoRA.
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-25T13:29:38.211508Z","iopub.execute_input":"2025-02-25T13:29:38.212177Z","iopub.status.idle":"2025-02-25T13:29:38.221161Z","shell.execute_reply.started":"2025-02-25T13:29:38.212144Z","shell.execute_reply":"2025-02-25T13:29:38.220221Z"}}
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# %% [markdown]
# # Utils

# %% [code] {"execution":{"iopub.status.busy":"2025-02-25T13:37:48.627954Z","iopub.execute_input":"2025-02-25T13:37:48.628281Z","iopub.status.idle":"2025-02-25T13:37:48.639437Z","shell.execute_reply.started":"2025-02-25T13:37:48.628259Z","shell.execute_reply":"2025-02-25T13:37:48.638514Z"}}
import bisect
import functools
import re


def to_codepoint(s):
    if isinstance(s, str) and s.startswith("\\u"):
        return int(s[2:], 16)
    elif isinstance(s, str) and s.startswith("U+"):
        return int(s[2:], 16)
    else:
        return int(s)


@functools.lru_cache(maxsize=1)
def load_script_ranges():
    df = pd.read_csv("/kaggle/input/unicode-ranges/unicode_ranges.csv")
    df["start"] = df["range_start"].apply(to_codepoint)
    df["end"] = df["range_end"].apply(to_codepoint)
    # Sort ranges by the start value.
    ranges = sorted(df.itertuples(index=False), key=lambda r: r.start)
    # Cache the unique language group names from the CSV.
    lang_names = df["language_group_name"].unique().tolist()
    # Also return a list of all start values for binary search.
    range_starts = [r.start for r in ranges]
    return ranges, range_starts, lang_names


def detect_script(text):
    ranges, range_starts, lang_names = load_script_ranges()

    # Initialize counts for each language group and "Unknown".
    script_counts = {lang: 0 for lang in lang_names}
    script_counts["Unknown"] = 0

    # Clean text (removing leading/trailing spaces and internal spaces).
    text = text.strip().replace(" ", "")
    for char in text:
        code = ord(char)
        # Locate the rightmost range whose start is <= code.
        idx = bisect.bisect_right(range_starts, code) - 1
        if idx >= 0:
            r = ranges[idx]
            if r.start <= code <= r.end:
                script_counts[r.language_group_name] += 1
                continue  # Skip the "Unknown" count.
        script_counts["Unknown"] += 1

    return script_counts


def filter_majority_script(text):
    """
    Keeps only the characters in the majority script so that text is uniform in its script.
    """
    script_counts = detect_script(text)
    # Find the majority script (excluding "Unknown").
    majority_script = max(
        (script for script in script_counts if script != "Unknown"),
        key=script_counts.get,
    )

    # Filter text to keep only characters in the majority script.
    ranges, range_starts, _ = load_script_ranges()
    filtered_text = []
    for char in text:
        code = ord(char)
        if char.isspace() or char in "()[]{}":
            filtered_text.append(char)
            continue
        idx = bisect.bisect_right(range_starts, code) - 1
        if idx >= 0:
            r = ranges[idx]
            if r.start <= code <= r.end and r.language_group_name == majority_script:
                filtered_text.append(char)

    return "".join(filtered_text)


def remove_links_and_tags(text: str):
    """
    Removes internet links, tags that begin with @, and hashtags.
    """
    # Remove URLs (e.g., starting with http://, https://, or www.)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove @tags (words that begin with @ followed by alphanumeric or underscore characters)
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags (words that begin with # followed by alphanumeric or underscore characters)
    text = re.sub(r"#\w+", "", text)
    # Remove multiple spaces between words.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_emojis(text: str):
    """
    Removes emojis from the text.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Hyperparameters

# %% [code] {"execution":{"iopub.status.busy":"2025-02-25T13:37:50.905562Z","iopub.execute_input":"2025-02-25T13:37:50.905880Z","iopub.status.idle":"2025-02-25T13:37:50.910192Z","shell.execute_reply.started":"2025-02-25T13:37:50.905853Z","shell.execute_reply":"2025-02-25T13:37:50.909432Z"},"jupyter":{"outputs_hidden":false}}
# ----- Model and Data Paths -----
# MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
MODEL_NAME = "FacebookAI/xlm-roberta-large"
CSV_PATH = "/kaggle/input/nlp-cs-2025/train_submission.csv"
SAVE_PATH = "/kaggle/input/fine-tune-xlm-roberta"

# ----- Data Processing -----
MAX_SEQ_LENGTH = 64
AUGMENT = True  # Enable data augmentation for training data

# ----- Training Hyperparameters -----
NUM_TRAIN_EPOCHS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
PER_DEVICE_TRAIN_BATCH_SIZE = 160
PER_DEVICE_EVAL_BATCH_SIZE = 160
LOGGING_STEPS = 100
EARLY_STOPPING_PATIENCE = 5  # Epochs to wait for improvement before stopping

# ----- Learning Rate Scheduling -----
WARMUP_EPOCHS = 4
STABLE_EPOCHS = 12
DECAY_EPOCHS = NUM_TRAIN_EPOCHS - WARMUP_EPOCHS - STABLE_EPOCHS

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Dataset

# %% [code] {"execution":{"iopub.status.busy":"2025-02-25T13:38:23.204358Z","iopub.execute_input":"2025-02-25T13:38:23.204645Z","iopub.status.idle":"2025-02-25T13:38:23.212837Z","shell.execute_reply.started":"2025-02-25T13:38:23.204624Z","shell.execute_reply":"2025-02-25T13:38:23.212031Z"},"jupyter":{"outputs_hidden":false}}
import random

import torch


class TextDataset(Dataset):
    def __init__(self, CSV_PATH, tokenizer, MAX_SEQ_LENGTH, AUGMENT=False, test=False):
        """
        Args:
            CSV_PATH (str): Path to the CSV file. The CSV should have columns 'Text' and optionally 'Label'.
            tokenizer: A Huggingface tokenizer (e.g., RobertaTokenizerFast).
            MAX_SEQ_LENGTH (int): Maximum tokenized sequence length.
            AUGMENT (bool): Whether to apply data augmentation.
            test (bool): Whether the dataset is for testing (no labels required).
        """
        print(f"Loading data from {CSV_PATH} ...")
        self.data = pd.read_csv(CSV_PATH)

        # ==== Clean data ====
        self.data["Text"] = self.data["Text"].apply(remove_links_and_tags)
        self.data["Text"] = self.data["Text"].apply(remove_emojis)
        self.data["Text"] = self.data["Text"].apply(filter_majority_script)
        # ====================

        self.tokenizer = tokenizer
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.AUGMENT = AUGMENT
        self.test = test  # If True, labels are not required

        if not self.test:
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

        # Tokenize text without automatic truncation.
        token_ids = self.tokenizer.encode(
            text, add_special_tokens=True, truncation=True
        )

        # Data augmentation: take a random contiguous slice if augmentation is enabled.
        if self.AUGMENT and len(token_ids) > self.MAX_SEQ_LENGTH:
            start_idx = random.randint(0, len(token_ids) - self.MAX_SEQ_LENGTH)
            token_ids = token_ids[start_idx : start_idx + self.MAX_SEQ_LENGTH]
        else:
            token_ids = token_ids[: self.MAX_SEQ_LENGTH]

        # Create an attention mask (1 for each token).
        attention_mask = [1] * len(token_ids)

        item = {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }

        # Include labels only if not in test mode
        if not self.test:
            label_idx = self.label2idx[row["Label"]]
            item["labels"] = label_idx

        return item


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training preparation


# %% [code] {"execution":{"iopub.status.busy":"2025-02-25T13:38:25.065669Z","iopub.execute_input":"2025-02-25T13:38:25.065950Z","iopub.status.idle":"2025-02-25T13:39:25.415155Z","shell.execute_reply.started":"2025-02-25T13:38:25.065931Z","shell.execute_reply":"2025-02-25T13:39:25.414473Z"},"jupyter":{"outputs_hidden":false}}
# Learning rate scheduling function
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS  # Linear warmup
    elif epoch < WARMUP_EPOCHS + STABLE_EPOCHS:
        return 1.0  # Keep it stable
    else:
        return max(
            0.1, 1.0 - (epoch - WARMUP_EPOCHS - STABLE_EPOCHS) / DECAY_EPOCHS
        )  # Linear decay


# ----- Load Tokenizer and Prepare Dataset -----
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Instantiating the full dataset...")
full_dataset = TextDataset(CSV_PATH, tokenizer, MAX_SEQ_LENGTH, AUGMENT=AUGMENT)
num_labels = len(full_dataset.label2idx)
print(f"Number of labels: {num_labels}")

# Split into training (90%) and validation (10%) sets.
dataset_size = len(full_dataset)
train_size = int(0.999 * dataset_size)
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
    batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

# ----- Load the Model and Wrap with LoRA (PEFT) -----
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True
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
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Define scheduler
scheduler = LambdaLR(optimizer, lr_lambda)

best_val_accuracy = 0.0
global_step = 0
epochs_without_improvement = 0

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Load previous trained model

# %% [code] {"execution":{"iopub.status.busy":"2025-02-25T07:45:45.664579Z","iopub.execute_input":"2025-02-25T07:45:45.664920Z","iopub.status.idle":"2025-02-25T07:45:50.610172Z","shell.execute_reply.started":"2025-02-25T07:45:45.664893Z","shell.execute_reply":"2025-02-25T07:45:50.609341Z"},"jupyter":{"outputs_hidden":false}}
print("Loading previous model...")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True
)
model = PeftModel.from_pretrained(base_model, SAVE_PATH)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Previous model loaded")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training Loop

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-20T10:00:40.108517Z","iopub.execute_input":"2025-02-20T10:00:40.108867Z","iopub.status.idle":"2025-02-20T10:01:11.381100Z","shell.execute_reply.started":"2025-02-20T10:00:40.108838Z","shell.execute_reply":"2025-02-20T10:01:11.379887Z"}}
for epoch in range(NUM_TRAIN_EPOCHS):
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

        if global_step % LOGGING_STEPS == 0:
            avg_loss = running_loss / LOGGING_STEPS
            print(
                f"Epoch [{epoch + 1}/{NUM_TRAIN_EPOCHS}], "
                f"Step [{step + 1}/{len(train_dataloader)}], "
                f"Loss: {avg_loss:.4f}"
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
        f"Epoch [{epoch + 1}/{NUM_TRAIN_EPOCHS}] Validation Loss: {avg_eval_loss:.4f} | Accuracy: {val_accuracy:.4f}"
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
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Save model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-02-18T14:12:35.321598Z","iopub.execute_input":"2025-02-18T14:12:35.321842Z","iopub.status.idle":"2025-02-18T14:12:35.864413Z","shell.execute_reply.started":"2025-02-18T14:12:35.321822Z","shell.execute_reply":"2025-02-18T14:12:35.863539Z"}}
# (reload the best model state)
output_dir = "."
# model.load_state_dict(best_model_state)

# ----- Save the Model and Tokenizer -----
os.makedirs(output_dir, exist_ok=True)
print("Saving model and tokenizer...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Final evaluation

# %% [code] {"execution":{"iopub.status.busy":"2025-02-25T07:45:56.868300Z","iopub.execute_input":"2025-02-25T07:45:56.868600Z","iopub.status.idle":"2025-02-25T07:45:57.305999Z","shell.execute_reply.started":"2025-02-25T07:45:56.868576Z","shell.execute_reply":"2025-02-25T07:45:57.305116Z"},"jupyter":{"outputs_hidden":false}}
# ----- Final Evaluation on the Validation Set -----
model.eval()
all_preds = []
all_labels = []
probs = []

with torch.no_grad():
    for batch_idx, batch in enumerate(val_dataloader):
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

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1} batches...")

print("Final evaluation completed!")

# %% [code] {"execution":{"iopub.status.busy":"2025-02-25T07:45:58.401033Z","iopub.execute_input":"2025-02-25T07:45:58.401321Z","iopub.status.idle":"2025-02-25T07:45:58.416290Z","shell.execute_reply.started":"2025-02-25T07:45:58.401300Z","shell.execute_reply":"2025-02-25T07:45:58.415421Z"},"jupyter":{"outputs_hidden":false},"scrolled":true}
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os

test_CSV_PATH = "/kaggle/input/nlp-cs-2025/test_without_labels.csv"

# Load test dataset
test_dataset = TextDataset(
    test_CSV_PATH, tokenizer, MAX_SEQ_LENGTH, AUGMENT=False, test=True
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

# Predict labels for the test dataset
model.eval()
test_preds = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        test_preds.extend(preds.cpu().numpy())

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches...")

print("Test prediction completed!")

# %% [code] {"execution":{"iopub.status.busy":"2025-02-18T14:22:53.956767Z","iopub.execute_input":"2025-02-18T14:22:53.957117Z","iopub.status.idle":"2025-02-18T14:22:54.822796Z","shell.execute_reply.started":"2025-02-18T14:22:53.957091Z","shell.execute_reply":"2025-02-18T14:22:54.822066Z"},"jupyter":{"outputs_hidden":false}}
# Create submission DataFrame
submission_df = pd.DataFrame(
    {
        "ID": range(1, len(test_preds) + 1),
        "Label": [full_dataset.idx2label[pred] for pred in test_preds],
    }
)

# Save submission file
submission_file_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(submission_file_path, index=False)
print(f"Submission file saved to {submission_file_path}")
