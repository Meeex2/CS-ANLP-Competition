import os
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from torch.utils.data import Dataset, DataLoader, random_split
import random
from scripts.utils import remove_links_and_tags, remove_emojis, filter_majority_script


class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length, augment=False, test=False):
        """
        Args:
            csv_path (str): Path to the CSV file. The CSV should have a 'Text' column.
            tokenizer: A Huggingface tokenizer (e.g., RobertaTokenizerFast).
            max_length (int): Maximum tokenized sequence length.
            augment (bool): Whether to apply data augmentation.
            test (bool): Whether this is a test dataset (no labels).
        """
        print(f"Loading data from {csv_path} ...")
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.test = test  # Flag to indicate if this is a test dataset

        self.data["Text"] = self.data["Text"].apply(remove_links_and_tags)
        # self.data["Text"] = self.data["Text"].apply(remove_emojis)
        self.data["Text"] = self.data["Text"].apply(filter_majority_script)

        if not self.test:
            # Remove rows with missing labels and create a deterministic label mapping.
            self.data = self.data[self.data["Label"].notnull()]
            unique_labels = sorted(self.data["Label"].unique())
            self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        else:
            # For test data, create dummy labels (not used during inference)
            self.data["Label"] = 0  # Dummy label
            self.label2idx = {"dummy": 0}
            self.idx2label = {0: "dummy"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["Text"]

        # Tokenize text without automatic truncation.
        token_ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)

        # Data augmentation: take a random contiguous slice if augmentation is enabled.
        if self.augment and len(token_ids) > self.max_length:
            start_idx = random.randint(0, len(token_ids) - self.max_length)
            token_ids = token_ids[start_idx : start_idx + self.max_length]
        else:
            token_ids = token_ids[: self.max_length]

        # Create an attention mask (1 for each token).
        attention_mask = [1] * len(token_ids)

        if not self.test:
            label = row["Label"]
            label_idx = self.label2idx[label]
        else:
            # For test data, use a dummy label (not used during inference)
            label_idx = 0

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": label_idx,  # Dummy label for test data
        }


def find_best_checkpoint(base_dir):
    """
    Looks for checkpoint directories under the base_dir that match the pattern:
    checkpoint_epoch_{epoch+1}_acc_{val_accuracy:.4f} and returns the one with the highest accuracy.
    """
    best_acc = -1.0
    best_checkpoint = None
    pattern = re.compile(r"checkpoint_epoch_\d+_acc_([0-9]+\.[0-9]+)$")
    
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if os.path.isdir(entry_path):
            match = pattern.match(entry)
            if match:
                acc = float(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_checkpoint = entry_path
    
    if best_checkpoint is None:
        print("No valid checkpoint found; using base directory instead.")
        return base_dir
    else:
        print(f"Selected checkpoint: {best_checkpoint} with accuracy {best_acc}")
        return best_checkpoint


# -----------------------------
# 1. Load the model and tokenizer
# -----------------------------
def load_model_and_tokenizer(model_dir, num_labels):
    """
    Load the fine-tuned model and tokenizer from the specified directory.
    """
    print(f"Loading model and tokenizer from {model_dir}...")

    # Load the tokenizer from the local directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # Load the fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        local_files_only=True
    )

    model.eval()
    return model, tokenizer


# Define base model directory and number of labels used during training
base_model_dir = "roberta_finetuned_preprocess"
num_labels = 389  # Replace with your actual number of labels

# Use the function to pick the checkpoint with the highest accuracy
best_checkpoint_dir = find_best_checkpoint(base_model_dir)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer from the best checkpoint, then move the model to the device
model, tokenizer = load_model_and_tokenizer(best_checkpoint_dir, num_labels)
model.to(device)

# -----------------------------
# 2. Prepare the test data
# -----------------------------
# Define test CSV path and parameters
test_CSV_PATH = "data/test_without_labels.csv"
MAX_SEQ_LENGTH = 128              # Adjust as needed
PER_DEVICE_EVAL_BATCH_SIZE = 720   # Adjust as needed
csv_path = "data/train_submission.csv"

# Setup data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create the test dataset and dataloader
test_dataset = TextDataset(test_CSV_PATH, tokenizer, MAX_SEQ_LENGTH, augment=False, test=True)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

max_length = 128

full_dataset = TextDataset(csv_path, tokenizer, max_length, augment=False)

# -----------------------------
# 3. Run predictions on the test dataset
# -----------------------------
model.eval()
test_preds = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        # Move all batch tensors to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        test_preds.extend(preds.cpu().numpy())

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches...")

print("Test prediction completed!")

# -----------------------------
# 4. Create the submission file
# -----------------------------
# Make sure you have a mapping from numeric predictions to label names.
# Here, we assume that `full_dataset.idx2label` exists and maps prediction indices to label strings.
submission_df = pd.DataFrame({
    "ID": range(1, len(test_preds) + 1),
    "Label": [full_dataset.idx2label[pred] for pred in test_preds],
})

# Define the output directory and save the CSV file
output_dir = "./output"  # Change as needed
os.makedirs(output_dir, exist_ok=True)
submission_file_path = os.path.join(output_dir, "submission_pre_full.csv")
submission_df.to_csv(submission_file_path, index=False)
print(f"Submission file saved to {submission_file_path}")