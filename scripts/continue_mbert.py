def main():
    import os
    import random
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    from accelerate import Accelerator
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support
    )
    # Import PEFT components for LoRA.
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from peft import PeftConfig, PeftModel
    # clear cuda cache
    torch.cuda.empty_cache()
    # clear cuda memory
    torch.cuda.memory_summary(device=None, abbreviated=False)
    # set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



    # -------------------------------
    # 1. Dataset Definition
    # -------------------------------
    class TextDataset(Dataset):
        def __init__(self, CSV_PATH, tokenizer, MAX_SEQ_LENGTH, AUGMENT=False, test=False):
            """
            Args:
                CSV_PATH (str): Path to the CSV file. The CSV should have columns 'Text' and optionally 'Label'.
                tokenizer: A Huggingface tokenizer.
                MAX_SEQ_LENGTH (int): Maximum tokenized sequence length.
                AUGMENT (bool): Whether to apply data augmentation.
                test (bool): Whether the dataset is for testing (no labels required).
            """
            print(f"Loading data from {CSV_PATH} ...")
            self.data = pd.read_csv(CSV_PATH)
            # ==== Clean data ====
            #from utils import remove_links_and_tags, filter_majority_script
            #self.data["Text"] = self.data["Text"].apply(remove_links_and_tags)
            #self.data["Text"] = self.data["Text"].apply(filter_majority_script)
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

            # Tokenize text (using automatic truncation if needed)
            token_ids = self.tokenizer.encode(
                text, add_special_tokens=True, truncation=True
            )
            if self.AUGMENT and len(token_ids) > self.MAX_SEQ_LENGTH:
                start_idx = random.randint(0, len(token_ids) - self.MAX_SEQ_LENGTH)
                token_ids = token_ids[start_idx : start_idx + self.MAX_SEQ_LENGTH]
            else:
                token_ids = token_ids[: self.MAX_SEQ_LENGTH]

            attention_mask = [1] * len(token_ids)
            item = {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
            }
            if not self.test:
                label_idx = self.label2idx[row["Label"]]
                item["labels"] = label_idx
            return item

    def remove_prefix_and_handle_classifier(state_dict):
        """
        Remove 'module.' prefix from state dict keys.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        return new_state_dict

    # -------------------------------
    # 2. Utility Function to Plot Confusion Matrix
    # -------------------------------
    def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues):
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap=cmap)
        plt.title(title)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    # -------------------------------
    # 3. Main Training and Evaluation Loop
    # -------------------------------
    # Settings and Hyperparameters
    model_name = "google-bert/bert-base-multilingual-cased"
    csv_path = "data/train_submission.csv"
    max_length = 128 
    augment = False

    accelerator = Accelerator(gradient_accumulation_steps=1)

    num_train_epochs = 50
    learning_rate = 2e-3
    weight_decay = 0.01
    per_device_train_batch_size = 732
    per_device_eval_batch_size = 732
    logging_steps = 12

    accelerator.print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accelerator.print("Instantiating the full dataset...")
    full_dataset = TextDataset(csv_path, tokenizer, max_length, AUGMENT=augment)
    num_labels = len(full_dataset.label2idx)
    accelerator.print(f"Number of labels: {num_labels}")

    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    accelerator.print(
        f"Total dataset size: {dataset_size}, Training size: {train_size}, Validation size: {val_size}"
    )
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

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
    accelerator.print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    # Add checkpoint loading code here
    checkpoint_path = "lora_bert_finetuned/checkpoint_epoch_9_acc_0.8451"  # Replace with your checkpoint path
    if os.path.exists(checkpoint_path):
        accelerator.print(f"Loading LoRA fine-tuned model and tokenizer from {checkpoint_path}...")
        # Load the tokenizer from the checkpoint directory
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)

        # Load the base model with the correct number of labels.
        base_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=False
        )

        # Load the LoRA configuration and apply it with PeftModel.
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        model = PeftModel.from_pretrained(base_model, checkpoint_path, config=peft_config)

        # Extract the starting epoch from the checkpoint directory name (assuming the naming structure)
        checkpoint_epoch = int(checkpoint_path.split("_")[-3])
        start_epoch = checkpoint_epoch  # Continue from this epoch
    else:
        accelerator.print("No checkpoint found. Starting training from scratch.")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        model = get_peft_model(model, lora_config)
        start_epoch = 0

    accelerator.print("LoRA-modified model loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-4)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    best_val_accuracy = 0.0
    global_step = 0

    # ----- Training Loop -----
    for epoch in range(start_epoch,num_train_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    accelerator.print(f"Epoch [{epoch+1}/{num_train_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}")
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
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                num_batches += 1

        avg_eval_loss = eval_loss / num_batches
        val_accuracy = accuracy_score(all_labels, all_preds)
        accelerator.print(f"Epoch [{epoch+1}/{num_train_epochs}] Validation Loss: {avg_eval_loss:.4f} | Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = remove_prefix_and_handle_classifier(model.state_dict())
            accelerator.print("Best model updated.")

        # Save checkpoint for current epoch
        checkpoint_dir = os.path.join("./lora_bert_finetuned", f"checkpoint_epoch_{epoch+1}_acc_{val_accuracy:.4f}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        accelerator.print(f"Saving checkpoint to {checkpoint_dir} ...")
        if hasattr(model, "module"):
            model.module.save_pretrained(checkpoint_dir)
        else:
            model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    try:
        cleaned_state_dict = remove_prefix_and_handle_classifier(best_model_state)
        accelerator.unwrap_model(model).load_state_dict(cleaned_state_dict, strict=False)
        accelerator.print("Successfully loaded best model state")
    except Exception as e:
        accelerator.print(f"Warning: Could not load best model state: {str(e)}")
        accelerator.print("Continuing with current model state")

    # ----- Save the Final Model and Tokenizer -----
    output_dir = "./lora_bert_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    accelerator.print("Saving model and tokenizer...")
    if hasattr(model, "module"):
        model.module.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    accelerator.print(f"Model saved to {output_dir}")

    # ----- Final Evaluation on the Validation Set -----
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    accelerator.print(f"\nValidation Accuracy: {acc:.4f}")
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    accelerator.print(f"Validation Precision (macro): {precision:.4f}")
    accelerator.print(f"Validation Recall (macro):    {recall:.4f}")
    accelerator.print(f"Validation F1 Score (macro):    {f1:.4f}")

    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    accelerator.print("\nClassification Report:\n", classification_report(all_labels, all_preds, zero_division=0))
    
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path, index=True)
    accelerator.print(f"Classification report saved to {report_csv_path}")

    cm = confusion_matrix(all_labels, all_preds)
    accelerator.print("Confusion Matrix shape:", cm.shape)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, None]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, cmap="Blues", cbar=True)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.show()

    # ----- Testing on Unlabeled Data -----
    test_CSV_PATH = "data/test_without_labels.csv"
    test_dataset = TextDataset(test_CSV_PATH, tokenizer, max_length, AUGMENT=False, test=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model.eval()
    test_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            test_preds.extend(preds.cpu().numpy())
            if (batch_idx + 1) % 100 == 0:
                accelerator.print(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches...")

    accelerator.print("Test prediction completed!")

    submission_df = pd.DataFrame(
        {
            "ID": range(1, len(test_preds) + 1),
            "Label": [full_dataset.idx2label[pred] for pred in test_preds],
        }
    )
    submission_file_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")

if __name__ == "__main__":
    main()
