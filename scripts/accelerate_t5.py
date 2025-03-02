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
        precision_recall_fscore_support,
    )
    from transformers import AutoTokenizer, T5EncoderModel
    from peft import LoraConfig, TaskType, get_peft_model
    import torch.nn as nn

    # -------------------------------
    # 1. Custom Model: MADLAD-T5 Encoder + Classification Head
    # -------------------------------
    class T5ForSequenceClassification(nn.Module):
        def __init__(self, model_name, num_labels):
            super().__init__()
            # Load only the encoder portion of T5
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            # Add the encoder's config to satisfy PEFT's requirements
            self.config = self.encoder.config
            self.dropout = nn.Dropout(0.1)
            # Classification head projecting from encoder hidden size to num_labels
            self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)

        def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, 
            output_attentions=False, output_hidden_states=False, **kwargs):
            # Pass inputs_embeds if provided; otherwise use input_ids
            if inputs_embeds is None:
                outputs = self.encoder(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs
                )
            else:
                outputs = self.encoder(
                    inputs_embeds=inputs_embeds, 
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs
                )
            
            # Mean pooling over sequence length
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            
            # Create a simple output class that mimics transformers ModelOutput
            output = type("ModelOutput", (), {"loss": loss, "logits": logits})
            return output

    # -------------------------------
    # 2. Dataset Definition
    # -------------------------------
    class TextDataset(Dataset):
        def __init__(self, CSV_PATH, tokenizer, MAX_SEQ_LENGTH, AUGMENT=False, test=False):
            print(f"Loading data from {CSV_PATH} ...")
            self.data = pd.read_csv(CSV_PATH)
            # Data cleaning functions are commented out in original code
            # self.data["Text"] = self.data["Text"].apply(remove_links_and_tags)
            # self.data["Text"] = self.data["Text"].apply(filter_majority_script)
            self.tokenizer = tokenizer
            self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
            self.AUGMENT = AUGMENT
            self.test = test  # In test mode, labels are not required
            if not self.test:
                self.data = self.data[self.data["Label"].notnull()]
                unique_labels = sorted(self.data["Label"].unique())
                self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
                self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            text = row["Text"]
            # Tokenize text (without padding here; we'll use a collate_fn later)
            token_ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=True,max_length = self.MAX_SEQ_LENGTH)
            if self.AUGMENT and len(token_ids) > self.MAX_SEQ_LENGTH:
                start_idx = random.randint(0, len(token_ids) - self.MAX_SEQ_LENGTH)
                token_ids = token_ids[start_idx : start_idx + self.MAX_SEQ_LENGTH]
            else:
                token_ids = token_ids[: self.MAX_SEQ_LENGTH]
            attention_mask = [1] * len(token_ids)
            item = {"input_ids": token_ids, "attention_mask": attention_mask}
            if not self.test:
                label_idx = self.label2idx[row["Label"]]
                item["labels"] = label_idx
            return item

    # Utility function to remove 'module.' prefix in state dict keys
    def remove_prefix_and_handle_classifier(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        return new_state_dict

    # Utility to plot confusion matrix
    def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues):
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap=cmap)
        plt.title(title)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()  # Close the figure to prevent display in headless environments

    # -------------------------------
    # 3. Settings and Hyperparameters
    # -------------------------------
    model_name = "google/madlad400-3b-mt"  # MADLAD-T5 model for machine translation (encoder part used here)
    csv_path = "data/train_submission.csv"
    output_dir = "./lora_madlad_finetuned"
    max_length = 128
    augment = False
    

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=2,mixed_precision="fp16")

    num_train_epochs = 40
    learning_rate = 2e-3
    weight_decay = 0.01
    per_device_train_batch_size = 32
    per_device_eval_batch_size = 32
    logging_steps = 12

    accelerator.print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    accelerator.print("Instantiating the full dataset...")
    full_dataset = TextDataset(csv_path, tokenizer, max_length, AUGMENT=augment)
    num_labels = len(full_dataset.label2idx)
    accelerator.print(f"Number of labels: {num_labels}")

    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    accelerator.print(f"Total dataset size: {dataset_size}, Training size: {train_size}, Validation size: {val_size}")
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Use the tokenizer's pad method in the collate_fn for dynamic padding
    def collate_fn(batch):
        return tokenizer.pad(batch, return_tensors="pt")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # -------------------------------
    # 4. Load the Model and Wrap with LoRA
    # -------------------------------
    accelerator.print("Loading model...")
    model = T5ForSequenceClassification(model_name, num_labels)
    
    # Configure LoRA for T5 - target different modules based on T5's architecture
    # 6. More memory-efficient LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=4,            # Reduce LoRA rank from 8 to 4
        lora_alpha=16,  # Adjust scaling factor
        lora_dropout=0.1,
        target_modules=["q", "k", "v"],  # Only target attention modules
    )
    model = get_peft_model(model, lora_config)
    accelerator.print("LoRA-modified model loaded.")
    model.print_trainable_parameters()  # Add this to see trainable vs total parameters

    # Use accelerator to detect appropriate device
    device = accelerator.device
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Number of training steps
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    # LR scheduler with warmup and cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5 * num_update_steps_per_epoch,  # Restart every 5 epochs 
        T_mult=2,  # Double period after each restart
        eta_min=1e-5  # Minimum learning rate
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    best_val_accuracy = 0.0
    global_step = 0

    # -------------------------------
    # 5. Training Loop
    # -------------------------------
    for epoch in range(num_train_epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"], 
                    labels=batch.get("labels")
                )
                loss = outputs.loss
                
                # Backward pass with accelerator
                accelerator.backward(loss)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Track loss
                running_loss += loss.item()
                global_step += 1
                
                # Log progress
                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    accelerator.print(f"Epoch [{epoch+1}/{num_train_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                    running_loss = 0.0

        # ----- Validation at the End of Each Epoch -----
        model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"], 
                    labels=batch["labels"]
                )
                eval_loss += outputs.loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                # Gather predictions and labels across processes
                preds = accelerator.gather(preds)
                gathered_labels = accelerator.gather(batch["labels"])
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(gathered_labels.cpu().numpy())
                num_batches += 1
                
        avg_eval_loss = eval_loss / num_batches
        val_accuracy = accuracy_score(all_labels, all_preds)
        accelerator.print(f"Epoch [{epoch+1}/{num_train_epochs}] Validation Loss: {avg_eval_loss:.4f} | Accuracy: {val_accuracy:.4f}")

        # Save if best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            # Save best model checkpoint
            accelerator.print(f"New best model with accuracy: {val_accuracy:.4f}")
            
            # Get unwrapped model
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Save checkpoint using PEFT's save_pretrained
            best_model_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            unwrapped_model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            
            # Save additional metadata
            with open(os.path.join(best_model_dir, "best_val_accuracy.txt"), "w") as f:
                f.write(f"{best_val_accuracy:.4f}")

        # Save regular checkpoint (optional - can be removed if disk space is a concern)
        if (epoch + 1) % 2 == 0 or epoch == num_train_epochs - 1:  # Save every 5 epochs and last epoch
            checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            accelerator.print(f"Saving checkpoint to {checkpoint_dir}")
            
            # Get unwrapped model
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Save checkpoint using PEFT's save_pretrained
            unwrapped_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    # ----- Load Best Model for Final Evaluation -----
    accelerator.print("\nLoading best model for final evaluation...")
    best_model_dir = os.path.join(output_dir, "best_model")
    
    # Load best model with PEFT
    best_model = T5ForSequenceClassification(model_name, num_labels)
    best_model = get_peft_model(best_model, lora_config)
    best_model = accelerator.prepare(best_model)
    
    # Try to load best model weights
    try:
        best_model.load_state_dict(torch.load(os.path.join(best_model_dir, "adapter_model.bin")))
        accelerator.print("Successfully loaded best model weights")
    except Exception as e:
        accelerator.print(f"Warning: Could not load best model state: {str(e)}")
        accelerator.print("Continuing with current model state")
        best_model = model  # Use current model if loading fails

    # ----- Final Evaluation on Validation Set -----
    best_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = best_model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                labels=batch["labels"]
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            # Gather predictions and labels across processes
            preds = accelerator.gather(preds)
            gathered_labels = accelerator.gather(batch["labels"])
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(gathered_labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    
    # Print final metrics
    accelerator.print(f"\nFinal Validation Metrics:")
    accelerator.print(f"Accuracy: {acc:.4f}")
    accelerator.print(f"Precision (macro): {precision:.4f}")
    accelerator.print(f"Recall (macro): {recall:.4f}")
    accelerator.print(f"F1 Score (macro): {f1:.4f}")
    
    # Generate and save classification report
    if accelerator.is_main_process:
        report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        accelerator.print("\nClassification Report:\n", classification_report(all_labels, all_preds, zero_division=0))
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv_path = os.path.join(output_dir, "classification_report.csv")
        report_df.to_csv(report_csv_path, index=True)
        accelerator.print(f"Classification report saved to {report_csv_path}")
        
        # Generate and save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        accelerator.print("Confusion Matrix shape:", cm.shape)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, None]
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, cmap="Blues", cbar=True)
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()  # Close the figure to prevent display in headless environments

    # ----- Testing on Unlabeled Data -----
    test_CSV_PATH = "data/test_without_labels.csv"  # Fixed path
    
    # Check if test file exists
    if not os.path.exists(test_CSV_PATH):
        accelerator.print(f"Warning: Test file {test_CSV_PATH} not found. Skipping test prediction.")
    else:
        test_dataset = TextDataset(test_CSV_PATH, tokenizer, max_length, AUGMENT=False, test=True)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_dataloader = accelerator.prepare(test_dataloader)
        
        best_model.eval()
        test_preds = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                outputs = best_model(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                # Gather predictions across processes
                preds = accelerator.gather(preds)
                test_preds.extend(preds.cpu().numpy())
                
                if (batch_idx + 1) % 100 == 0:
                    accelerator.print(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches...")
        
        if accelerator.is_main_process:
            accelerator.print("Test prediction completed!")
            submission_df = pd.DataFrame(
                {
                    "ID": range(1, len(test_preds) + 1),
                    "Label": [full_dataset.idx2label[pred] for pred in test_preds],
                }
            )
            submission_file_path = os.path.join(output_dir, "submission.csv")
            submission_df.to_csv(submission_file_path, index=False)
            accelerator.print(f"Submission file saved to {submission_file_path}")

if __name__ == "__main__":
    main()