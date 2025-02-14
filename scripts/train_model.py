import torch
import numpy as np
from datasets import load_dataset, ClassLabel
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,

    EarlyStoppingCallback
)
import evaluate

#config
MODEL_NAME = "answerdotai/ModernBERT-base"
CSV_FILE = "data/train_submission.csv"  # CSV path
MAX_LENGTH = 256  # Adjust based on analysis or remove truncation to use data collator alone
BATCH_SIZE = 128
NUM_EPOCHS = 10

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data loading from CSV
dataset = load_dataset("csv", data_files=CSV_FILE)
# Note: The CSV columns are: ID,Usage,Text,Label

# use the text for model input and label for training;
# if you want to train, you need to convert labels into numeric IDs
unique_labels = list(set(dataset["train"]["Label"]))

class_feature = ClassLabel(names=unique_labels)
# Map the "Label" column to numeric ids while keeping the original text in a separate field if needed
dataset = dataset.map(lambda ex: {"Label": class_feature.str2int(ex["Label"])}, batched=True)
dataset = dataset.cast_column("Label", class_feature)

# rename "Label" to "labels" for Trainer compatibility
dataset = dataset.rename_column("Label", "labels")
# Optionally, you can remove unnecessary columns like "ID" and "Usage"
dataset = dataset.remove_columns(["ID", "Usage"])

#train test split
split_dataset = dataset["train"].train_test_split(test_size=0.2, stratify_by_column="labels")

#tokenization using "Text" column
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["Text"],
        padding=False,  # or remove truncation/limit max_length if using dynamic padding fully
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_datasets = split_dataset.map(tokenize_function, batched=True)

# ... rest of your code remains the same ...

#class balancing
train_labels = tokenized_datasets["train"]["labels"]
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Custom Trainer for class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.class_weights_on_device = None

    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device
        labels = inputs.pop("labels").to(device)
        if self.class_weights_on_device is None:
            self.class_weights_on_device = self.class_weights.to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights_on_device)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    

#load the model
#the %%forsequenceclassification class extends the checkpoints with a softmax classification head with num_labels neurons

model = ModernBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label={i: lbl for i, lbl in enumerate(unique_labels)},
    label2id={lbl: i for i, lbl in enumerate(unique_labels)}
).to(device)



#experiment N 1
training_args = TrainingArguments(
    output_dir="./results/exp2",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=200,
    eval_steps=20,
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,
    # lr_scheduler_type = "cosine",
    # warmup_ratio = 0.1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    report_to="none",
    logging_steps=50,
    gradient_accumulation_steps = 10,
    logging_dir="./train_logs/exp2"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "macro_f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    }


# Initialize Trainer
trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_datasets["train"],
     eval_dataset=tokenized_datasets["test"],
     compute_metrics=compute_metrics,
     data_collator=data_collator,
     callbacks=[EarlyStoppingCallback(early_stopping_patience=20, early_stopping_threshold=0.001)]
)

'''
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)]
)
'''

#training
trainer.train()
trainer.save_model("./checkpoints/exp2")
tokenizer.save_pretrained("./checkpoints/exp2")
