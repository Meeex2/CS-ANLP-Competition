import pandas as pd
import numpy as np
import fasttext
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report

# Download and load the model
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)

# Patch the model's predict function by temporarily overriding np.array
original_predict = model.predict
def patched_predict(text, k=-1, threshold=0.0):
    # Backup the original np.array function
    original_array = np.array
    # Define a replacement that accepts the 'copy' keyword but always uses np.asarray
    def new_array(a, copy=False):
        return np.asarray(a)
    # Override np.array with the new version
    np.array = new_array
    try:
        labels, probs = original_predict(text, k=k, threshold=threshold)
    finally:
        # Restore the original np.array
        np.array = original_array
    return labels, np.asarray(probs)

model.predict = patched_predict

# Read the training data
df = pd.read_csv("data/train_submission.csv")

# Sample 10% of the data using a fixed random seed for reproducibility
sample_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Get predictions using the fastText model.
predicted_labels = []
for i, text in enumerate(sample_df["Text"]):
    if i % 100 == 0:
        print(f"Processing text {i}/{len(sample_df)}")
        if i > 0:
            print("Predicted: ",label)
            print("Actual: ",sample_df["Label"][i-1])
    label = model.predict(text)[0][0]
    # Remove common fastText label prefixes if present (e.g., '__label__')
    label = label.replace("__label__", "")
    label = label[:3]
    predicted_labels.append(label)
# Print the classification report comparing ground truth and model predictions
print(classification_report(sample_df["Label"].astype(str), [str(lbl) for lbl in predicted_labels]))

# Uncomment below to create submission file:
# submission_df = pd.DataFrame({
#     "ID": range(1, len(predicted_labels) + 1),
#     "Label": predicted_labels
# })
# submission_df.to_csv("submission.csv", index=False)