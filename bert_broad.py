# bert_broad.py
# Fine-tunes DistilBERT on the full 40k multi-category dataset
# Saves the model afterward so the app can load it instantly

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# ---------- Load data ----------
df = pd.read_csv("fake reviews dataset.csv")
df = df.dropna(subset=["text_"])

X = df["text_"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

label_map = {"OR": 0, "CG": 1}
y_train_numeric = y_train.map(label_map).tolist()
y_test_numeric = y_test.map(label_map).tolist()

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

# ---------- Tokenize ----------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=256)

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, y_train_numeric)
test_dataset = ReviewDataset(test_encodings, y_test_numeric)

# ---------- Load model ----------
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./results_broad",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    logging_steps=100,
    save_strategy="no",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print()
print("Starting training on", device, "... this will take a while, grab a coffee")
trainer.train()

# ---------- Final evaluation ----------
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

print()
print("=== FINAL RESULTS ===")
print("Accuracy:", accuracy_score(y_test_numeric, y_pred))
print()
print(classification_report(y_test_numeric, y_pred, target_names=["OR", "CG"]))
print()
print("Confusion matrix:")
print(confusion_matrix(y_test_numeric, y_pred))

# ---------- Save the trained model + tokenizer ----------
model.save_pretrained("./bert_broad_model")
tokenizer.save_pretrained("./bert_broad_model")
print()
print("Saved model + tokenizer to ./bert_broad_model")