import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# =========================
# 🔥 IMPORTANT FIXES
# =========================
torch.set_num_threads(1)  # prevent crash / memory overload

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# STEP 1: LOAD DATA
# =========================
df = pd.read_csv("train.csv")

df = df.rename(columns={
    "Statement": "text",
    "Label": "label"
})

df = df.dropna()
df.reset_index(drop=True, inplace=True)

df['label'] = df['label'].astype(int)

print("Dataset Loaded:", df.shape)

# =========================
# STEP 2: TRAIN-TEST SPLIT
# =========================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# =========================
# STEP 3: TOKENIZER
# =========================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=128   # 🔥 reduced from 256 → saves memory
)

val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding=True,
    max_length=128
)

# =========================
# STEP 4: DATASET CLASS
# =========================
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# =========================
# STEP 5: LOAD MODEL
# =========================
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

model.to(device)

# =========================
# STEP 6: TRAINING SETUP
# =========================
training_args = TrainingArguments(
    output_dir='./results',

    # 🔥 REDUCED FOR STABILITY
    num_train_epochs=1,                 # was 2
    per_device_train_batch_size=2,      # was 8 → major fix
    per_device_eval_batch_size=2,

    # 🔥 MEMORY SAFE SETTINGS
    dataloader_pin_memory=False,
    save_strategy="no",                 # avoid extra memory usage
    logging_steps=50,

    report_to="none"                    # disables tensorboard
)

# =========================
# STEP 7: TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# =========================
# STEP 8: TRAIN
# =========================
print("🚀 Training started...")
trainer.train()

# =========================
# STEP 9: SAVE MODEL
# =========================
model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_model")

print("✅ Training complete. Model saved in ./bert_model")