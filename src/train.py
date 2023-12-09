from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    Trainer,
    TrainingArguments,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer
import numpy as np

from process import TextNormalizer

RANDOM_SEED = 42
EPOCHS = 4
LEARNING_RATE = 1e-6
BATCH_SIZE = 8

model_name = "mideind/IceBERT"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


def tokenize_data(data, tokenizer, max_len=512):
    print(len(data))
    return tokenizer(
        data.tolist(), padding="max_length", truncation=True, max_length=max_len
    )


# df = pd.read_csv("IMDB-Dataset-GoogleTranslate.csv")
df = pd.read_csv("IMDB-Dataset-GoogleTranslate.csv")
# df = pd.read_csv("IMDB-Dataset.csv")


def convert(sentiment):
    return 1 if sentiment == "positive" else 0


tn = TextNormalizer(None)
df["sentiment"] = df.sentiment.apply(convert)
df["review"] = df.review.apply(tn.remove_noise)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_temp, y_train, y_temp = train_test_split(
    df["review"], df["sentiment"], test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return {
        "acc": (predictions == labels).mean(),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

# Tokenize the training data
train_data = tokenize_data(X_train, tokenizer)

# Tokenize the validation data
val_data = tokenize_data(X_val, tokenizer)

# Tokenize the test data
test_data = tokenize_data(X_test, tokenizer)

train_dataset = SentimentDataset(train_data, y_train)
val_dataset = SentimentDataset(val_data, y_val)
test_dataset = SentimentDataset(test_data, y_test)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

total_steps = len(train_dataset) * EPOCHS
print(total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_training_steps=total_steps, num_warmup_steps=0
)

log_dir = "./logs"

training_args = TrainingArguments(
    output_dir="./results/Icebert-google-batch8-test",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=log_dir,
    load_best_model_at_end=True,
    learning_rate=LEARNING_RATE,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[
        # EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01),
    ],
    optimizers=(optimizer, scheduler),
    tokenizer=tokenizer,
)


trainer.train()

results = trainer.evaluate(test_dataset)
print("test results:", results)

model.save_pretrained("./Icebert-google-batch8-test-model")
tokenizer.save_pretrained("./Icebert-google-batch8-test-model")
