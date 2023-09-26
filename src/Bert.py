from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

# Load Dataset
print("Loading data...")
df = pd.read_csv("../IMDB-Dataset.csv")
texts = df["review"].tolist()
labels = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0).tolist()

# Print GPU info
# print("GPU info:")
# print(torch.cuda.get_device_name(0))
# print(f"Is available: {torch.cuda.is_available()}")


# Train-Test Split
print("Splitting data...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2
)

# Tokenization
print("Tokenizing data...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


# Dataset class
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


print("Preparing dataset...")
train_dataset = IMDbDataset(train_encodings, train_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

# Initialize TensorBoard
print("Preparing dataset...")
writer = SummaryWriter()

# TrainingArguments & Trainer
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    report_to="tensorboard",
    logging_steps=10,
    num_train_epochs=3,
)

print("Loading model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
print("Starting training...")
trainer.train()

# Save Model
print("Saving trained model...")
model.save_pretrained("trained_model")

# Close TensorBoard writer
print("Closing TensorBoard...")
writer.close()

print("Training completed.")
