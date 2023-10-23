import os
import tempfile
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    Trainer,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from ray import tune
from ray.tune import CLIReporter
from ray.tune.tuner import Tuner
from ray.tune.schedulers import ASHAScheduler
from transformers import Trainer
import numpy as np
from ray import train
from ray.train import get_context


RANDOM_SEED = 42


def tokenize_data(data, tokenizer, max_len=512):
    return tokenizer(
        data.tolist(), padding="max_length", truncation=True, max_length=max_len
    )


# df = pd.read_csv("Google-without-lem.csv")
df = pd.read_csv("IMDB-Dataset-Processed.csv")

df = df.sample(n=10000, random_state=RANDOM_SEED)
# df.drop(["Unnamed: 0"], axis=1, inplace=True)


def convert(sentiment):
    return 1 if sentiment == "positive" else 0


df["sentiment"] = df.sentiment.apply(convert)

# show how many positive and negative reviews we have
# print(df.sentiment.value_counts())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# df["review"] = df.review.apply(lambda x: x.replace("_NEG", ""))
torch.manual_seed(RANDOM_SEED)

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
        item["labels"] = torch.tensor(
            self.labels.iloc[idx]
        )  # ensure labels are included
        return item

    def __len__(self):
        return len(self.labels)


# Set GPU


# class CustomTrainer(Trainer):
#     def log(self, logs: dict):
#         # Get the last evaluation loss.
#         eval_loss = self.evaluate()["eval_loss"]
#         logs["eval_loss"] = eval_loss

#         # Call the parent class log method to handle the rest.
#         super().log(logs)


# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="steps",  # "steps" to evaluate each `logging_steps` or "epoch" to evaluate each epoch
#     eval_steps=1000,  # Evaluation and Save happens every 500 steps
#     num_train_epochs=10,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir="./logs",
# )
class CustomTuneReportCallback(TrainerCallback):
    def __init__(self) -> None:
        self.train_loss = 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        train.report(
            {
                "train_loss": self.train_loss,
                "eval_acc": metrics["eval_acc"],
                "eval_loss": metrics["eval_loss"],
                "eval_f1": metrics["eval_f1"],
            }
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        self.train_loss = logs.get("loss", self.train_loss)


class SaveModelOnFinishCallback(TrainerCallback):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            output_dir = os.path.join(get_context().get_trial_dir(), "final_model")
            print(f"Saving model checkpoint to {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)


def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return {
        "acc": (predictions == labels).mean(),
        "f1": f1_score(labels, predictions, average="weighted"),
    }
    # return {"acc": (np.argmax(p.predictions, axis=1) == p.label_ids).mean()}


def tune_model(config):
    # model_name = "mideind/IceBERT"
    model_name = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the training data
    train_data = tokenize_data(X_train, tokenizer)

    # Tokenize the validation data
    val_data = tokenize_data(X_val, tokenizer)

    # Tokenize the test data
    test_data = tokenize_data(X_test, tokenizer)

    train_dataset = SentimentDataset(train_data, y_train)
    val_dataset = SentimentDataset(val_data, y_val)
    test_dataset = SentimentDataset(test_data, y_test)

    log_dir = "D:\\HR\\LOKA\\sentiment-analysis\\logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    training_args = TrainingArguments(
        output_dir="D:\\HR\\LOKA\\sentiment-analysis\\results",
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_steps=500,
        evaluation_strategy="steps",
        logging_dir=log_dir,
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
        learning_rate=config["learning_rate"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            # EarlyStoppingCallback(
            #     early_stopping_patience=5, early_stopping_threshold=0.01
            # ),
            CustomTuneReportCallback(),
            SaveModelOnFinishCallback(model=model, tokenizer=tokenizer),
        ],
    )

    trainer.train()


config = {
    "num_train_epochs": tune.choice([3, 5]),
    "per_device_train_batch_size": tune.choice([8, 8]),
    "per_device_eval_batch_size": tune.choice([8, 8]),
    "warmup_steps": tune.choice([0, 500]),
    "weight_decay": tune.loguniform(0.001, 0.06),
    "learning_rate": tune.loguniform(1e-7, 5e-7),
}

scheduler = ASHAScheduler(metric="eval_loss", mode="min", max_t=10, grace_period=2)
reporter = CLIReporter(
    metric_columns=[
        "eval_acc",
        "eval_loss",
        "eval_f1",
        "training_iteration",
        "train_loss",
    ]
)

# Wrap your function with tune.with_parameters to pass the large objects
trainable = tune.with_parameters(tune_model)

tuner = Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        num_samples=10,
        scheduler=scheduler,
        max_concurrent_trials=1,
    ),
    param_space=config,
    run_config=train.RunConfig(
        progress_reporter=reporter,
        local_dir="D:/HR/LOKA/sentiment-analysis/ray_results",
        name="tune_eng5",
    ),
)

results = tuner.fit()
best_result = results.get_best_result("eval_f1", "max", "all")
best_config = best_result.config  # Get best trial's hyperparameters
best_logdir = best_result.path  # Get best trial's result directory
best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
best_metrics = best_result.metrics  # Get best trial's last results
print("Best checkpoint is", best_checkpoint)
print("Best result is", best_result)
print("Best config is", best_config)
print("Best logdir is", best_logdir)
print("Best metrics are", best_metrics)

# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=10,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.004377621058488341,
#     logging_dir="./logs",
#     logging_steps=500,
#     evaluation_strategy="steps",
#     eval_steps=500,
#     save_steps=1000,
#     load_best_model_at_end=True,  # This will ensure that the best model is loaded at the end of training
#     learning_rate=1.259717850250621e-05,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     callbacks=[
#         EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01)
#     ],
# )

# trainer.train()


# results = trainer.evaluate()
# # print(results)
# best_trial = analysis.get_best_trial("eval_acc", mode="max", scope="all")
# best_checkpoint_dir = best_trial.get_best_checkpoint(
#     best_trial.metric_analysis["eval_acc"]["max"]
# )
# model_state, optimizer_state, scheduler_state, trainer_state = torch.load(
#     os.path.join(best_checkpoint_dir, "checkpoint.pth")
# )
# model.load_state_dict(model_state)


# model.save_pretrained("./sentiment_model")
# tokenizer.save_pretrained("./sentiment_model")
