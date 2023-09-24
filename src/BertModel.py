from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

import os
import shutil

import tensorflow as tf
import pandas as pd

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.summary()

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file(
    fname="aclImdb_v1.tar.gz", origin=URL, untar=True, cache_dir=".", cache_subdir=""
)
# The shutil module offers a number of high-level
# operations on files and collections of files.

# Create main directory path ("/aclImdb")
main_dir = os.path.join(os.path.dirname(dataset), "aclImdb")
# Create sub directory path ("/aclImdb/train")
train_dir = os.path.join(main_dir, "train")
# Remove unsup folder since this is a supervised learning task
remove_dir = os.path.join(train_dir, "unsup")
shutil.rmtree(remove_dir)
# View the final train folder
print(os.listdir(train_dir))


# We create a training dataset and a validation
# dataset from our "aclImdb/train" directory with a 80/20 split.
train = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train", batch_size=30000, validation_split=0.2, subset="training", seed=123
)
test = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=30000,
    validation_split=0.2,
    subset="validation",
    seed=123,
)


for i in train.take(1):
    train_feat = i[0].numpy()
    train_lab = i[1].numpy()

train = pd.DataFrame([train_feat, train_lab]).T
train.columns = ["DATA_COLUMN", "LABEL_COLUMN"]
train["DATA_COLUMN"] = train["DATA_COLUMN"].str.decode("utf-8")
train.head()

for j in test.take(1):
    test_feat = j[0].numpy()
    test_lab = j[1].numpy()

test = pd.DataFrame([test_feat, test_lab]).T
test.columns = ["DATA_COLUMN", "LABEL_COLUMN"]
test["DATA_COLUMN"] = test["DATA_COLUMN"].str.decode("utf-8")
test.head()


def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(
        lambda x: InputExample(
            guid=None,  # Globally unique ID for bookkeeping, unused in this case
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN],
        ),
        axis=1,
    )

    validation_InputExamples = test.apply(
        lambda x: InputExample(
            guid=None,  # Globally unique ID for bookkeeping, unused in this case
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN],
        ),
        axis=1,
    )

    return train_InputExamples, validation_InputExamples

    train_InputExamples, validation_InputExamples = convert_data_to_examples(
        train, test, "DATA_COLUMN", "LABEL_COLUMN"
    )


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []  # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,  # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,  # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True,
        )

        input_ids, token_type_ids, attention_mask = (
            input_dict["input_ids"],
            input_dict["token_type_ids"],
            input_dict["attention_mask"],
        )

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=e.label,
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        (
            {
                "input_ids": tf.int32,
                "attention_mask": tf.int32,
                "token_type_ids": tf.int32,
            },
            tf.int64,
        ),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = "DATA_COLUMN"
LABEL_COLUMN = "LABEL_COLUMN"

InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)