from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pandas as pd
import csv
import logging
import time

device = torch.cuda.current_device() if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(
    "../nmt-doc-en-is-2022-10", src_lang="en_XX", tgt_lang="is_IS"
)

model = AutoModelForSeq2SeqLM.from_pretrained("../nmt-doc-en-is-2022-10")

translate = pipeline(
    "translation_XX_to_YY",
    model=model,
    tokenizer=tokenizer,
    device=device,
    src_lang="en_XX",
    tgt_lang="is_IS",
)

logging.basicConfig(level=logging.INFO)


def machine_translate_review(review, index):
    review = review.replace("<br />", "")
    token_length = len(tokenizer.encode(review))
    if token_length < 1025:  # 1024 is the max length supported by the model
        try:
            target_seq = translate(
                review, src_lang="en_XX", tgt_lang="is_IS", max_length=1024
            )
            return target_seq[0]["translation_text"].strip("YY ")
        except Exception as e:
            logging.warning(f"Translation failed for index: {index} with error: {e}")
            return None
    else:
        logging.warning(
            f"Token length too long for index: {index} with length: {token_length}"
        )
        return None


def save_review(index, review, sentiment, writer, failed_writer):
    start = time.time()
    translated_review = machine_translate_review(review, index)

    if translated_review is None:
        failed_writer.writerow([review, sentiment])
    else:
        writer.writerow([translated_review, sentiment])
    end = time.time()
    logging.info(f"Processed index: {index} in {end-start} seconds.")


df = pd.read_csv("IMDB Dataset copy.csv")
reviews = df["review"]
sentiments = df["sentiment"]

with open(
    "IMDB-Dataset-MideindTranslate.csv", "a", newline="", encoding="utf-8"
) as trans_file, open(
    "failed-IMDB-Dataset-MideindTranslate.csv", "a", newline="", encoding="utf-8"
) as failed_file:
    writer = csv.writer(trans_file)
    writer.writerow(["review", "sentiment"])

    failed_writer = csv.writer(failed_file)
    failed_writer.writerow(["review", "sentiment"])
    start = time.time()

    for index, (review, sentiment) in enumerate(zip(reviews, sentiments)):
        save_review(index, review, sentiment, writer, failed_writer)

    end = time.time()
    print(f"Time taken: {end-start}")
