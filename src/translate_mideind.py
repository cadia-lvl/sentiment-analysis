import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pandas as pd
import csv
import logging
import time
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)


def initialize_model():
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(
        "mideind/nmt-doc-en-is-2022-10", src_lang="en_XX", tgt_lang="is_IS"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("mideind/nmt-doc-en-is-2022-10")

    translate = pipeline(
        "translation_XX_to_YY",
        model=model,
        tokenizer=tokenizer,
        device=device,
        src_lang="en_XX",
        tgt_lang="is_IS",
    )
    return translate, tokenizer


def machine_translate_review(review, index, tokenizer, translate):
    clean = re.compile("<.*?>")
    review = re.sub(clean, "", review)
    # Replace multiple punctuations with a single one
    review = re.sub(r"([.!?,])\1+", r"\1", review)

    # Ensure there's a space after punctuation
    review = re.sub(r"([.!?,])([^\s])", r"\1 \2", review)

    # Remove *** from the text
    review = review.replace("*", "")

    # token_length = len(tokenizer.encode(review))
    # if token_length < 1025:  # 1024 is the max length supported by the model
    try:
        translated_chunks = []
        for chunk in smart_split(review, tokenizer):
            target_seq = translate(
                chunk, src_lang="en_XX", tgt_lang="is_IS", max_length=1024
            )
            translated_chunks.append(target_seq[0]["translation_text"].strip("YY "))
        return " ".join(translated_chunks)
    except Exception as e:
        logging.warning(f"Translation failed for index: {index} with error: {e}")
        return None
    # else:
    #     logging.warning(
    #         f"Token length too long for index: {index} with length: {token_length}"
    #     )
    #     return None


def save_review(index, review, sentiment, writer, failed_writer, tokenizer, translate):
    start = time.time()
    translated_review = machine_translate_review(review, index, tokenizer, translate)

    if translated_review is None:
        failed_writer.writerow([review, sentiment])
    else:
        writer.writerow([translated_review, sentiment])
    end = time.time()
    logging.info(f"Processed index: {index} in {end-start} seconds.")


def smart_split(text, tokenizer, limit=128):
    raw_sentences = re.split("([.!?])", text)
    sentences = ["".join(x) for x in zip(raw_sentences[::2], raw_sentences[1::2])]

    if len(raw_sentences) % 2 == 1:
        sentences.append(raw_sentences[-1])

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence))
        if current_length + sentence_length <= limit:
            current_length += sentence_length
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    chunks.append(" ".join(current_chunk))

    return chunks


def main():
    try:
        translate, tokenizer = initialize_model()
    except Exception:
        print("Invalid model path")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dataset = os.path.join(parent_dir, "Datasets/IMDB-Dataset.csv")
    if not dataset or not Path(dataset).exists() or not Path(dataset).is_file():
        print("Invalid dataset path")
        return

    try:
        df = pd.read_csv(dataset)
        reviews = df["review"]
        sentiments = df["sentiment"]
    except Exception:
        print("Invalid dataset")
        return

    with open(
        os.path.join(parent_dir, "Datasets/IMDB-Dataset-MideindTranslate.csv"),
        "a",
        newline="",
        encoding="utf-8",
    ) as trans_file, open(
        os.path.join(parent_dir, "Datasets/failed-IMDB-Dataset-MideindTranslate.csv"),
        "a",
        newline="",
        encoding="utf-8",
    ) as failed_file:
        writer = csv.writer(trans_file)
        writer.writerow(["review", "sentiment"])

        failed_writer = csv.writer(failed_file)
        failed_writer.writerow(["review", "sentiment"])
        start = time.time()

        for index, (review, sentiment) in enumerate(zip(reviews, sentiments)):
            save_review(
                index, review, sentiment, writer, failed_writer, tokenizer, translate
            )

        end = time.time()
        print(f"Time taken: {end-start}")


if __name__ == "__main__":
    main()
